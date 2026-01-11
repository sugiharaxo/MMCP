"""Implements LLMPrompt protocol for end-to-end BAML processing.

This layer handles the complete BAML pipeline:
1. Convert Pydantic schemas to BAML TypeBuilder
2. Build ClientRegistry with provider configuration and caching
3. Call BAML function directly (prompt → transport → SAP)
4. Return structured FinalResponse or ToolCall

Has zero knowledge of networking or API keys - BAML handles everything.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from app.core.config import UserSettings, parse_llm_model_string
from app.core.errors import AgentLogicError, map_provider_error

if TYPE_CHECKING:
    from app.services.type_mapper import TypeMapper
    from baml_client.types import FinalResponse, ToolCall

# Import BAML exceptions for proper error handling
try:
    from baml_py import baml_exceptions as baml_errors
except ImportError:
    # Fallback if baml_exceptions module structure differs
    baml_errors = None


@dataclass
class ToolInfo:
    """Complete tool metadata for prompt compilation."""

    name: str
    description: str
    schema: type[BaseModel]
    classification: Literal["INTERNAL", "EXTERNAL"] = "EXTERNAL"


class PromptService:
    """Prompt service implementing LLMPrompt protocol.

    Uses BAML end-to-end: TypeBuilder + ClientRegistry caching + direct BAML calls.
    Maintains HTTP Keep-Alive through ClientRegistry caching to avoid connection overhead.
    Imports are lazy-loaded inside methods to maintain <50MB idle target.
    """

    def __init__(self, type_mapper: "TypeMapper"):
        self._type_mapper = type_mapper
        # Cache ClientRegistry instances by user config hash to avoid HTTP client thrashing
        # This prevents creating new HTTP clients (with TCP handshake) on every request
        self._client_registry_cache: dict[str, Any] = {}

    def _build_client_registry(self, user_settings: UserSettings) -> Any:
        """
        Build BAML ClientRegistry from user_settings with caching.

        CRITICAL: ClientRegistry contains HTTP clients that should be reused to maintain
        HTTP Keep-Alive connections. Creating new registries per request causes:
        - Full TCP handshake + TLS negotiation on every request (hundreds of ms latency)
        - File descriptor exhaustion under load

        Args:
            user_settings: UserSettings instance with llm_model, llm_api_key, llm_base_url

        Returns:
            Cached ClientRegistry configured for the user's provider/model
        """

        # Create stable hash for this user's configuration
        config_id = hashlib.sha256(
            f"{user_settings.llm_model}{user_settings.llm_api_key}{user_settings.llm_base_url}".encode()
        ).hexdigest()

        # Return cached registry if it exists
        if config_id in self._client_registry_cache:
            return self._client_registry_cache[config_id]

        # Build new registry (lazy load heavy import)
        from baml_py import ClientRegistry

        provider, model = parse_llm_model_string(user_settings.llm_model)
        registry = ClientRegistry()

        # Build options dict for add_llm_client
        options = {"model": model}
        if user_settings.llm_api_key:
            options["api_key"] = user_settings.llm_api_key
        if user_settings.llm_base_url:
            options["base_url"] = user_settings.llm_base_url

        registry.add_llm_client(name="user_client", provider=provider, options=options)
        registry.set_primary("user_client")

        # Cache for reuse
        self._client_registry_cache[config_id] = registry
        return registry

    async def call_llm(
        self,
        tool_infos: list[ToolInfo],
        context_data: dict[str, Any],
        user_settings: UserSettings,
        user_input: str | None = None,
        history: list[dict[str, Any]] | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> "FinalResponse | ToolCall":
        """
        Execute an end-to-end BAML prompt flow and return either a final response or a tool invocation.
        
        Parameters:
            tool_infos (list[ToolInfo]): Metadata and schemas for available tools.
            context_data (dict[str, Any]): Context provided by context sources to include in the prompt.
            user_settings (UserSettings): LLM configuration (model, keys, provider-specific options).
            user_input (str | None): Optional new user message; if None, conversation history is expected to contain the latest user turn.
            history (list[dict[str, Any]] | None): Optional conversation history formatted for BAML.
            stream_callback (Callable[[str], None] | None): If provided, receives JSON chunk strings from the agent as they arrive for real-time streaming UI updates.
        
        Returns:
            FinalResponse | ToolCall: `FinalResponse` when the agent produced a final textual answer; `ToolCall` when the agent selected a tool and provided arguments for execution.
        
        Raises:
            AgentLogicError: For validation or formatting errors reported by BAML (non-retryable).
            ProviderError: For network, timeout, or provider-level errors (may be retryable).
        """
        # Lazy load heavy BAML client types only when needed
        from baml_client import b

        provider, model_name = parse_llm_model_string(user_settings.llm_model)
        merge_system = any(
            model_keyword.lower() in model_name.lower()
            for model_keyword in user_settings.models_requiring_system_merge
        )

        # Build TypeBuilder from Pydantic schemas using TypeMapper
        tb = self._type_mapper.build_type_builder(tool_infos)

        # Build tool descriptions (same as before)
        tools_description = self._build_tool_descriptions(tool_infos)

        # Prepare context and history
        context: dict[str, Any] = {**context_data}
        baml_history: list[Any] = history if history else []

        # Get cached ClientRegistry (avoids HTTP client thrashing)
        client_registry = self._build_client_registry(user_settings)

        # Call BAML function - streaming if callback provided, non-streaming otherwise
        try:
            if stream_callback:
                # Streaming mode: use BAML's built-in streaming with on_tick callback
                result = await b.UniversalAgent(
                    user_input=user_input or "",
                    tools=tools_description,
                    context=context,
                    history=baml_history,
                    merge_system=merge_system,
                    baml_options={
                        "type_builder": tb,
                        "client_registry": client_registry,
                        "on_tick": lambda chunk_json, function_log: stream_callback(chunk_json),
                    },
                )
            else:
                # Non-streaming mode: direct call
                result = await b.UniversalAgent(
                    user_input=user_input or "",
                    tools=tools_description,
                    context=context,
                    history=baml_history,
                    merge_system=merge_system,
                    baml_options={
                        "type_builder": tb,
                        "client_registry": client_registry,
                    },
                )
        except (TimeoutError, ConnectionError) as e:
            # Network/timeout errors should be mapped to ProviderError
            raise map_provider_error(e) from e
        except Exception as e:
            # Map BAML exceptions to MMCP error hierarchy
            if baml_errors and isinstance(
                e, (baml_errors.BamlValidationError, baml_errors.BamlClientError)
            ):
                # These are typically non-retryable (format/validation issues)
                raise AgentLogicError(
                    f"BAML validation/client error: {str(e)}",
                    retryable=False,
                ) from e

            # Fallback: map generic exceptions
            raise map_provider_error(e) from e

        return result

    def _build_tool_descriptions(self, tool_infos: list[ToolInfo]) -> str:
        """
        Build human-readable tool descriptions using actual tool names and metadata.

        Uses explicit labels (Tool:, Type:, Description:, Arguments:) for better
        LLM attention, especially for smaller models trained on code.
        """
        tool_descriptions = []

        for tool_info in tool_infos:
            schema = tool_info.schema
            json_schema = schema.model_json_schema()

            fields_info = []
            # 1. Standard Fields from Pydantic schema
            if "properties" in json_schema:
                for field_name, field_def in json_schema["properties"].items():
                    field_type = field_def.get("type", "unknown")
                    # Handle enum types (show as union)
                    if "enum" in field_def:
                        enum_values = " | ".join(field_def["enum"])
                        field_type = enum_values

                    description = field_def.get("description", "")
                    required = field_name in json_schema.get("required", [])
                    field_marker = "(required)" if required else "(optional)"

                    field_line = f"  - {field_name}: {field_type} {field_marker}"
                    if description:
                        field_line += f" - {description}"
                    fields_info.append(field_line)

            # 2. DYNAMIC INJECTION: Add rationale to the text signature for EXTERNAL tools
            # The LLM doesn't need to know it's "EXTERNAL", it just sees the required rationale field.
            if tool_info.classification == "EXTERNAL":
                fields_info.append(
                    "  - rationale: string (required) - User-facing explanation: Why is this action necessary?"
                )

            # 3. Clean Signature (Removed "Type: {classification}")
            # Agent doesn't need to know about protocol-level classifications
            tool_desc = (
                f"Tool: {tool_info.name}\n"
                f"Description: {tool_info.description}\n"
                f"Arguments:\n" + "\n".join(fields_info)
            )
            tool_descriptions.append(tool_desc)

        return "\n\n".join(tool_descriptions)

    async def close(self) -> None:
        """Clean up any resources."""
        # Explicitly close cached HTTP clients before clearing the cache
        for registry in self._client_registry_cache.values():
            if hasattr(registry, "close"):
                await registry.close()
        self._client_registry_cache.clear()