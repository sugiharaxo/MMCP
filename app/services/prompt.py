"""Implements LLMPrompt protocol for end-to-end BAML processing.

This layer handles the complete BAML pipeline:
1. Convert Pydantic schemas to BAML TypeBuilder
2. Build ClientRegistry with provider configuration and caching
3. Call BAML function directly (prompt → transport → SAP)
4. Return structured FinalResponse or ToolCall

Has zero knowledge of networking or API keys - BAML handles everything.
"""

import hashlib
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, get_args, get_origin

from baml_py import ClientRegistry
from pydantic import BaseModel

from app.core.errors import AgentLogicError, map_provider_error
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

    def __init__(self):
        # Cache ClientRegistry instances by user config hash to avoid HTTP client thrashing
        # This prevents creating new HTTP clients (with TCP handshake) on every request
        #
        # FUTURE: Consider adding TTL (Time To Live) or size limit if application runs
        # for extended periods without restart to prevent memory leaks from stale configs.
        # For typical web server scenarios, this is not immediately necessary.
        self._client_registry_cache: dict[str, Any] = {}

    def _build_client_registry(self, user_settings: Any) -> Any:
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
        from app.core.config import parse_llm_model_string

        # Create stable hash for this user's configuration
        config_id = hashlib.sha256(
            f"{user_settings.llm_model}{user_settings.llm_api_key}{user_settings.llm_base_url}".encode()
        ).hexdigest()

        # Return cached registry if it exists
        if config_id in self._client_registry_cache:
            return self._client_registry_cache[config_id]

        # Build new registry
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
        user_input: str | None = None,
        history: list[dict[str, Any]] | None = None,
        user_settings: Any | None = None,
    ) -> FinalResponse | ToolCall:
        """
        Single method that uses BAML end-to-end: prompt → transport → SAP.

        Replaces the old compile_prompt() + transport.send_message() + parse_response() flow.

        Args:
            tool_infos: List of ToolInfo containing tool metadata
            context_data: Raw context dictionary from context providers
            user_input: Optional new user message (None if in history)
            history: Optional conversation history
            user_settings: UserSettings instance for LLM configuration (defaults to global)

        Returns:
            FinalResponse | ToolCall: Parsed response from BAML. FinalResponse contains
            the agent's final answer, ToolCall contains the tool name and arguments
            for execution.

        Raises:
            AgentLogicError: For validation/format errors (non-retryable)
            ProviderError: For network/timeout/rate limit errors (may be retryable)
        """
        from app.core.config import user_settings as default_settings
        from baml_client import b
        from baml_client.type_builder import TypeBuilder

        # Use provided user_settings or fall back to global
        settings = user_settings or default_settings

        # Determine if we need to merge system instructions (for Gemma models)
        _, model_name = settings.llm_model.split(":", 1)
        merge_system = model_name.lower().find("gemma") != -1

        # Build TypeBuilder from Pydantic schemas
        # NOTE: If tool_infos is static (same for all users), consider caching TypeBuilder
        # in __init__ to avoid re-compilation on every request
        tb = TypeBuilder()

        # Map Pydantic tools to BAML classes (existing logic from compile_prompt)
        for tool_info in tool_infos:
            schema = tool_info.schema
            cls = tb.add_class(schema.__name__)

            # Map existing Pydantic fields
            for field_name, field_info in schema.model_fields.items():
                baml_type = self._map_pydantic_to_baml(tb, field_info.annotation, field_info)
                prop = cls.add_property(field_name, baml_type)

                # Preserve field descriptions
                if field_info.description:
                    prop.description(field_info.description)

            # Add rationale for EXTERNAL tools
            if tool_info.classification == "EXTERNAL":
                cls.add_property("rationale", tb.string()).description(
                    "User-facing explanation: Why is this action necessary?"
                )

        # Build tool descriptions (same as before)
        tools_description = self._build_tool_descriptions(tool_infos)

        # Prepare context and history
        context: dict[str, Any] = {**context_data}
        baml_history: list[Any] = history if history else []

        # Get cached ClientRegistry (avoids HTTP client thrashing)
        client_registry = self._build_client_registry(settings)

        # Call BAML function directly - BAML handles everything
        try:
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
        except Exception as e:
            # Map BAML exceptions to MMCP error hierarchy
            if baml_errors:
                if isinstance(e, (baml_errors.BamlValidationError, baml_errors.BamlClientError)):
                    # These are typically non-retryable (format/validation issues)
                    raise AgentLogicError(
                        f"BAML validation/client error: {str(e)}",
                        retryable=False,
                    ) from e
                # Network/timeout errors should be mapped to ProviderError
                if isinstance(e, (TimeoutError, ConnectionError)):
                    raise map_provider_error(e) from e

            # Fallback: map generic exceptions
            raise map_provider_error(e) from e

        return result

    def _map_pydantic_to_baml(self, tb: Any, field_type: type, field_info: Any) -> Any:
        """
        Recursively convert Pydantic types to BAML types.

        Handles: primitives, Union/Optional, list, dict, Enum, nested BaseModel.
        Based on cognee's approach, adapted for MMCP.
        """

        origin = get_origin(field_type)
        args = get_args(field_type)

        # Literal (treated as union of string literals)
        if origin is Literal:
            # Literal["a", "b"] -> union of string values
            literal_values = args
            if all(isinstance(v, str) for v in literal_values):
                # All string literals - create union of strings
                return tb.union([tb.string() for _ in literal_values])
            # Mixed types - create union of mapped types
            return tb.union(
                [self._map_pydantic_to_baml(tb, type(v), field_info) for v in literal_values]
            )

        # Union / Optional
        # Handle both typing.Union (old syntax) and types.UnionType (PEP 604: int | None)
        if origin is typing.Union or origin is types.UnionType:
            non_none_args = [t for t in args if t is not type(None)]
            if len(args) == 2 and len(non_none_args) == 1:
                # Optional[T] -> T.optional()
                return self._map_pydantic_to_baml(tb, non_none_args[0], field_info).optional()
            # Union[A, B, ...] -> union(A, B, ...)
            return tb.union([self._map_pydantic_to_baml(tb, t, field_info) for t in args])

        # List
        if origin is list:
            inner_type = args[0] if args else str
            return self._map_pydantic_to_baml(tb, inner_type, field_info).list()

        # Dict
        if origin is dict:
            key_type, value_type = args if args else (str, str)
            # BAML maps only allow 'str' or Enum as keys
            if key_type is not str and not (
                isinstance(key_type, type) and issubclass(key_type, Enum)
            ):
                raise ValueError(f"BAML maps only allow 'str' or Enum as keys, got {key_type}")
            return tb.map(
                self._map_pydantic_to_baml(tb, key_type, field_info),
                self._map_pydantic_to_baml(tb, value_type, field_info),
            )

        # Enum
        if isinstance(field_type, type) and issubclass(field_type, Enum):
            enum_builder = tb.add_enum(field_type.__name__)
            for member in field_type:
                enum_builder.add_value(member.name)
            return enum_builder.type()

        # Nested Pydantic Model
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            try:
                nested_class = tb.add_class(field_type.__name__)
                # Recursively map nested model fields
                for nested_field_name, nested_field_info in field_type.model_fields.items():
                    nested_type = self._map_pydantic_to_baml(
                        tb, nested_field_info.annotation, nested_field_info
                    )
                    prop = nested_class.add_property(nested_field_name, nested_type)
                    if nested_field_info.description:
                        prop.description(nested_field_info.description)
            except ValueError:
                # Class already exists, get it
                nested_class = tb._tb.class_(field_type.__name__)

            # Return the nested class type
            if hasattr(nested_class, "field"):
                return nested_class.field()
            return nested_class.type()

        # Primitives
        primitive_map = {
            str: tb.string(),
            int: tb.int(),
            float: tb.float(),
            bool: tb.bool(),
        }
        if field_type in primitive_map:
            return primitive_map[field_type]

        raise ValueError(f"Unsupported type for BAML mapping: {field_type}")

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
        pass
