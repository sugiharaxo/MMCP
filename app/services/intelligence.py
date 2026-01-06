"""Intelligence Layer: Implements LLMIntelligence protocol for prompt compilation and parsing.

This layer handles:
1. Compile: Take Pydantic models from Plugins and convert them into formatted prompts
2. Parse: Take raw strings from Transport and return structured data

Has zero knowledge of networking or API keys.

Currently implements a dummy service. BAML TypeBuilder integration will be added in a future phase.
"""

from typing import Any

from pydantic import BaseModel

from app.agent.schemas import FinalResponse
from app.core.logger import logger


class IntelligenceService:
    """Intelligence service implementing LLMIntelligence protocol.

    Dummy implementation: Returns simple formatted strings for compile,
    and hardcoded FinalResponse for parse.
    BAML TypeBuilder is imported lazily inside methods (not at module level).
    """

    def __init__(self) -> None:
        """Initialize the intelligence service."""
        self._baml_client = None
        self._type_builder = None

    async def _get_baml_client(self) -> Any:
        """Lazy initialization of BAML client (not used in dummy mode)."""
        if self._baml_client is None:
            try:
                # BAML client initialization - lazy import when actually needed
                # import baml_py  # noqa: F401
                logger.debug("BAML client available (dummy mode active)")
                self._baml_client = "baml_shell"  # Placeholder
            except ImportError:
                logger.debug("BAML package not installed (dummy mode active)")
                self._baml_client = "baml_shell"

        return self._baml_client

    async def _get_type_builder(self) -> Any:
        """Lazy initialization of BAML TypeBuilder (not used in dummy mode)."""
        if self._type_builder is None:
            try:
                # BAML TypeBuilder initialization - lazy import when actually needed
                # import baml_py  # noqa: F401
                logger.debug("BAML TypeBuilder available (dummy mode active)")
                self._type_builder = "type_builder_shell"  # Placeholder
            except ImportError:
                logger.debug("BAML TypeBuilder not available (dummy mode active)")
                self._type_builder = "type_builder_shell"

        return self._type_builder

    async def compile_prompt(
        self,
        tool_schemas: list[type[BaseModel]],
        system_prompt: str,
        user_input: str,
    ) -> str:
        """
        Compile Pydantic tool schemas into a formatted prompt.

        Dummy implementation: Returns a simple formatted string.
        In real implementation, this would use BAML's TypeBuilder to convert
        Pydantic schemas to BAML-DSL prompts.

        Args:
            tool_schemas: List of Pydantic models representing available tools
            system_prompt: Base system prompt template
            user_input: User's current message

        Returns:
            Compiled prompt string ready for transport layer
        """
        try:
            # Lazy import check (not used in dummy mode)
            await self._get_type_builder()

            logger.debug(
                f"Dummy compile: {len(tool_schemas)} tool schemas, user input length: {len(user_input)}"
            )

            # Dummy implementation: Simple formatted string
            # In real implementation, this would use BAML's TypeBuilder
            tool_descriptions = []
            for schema in tool_schemas:
                # Extract basic info from Pydantic schema
                fields = []
                if hasattr(schema, "model_fields"):
                    for field_name, field_info in schema.model_fields.items():
                        field_type = str(field_info.annotation).replace("typing.", "")
                        description = getattr(field_info, "description", "")
                        fields.append(
                            f"  - {field_name}: {field_type} ({description})"
                            if description
                            else f"  - {field_name}: {field_type}"
                        )

                tool_desc = f"Tool: {schema.__name__}\n" + "\n".join(fields)
                tool_descriptions.append(tool_desc)

            # Build the final prompt
            compiled_prompt = f"""{system_prompt}

Available Tools:
{chr(10).join(tool_descriptions)}

User Input: {user_input}

Please respond with either a tool call or a final answer."""

            logger.debug(f"Compiled prompt (length: {len(compiled_prompt)})")
            return compiled_prompt

        except Exception as e:
            logger.error(f"Prompt compilation failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    async def parse_response(
        self, raw_response: str, expected_schema: type[BaseModel]
    ) -> BaseModel:
        """
        Parse raw text response into structured Pydantic model.

        Dummy implementation: Returns a hardcoded FinalResponse.
        In real implementation, this would use BAML's Schema-Aligned Parser
        to extract structured data from the raw response.

        Args:
            raw_response: Raw text response from transport layer
            expected_schema: Pydantic model to parse the response into
                (typically a Union of FinalResponse | ToolSchema1 | ToolSchema2 | ...)

        Returns:
            Validated Pydantic instance (dummy: always returns FinalResponse)
        """
        try:
            # Lazy import check (not used in dummy mode)
            await self._get_baml_client()

            logger.debug(
                f"Dummy parse: response length {len(raw_response)}, expected schema: {expected_schema.__name__}"
            )

            # Dummy implementation: Return hardcoded FinalResponse
            # In real implementation, this would use BAML's Schema-Aligned Parser
            # to parse the raw_response into the expected_schema (Union type)
            return FinalResponse(
                thought="Dummy parsing: This is a placeholder response.",
                answer="Dummy Answer: The intelligence layer is in dummy mode.",
                acknowledged_ids=None,
            )

        except Exception as e:
            logger.error(f"Response parsing failed: {type(e).__name__}: {e}", exc_info=True)
            raise

    async def close(self) -> None:
        """Clean up any resources."""
        if self._baml_client:
            self._baml_client = None
        if self._type_builder:
            self._type_builder = None
