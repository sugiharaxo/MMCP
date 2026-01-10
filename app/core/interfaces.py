"""Protocol-Driven Interfaces: Architectural contracts for the two-layer system.

This module defines the formal contracts between Prompt and Orchestrator
layers using Python's typing.Protocol. Transport is now handled by BAML end-to-end.
These are purely conceptual interfaces that enforce the "Rules of the House"
without dictating implementation details.
"""

from typing import Any, Protocol

from pydantic import BaseModel


class LLMPrompt(Protocol):
    """Prompt service implementing LLMPrompt protocol.

    Contract 1 (Compile):
        - Take user input and dynamic tool metadata
        - Return a formatted prompt string ready for transport

    Contract 2 (Parse):
        - Take a raw response string from transport
        - Return a structured Union of tool call or final answer

    Constraint:
        - Zero knowledge of networking or API keys
        - Stateless: compilation and parsing are pure functions

    This protocol ensures the prompt layer handles all schema-aware
    operations without touching network code.
    """

    async def compile_prompt(
        self,
        tool_schemas: list[type[BaseModel]],
        system_prompt: str,
        user_input: str,
        history: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compile tool schemas and user input into a formatted messages array.

        Args:
            tool_schemas: List of Pydantic models representing available tools
            system_prompt: Base system prompt template
            user_input: User's current message
            history: Optional conversation history

        Returns:
            List of message dicts in OpenAI format ready for transport layer
        """
        ...

    async def parse_response(
        self, raw_response: str, expected_schema: type[BaseModel]
    ) -> BaseModel:
        """
        Parse raw text response into structured Pydantic model.

        Args:
            raw_response: Raw text response from transport layer
            expected_schema: Pydantic model to parse the response into
                (typically a Union of FinalResponse | ToolSchema1 | ToolSchema2 | ...)

        Returns:
            Validated Pydantic instance (either tool call or final answer)
        """
        ...

    async def close(self) -> None:
        """
        Clean up any resources held by the prompt service.

        This method should be called when the service is shutting down
        to ensure proper cleanup of cached resources, HTTP connections, etc.
        """
        ...


class AgentOrchestrator(Protocol):
    """Orchestrator Protocol: Coordinates the ReAct loop between layers.

    Contract:
        - Coordinate the loop between Transport and Prompt
        - Manage the single-turn or multi-turn ReAct pattern
        - Resolve user requests by routing tool calls and final answers

    This protocol ensures the orchestrator remains focused on coordination
    logic without implementing transport or prompt details.
    """

    async def process_message(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> dict:
        """
        Process a user message through the agent loop.

        This coordinates: prompt.compile -> transport.send -> prompt.parse

        Args:
            user_input: The user's message
            session_id: Optional session identifier
            turn_instructions: Optional temporary instructions for this specific turn
            **kwargs: Additional parameters for transport layer

        Returns:
            Response dict with result and metadata
        """
        ...
