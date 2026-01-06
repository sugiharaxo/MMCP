"""Protocol-Driven Interfaces: Architectural contracts for the three-layer system.

This module defines the formal contracts between Transport, Intelligence, and Orchestrator
layers using Python's typing.Protocol. These are purely conceptual interfaces that enforce
the "Rules of the House" without dictating implementation details.
"""

from typing import Protocol

from pydantic import BaseModel


class LLMTransport(Protocol):
    """Transport Layer Protocol: Pure I/O wrapper for LLM communication.

    Contract:
        - Must accept a prompt (string) and return a response (string)
        - Zero knowledge of schemas, tools, or parsing logic
        - Stateless: no internal state that affects behavior

    This protocol ensures the transport layer remains a "dumb pipe" that only
    handles network I/O and API communication.
    """

    async def send_message(self, prompt: str, **kwargs) -> str:
        """
        Send a raw text prompt and receive a raw text response.

        Args:
            prompt: Raw text prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            Raw text response from the LLM
        """
        ...


class LLMIntelligence(Protocol):
    """Intelligence Layer Protocol: Prompt compilation and response parsing.

    Contract 1 (Compile):
        - Take user input and dynamic tool metadata
        - Return a formatted prompt string ready for transport

    Contract 2 (Parse):
        - Take a raw response string from transport
        - Return a structured Union of tool call or final answer

    Constraint:
        - Zero knowledge of networking or API keys
        - Stateless: compilation and parsing are pure functions

    This protocol ensures the intelligence layer handles all schema-aware
    operations without touching network code.
    """

    async def compile_prompt(
        self,
        tool_schemas: list[type[BaseModel]],
        system_prompt: str,
        user_input: str,
    ) -> str:
        """
        Compile tool schemas and user input into a formatted prompt.

        Args:
            tool_schemas: List of Pydantic models representing available tools
            system_prompt: Base system prompt template
            user_input: User's current message

        Returns:
            Compiled prompt string ready for transport layer
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


class AgentOrchestrator(Protocol):
    """Orchestrator Protocol: Coordinates the ReAct loop between layers.

    Contract:
        - Coordinate the loop between Transport and Intelligence
        - Manage the single-turn or multi-turn ReAct pattern
        - Resolve user requests by routing tool calls and final answers

    This protocol ensures the orchestrator remains focused on coordination
    logic without implementing transport or intelligence details.
    """

    async def process_message(
        self,
        user_input: str,
        session_id: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        **kwargs,
    ) -> dict:
        """
        Process a user message through the agent loop.

        This coordinates: intelligence.compile -> transport.send -> intelligence.parse

        Args:
            user_input: The user's message
            session_id: Optional session identifier
            system_prompt: System prompt template
            **kwargs: Additional parameters for transport layer

        Returns:
            Response dict with result and metadata
        """
        ...
