"""Agent Service: The Orchestrator coordinating Transport and Intelligence layers.

This layer manages:
- State management for the ReAct loop
- Coordination between Transport (I/O) and Intelligence (compilation/parsing)
- The main while loop and HITL (Human-in-the-loop) approvals

Implements AgentOrchestrator protocol with dependency injection for Transport and Intelligence.
"""

import asyncio
import uuid
from typing import Any

from app.agent.schemas import FinalResponse
from app.core.interfaces import LLMIntelligence, LLMTransport
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader


class AgentService:
    """Main agent orchestrator implementing AgentOrchestrator protocol.

    Pure orchestrator: Coordinates the loop between Transport and Intelligence.
    Uses dependency injection to accept any implementation of LLMTransport and LLMIntelligence.
    """

    def __init__(
        self,
        plugin_loader: PluginLoader,
        transport: LLMTransport | None = None,
        intelligence: LLMIntelligence | None = None,
    ):
        """
        Initialize the agent service with dependency injection.

        Args:
            plugin_loader: PluginLoader with discovered tools
            transport: LLMTransport implementation (defaults to TransportService)
            intelligence: LLMIntelligence implementation (defaults to IntelligenceService)
        """
        self.plugin_loader = plugin_loader

        # Dependency injection: Use provided implementations or create defaults
        if transport is None:
            from app.services.transport import TransportService

            self.transport: LLMTransport = TransportService()
        else:
            self.transport = transport

        if intelligence is None:
            from app.services.intelligence import IntelligenceService

            self.intelligence: LLMIntelligence = IntelligenceService()
        else:
            self.intelligence = intelligence

        # Session storage (in-memory for now)
        self._sessions: dict[str, dict[str, Any]] = {}

    def _get_session(self, session_id: str | None) -> dict[str, Any]:
        """Get or create a session."""
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "session_id": session_id,
                "conversation_history": [],
                "tool_results": [],
                "pending_actions": [],
            }

        return self._sessions[session_id]

    async def process_message(
        self,
        user_input: str,
        session_id: str | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process a user message through the agent loop.

        Single-turn ReAct skeleton:
        1. Intelligence.compile: Convert user input + tool metadata -> prompt string
        2. Transport.send: Send prompt -> receive raw response string
        3. Intelligence.parse: Parse raw response -> structured Union (tool call or final answer)

        This is the main entry point that coordinates Transport and Intelligence layers.

        Args:
            user_input: The user's message
            session_id: Optional session identifier
            system_prompt: System prompt template
            **kwargs: Additional parameters for transport layer

        Returns:
            Response dict with result and metadata
        """
        session = self._get_session(session_id)
        session_id = session["session_id"]

        logger.info(f"Processing message for session {session_id} (length: {len(user_input)})")

        try:
            # Get available tool schemas from plugin loader
            tool_schemas = []
            for tool in self.plugin_loader.tools.values():
                if hasattr(tool, "input_schema") and tool.input_schema:
                    tool_schemas.append(tool.input_schema)

            # Step 1: Intelligence.compile - Convert user input + tool metadata -> prompt string
            compiled_prompt = await self.intelligence.compile_prompt(
                tool_schemas=tool_schemas,
                system_prompt=system_prompt,
                user_input=user_input,
            )

            # Step 2: Transport.send - Send prompt -> receive raw response string
            raw_response = await self.transport.send_message(compiled_prompt, **kwargs)

            # Step 3: Intelligence.parse - Parse raw response -> structured Union
            # Build the Union schema for parsing (FinalResponse | ToolSchema1 | ToolSchema2 | ...)
            # For now, we'll use FinalResponse as the expected schema in dummy mode
            # In full implementation, this would be a Union type built from tool_schemas
            parsed_response = await self.intelligence.parse_response(
                raw_response=raw_response,
                expected_schema=FinalResponse,  # Dummy: will be Union type in full implementation
            )

            # Route based on parsed response type
            if isinstance(parsed_response, FinalResponse):
                # Final answer: return to user
                logger.debug(f"Received final response for session {session_id}")
                return {
                    "response": parsed_response.answer,
                    "type": "final_response",
                    "session_id": session_id,
                    "thought": parsed_response.thought,
                }
            else:
                # Tool call: would execute tool and loop back (not implemented in dummy mode)
                logger.debug(
                    f"Received tool call for session {session_id} (not implemented in dummy mode)"
                )
                return {
                    "response": f"Tool call detected: {type(parsed_response).__name__} (tool execution not implemented in dummy mode)",
                    "type": "tool_call",
                    "session_id": session_id,
                }

        except Exception as e:
            logger.error(
                f"Agent processing failed for session {session_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    async def execute_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        """
        Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            session_id: Session identifier

        Returns:
            Tool execution result
        """
        session = self._get_session(session_id)

        if tool_name not in self.plugin_loader.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.plugin_loader.tools[tool_name]

        try:
            logger.info(f"Executing tool '{tool_name}' for session {session_id}")

            # Execute the tool
            result = await tool.execute(tool_args)

            # Store result in session
            session["tool_results"].append(
                {
                    "tool_name": tool_name,
                    "args": tool_args,
                    "result": result,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            )

            return {
                "tool_name": tool_name,
                "result": result,
                "session_id": session_id,
            }

        except Exception as e:
            logger.error(
                f"Tool execution failed for '{tool_name}': {type(e).__name__}: {e}", exc_info=True
            )
            raise

    async def get_session_history(self, session_id: str) -> dict[str, Any]:
        """
        Get the conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            Session data including history
        """
        session = self._get_session(session_id)
        return session.copy()

    async def clear_session(self, session_id: str) -> None:
        """
        Clear a session's data.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session {session_id}")

    async def close(self) -> None:
        """Clean up all resources."""
        # Close transport and intelligence if they have close methods
        if hasattr(self.transport, "close"):
            await self.transport.close()
        if hasattr(self.intelligence, "close"):
            await self.intelligence.close()

        # Clear all sessions
        self._sessions.clear()
        logger.info("Agent service shut down")
