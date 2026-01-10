"""ReAct Loop: Handles the Reasoning + Acting loop execution.

This class encapsulates the ReAct loop logic that was previously in AgentService.
It manages the iterative process of LLM reasoning and tool execution until a final response.
"""

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

from app.agent.history_manager import HistoryManager, HistoryMessage
from app.agent.schemas import ActionRequestResponse
from app.agent.session_manager import SessionManager as AgentSessionManager
from app.api.base import Tool
from app.core.config import UserSettings
from app.core.errors import AgentLogicError, ProviderError, map_provider_error
from app.core.interfaces import LLMPrompt
from app.core.logger import logger
from app.services.prompt import ToolInfo
from baml_client.types import FinalResponse, ToolCall


class ToolExecutionTimeout(Exception):
    """Raised when a tool execution times out."""

    pass


if TYPE_CHECKING:
    from app.agent.history_manager import HistoryManager
    from app.core.plugin_loader import PluginLoader


class ReActLoop:
    """Handles the ReAct (Reasoning + Acting) loop execution.

    This class manages the iterative process where the agent alternates between
    reasoning (LLM calls) and acting (tool execution) until a final response is reached.
    """

    def __init__(
        self,
        prompt: LLMPrompt,
        session_manager: AgentSessionManager,
        user_settings: UserSettings,
        plugin_loader: "PluginLoader",
        history_manager: "HistoryManager",
    ):
        """
        Initialize the ReAct loop handler.

        Args:
            prompt: LLMPrompt implementation for LLM calls
            session_manager: Session manager for database persistence
            user_settings: User settings configuration
            plugin_loader: Plugin loader for tool access
            history_manager: History manager for conversation tracking
        """
        self.prompt = prompt
        self.session_manager = session_manager
        self.user_settings = user_settings
        self.plugin_loader = plugin_loader
        self.history_manager = history_manager

    def stringify_tool_result(self, tool_result: Any) -> str:
        """Convert tool result to string for history."""
        if isinstance(tool_result, dict):
            return json.dumps(tool_result, indent=2)
        return str(tool_result)

    async def call_llm_with_error_handling(
        self,
        tool_infos: list[ToolInfo],
        context_data: dict[str, Any],
        history: list[HistoryMessage],
        step: int | None = None,
    ) -> FinalResponse | ToolCall:
        """
        Call LLM with consistent error handling.

        Note: This method converts history to dict list only for the API call.
        The original `history` list is not modified by this conversion.

        Returns:
            FinalResponse | ToolCall from LLM

        Raises:
            AgentLogicError: For validation/format errors (non-retryable)
            ProviderError: For network/timeout/rate limit errors (may be retryable)
        """
        try:
            return await self.prompt.call_llm(
                tool_infos=tool_infos,
                context_data=context_data,
                user_input=None,  # No new user input, tool result is in history
                history=HistoryManager.to_dict_list(history),  # Convert only for API transport
                user_settings=self.user_settings,
            )
        except (AgentLogicError, ProviderError) as llm_error:
            logger.error(
                f"LLM call failed{' in ReAct loop (step ' + str(step) + ')' if step else ''}: {llm_error}",
                exc_info=True,
            )
            raise  # Re-raise to be handled by caller
        except Exception as llm_error:
            # Map unmapped errors to ProviderError
            logger.error(
                f"Unexpected LLM error{' in ReAct loop (step ' + str(step) + ')' if step else ''}: {llm_error}",
                exc_info=True,
            )
            raise map_provider_error(llm_error) from llm_error

    def handle_llm_error(
        self,
        llm_error: Exception,
        history: list[HistoryMessage],
        session_id: str,  # Used in result dict
    ) -> dict[str, Any]:
        """Handle LLM errors consistently."""
        if isinstance(llm_error, (AgentLogicError, ProviderError)):
            error_msg = f"LLM call failed: {llm_error.message}"
            retryable = llm_error.retryable
        else:
            error_msg = f"LLM call failed: {str(llm_error)}"
            retryable = None

        self.history_manager.add_error_message(history, error_msg)
        if retryable is not None:
            return {
                "response": error_msg,
                "type": "error",
                "session_id": session_id,
                "retryable": retryable,
            }
        return {
            "response": error_msg,
            "type": "error",
            "session_id": session_id,
        }

    async def execute_tool(
        self,
        tool: Tool,
        tool_args: dict[str, Any],
    ) -> Any:
        """Execute a tool with proper validation and timeout enforcement."""
        if hasattr(tool, "input_schema") and tool.input_schema:
            validated_args = tool.input_schema.model_validate(tool_args)
            args = validated_args.model_dump()
        else:
            args = tool_args

        # Apply tool execution timeout from user settings
        try:
            return await asyncio.wait_for(
                tool.execute(**args), timeout=self.user_settings.tool_execution_timeout_seconds
            )
        except TimeoutError as timeout_error:
            raise ToolExecutionTimeout(
                f"Tool execution timed out after {self.user_settings.tool_execution_timeout_seconds} seconds"
            ) from timeout_error

    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        context_data: dict[str, Any],
        history: list[HistoryMessage],
        session_id: str,
        step: int,
        tool_infos: list[ToolInfo],
    ) -> tuple[dict[str, Any] | None, FinalResponse | ToolCall | None]:
        """
        Handle a tool call: execute if INTERNAL, request approval if EXTERNAL.

        Returns:
            Tuple of (error/approval dict or None, next response or None)
            - If approval needed or error: (dict, None)
            - If tool executed successfully: (None, next_response)
        """
        tool_name = tool_call.tool_name
        # Convert args to mutable dict - handle both dict and Pydantic models
        if isinstance(tool_call.args, dict):
            tool_args = dict(tool_call.args)
        elif hasattr(tool_call.args, "model_dump"):
            tool_args = tool_call.args.model_dump()
        else:
            tool_args = dict(tool_call.args) if hasattr(tool_call.args, "__dict__") else {}

        logger.info(f"Agent Thought: {tool_call.thought}")
        logger.debug(f"Received tool call for session {session_id}: {tool_name}")

        tool_call_dict = tool_call.model_dump()
        self.history_manager.add_tool_call(history, tool_call_dict)

        tool_call_id = str(uuid.uuid4())

        tool = self.plugin_loader.get_tool(tool_name)
        if tool is None:
            logger.warning(f"Tool '{tool_name}' not found in plugin loader")
            error_msg = f"Tool '{tool_name}' not found or not available"
            self.history_manager.add_error_message(history, error_msg)
            await self.session_manager.save_session(session_id, history)
            return (
                {
                    "response": error_msg,
                    "type": "error",
                    "session_id": session_id,
                },
                None,
            )

        # All tools should have classification (defaults to "EXTERNAL" in base class)
        tool_classification = getattr(tool, "classification", "EXTERNAL")
        if tool_classification == "EXTERNAL":
            hitl_rationale = tool_args.pop("rationale", None)
            if not hitl_rationale:
                error_msg = (
                    f"Protocol Error: External tool '{tool_name}' requires a 'rationale' field "
                    "explaining the action to the user. Please provide a rationale and try again."
                )
                logger.warning(error_msg)
                self.history_manager.add_error_message(history, error_msg)
                await self.session_manager.save_session(session_id, history)
                return (
                    {
                        "response": error_msg,
                        "type": "error",
                        "session_id": session_id,
                    },
                    None,
                )

            logger.info(f"HITL Rationale for '{tool_name}': {hitl_rationale}")

            approval_id = str(uuid.uuid4())
            action_request = ActionRequestResponse(
                approval_id=approval_id,
                explanation=hitl_rationale,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
            )

            pending_action = {
                "approval_id": approval_id,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_call_id": tool_call_id,
                "explanation": hitl_rationale,
            }
            await self.session_manager.save_session_with_pending_action(
                session_id, history, pending_action
            )

            logger.info(
                f"EXTERNAL tool '{tool_name}' requires approval (approval_id: {approval_id})"
            )

            return (
                {
                    "response": action_request.explanation,
                    "explanation": action_request.explanation,
                    "type": "action_request",
                    "session_id": session_id,
                    "approval_id": approval_id,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "content": tool_call_dict,
                },
                None,
            )

        # INTERNAL tool: Execute immediately
        try:
            tool_result = await self.execute_tool(tool, tool_args)
            tool_result_str = self.stringify_tool_result(tool_result)

            self.history_manager.add_tool_result(history, tool_name, tool_result_str)
            await self.session_manager.save_session(session_id, history)
            self.history_manager.trim_history(history, self.user_settings)

            try:
                next_response = await self.call_llm_with_error_handling(
                    tool_infos=tool_infos,
                    context_data=context_data,
                    history=history,
                    step=step,
                )
                return (None, next_response)  # Continue loop with next response
            except Exception as llm_error:
                error_result = self.handle_llm_error(llm_error, history, session_id)
                await self.session_manager.save_session(session_id, history)
                return (error_result, None)

        except ToolExecutionTimeout as timeout_error:
            # Handle tool timeout gracefully - add as observation for LLM to reason about
            logger.warning(f"Tool execution timed out for '{tool_name}': {timeout_error}")
            timeout_observation = str(timeout_error)
            self.history_manager.add_tool_result(history, tool_name, timeout_observation)
            await self.session_manager.save_session(session_id, history)
            self.history_manager.trim_history(history, self.user_settings)

            try:
                next_response = await self.call_llm_with_error_handling(
                    tool_infos=tool_infos,
                    context_data=context_data,
                    history=history,
                    step=step,
                )
                return (None, next_response)  # Continue loop with next response
            except Exception as llm_error:
                error_result = self.handle_llm_error(llm_error, history, session_id)
                await self.session_manager.save_session(session_id, history)
                return (error_result, None)

        except Exception as tool_error:
            logger.error(
                f"Tool execution failed for '{tool_name}': {tool_error}",
                exc_info=True,
            )
            error_msg = f"Tool execution failed: {str(tool_error)}"
            self.history_manager.add_error_message(history, error_msg)
            await self.session_manager.save_session(session_id, history)
            return (
                {
                    "response": error_msg,
                    "type": "error",
                    "session_id": session_id,
                },
                None,
            )

    async def execute(
        self,
        parsed_response: FinalResponse | ToolCall,
        context_data: dict[str, Any],
        history: list[HistoryMessage],
        session_id: str,
        max_steps: int,
        tool_infos: list[ToolInfo],
    ) -> dict[str, Any]:
        """
        Execute ReAct loop until FinalResponse or max steps reached.

        **Important**: This method mutates the `history` list in-place by appending
        assistant responses, tool calls, and tool results. The caller's original
        list is modified, ensuring that SessionManager can persist the updated history.

        Args:
            parsed_response: Initial LLM response (FinalResponse or ToolCall)
            context_data: Context data for LLM calls
            history: Conversation history (mutated in-place)
            session_id: Session identifier
            max_steps: Maximum ReAct loop iterations
            tool_infos: Tool metadata for LLM

        Returns:
            Response dict with result or error
        """
        step = 0
        current_response = parsed_response

        while step < max_steps:
            step += 1
            logger.debug(f"ReAct loop step {step}/{max_steps} for session {session_id}")

            if isinstance(current_response, FinalResponse):
                final_response_dict = current_response.model_dump()
                # Import here to avoid circular imports
                from app.agent.history_manager import HistoryManager

                self.history_manager.add_final_response(history, final_response_dict)
                await self.session_manager.save_session(session_id, history)
                return {
                    "response": current_response.answer,
                    "type": "final_response",
                    "session_id": session_id,
                    "thought": current_response.thought,
                    "steps": step,
                    "content": final_response_dict,
                }

            if isinstance(current_response, ToolCall):
                action_result, next_response = await self._handle_tool_call(
                    current_response, context_data, history, session_id, step, tool_infos
                )
                if action_result is not None:
                    # Action needed (approval request or error)
                    return action_result
                if next_response is not None:
                    # Continue loop with next response
                    current_response = next_response
                    continue
                # Critical: Tool handler returned None for both - this indicates a logic error
                # Do not continue as it would cause an infinite loop
                logger.error(
                    f"Agent stalled during tool execution for session {session_id} at step {step} - "
                    "tool handler returned None for both action and response"
                )
                error_msg = "Agent stalled during tool execution - internal logic error"
                from app.agent.history_manager import HistoryManager

                self.history_manager.add_error_message(history, error_msg)
                await self.session_manager.save_session(session_id, history)
                return self.handle_llm_error(
                    AgentLogicError(error_msg, retryable=False), history, session_id
                )

            # Unknown response type
            logger.warning(
                f"Unknown response type for session {session_id}: {type(current_response)}"
            )
            error_msg = f"Unknown response type: {type(current_response).__name__}"
            self.history_manager.add_error_message(history, error_msg)
            await self.session_manager.save_session(session_id, history)
            return {
                "response": error_msg,
                "type": "error",
                "session_id": session_id,
            }

        # Max steps reached
        logger.warning(
            f"ReAct loop reached max steps ({max_steps}) without final response for session {session_id}"
        )
        error_msg = f"Maximum ReAct steps ({max_steps}) reached without final answer"
        from app.agent.history_manager import HistoryManager

        history_manager = HistoryManager()
        history_manager.add_error_message(history, error_msg)
        await self.session_manager.save_session(session_id, history)
        return {
            "response": error_msg,
            "type": "error",
            "session_id": session_id,
        }

    async def resume_after_tool_execution(
        self,
        context_data: dict[str, Any],
        history: list[HistoryMessage],
        session_id: str,
        max_steps: int,
        tool_infos: list[ToolInfo],
    ) -> dict[str, Any]:
        """
        Resume the ReAct loop after a tool has been executed.

        This method handles the LLM call and continues the ReAct loop
        after tool execution (used in HITL resume scenarios).

        Args:
            context_data: Context data for the LLM
            history: Current conversation history (including tool result)
            session_id: Session identifier
            max_steps: Maximum steps for the ReAct loop

        Returns:
            Response dict with result from continued agent loop
        """
        # Use provided tool metadata

        # Call LLM to get next response after tool execution
        try:
            next_response = await self.call_llm_with_error_handling(
                tool_infos=tool_infos,
                context_data=context_data,
                history=history,
            )
        except Exception as llm_error:
            error_result = self.handle_llm_error(llm_error, history, session_id)
            await self.session_manager.save_session(session_id, history)
            return error_result

        # Continue the ReAct loop
        return await self.execute(
            next_response, context_data, history, session_id, max_steps, tool_infos
        )
