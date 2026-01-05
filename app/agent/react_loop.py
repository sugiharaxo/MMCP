"""ReAct Loop - handles the main reasoning and tool execution loop."""

import asyncio
import json
import uuid
from typing import Any

import instructor
from pydantic import BaseModel

from app.agent.history_manager import HistoryManager
from app.agent.llm_interface import LLMInterface
from app.agent.schemas import (
    ActionRequestResponse,
    FinalResponse,
    get_reasoned_model,
)
from app.core.config import internal_settings, user_settings
from app.core.context import MMCPContext
from app.core.errors import MMCPError, map_agent_error, map_provider_error
from app.core.llm import get_instructor_mode, safe_get
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.core.security import sanitize_explanation


class ReActLoop:
    """Handles the main ReAct reasoning loop and tool execution."""

    def __init__(self, loader: PluginLoader, status_callback=None):
        self.loader = loader
        self.llm = LLMInterface()
        self.history_manager = HistoryManager()
        self.status_callback = status_callback

    async def execute_turn(
        self,
        user_input: str,
        session_history: list[dict[str, Any]],
        context: MMCPContext,
        system_prompt: str | tuple[str, str],
        notification_injector=None,
    ) -> tuple[str | ActionRequestResponse, list[dict[str, Any]]]:
        """
        Execute a single ReAct turn.

        Args:
            user_input: The user's message
            session_history: Current conversation history
            context: MMCP context
            system_prompt: The system prompt(s) to use.
                          Can be a single string (legacy) or tuple of (static, dynamic) strings.

        Returns:
            Tuple of (response object, updated history list)
        """
        # Reconstruct history for this turn (supports both legacy and cache-safe formats)
        history = self.history_manager.reconstruct_history(
            session_history, system_prompt, user_input
        )

        # Build the reasoned Union response model (FinalResponse | flattened tool schemas)
        ResponseModel = self._get_reasoned_response_model()

        instructor_mode = get_instructor_mode(user_settings)

        # ReAct loop: allow multiple reasoning steps
        max_steps = user_settings.react_max_steps
        llm_retry_count = 0
        max_llm_retries = user_settings.react_max_llm_retries
        rate_limit_retry_count = 0
        max_rate_limit_retries = user_settings.react_max_rate_limit_retries

        for _ in range(max_steps):
            context.increment_step()
            self.history_manager.trim_history(history)

            # Turn 1 (Reasoning): Single-turn reasoned decision
            try:
                context.increment_llm_call()
                if self.status_callback:
                    await self.status_callback(
                        "Thinking about next action...", context.runtime.trace_id, "thought"
                    )
                response, raw_completion = await self.llm.get_reasoned_decision(
                    history, ResponseModel, context.runtime.trace_id
                )
                llm_retry_count = 0

            except Exception as e:
                # Handle errors with retry logic
                (
                    should_retry,
                    llm_retry_count,
                    rate_limit_retry_count,
                ) = await self._handle_llm_error(
                    e,
                    history,
                    context,
                    llm_retry_count,
                    rate_limit_retry_count,
                    max_llm_retries,
                    max_rate_limit_retries,
                )
                if should_retry:
                    continue
                else:
                    # Return error message
                    error_msg = self._extract_error_message(e)
                    return error_msg, history

            # Route based on response type (outside try/except)
            if isinstance(response, FinalResponse):
                # For FinalResponse: Do NOT use add_llm_message (which adds tool_calls metadata)
                # Instead, _handle_final_response will use add_assistant_response to record
                # the plain text answer, closing the tool-calling protocol for the next turn.
                # Generate rich dialogue response using DialogueProfile
                result = await self._handle_final_response(
                    response, history, context, notification_injector
                )
                return result, history
            else:
                # Flattened tool call structure: response is directly a tool schema with rationale
                tool_call_id = None
                all_tool_calls = []

                # Extract protocol metadata from the RAW completion (Mode-Agnostic)
                choices = safe_get(raw_completion, "choices")
                message = safe_get(choices[0], "message") if choices else None
                tool_calls = safe_get(message, "tool_calls")

                # Add assistant message to history
                if message:
                    self.history_manager.add_llm_message(history, message, instructor_mode)

                # Extract tool_call_id from raw completion (works for all modes)
                if tool_calls:
                    # We found tool calls in the transport layer.
                    # Use the ID provided by the model (works for wrappers OR native calls).
                    all_tool_calls = tool_calls
                    call_id = safe_get(tool_calls[0], "id")
                    if call_id:
                        tool_call_id = call_id

                    if len(tool_calls) > 1:
                        logger.warning(
                            f"Multiple tool calls detected ({len(tool_calls)}). "
                            f"Executing primary tool, additional calls will receive error responses.",
                            extra={"trace_id": context.runtime.trace_id},
                        )

                # STRICT STATE PERSISTENCE: If TOOLS mode, tool_call_id is mandatory
                self._validate_protocol(tool_calls, tool_call_id, instructor_mode, context)

                # Handle flattened tool call (has rationale field directly)
                result = await self._handle_flattened_tool_call(
                    response,
                    history,
                    context,
                    tool_call_id=tool_call_id,
                    instructor_mode=instructor_mode,
                )

                # Handle multiple tool calls: provide error results for additional tool calls
                if instructor_mode == instructor.Mode.TOOLS and len(all_tool_calls) > 1:
                    for additional_tc in all_tool_calls[1:]:
                        if not additional_tc.id:
                            raise MMCPError(
                                "Protocol Corruption: Additional tool_call missing ID in TOOLS mode",
                                trace_id=context.runtime.trace_id,
                            )

                        # If we have an ID, we MUST respond to it to satisfy the protocol
                        error_result = (
                            "Parallel tool execution not supported. "
                            "Please retry with one tool at a time."
                        )
                        self.history_manager.add_tool_result(
                            history,
                            additional_tc.id,
                            error_result,
                            instructor_mode=instructor_mode,
                        )
                if isinstance(result, ActionRequestResponse):
                    return result, history  # HITL interruption
                elif isinstance(result, str):
                    return result, history  # Circuit breaker or error response
                # Continue loop with tool result in history

        # Exhausted max steps
        logger.warning(
            f"ReAct loop exhausted {max_steps} steps without final response (trace_id={context.runtime.trace_id})",
            extra={"trace_id": context.runtime.trace_id, "max_steps": max_steps},
        )
        return (
            "I attempted to solve this but reached my reasoning limit. Please try being more specific or breaking down your request.",
            history,
        )

    def _validate_protocol(
        self,
        tool_calls: list | None,
        tool_call_id: str | None,
        instructor_mode: instructor.Mode,
        context: MMCPContext,
    ) -> None:
        """
        Standardized protocol validation for TOOLS mode.

        Ensures that when using Mode.TOOLS, the LLM response contains valid tool_calls
        with proper IDs for protocol compliance (e.g., DeepSeek 1:1 pairing requirement).
        """
        if instructor_mode != instructor.Mode.TOOLS:
            return

        if not tool_calls:
            raise MMCPError(
                "Protocol Corruption: LLM response missing tool_calls in TOOLS mode",
                trace_id=context.runtime.trace_id,
            )
        if not tool_call_id:
            raise MMCPError(
                "Protocol Corruption: tool_call missing ID in TOOLS mode",
                trace_id=context.runtime.trace_id,
            )

    async def _handle_llm_error(
        self,
        error: Exception,
        history: list[dict[str, Any]],
        context: MMCPContext,
        llm_retry_count: int,
        rate_limit_retry_count: int,
        max_llm_retries: int,
        max_rate_limit_retries: int,
    ) -> tuple[bool, int, int]:
        """Handle LLM errors. Returns (should_retry, updated_llm_count, updated_rate_limit_count)."""
        agent_error = map_agent_error(error, trace_id=context.runtime.trace_id)

        # Validation/parsing errors - feed back to LLM for self-correction
        if agent_error.retryable and llm_retry_count < max_llm_retries:
            self.history_manager.add_error_message(history, agent_error.message)
            return True, llm_retry_count + 1, rate_limit_retry_count  # Only increment LLM count

        # Provider errors
        provider_error = map_provider_error(error, trace_id=context.runtime.trace_id)
        if isinstance(provider_error, MMCPError) and not provider_error.retryable:
            raise provider_error from error

        # Rate limit errors with exponential backoff
        is_rate_limit = "rate limit" in str(error).lower() or "429" in str(error)
        if is_rate_limit and rate_limit_retry_count < max_rate_limit_retries:
            backoff_base = internal_settings["react_loop"]["backoff_base"]
            backoff_seconds = backoff_base ** (rate_limit_retry_count - 1)
            logger.warning(
                f"Rate limit hit (attempt {rate_limit_retry_count}/{max_rate_limit_retries}), "
                f"backing off for {backoff_seconds}s (trace_id={context.runtime.trace_id})"
            )
            await asyncio.sleep(backoff_seconds)
            return (
                True,
                llm_retry_count,
                rate_limit_retry_count + 1,
            )  # Only increment Rate Limit count

        return False, llm_retry_count, rate_limit_retry_count

    async def _handle_final_response(
        self,
        response: FinalResponse,
        history: list[dict[str, Any]],
        context: MMCPContext,
        notification_injector=None,
    ) -> str:
        """
        Handle final response from LLM reasoning phase.

        If the reasoning phase returned a FinalResponse, we use the DialogueProfile
        to generate a rich, conversational response to the user.
        """
        # Process agent ACKs for asynchronous notifications
        if response.acknowledged_ids and notification_injector:
            user_id = "default"  # Single-user system for now
            await notification_injector.mark_agent_processed(response.acknowledged_ids, user_id)

        # Generate rich dialogue response using DialogueProfile
        # Use the thought and answer from reasoning as context
        dialogue_messages = history + [
            {
                "role": "system",
                "content": f"Internal reasoning: {response.thought}\n\nGenerate a natural, helpful response to the user based on this reasoning.",
            },
            {"role": "user", "content": "Please provide your response."},
        ]

        try:
            dialogue_response = await self.llm.generate_dialogue_response(
                dialogue_messages, context.runtime.trace_id
            )
            self.history_manager.add_assistant_response(history, dialogue_response)
            context.log_inspection()
            return dialogue_response
        except Exception as e:
            # Fallback to the answer from reasoning if dialogue generation fails
            logger.warning(
                f"Dialogue generation failed, using reasoning answer (trace_id={context.runtime.trace_id}): {e}",
                extra={"trace_id": context.runtime.trace_id},
            )
            self.history_manager.add_assistant_response(history, response.answer)
            context.log_inspection()
            return response.answer

    async def _handle_flattened_tool_call(
        self,
        response: Any,
        history: list[dict[str, Any]],
        context: MMCPContext,
        tool_call_id: str | None = None,
        instructor_mode: instructor.Mode | None = None,
    ) -> str | ActionRequestResponse | None:
        """
        Handle flattened tool call structure (tool schema with rationale field directly).

        With the flattened Union structure, tool schemas include rationale field
        from MMCPAction base class for clean, direct tool execution.
        """
        if instructor_mode is None:
            instructor_mode = get_instructor_mode(user_settings)

        rationale = response.rationale

        # Route using mandatory discriminator pattern
        tool_name = response.tool_call_id
        tool = self.loader.get_tool(tool_name)

        if not tool:
            error_msg = f"Tool '{tool_name}' not found in registry."
            self.history_manager.add_tool_result(history, tool_call_id, error_msg, instructor_mode)
            return None

        # Check if tool is in standby
        if tool.name in self.loader.standby_tools:
            plugin_name = tool.plugin_name
            config_error = self.loader._plugin_config_errors.get(plugin_name, "Setup required")
            result = f"Tool '{tool.name}' is on standby. {config_error}."
            self.history_manager.add_tool_result(
                history,
                tool_call_id,
                result,
                instructor_mode=instructor_mode,
            )
            return None

        tool_name = tool.name

        # Prepare tool_args for EXTERNAL tools (needed for ActionRequestResponse)
        # Discriminator will be stripped in safe_tool_call
        tool_args = response.model_dump(exclude={"rationale", "type"})

        # Check classification
        if tool.classification == "EXTERNAL":
            # Sanitize rationale to prevent injection attacks
            sanitized_rationale = sanitize_explanation(rationale)

            # Template-based fallback if rationale is missing or empty
            if not sanitized_rationale or not sanitized_rationale.strip():
                sanitized_rationale = (
                    f"I'm requesting to use {tool_name} to proceed with your request."
                )

            approval_id = str(uuid.uuid4())
            return ActionRequestResponse(
                approval_id=approval_id,
                explanation=sanitized_rationale,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
            )

        logger.info(f"Executing {tool.name} | Rationale: {rationale}")

        result, is_error = await self.safe_tool_call(tool_call_data=response, context=context)
        self.history_manager.add_tool_result(history, tool_call_id, result, instructor_mode)
        return None

    async def safe_tool_call(
        self,
        tool_call_data: BaseModel,
        context: MMCPContext,
    ) -> tuple[Any, bool] | ActionRequestResponse:
        """
        Primary tool entry point.
        Handles: Lookup -> HITL Check -> Execution -> Result.

        Args:
            tool_call_data: Pydantic model instance with tool_call_id field (mandatory)
            context: MMCP context for error handling
        """
        tool_name = tool_call_data.tool_call_id
        tool = self.loader.get_tool(tool_name)

        if not tool:
            error_msg = f"Tool '{tool_name}' not found in registry."
            logger.error(error_msg, extra={"trace_id": context.runtime.trace_id})
            return error_msg, True

        if tool.classification == "EXTERNAL" and not getattr(context, "is_approved", False):
            return ActionRequestResponse(
                approval_id=str(uuid.uuid4()),
                explanation=sanitize_explanation(tool_call_data.rationale),
                tool_name=tool_name,
                # Be explicit: tool_args is ONLY the payload - exclude all system fields
                tool_args=tool_call_data.model_dump(exclude={"tool_call_id", "rationale", "type"}),
                tool_call_id=getattr(tool_call_data, "id", None),
            )

        args = tool_call_data.model_dump(mode="json")
        # Exclude fields from MMCPToolAction base class and tool_call_id
        args.pop("tool_call_id", None)
        args.pop("rationale", None)
        args.pop("type", None)

        try:
            if self.status_callback:
                await self.status_callback(
                    f"Executing: {tool_name}", context.runtime.trace_id, "tool_start"
                )

            result = await asyncio.wait_for(
                tool.execute(**args), timeout=user_settings.tool_execution_timeout_seconds
            )

            if self.status_callback:
                await self.status_callback(
                    f"Finished: {tool_name}", context.runtime.trace_id, "tool_end"
                )

            # We only stringify because LiteLLM/APIs require a string/serializable content
            if result is None:
                return "Success", False
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False), False
            return str(result), False
        except Exception as e:
            from app.core.errors import map_tool_error

            tool_error = map_tool_error(e, tool_name=tool_name, trace_id=context.runtime.trace_id)
            logger.error(
                f"Tool {tool_name} failed (trace_id={context.runtime.trace_id}): {e}",
                exc_info=True,
                extra={"trace_id": context.runtime.trace_id, "tool_name": tool_name},
            )
            context.increment_tool_failures(tool_name)

            # Sanitized error message
            if isinstance(e, AttributeError):
                error_msg = f"Tool '{tool_name}' has a code error. Admin needs to check the logs."
            elif "401" in str(e):
                error_msg = f"Tool '{tool_name}' authentication failed. Is the API Key configured?"
            else:
                error_msg = (
                    f"Tool '{tool_name}' failed: {tool_error.message}. Try a different approach."
                )
            return error_msg, True

    def _get_reasoned_response_model(self):
        """
        Build the reasoned response model list for single-turn HITL flow.

        Returns a list of Pydantic models (FinalResponse + extended tool schemas).
        Passing a list to Instructor ensures native tool names without the "Response" wrapper.
        """
        from inspect import isclass

        from pydantic import BaseModel

        tools_map = {}
        for tool_name, tool in self.loader.tools.items():
            if hasattr(tool, "input_schema"):
                schema = tool.input_schema
                if schema and isclass(schema) and issubclass(schema, BaseModel):
                    tools_map[tool_name] = schema

        # Use get_reasoned_model to build the list of models
        return get_reasoned_model(tools_map)

    def _extract_error_message(self, error: Exception) -> str:
        """Extract user-friendly error message from exception."""
        from app.core.errors import MMCPError, map_provider_error

        provider_error = map_provider_error(error)
        if isinstance(provider_error, MMCPError):
            return (
                f"I encountered an issue processing your request. Error: {provider_error.message}"
            )
        return f"I encountered an issue processing your request. Error: {str(error)}"
