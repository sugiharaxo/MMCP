"""ReAct Loop - handles the main reasoning and tool execution loop."""

import asyncio
import uuid
from typing import Any

import instructor

from app.agent.history_manager import HistoryManager
from app.agent.llm_interface import LLMInterface
from app.agent.schemas import (
    ActionRequestResponse,
    FinalResponse,
    ReasonedToolCall,
    get_reasoned_model,
)
from app.core.config import settings
from app.core.context import MMCPContext
from app.core.errors import MMCPError, map_agent_error, map_provider_error
from app.core.llm import get_instructor_mode
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
        system_prompt: str,
        notification_injector=None,
    ) -> tuple[str | ActionRequestResponse, list[dict[str, Any]]]:
        """
        Execute a single ReAct turn.

        Args:
            user_input: The user's message
            session_history: Current conversation history
            context: MMCP context
            system_prompt: The system prompt to use

        Returns:
            Tuple of (response object, updated history list)
        """
        # Reconstruct history for this turn
        history = self.history_manager.reconstruct_history(
            session_history, system_prompt, user_input
        )

        # Build the reasoned Union response model (FinalResponse | ReasonedToolCall)
        ResponseModel = self._get_reasoned_response_model()

        instructor_mode = get_instructor_mode(settings.llm_model)

        # ReAct loop: allow multiple reasoning steps
        max_steps = 5
        llm_retry_count = 0
        max_llm_retries = 2
        rate_limit_retry_count = 0
        max_rate_limit_retries = 3

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
            elif isinstance(response, ReasonedToolCall):
                # For ReasonedToolCall: Extract tool_call_id and add LLM message to preserve
                # tool_calls metadata for DeepSeek 1:1 pairing requirement
                tool_call_id = None
                all_tool_calls = []

                # CRITICAL: Add assistant message to history IMMEDIATELY before tool execution
                # This preserves the tool_calls metadata for Mode.TOOLS compatibility
                assistant_msg = raw_completion.choices[0].message
                self.history_manager.add_llm_message(history, assistant_msg, instructor_mode)

                # Extract tool_call_id if in TOOLS mode
                # CRITICAL: Extract from raw_completion.choices[0].message.tool_calls[0].id
                # for DeepSeek 1:1 pairing requirement
                # STRICT STATE PERSISTENCE: If TOOLS mode, tool_call_id is mandatory
                if instructor_mode == instructor.Mode.TOOLS:
                    # If the LLM failed to give us an ID in TOOLS mode,
                    # that is a system failure, not a scenario for a fallback.
                    if not hasattr(raw_completion.choices[0].message, "tool_calls"):
                        raise MMCPError(
                            "Protocol Corruption: LLM response missing tool_calls in TOOLS mode",
                            trace_id=context.runtime.trace_id,
                        )
                    tool_calls = raw_completion.choices[0].message.tool_calls
                    if not tool_calls or len(tool_calls) == 0:
                        raise MMCPError(
                            "Protocol Corruption: LLM response has empty tool_calls in TOOLS mode",
                            trace_id=context.runtime.trace_id,
                        )
                    # Extract all tool call IDs for protocol compliance
                    # LLM providers require results for ALL tool calls, not just the first
                    all_tool_calls = tool_calls
                    tool_call_id = tool_calls[0].id
                    if not tool_call_id:
                        raise MMCPError(
                            "Protocol Corruption: tool_call missing ID in TOOLS mode",
                            trace_id=context.runtime.trace_id,
                        )
                    if len(tool_calls) > 1:
                        logger.warning(
                            f"Multiple tool calls detected ({len(tool_calls)}). "
                            f"Executing primary tool, additional calls will receive error responses.",
                            extra={"trace_id": context.runtime.trace_id},
                        )
                # Handle reasoned tool call (single-turn atomic consistency)
                result = await self._handle_reasoned_tool_call(
                    response,
                    history,
                    context,
                    tool_call_id=tool_call_id,
                    instructor_mode=instructor_mode,
                )

                # Handle multiple tool calls: provide error results for additional tool calls
                if instructor_mode == instructor.Mode.TOOLS and len(all_tool_calls) > 1:
                    for additional_tc in all_tool_calls[1:]:
                        if additional_tc.id:
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
            else:
                # Fallback: treat as regular tool call (for backward compatibility)
                # Extract tool_call_id if in TOOLS mode (similar to ReasonedToolCall)
                tool_call_id = None
                all_tool_calls = []

                # Add LLM message to preserve tool_calls metadata
                assistant_msg = raw_completion.choices[0].message
                self.history_manager.add_llm_message(history, assistant_msg, instructor_mode)

                # If TOOLS mode, tool_call_id is mandatory
                if instructor_mode == instructor.Mode.TOOLS:
                    if not hasattr(raw_completion.choices[0].message, "tool_calls"):
                        raise MMCPError(
                            "Protocol Corruption: LLM response missing tool_calls in TOOLS mode",
                            trace_id=context.runtime.trace_id,
                        )
                    tool_calls = raw_completion.choices[0].message.tool_calls
                    if not tool_calls or len(tool_calls) == 0:
                        raise MMCPError(
                            "Protocol Corruption: LLM response has empty tool_calls in TOOLS mode",
                            trace_id=context.runtime.trace_id,
                        )
                    # Extract all tool call IDs for protocol compliance
                    all_tool_calls = tool_calls
                    tool_call_id = tool_calls[0].id
                    if not tool_call_id:
                        raise MMCPError(
                            "Protocol Corruption: tool_call missing ID in TOOLS mode",
                            trace_id=context.runtime.trace_id,
                        )
                    if len(tool_calls) > 1:
                        logger.warning(
                            f"Multiple tool calls detected ({len(tool_calls)}). "
                            f"Executing primary tool, additional calls will receive error responses.",
                            extra={"trace_id": context.runtime.trace_id},
                        )

                result = await self._handle_tool_call(
                    response,
                    history,
                    context,
                    tool_call_id=tool_call_id,
                    instructor_mode=instructor_mode,
                )

                # Handle multiple tool calls: provide error results for additional tool calls
                if instructor_mode == instructor.Mode.TOOLS and len(all_tool_calls) > 1:
                    for additional_tc in all_tool_calls[1:]:
                        if additional_tc.id:
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
            backoff_seconds = 2 ** (rate_limit_retry_count - 1)
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

    async def _handle_reasoned_tool_call(
        self,
        response: ReasonedToolCall,
        history: list[dict[str, Any]],
        context: MMCPContext,
        tool_call_id: str | None = None,
        instructor_mode: instructor.Mode | None = None,
    ) -> str | ActionRequestResponse | None:
        """
        Handle reasoned tool call from single-turn reasoning phase.

        Extracts the tool and its rationale, then routes based on classification:
        - EXTERNAL: Sanitize rationale and return ActionRequestResponse for HITL
        - INTERNAL: Log rationale for auditability and execute tool
        """
        tool_call = response.tool_call
        rationale = response.rationale

        if instructor_mode is None:
            instructor_mode = get_instructor_mode(settings.llm_model)

        # Extract tool from registry
        tool = self.loader.get_tool_by_schema(type(tool_call))
        if not tool:
            tool_name = type(tool_call).__name__
            result = f"Error: Tool for schema '{tool_name}' not found."
            self.history_manager.add_tool_result(
                history,
                tool_call_id,
                result,
                instructor_mode=instructor_mode,
            )
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

        # Prepare tool execution
        tool_args = tool_call.model_dump(mode="json")
        tool_name = tool.name

        # Check classification
        if tool.classification == "EXTERNAL":
            # Sanitize rationale to prevent injection attacks
            sanitized_rationale = sanitize_explanation(rationale)

            # Template-based fallback if rationale is missing or empty
            if not sanitized_rationale or not sanitized_rationale.strip():
                sanitized_rationale = (
                    f"I'm requesting to use {tool_name} to proceed with your request."
                )
            # TODO Implement TTL (Time-To-Live) for HITL requests.
            # Stale requests (> 24h for example) should be auto-expired to prevent database bloat
            # and unintended actions from a shifted context.

            # TODO Add 'Contextual Re-validation' for approvals older than 1 hour (example).
            # If a user takes several hours to approve, the agent should perform
            # a silent check (e.g., verify URL still works) before final execution.

            # TODO(ANP): Internal Turn Escalation - ANP Spec Section 6.1
            # When an EXTERNAL tool is invoked during an Internal Turn (triggered by ANP Channel C
            # with Target=AGENT), the protocol requires automatic promotion of Target→USER to ensure
            # user visibility. Currently, all EXTERNAL tools trigger HITL (ActionRequestResponse),
            # which is safer but doesn't implement the ANP internal turn escalation mechanism.
            #
            # Required implementation:
            # 1. Detect if current turn is "internal" (triggered by ANP notification with Target=AGENT)
            # 2. If internal turn + EXTERNAL tool: call notification_injector.promote_external_tool()
            #    to promote Target→USER and re-route to Channel A/B
            # 3. Set FinalResponse.internal_turn=True when appropriate
            # 4. Ensure user visibility is guaranteed per ANP spec Section 6.1.1
            #
            # See: docs/specs/anp-v1.0.md Section 6.1 "The Internal Turn Protection"

            approval_id = str(uuid.uuid4())
            return ActionRequestResponse(
                approval_id=approval_id,
                explanation=sanitized_rationale,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
            )

        # Execute internal tool
        # Log rationale for auditability
        logger.info(
            f"Executing INTERNAL tool '{tool_name}' with rationale: {rationale} "
            f"(trace_id={context.runtime.trace_id})",
            extra={
                "trace_id": context.runtime.trace_id,
                "tool_name": tool_name,
                "rationale": rationale,
            },
        )

        if self.status_callback:
            await self.status_callback(
                f"Executing tool: {tool_name}", context.runtime.trace_id, "tool_start"
            )
        result, is_error = await self.safe_tool_call(tool, tool_name, tool_args, context)
        if self.status_callback:
            await self.status_callback(
                f"Completed tool: {tool_name}", context.runtime.trace_id, "tool_end"
            )

        # Add result to history using provided tool_call_id
        # In JSON mode, tool_call_id is None and ignored by add_tool_result
        self.history_manager.add_tool_result(
            history, tool_call_id, result, instructor_mode=instructor_mode
        )

        # Check for circuit breaker
        if is_error and context.is_tool_circuit_breaker_tripped(tool_name, threshold=3):
            failure_count = context.get_tool_failure_count(tool_name)
            msg = f"I encountered a persistent issue with the {tool_name} tool. {result}"
            logger.error(
                f"Circuit breaker triggered for {tool_name} after {failure_count} failures "
                f"(trace_id={context.runtime.trace_id})",
                extra={
                    "trace_id": context.runtime.trace_id,
                    "tool_name": tool_name,
                    "failure_count": failure_count,
                },
            )
            return msg

        return None

    async def _handle_tool_call(
        self,
        response: Any,
        history: list[dict[str, Any]],
        context: MMCPContext,
        tool_call_id: str | None = None,
        instructor_mode: instructor.Mode | None = None,
    ) -> str | ActionRequestResponse | None:
        """
        Handle tool call (legacy method for backward compatibility).

        This method is used when the response is not a ReasonedToolCall,
        maintaining backward compatibility with the old flow.
        """
        if instructor_mode is None:
            instructor_mode = get_instructor_mode(settings.llm_model)

        tool = self.loader.get_tool_by_schema(type(response))
        if not tool:
            tool_name = type(response).__name__
            result = f"Error: Tool for schema '{tool_name}' not found."
            self.history_manager.add_tool_result(
                history,
                tool_call_id,
                result,
                instructor_mode=instructor_mode,
            )
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

        # Prepare tool execution
        tool_args = response.model_dump(mode="json")
        tool_name = tool.name

        # Check classification
        if tool.classification == "EXTERNAL":
            # Template-based fallback for legacy flow
            explanation = f"I'm requesting to use {tool_name} to proceed with your request."
            sanitized_explanation = sanitize_explanation(explanation)

            approval_id = str(uuid.uuid4())
            return ActionRequestResponse(
                approval_id=approval_id,
                explanation=sanitized_explanation,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_call_id=tool_call_id,
            )

        # Execute internal tool
        if self.status_callback:
            await self.status_callback(
                f"Executing tool: {tool_name}", context.runtime.trace_id, "tool_start"
            )
        result, is_error = await self.safe_tool_call(tool, tool_name, tool_args, context)
        if self.status_callback:
            await self.status_callback(
                f"Completed tool: {tool_name}", context.runtime.trace_id, "tool_end"
            )

        # Add result to history using provided tool_call_id
        # In JSON mode, tool_call_id is None and ignored by add_tool_result
        self.history_manager.add_tool_result(
            history, tool_call_id, result, instructor_mode=instructor_mode
        )

        # Check for circuit breaker
        if is_error and context.is_tool_circuit_breaker_tripped(tool_name, threshold=3):
            failure_count = context.get_tool_failure_count(tool_name)
            msg = f"I encountered a persistent issue with the {tool_name} tool. {result}"
            logger.error(
                f"Circuit breaker triggered for {tool_name} after {failure_count} failures "
                f"(trace_id={context.runtime.trace_id})",
                extra={
                    "trace_id": context.runtime.trace_id,
                    "tool_name": tool_name,
                    "failure_count": failure_count,
                },
            )
            return msg

        return None

    async def safe_tool_call(
        self, tool, tool_name: str, args: dict[str, Any], context: MMCPContext
    ):
        """Safely execute a tool with error handling."""
        try:
            result = await asyncio.wait_for(tool.execute(**args), timeout=30.0)
            return str(result) if result is not None else "Tool executed successfully.", False
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
        Build the reasoned Union response model for single-turn HITL flow.

        Returns Union[FinalResponse, ReasonedToolCall] where tool_call is
        a Union of all available tool schemas.
        """
        from inspect import isclass

        from pydantic import BaseModel

        tool_schemas = []
        for tool in self.loader.tools.values():
            if hasattr(tool, "input_schema"):
                schema = tool.input_schema
                if schema and isclass(schema) and issubclass(schema, BaseModel):
                    tool_schemas.append(schema)

        # Use get_reasoned_model to build the Union with ReasonedToolCall
        return get_reasoned_model(tool_schemas)

    def _extract_error_message(self, error: Exception) -> str:
        """Extract user-friendly error message from exception."""
        from app.core.errors import MMCPError, map_provider_error

        provider_error = map_provider_error(error)
        if isinstance(provider_error, MMCPError):
            return (
                f"I encountered an issue processing your request. Error: {provider_error.message}"
            )
        return f"I encountered an issue processing your request. Error: {str(error)}"
