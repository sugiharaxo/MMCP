"""ReAct Loop - handles the main reasoning and tool execution loop."""

import asyncio
import uuid
from typing import Any

from app.agent.history_manager import HistoryManager
from app.agent.llm_interface import LLMInterface
from app.agent.schemas import ActionRequestResponse, FinalResponse
from app.core.context import MMCPContext
from app.core.errors import MMCPError, map_agent_error, map_provider_error
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader


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

        # Build the Union response model
        ResponseModel = self._get_response_model()

        # ReAct loop: allow multiple reasoning steps
        max_steps = 5
        llm_retry_count = 0
        max_llm_retries = 2
        rate_limit_retry_count = 0
        max_rate_limit_retries = 3

        for _ in range(max_steps):
            context.increment_step()
            self.history_manager.trim_history(history)

            # Get LLM decision
            try:
                context.increment_llm_call()
                if self.status_callback:
                    await self.status_callback(
                        "Thinking about next action...", context.runtime.trace_id, "thought"
                    )
                response = await self.llm.get_agent_decision(
                    history, ResponseModel, context.runtime.trace_id
                )
                llm_retry_count = 0
            except Exception as e:
                # Handle errors with retry logic
                error_handled = await self._handle_llm_error(
                    e,
                    history,
                    context,
                    llm_retry_count,
                    rate_limit_retry_count,
                    max_llm_retries,
                    max_rate_limit_retries,
                )
                if error_handled:
                    llm_retry_count += 1
                    rate_limit_retry_count += 1
                    continue
                else:
                    # Return error message
                    error_msg = self._extract_error_message(e)
                    return error_msg, history

            # Route based on response type (outside try/except)
            if isinstance(response, FinalResponse):
                result = await self._handle_final_response(
                    response, history, context, notification_injector
                )
                return result, history
            else:
                # It's a tool call
                result = await self._handle_tool_call(response, history, context)
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
    ) -> bool:
        """Handle LLM errors. Returns True if should retry."""
        agent_error = map_agent_error(error, trace_id=context.runtime.trace_id)

        # Validation/parsing errors - feed back to LLM for self-correction
        if agent_error.retryable and llm_retry_count < max_llm_retries:
            self.history_manager.add_error_message(history, agent_error.message)
            return True

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
            return True

        return False

    async def _handle_final_response(
        self,
        response: FinalResponse,
        history: list[dict[str, Any]],
        context: MMCPContext,
        notification_injector=None,
    ) -> str:
        """Handle final response from LLM."""
        self.history_manager.add_assistant_response(history, response.answer)

        # Process agent ACKs for asynchronous notifications
        if response.acknowledged_ids and notification_injector:
            user_id = "default"  # Single-user system for now
            await notification_injector.mark_agent_processed(response.acknowledged_ids, user_id)

        context.log_inspection()
        return response.answer

    async def _handle_tool_call(
        self, response: Any, history: list[dict[str, Any]], context: MMCPContext
    ) -> str | ActionRequestResponse | None:
        """Handle tool call. Returns ActionRequestResponse for HITL or None to continue."""
        tool = self.loader.get_tool_by_schema(type(response))
        if not tool:
            tool_name = type(response).__name__
            result = f"Error: Tool for schema '{tool_name}' not found."
            self.history_manager.add_tool_result(history, f"error-{tool_name}", result)
            return None

        # Check if tool is in standby
        if tool.name in self.loader.standby_tools:
            plugin_name = tool.plugin_name
            config_error = self.loader._plugin_config_errors.get(plugin_name, "Setup required")
            result = f"Tool '{tool.name}' is on standby. {config_error}."
            self.history_manager.add_tool_result(history, f"error-{tool.name}", result)
            return None

        # Prepare tool execution
        tool_args = response.model_dump(mode="json")
        tool_name = tool.name

        # Check classification
        if tool.classification == "EXTERNAL":
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
            # Generate HITL request
            return await self._generate_hitl_request(tool_name, tool_args, history, context)

        # Execute internal tool
        if self.status_callback:
            await self.status_callback(
                f"Executing tool: {tool_name}", context.runtime.trace_id, "tool_start"
            )
        result, is_error = await self._safe_tool_call(tool, tool_name, tool_args, context)
        if self.status_callback:
            await self.status_callback(
                f"Completed tool: {tool_name}", context.runtime.trace_id, "tool_end"
            )

        # Add result to history
        call_id = getattr(response, "id", f"internal-{uuid.uuid4()}")
        self.history_manager.add_tool_result(history, call_id, result)

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

    async def _generate_hitl_request(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        history: list[dict[str, Any]],
        context: MMCPContext,
    ) -> ActionRequestResponse:
        """Generate HITL request for external tools."""
        # Contextualization turn to get explanation
        explanation_prompt = (
            "You are about to execute an external tool that requires user approval. "
            "You are currently in an interrupted state - the tool has NOT been executed yet. "
            "Explain to the user what you are about to do in a friendly, contextual manner. "
            "Do not mention the tool name literally - contextualize the action. "
            "Make it clear this is an action you want to take with their permission. "
            f"You are about to execute: {tool_name} with arguments: {tool_args}"
        )

        context_messages = history + [{"role": "system", "content": explanation_prompt}]

        explanation = await self.llm.generate_hitl_explanation(
            context_messages, context.runtime.trace_id
        )

        approval_id = str(uuid.uuid4())
        return ActionRequestResponse(
            approval_id=approval_id,
            explanation=explanation,
            tool_name=tool_name,
            tool_args=tool_args,
        )

    async def _safe_tool_call(
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

    def _get_response_model(self):
        """Build the discriminated Union response model."""
        from functools import reduce
        from inspect import isclass

        from pydantic import BaseModel

        tool_schemas = []
        for tool in self.loader.tools.values():
            if hasattr(tool, "input_schema"):
                schema = tool.input_schema
                if schema and isclass(schema) and issubclass(schema, BaseModel):
                    tool_schemas.append(schema)

        if not tool_schemas:
            return FinalResponse

        ResponseModel = reduce(lambda acc, schema: acc | schema, tool_schemas, FinalResponse)
        return ResponseModel

    def _extract_error_message(self, error: Exception) -> str:
        """Extract user-friendly error message from exception."""
        from app.core.errors import MMCPError, map_provider_error

        provider_error = map_provider_error(error)
        if isinstance(provider_error, MMCPError):
            return (
                f"I encountered an issue processing your request. Error: {provider_error.message}"
            )
        return f"I encountered an issue processing your request. Error: {str(error)}"
