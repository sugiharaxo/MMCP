"""Agent Orchestrator - manages the ReAct loop and conversation context."""

import asyncio
import uuid
from functools import reduce
from inspect import isclass
from typing import Any, Union

from pydantic import BaseModel

from app.agent.schemas import FinalResponse
from app.core.config import settings
from app.core.context import MMCPContext
from app.core.errors import (
    MMCPError,
    map_agent_error,
    map_provider_error,
    map_tool_error,
)
from app.core.health import HealthMonitor
from app.core.llm import get_agent_decision
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.core.utils.pruner import ContextPruner


class AgentOrchestrator:
    """
    Manages the agent's conversation history and tool execution loop.

    Implements character-based context management for efficient memory usage.
    Uses Instructor's discriminated Union pattern: LLM returns FinalResponse or a tool input schema directly.
    """

    def __init__(self, loader: PluginLoader, health: HealthMonitor | None = None):
        """
        Initialize the orchestrator.

        Args:
            loader: PluginLoader instance with loaded tools.
            health: HealthMonitor instance (singleton from app state). If None, creates a new one
                   (for backward compatibility, but not recommended for production).
        """
        self.loader = loader
        self.history: list[dict] = []
        self.health = health or HealthMonitor()

    def _get_system_prompt(self, context: MMCPContext | None = None) -> str:
        """
        Generate system prompt with dynamic tool descriptions and context-aware state.

        Simplified prompt focusing on "when to use tools" rather than formatting rules.
        The Union response_model handles the "how" of formatting automatically.

        Injects dynamic state from context (user preferences, media state) to make
        the LLM aware of current server state.
        """
        if not self.loader:
            tool_desc = "No tools are currently available."
        else:
            tool_descriptions = []
            for idx, (name, tool) in enumerate(self.loader.tools.items(), start=1):
                desc = f"{idx}. {name}: {tool.description}"

                # Add clean argument descriptions instead of raw JSON schema
                if hasattr(tool, "input_schema"):
                    schema = tool.input_schema
                    schema_json = schema.model_json_schema()
                    properties = schema_json.get("properties", {})
                    required = schema_json.get("required", [])

                    if properties:
                        arg_list = []
                        for prop_name, prop_info in properties.items():
                            prop_type = prop_info.get("type", "any")
                            is_required = prop_name in required
                            prop_desc = prop_info.get("description", "")
                            default = prop_info.get("default", None)

                            if is_required:
                                arg_list.append(
                                    f"  - {prop_name} ({prop_type}): {prop_desc} [REQUIRED]"
                                )
                            else:
                                default_str = f", default: {default}" if default is not None else ""
                                arg_list.append(
                                    f"  - {prop_name} ({prop_type}): {prop_desc} [optional{default_str}]"
                                )

                        if arg_list:
                            desc += "\n   Args:\n" + "\n".join(arg_list)

                tool_descriptions.append(desc)

            tool_desc = "\n".join(tool_descriptions)

        # Get the sanitized LLM payload for context-aware prompts
        context_section = ""
        if context:
            state = context.get_llm_payload()
            # Build human-readable state description
            state_parts = []
            if state.get("user_preferences"):
                state_parts.append(f"User Preferences: {state['user_preferences']}")
            if state.get("media_state"):
                state_parts.append(f"Media State: {state['media_state']}")
            if state_parts:
                state_desc = "\n".join(state_parts)
                context_section = f"CONTEXT:\n{state_desc}"

        # Add standby alerts for system awareness
        standby_alerts = self._get_standby_alerts()

        return f"""You are MMCP (Modular Media Control Plane), an intelligent media assistant.
You help users manage their media library, search for metadata, and handle downloads.

IDENTITY:
- Be concise and helpful.
- Use tools to fetch data before making recommendations.
- If a tool fails, explain why to the user and try a different approach.

{context_section}

{standby_alerts}

AVAILABLE TOOLS:
{tool_desc}

Use tools when you need specific information or actions. When you have enough information to answer the user, provide a FinalResponse with your answer."""

    def _get_standby_alerts(self) -> str:
        """
        Generate system prompt section for plugins that failed to load.

        Only reports plugins that FAILED to load, providing the 'Proprioception'.
        This keeps the context lean while enabling self-resolution.
        """
        standby = self.loader.standby_tools
        if not standby:
            return ""

        alerts = ["DISABLED CAPABILITIES (SYSTEM ERRORS):"]
        for name, tool in standby.items():
            # Get the system-managed error (e.g., 'Configuration validation failed')
            reason = self.loader._plugin_config_errors.get(name, "Unknown system error")
            alerts.append(f"- {name}: {reason}")

        return "\n".join(alerts)

    def _get_response_model(self) -> type:
        """
        Build the discriminated Union response model: Union[FinalResponse, ToolSchema1, ToolSchema2, ...]

        The LLM will return one of these types directly. We use isinstance() to route.
        """
        tool_schemas = []
        for tool in self.loader.tools.values():
            if hasattr(tool, "input_schema"):
                schema = tool.input_schema
                # Verify it's actually a class before adding
                if schema and isclass(schema) and issubclass(schema, BaseModel):
                    tool_schemas.append(schema)
                else:
                    logger.warning(
                        f"Tool '{tool.name}' has invalid input_schema: {schema}. "
                        f"Expected a BaseModel subclass."
                    )

        if not tool_schemas:
            # No tools available - can only return FinalResponse
            return FinalResponse

        # Build Union using typing.Union for dynamic construction in reduce()
        # Note: Python automatically normalizes nested Unions (Union[Union[A, B], C] -> Union[A, B, C])
        # We use typing.Union (not | operator) because:
        # 1. The | operator cannot be used in reduce() lambda expressions
        # 2. typing.Union is more explicit and compatible with Instructor/Pydantic
        # 3. Instructor (via Pydantic) handles Union types correctly for discriminated unions
        ResponseModel = reduce(lambda acc, schema: Union[acc, schema], tool_schemas, FinalResponse)  # noqa: UP007

        return ResponseModel

    def _emit_status(
        self,
        message: str,
        trace_id: str | None = None,
        status_type: str = "info",  # noqa: UP007
    ):
        """
        Emit status messages for visibility and future extensibility.

        Logs structured messages for observability. UI components should subscribe
        to log streams rather than scraping console output.

        Args:
            message: The status message to emit
            trace_id: Optional trace ID for log correlation
            status_type: Type of status update ('tool_start', 'tool_end', 'thought', 'info')
                        Used by frontend to filter and display appropriate updates
        """
        extra = {"is_status_update": True, "status_type": status_type}
        if trace_id:
            extra["trace_id"] = trace_id
        logger.info(message, extra=extra)

    def _trim_history(self):
        """
        Trims history based on character count.
        Always preserves the system prompt (index 0).
        """
        if len(self.history) <= 1:
            return

        while True:
            # Calculate total character count of the history
            current_chars = sum(len(m.get("content", "")) for m in self.history)

            # If we are under the limit or only have system prompt left, stop
            if current_chars <= settings.llm_max_context_chars or len(self.history) <= 2:
                break

            # Remove the oldest non-system message (index 1)
            # Index 0 is System, Index 1 is the oldest User/Assistant message
            self.history.pop(1)
            # Recalculate after pop for accurate logging
            current_chars = sum(len(m.get("content", "")) for m in self.history)
            logger.debug(f"Trimmed context to {current_chars} chars.")

    async def _safe_execute_provider(
        self, provider, user_input: str
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Safely execute a context provider with timeout and error handling.

        Args:
            provider: The MMCPContextProvider instance.
            user_input: The user's query for eligibility checking.

        Returns:
            Tuple of (provider_key, data_dict or None if failed).
        """
        provider_key = provider.context_key

        try:
            # Check eligibility first (lightweight)
            if not await provider.is_eligible(user_input):
                return provider_key, None

            # Check health (circuit breaker)
            if not self.health.is_available(provider_key):
                logger.debug(f"Context provider '{provider_key}' is circuit-broken, skipping")
                return provider_key, None

            # Execute with per-provider timeout using PluginContext facade
            timeout_seconds = settings.context_per_provider_timeout_ms / 1000.0
            plugin_context = self.loader.create_plugin_context()
            result = await asyncio.wait_for(
                provider.provide_context(plugin_context), timeout=timeout_seconds
            )

            # Extract data from ContextResponse
            data = result.data
            # TTL is available in ContextResponse but not currently used (future enhancement)
            # Could implement caching based on result.ttl if needed

            # Truncate if needed (Safety Fuse approach)
            truncated_data = ContextPruner.truncate_provider_data(data, provider_key, settings)

            # Record success
            self.health.record_success(provider_key)

            return provider_key, truncated_data

        except asyncio.TimeoutError:
            logger.warning(f"Context provider '{provider_key}' timed out after {timeout_seconds}s")
            self.health.record_failure(provider_key)
            return provider_key, None

        except Exception as e:
            logger.error(
                f"Context provider '{provider_key}' failed: {e}",
                exc_info=True,
            )
            self.health.record_failure(provider_key)
            return provider_key, None

    async def assemble_llm_context(self, user_input: str, context: MMCPContext) -> None:
        """
        Assemble LLM context by running all eligible context providers in parallel.

        This is the "ReAct preparation phase" - fetches dynamic state before the loop begins.
        Implements global timeout, per-provider timeouts, and circuit breaker protection.

        Args:
            user_input: The user's query (for eligibility filtering).
            context: The MMCPContext to update with provider data.
        """
        if not self.loader.context_providers:
            logger.debug("No context providers registered")
            return

        # Filter by eligibility and health
        active_providers = []
        for provider in self.loader.context_providers.values():
            # Quick eligibility check (synchronous check first, then async if needed)
            if not self.health.is_available(provider.context_key):
                continue
            active_providers.append(provider)

        if not active_providers:
            logger.debug("No eligible context providers available")
            return

        logger.info(
            f"Assembling context from {len(active_providers)} provider(s) "
            f"(global timeout: {settings.context_global_timeout_ms}ms)"
        )

        # Run all providers in parallel with global timeout
        tasks = [self._safe_execute_provider(p, user_input) for p in active_providers]

        try:
            global_timeout_seconds = settings.context_global_timeout_ms / 1000.0
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=global_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Context assembly timed out after {global_timeout_seconds}s "
                f"(global timeout exceeded)"
            )
            return

        # Update LLM media state with successful results
        # Note: _safe_execute_provider wraps all exceptions, so results will be tuples, not Exceptions
        for result in results:
            # Defensive check (shouldn't happen, but safe programming)
            if isinstance(result, Exception):
                logger.error(f"Context provider raised exception: {result}", exc_info=True)
                continue

            provider_key, data = result
            if data is not None:
                context.llm.media_state[provider_key] = data
                logger.debug(f"Updated media_state with data from '{provider_key}'")
            else:
                logger.debug(f"Provider '{provider_key}' returned no data (skipped or failed)")

    async def safe_tool_call(
        self, tool, tool_name: str, args: dict, context: MMCPContext
    ) -> tuple[str, bool]:
        """
        Safely execute a tool.

        The response from the LLM is already validated (it's a Pydantic model instance),
        so we can directly use model_dump() to get the arguments.

        Errors are mapped to ToolError and converted to a simple string that the LLM can process.
        Full error details are logged with trace_id for correlation.

        Args:
            tool: The tool instance to execute
            tool_name: Name of the tool (for error messages)
            args: Arguments dict (already validated by Pydantic)
            context: MMCPContext for this execution

        Returns:
            Tool result as string, or error message if execution failed
        """
        try:
            # Create PluginContext for tool execution (tools expect PluginContext, not MMCPContext)
            plugin_context = self.loader.create_plugin_context()

            # Retrieve plugin settings loaded in Phase 1
            settings = self.loader._plugin_settings.get(tool_name, None)

            # Execute tool with PluginContext, settings, and validated arguments
            result = await asyncio.wait_for(
                tool.execute(plugin_context, settings, **args), timeout=30.0
            )
            return str(result) if result is not None else "Tool executed successfully.", False

        except Exception as e:
            # Map to ToolError for logging, but return simple string for LLM
            tool_error = map_tool_error(e, tool_name=tool_name, trace_id=context.runtime.trace_id)
            logger.error(
                f"Tool {tool_name} failed (trace_id={context.runtime.trace_id}): {e}",
                exc_info=True,
                extra={"trace_id": context.runtime.trace_id, "tool_name": tool_name},
            )
            # Track tool failures to detect repeated issues
            context.increment_tool_failures(tool_name)

            # Return sanitized error message that LLM can understand and act on
            # Handle specific error types for better user experience
            if isinstance(e, AttributeError):
                error_msg = f"Tool '{tool_name}' has a code error. Admin needs to check the logs."
            elif "401" in str(e):
                error_msg = f"Tool '{tool_name}' authentication failed. Is the API Key configured?"
            else:
                error_msg = f"Tool '{tool_name}' failed: {tool_error.message}. Try a different approach or check if the tool is configured correctly."
            return error_msg, True

    async def chat(self, user_input: str, trace_id: str | None = None) -> str:
        """
        The main ReAct loop using native tool calling.

        Processes user input through multiple reasoning steps:
        1. LLM returns FinalResponse or a tool input schema (Union pattern)
        2. If tool schema: Execute tool and feed result back
        3. If FinalResponse: Return to user
        4. Repeat until FinalResponse or max steps reached

        Errors are handled gracefully:
        - Tool failures are fed back to the LLM so it can adapt
        - LLM decision failures are retried with error context
        - Fatal errors (config, auth) are raised as MMCPError

        Args:
            user_input: The user's message.
            trace_id: Optional trace ID for log correlation (generated if not provided)

        Returns:
            The final response message to the user.

        Raises:
            MMCPError: For fatal errors (configuration, authentication)
        """
        # Generate trace_id if not provided
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        # Create MMCP context for this conversation
        context = MMCPContext(trace_id=trace_id)

        # Populate LLM context with available tools
        context.set_available_tools(self.loader.list_tools())

        # Assemble dynamic context from providers (ReAct preparation phase)
        # This runs on EVERY user message to ensure fresh context is available
        await self.assemble_llm_context(user_input, context)

        # 2. Reconstruct history: PIN system at top, keep ALL other history
        # We only filter out 'system' to prevent duplicates.
        # We MUST keep 'user', 'assistant', and 'tool' roles.
        system_message = {"role": "system", "content": self._get_system_prompt(context)}
        messages = [m for m in self.history if m["role"] != "system"]
        self.history = [system_message] + messages

        # 3. Add the user's current message
        self.history.append({"role": "user", "content": user_input})

        # Build the Union response model: FinalResponse | ToolSchema1 | ToolSchema2 | ...
        ResponseModel = self._get_response_model()

        # ReAct loop: allow multiple reasoning steps
        max_steps = 5  # Prevent infinite loops
        llm_retry_count = 0
        max_llm_retries = 2  # Allow LLM to self-correct on validation errors
        rate_limit_retry_count = 0
        max_rate_limit_retries = 3  # Allow up to 3 retries for rate limits

        for step in range(max_steps):
            context.increment_step()
            self._trim_history()

            # 1. Ask the LLM what to do
            try:
                context.increment_llm_call()
                response = await get_agent_decision(self.history, response_model=ResponseModel)
                llm_retry_count = 0  # Reset retry count on success
            except Exception as e:
                # Map the error
                agent_error = map_agent_error(e, trace_id=trace_id)

                # For validation/parsing errors, feed back to LLM for self-correction
                if agent_error.retryable and llm_retry_count < max_llm_retries:
                    llm_retry_count += 1
                    logger.warning(
                        f"LLM validation error (attempt {llm_retry_count}/{max_llm_retries}), "
                        f"feeding back to LLM for self-correction (trace_id={trace_id})"
                    )
                    # Feed error back to LLM so it can correct itself
                    self.history.append(
                        {
                            "role": "assistant",
                            "content": f"Error: {agent_error.message} Please provide a valid response.",
                        }
                    )
                    continue  # Retry the LLM call

                # For provider errors, check if they're fatal
                provider_error = map_provider_error(e, trace_id=trace_id)
                if isinstance(provider_error, MMCPError) and not provider_error.retryable:
                    # Fatal error (e.g., config issue) - raise it
                    logger.error(
                        f"Fatal provider error (trace_id={trace_id}): {e}",
                        exc_info=True,
                        extra={"trace_id": trace_id},
                    )
                    raise provider_error from e

                # Special handling for rate limit errors - implement exponential backoff
                is_rate_limit = "rate limit" in str(e).lower() or "429" in str(e)
                if is_rate_limit and rate_limit_retry_count < max_rate_limit_retries:
                    rate_limit_retry_count += 1
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_seconds = 2 ** (rate_limit_retry_count - 1)
                    logger.warning(
                        f"Rate limit hit (attempt {rate_limit_retry_count}/{max_rate_limit_retries}), "
                        f"backing off for {backoff_seconds}s (trace_id={trace_id})"
                    )
                    await asyncio.sleep(backoff_seconds)
                    continue  # Retry the LLM call after backoff

                # Non-fatal provider error or max retries reached - return user-friendly message
                logger.error(
                    f"Failed to get agent decision after retries (trace_id={trace_id}): {e}",
                    exc_info=True,
                    extra={"trace_id": trace_id},
                )
                error_message = (
                    provider_error.message
                    if isinstance(provider_error, MMCPError)
                    else agent_error.message
                )
                return f"I encountered an issue processing your request. Error: {error_message}"

            # 2. Route based on response type using isinstance

            if isinstance(response, FinalResponse):
                # Agent wants to respond to user - we're done
                self._emit_status(
                    f"\n[Step {step + 1}] Final Response: {response.answer}",
                    trace_id,
                    status_type="thought",
                )
                context.log_inspection()
                self.history.append({"role": "assistant", "content": response.answer})
                return response.answer

            # It's a tool! Map the schema class to the tool instance
            tool = self.loader.get_tool_by_schema(type(response))
            if not tool:
                # Tool not found for this schema - this shouldn't happen if mapping is correct
                tool_name = type(response).__name__
                result = (
                    f"Error: Tool for schema '{tool_name}' not found. "
                    f"Available tools: {list(self.loader.list_tools().keys())}"
                )
                logger.error(
                    result,
                    extra={"trace_id": trace_id, "schema_class": tool_name},
                )
                # Feed error back and continue loop
                self.history.append(
                    {
                        "role": "user",
                        "content": f"Observation: {result}",
                    }
                )
                continue

            # Check if tool is in standby (not fully configured)
            tool_name = tool.name
            is_standby = tool_name in self.loader.standby_tools

            if is_standby:
                # Tool is on standby - return user-friendly error with setup instructions
                config_error = self.loader._plugin_config_errors.get(tool_name, "Setup required")
                # Build helpful error message with env var names
                env_prefix = f"MMCP_PLUGIN_{self.loader._slugify_plugin_name(tool_name)}_"
                result = (
                    f"Tool '{tool_name}' is on standby. {config_error}. "
                    f"Set environment variables starting with '{env_prefix}' or configure via the Web UI at /api/v1/settings/plugins/{self.loader._slugify_plugin_name(tool_name).lower()}."
                )
                logger.info(
                    f"Standby tool '{tool_name}' was called (trace_id={trace_id})",
                    extra={"trace_id": trace_id, "tool_name": tool_name},
                )
                # Feed error back to LLM so it can inform the user
                self.history.append(
                    {
                        "role": "user",
                        "content": f"Observation from {tool_name}: {result}",
                    }
                )
                continue

            # Execute the tool with validated arguments (response is already a Pydantic model)
            # Use mode='json' to ensure SecretStr fields are masked (not stringified as SecretStr('**********'))
            tool_args = response.model_dump(mode="json")
            self._emit_status(
                f"[Step {step + 1}] Action: Call tool '{tool_name}' with args {tool_args}",
                trace_id,
                status_type="tool_start",
            )
            context.increment_tool_execution()
            logger.info(
                f"Executing tool: {tool_name} with {tool_args}",
                extra={"trace_id": trace_id, "tool_name": tool_name},
            )

            result, is_error = await self.safe_tool_call(tool, tool_name, tool_args, context)
            self._emit_status(
                f"[Step {step + 1}] Observation: {result}",
                trace_id,
                status_type="tool_end",
            )

            # Check for repeated tool failures - if same tool fails 3+ times, force final response
            if is_error and context.is_tool_circuit_breaker_tripped(tool_name, threshold=3):
                failure_count = context.get_tool_failure_count(tool_name)
                msg = f"I encountered a persistent issue with the {tool_name} tool. {result}"
                logger.error(
                    f"Circuit breaker triggered for {tool_name} after {failure_count} failures "
                    f"(trace_id={trace_id})",
                    extra={
                        "trace_id": trace_id,
                        "tool_name": tool_name,
                        "failure_count": failure_count,
                    },
                )
                # Immediate return - bypasses the rest of the loop and prevents LLM from trying again
                context.log_inspection()
                return msg

            # Feed tool result back into history for next iteration
            # This allows the LLM to process errors and adapt its strategy
            self.history.append(
                {
                    "role": "user",
                    "content": f"Observation from {tool_name}: {result}",
                }
            )
            # Loop continues so LLM can process the result

        # If we've exhausted max steps, return a message
        logger.warning(
            f"ReAct loop exhausted {max_steps} steps without final response (trace_id={trace_id})",
            extra={"trace_id": trace_id, "max_steps": max_steps},
        )
        context.log_inspection()
        self._emit_status(
            f"[System] ReAct loop exhausted {max_steps} steps without resolution",
            trace_id,
            status_type="info",
        )
        return "I attempted to solve this but reached my reasoning limit. Please try being more specific or breaking down your request."
