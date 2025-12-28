"""Agent Orchestrator - manages the ReAct loop and conversation context."""

import asyncio
import json
import uuid
from typing import Any, Union

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
from app.core.plugin_interface import ContextResponse
from app.core.plugin_loader import PluginLoader


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
                context_section = f"CONTEXT:\n{state_desc}\n\n"

        return f"""You are MMCP (Modular Media Control Plane), an intelligent media assistant.
You help users manage their media library, search for metadata, and handle downloads.

IDENTITY:
- Be concise and helpful.
- Use tools to fetch data before making recommendations.
- If a tool fails, explain why to the user and try a different approach.

{context_section}AVAILABLE TOOLS:
{tool_desc}

Use tools when you need specific information or actions. When you have enough information to answer the user, provide a FinalResponse with your answer."""

    def _get_response_model(self) -> type:
        """
        Build the discriminated Union response model: FinalResponse | ToolSchema1 | ToolSchema2 | ...

        The LLM will return one of these types directly. We use isinstance() to route.
        """
        tool_schemas = []
        for tool in self.loader.tools.values():
            if hasattr(tool, "input_schema") and tool.input_schema:
                tool_schemas.append(tool.input_schema)

        if not tool_schemas:
            # No tools available - can only return FinalResponse
            return FinalResponse

        # Build Union: FinalResponse | Tool1 | Tool2 | ...
        # Start with FinalResponse, then add all tool schemas
        # Note: Using Union for dynamic runtime construction (linter warning is false positive)
        ResponseModel: type = FinalResponse
        for schema in tool_schemas:
            ResponseModel = Union[ResponseModel, schema]  # type: ignore[assignment]

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

    def _prune_dict(self, data: Any, max_chars: int, current_size: int = 0) -> tuple[Any, int]:
        """
        Recursively prune dictionary/list to fit within character limit.

        Uses "Lumberjack" approach: limits list lengths and string lengths
        before serialization to avoid JSON parsing errors.

        Args:
            data: The data structure to prune (dict, list, or primitive).
            max_chars: Maximum character count allowed.
            current_size: Current character count (for tracking).

        Returns:
            Tuple of (pruned_data, estimated_size).
        """
        # Base case: primitive types
        if isinstance(data, (str, int, float, bool, type(None))):
            str_repr = str(data)
            max_string_len = settings.context_max_string_length
            if len(str_repr) > max_string_len:
                return str_repr[:max_string_len] + "...", current_size + max_string_len + 3
            return data, current_size + len(str_repr)

        # List: limit to configured number of items
        if isinstance(data, list):
            pruned_list = []
            size = current_size + 2  # Account for brackets
            max_list_items = settings.context_max_list_items
            for idx, item in enumerate(data):
                if idx >= max_list_items:
                    pruned_list.append("... (truncated)")
                    size += 15
                    break
                pruned_item, size = self._prune_dict(item, max_chars, size)
                pruned_list.append(pruned_item)
                if size > max_chars:
                    pruned_list.append("... (truncated)")
                    break
            return pruned_list, size

        # Dict: recursively prune values
        if isinstance(data, dict):
            pruned_dict = {}
            size = current_size + 2  # Account for braces
            for key, value in data.items():
                key_str = str(key)
                size += len(key_str) + 3  # Key + quotes + colon
                if size > max_chars:
                    pruned_dict["... (truncated)"] = True
                    break
                pruned_value, size = self._prune_dict(value, max_chars, size)
                pruned_dict[key] = pruned_value
                if size > max_chars:
                    break
            return pruned_dict, size

        # Fallback: convert to string
        str_repr = str(data)
        if len(str_repr) > 200:
            return str_repr[:200] + "...", current_size + 203
        return data, current_size + len(str_repr)

    def _truncate_provider_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Truncate provider data to prevent context bloat.

        Uses recursive dictionary pruning ("Lumberjack" approach) instead of
        fragile JSON string slicing. This prevents JSONDecodeError on complex
        nested structures.

        Args:
            data: The provider data dictionary.

        Returns:
            Truncated data dictionary (pruned recursively to fit within limits).
        """
        max_chars = settings.context_max_chars_per_provider

        # Check size first
        json_str = json.dumps(data, default=str)
        if len(json_str) <= max_chars:
            return data

        # Prune recursively
        pruned_data, _ = self._prune_dict(data, max_chars)
        pruned_json = json.dumps(pruned_data, default=str)

        logger.warning(
            f"Provider data truncated from {len(json_str)} to {len(pruned_json)} chars "
            f"(recursive pruning applied)"
        )
        return pruned_data

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

            # Execute with per-provider timeout
            timeout_seconds = settings.context_per_provider_timeout_ms / 1000.0
            result = await asyncio.wait_for(provider.provide_context(), timeout=timeout_seconds)

            # Handle both dict (backward compat) and ContextResponse
            data = result.data if isinstance(result, ContextResponse) else result
            # TTL is available in ContextResponse but not currently used (future enhancement)
            # Could implement caching based on result.ttl if needed

            # Truncate if needed
            truncated_data = self._truncate_provider_data(data)

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
            # Execute tool with injected context and validated arguments
            result = await asyncio.wait_for(tool.execute(context, **args), timeout=30.0)
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

            # Execute the tool with validated arguments (response is already a Pydantic model)
            tool_name = tool.name
            tool_args = response.model_dump()
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
