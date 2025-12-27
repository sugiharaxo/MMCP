"""Agent Orchestrator - manages the ReAct loop and conversation context."""

import asyncio
import logging
import uuid
from typing import Union

from app.agent.schemas import FinalResponse
from app.core.config import settings
from app.core.errors import (
    MMCPError,
    map_agent_error,
    map_provider_error,
    map_tool_error,
)
from app.core.llm import get_agent_decision
from app.core.plugin_loader import PluginLoader

logger = logging.getLogger("mmcp.agent")


class AgentOrchestrator:
    """
    Manages the agent's conversation history and tool execution loop.

    Implements character-based context management for efficient memory usage.
    Uses Instructor's discriminated Union pattern: LLM returns FinalResponse or a tool input schema directly.
    """

    def __init__(self, loader: PluginLoader):
        """
        Initialize the orchestrator.

        Args:
            loader: PluginLoader instance with loaded tools.
        """
        self.loader = loader
        self.history: list[dict] = []

    def _get_system_prompt(self) -> str:
        """
        Generate system prompt with dynamic tool descriptions.

        Simplified prompt focusing on "when to use tools" rather than formatting rules.
        The Union response_model handles the "how" of formatting automatically.
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

        return f"""You are MMCP (Modular Media Control Plane), an intelligent media assistant.
You help users manage their media library, search for metadata, and handle downloads.

IDENTITY:
- Be concise and helpful.
- Use tools to fetch data before making recommendations.
- If a tool fails, explain why to the user and try a different approach.

AVAILABLE TOOLS:
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

    def _emit_status(self, message: str, trace_id: str | None = None):
        """
        Emit status messages for visibility and future extensibility.

        Currently prints to console and logs. In the future, this could broadcast
        to WebSocket clients or other UI components.

        Args:
            message: The status message to emit
            trace_id: Optional trace ID for log correlation
        """
        print(message)
        logger.info(message, extra={"trace_id": trace_id} if trace_id else {})

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

    async def safe_tool_call(
        self, tool, tool_name: str, args: dict, trace_id: str, failure_counts: dict[str, int]
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
            trace_id: Trace ID for log correlation

        Returns:
            Tool result as string, or error message if execution failed
        """
        try:
            # Execute tool with validated arguments
            result = await asyncio.wait_for(tool.execute(**args), timeout=30.0)
            return str(result) if result is not None else "Tool executed successfully.", False

        except Exception as e:
            # Map to ToolError for logging, but return simple string for LLM
            tool_error = map_tool_error(e, tool_name=tool_name, trace_id=trace_id)
            logger.error(
                f"Tool {tool_name} failed (trace_id={trace_id}): {e}",
                exc_info=True,
                extra={"trace_id": trace_id, "tool_name": tool_name},
            )
            # Track tool failures to detect repeated issues
            failure_counts[tool_name] = failure_counts.get(tool_name, 0) + 1

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

        # Initialize system prompt if this is the first message
        if not self.history:
            self.history.append({"role": "system", "content": self._get_system_prompt()})

        # Add user message to history
        self.history.append({"role": "user", "content": user_input})

        # Build the Union response model: FinalResponse | ToolSchema1 | ToolSchema2 | ...
        ResponseModel = self._get_response_model()

        # ReAct loop: allow multiple reasoning steps
        max_steps = 5  # Prevent infinite loops
        llm_retry_count = 0
        max_llm_retries = 2  # Allow LLM to self-correct on validation errors
        rate_limit_retry_count = 0
        max_rate_limit_retries = 3  # Allow up to 3 retries for rate limits
        tool_failure_counts: dict[str, int] = {}  # Track repeated tool failures

        for step in range(max_steps):
            self._trim_history()

            # 1. Ask the LLM what to do
            try:
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
                    f"\n[Step {step + 1}] Final Response: {response.answer}", trace_id
                )
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
            )
            logger.info(
                f"Executing tool: {tool_name} with {tool_args}",
                extra={"trace_id": trace_id, "tool_name": tool_name},
            )

            result, is_error = await self.safe_tool_call(
                tool, tool_name, tool_args, trace_id, tool_failure_counts
            )
            self._emit_status(f"[Step {step + 1}] Observation: {result}", trace_id)

            # Check for repeated tool failures - if same tool fails 3+ times, force final response
            failure_count = tool_failure_counts.get(tool_name, 0)
            if is_error and failure_count >= 3:
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
        self._emit_status(
            f"[System] ReAct loop exhausted {max_steps} steps without resolution", trace_id
        )
        return "I attempted to solve this but reached my reasoning limit. Please try being more specific or breaking down your request."
