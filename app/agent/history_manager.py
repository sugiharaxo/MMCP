"""History Management - handles conversation history operations."""

from typing import Any

from app.core.config import user_settings
from app.core.logger import logger


class HistoryManager:
    """Manages conversation history operations."""

    def trim_history(self, history: list[dict[str, Any]]) -> None:
        """
        Trims history based on character count.
        Always preserves the system prompt (index 0).

        Args:
            history: The history list to trim (modified in-place)
        """
        if len(history) <= 1:
            return

        while True:
            # Calculate total character count of the history
            # Handle dict content (tool calls and final responses have dict content)
            # Handle string content (user messages and tool results have string content)
            current_chars = sum(
                len(str(m.get("content") or "")) if isinstance(m.get("content"), (str, dict)) else 0
                for m in history
            )

            # If we are under the limit or only have system prompt left, stop
            if current_chars <= user_settings.llm_max_context_chars or len(history) <= 2:
                break

            # Remove the oldest non-system message (index 1)
            # Index 0 is System, Index 1 is the oldest User/Assistant message
            history.pop(1)
            # Recalculate after pop for accurate logging
            current_chars = sum(len(m.get("content") or "") for m in history)
            logger.debug(f"Trimmed context to {current_chars} chars.")

    def add_user_message(self, history: list[dict[str, Any]], content: str) -> None:
        """Add user message to history."""
        history.append({"role": "user", "content": content})

    def add_final_response(
        self,
        history: list[dict[str, Any]],
        final_response: dict[str, Any],
    ) -> None:
        """
        Add FinalResponse JSON object to history.

        Stores the full JSON object (thought + answer) so the model sees exactly
        what it outputted previously.

        Args:
            history: History list to append to
            final_response: Full FinalResponse dict with 'thought' and 'answer' keys
        """
        history.append(
            {
                "role": "assistant",
                "type": "final_response",
                "content": final_response,  # Full JSON dict, not stringified
            }
        )

    def add_tool_call(
        self,
        history: list[dict[str, Any]],
        tool_call: dict[str, Any],
    ) -> None:
        """
        Add ToolCall JSON object to history.

        Stores the full JSON object (thought + tool_name + args) so the model sees
        exactly what it outputted previously.

        Args:
            history: History list to append to
            tool_call: Full ToolCall dict with 'thought', 'tool_name', and 'args' keys
        """
        history.append(
            {
                "role": "assistant",
                "type": "tool_call",
                "content": tool_call,  # Full JSON dict, not stringified
            }
        )

    def add_tool_result(
        self,
        history: list[dict[str, Any]],
        tool_name: str,
        result: str,
    ) -> None:
        """
        Add tool execution result to history.

        Stores the raw result JSON string in content, which will be formatted
        by BAML template as an observation tag: <observation tool="tool_name">{result}</observation>

        Args:
            history: History list to append to
            tool_name: Name of the tool that was executed
            result: Tool execution result (already converted to JSON string)
        """
        # Store raw result JSON - BAML template will format as observation tag
        history.append(
            {
                "role": "user",  # Universal role for compatibility
                "content": result,  # Raw JSON result string
                "tool_result": True,  # Metadata for BAML to detect and format as observation
                "tool_name": tool_name,  # Tool name for observation tag
            }
        )

    def add_error_message(self, history: list[dict[str, Any]], error_message: str) -> None:
        """Add error message to history for LLM correction."""
        history.append(
            {
                "role": "assistant",
                "content": f"Error: {error_message} Please provide a valid response.",
            }
        )
