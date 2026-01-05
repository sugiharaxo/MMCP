"""History Management - handles conversation history operations."""

from typing import Any

import instructor

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
            # Handle None content (tool calls have content: None)
            current_chars = sum(len(m.get("content") or "") for m in history)

            # If we are under the limit or only have system prompt left, stop
            if current_chars <= user_settings.llm_max_context_chars or len(history) <= 2:
                break

            # Remove the oldest non-system message (index 1)
            # Index 0 is System, Index 1 is the oldest User/Assistant message
            history.pop(1)
            # Recalculate after pop for accurate logging
            current_chars = sum(len(m.get("content") or "") for m in history)
            logger.debug(f"Trimmed context to {current_chars} chars.")

    def reconstruct_history(
        self,
        base_history: list[dict[str, Any]],
        system_prompt: str | tuple[str, str],
        user_input: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Reconstruct conversation history for LLM consumption.

        Supports both legacy single system prompt and new cache-safe two-message format.

        Args:
            base_history: The base conversation history
            system_prompt: System prompt(s) to prepend.
                          Can be a single string (legacy) or tuple of (static, dynamic) strings.
            user_input: Optional user input to append

        Returns:
            Reconstructed history ready for LLM
        """
        # Handle cache-safe two-message format
        if isinstance(system_prompt, tuple):
            static_prompt, dynamic_prompt = system_prompt
            # PIN static system message at top (Instructor will append schema here)
            static_message = {"role": "system", "content": static_prompt}
            # Add dynamic state as second system message
            dynamic_message = {"role": "system", "content": dynamic_prompt}
            # Remove all system messages from base history
            messages = [m for m in base_history if m["role"] != "system"]
            history = [static_message, dynamic_message] + messages
        else:
            # Legacy single system prompt format
            system_message = {"role": "system", "content": system_prompt}
            messages = [m for m in base_history if m["role"] != "system"]
            history = [system_message] + messages

        # Add user input if provided
        if user_input and user_input.strip():
            history.append({"role": "user", "content": user_input})

        return history

    def add_assistant_response(self, history: list[dict[str, Any]], content: str) -> None:
        """Add assistant response to history."""
        history.append({"role": "assistant", "content": content})

    def add_tool_result(
        self,
        history: list[dict[str, Any]],
        tool_call_id: str | None,
        result: str,
        instructor_mode: instructor.Mode | None = None,
    ) -> None:
        """
        Add tool execution result to history.

        Mode-aware routing:
        - Mode.TOOLS: Uses role="tool" with tool_call_id (native tool calling format)
        - Mode.JSON: Uses role="user" with OBSERVATION prefix (for local models that don't support tool role)

        Args:
            history: History list to append to
            tool_call_id: Tool call identifier (may be generated UUID for JSON mode)
            result: Tool execution result
            instructor_mode: Instructor mode to determine message format
        """
        if instructor_mode is None:
            # Auto-detect mode if not provided
            from app.core.llm import get_instructor_mode

            instructor_mode = get_instructor_mode(user_settings)

        if instructor_mode == instructor.Mode.TOOLS:
            # Native tool calling format (OpenAI, Gemini, Claude, DeepSeek)
            history.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(result)})
        else:
            # JSON mode (Ollama/local models) - use user role with prefix
            history.append(
                {
                    "role": "user",
                    "content": f"OBSERVATION: {str(result)}",
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

    def add_llm_message(
        self,
        history: list[dict[str, Any]],
        completion_message: Any,
        instructor_mode: instructor.Mode,
    ) -> None:
        """
        Add LLM completion message to history, handling mode-specific formatting.

        Converts raw completion message to history dict format, preserving
        tool_calls metadata for Mode.TOOLS compatibility.

        Args:
            history: History list to append to
            completion_message: Message object from LLM completion
            instructor_mode: Instructor mode to determine message format
        """
        # If using Mode.TOOLS, preserve tool_calls structure
        if (
            instructor_mode == instructor.Mode.TOOLS
            and hasattr(completion_message, "tool_calls")
            and completion_message.tool_calls
        ):
            # DeepSeek requires content to be explicitly None (not missing) for tool_calls
            assistant_dict = {
                "role": "assistant",
                "content": None,  # Explicitly set to None for DeepSeek compatibility
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in completion_message.tool_calls
                ],
            }
        else:
            # Regular assistant message with content
            assistant_dict = {
                "role": "assistant",
                "content": completion_message.content or "",  # Ensure string, not None
            }

        history.append(assistant_dict)
