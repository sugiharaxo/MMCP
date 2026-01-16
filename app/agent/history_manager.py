"""History Management - handles conversation operations with cached size invariants."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import UserSettings
from app.core.logger import logger


class HistoryMessage(BaseModel):
    """Schema for history messages including cached serialization size."""

    role: Literal["user", "assistant", "system"]
    content: str | dict[str, Any]
    type: Literal["final_response", "tool_call", "action_request"] | None = None
    tool_result: bool | None = None
    tool_name: str | None = None
    anp_event: bool | None = None
    approval_id: str | None = None  # For action_request messages
    action_status: Literal["pending", "approved", "denied"] | None = None  # For action_request messages
    size: int = Field(..., alias="_size")

    model_config = ConfigDict(populate_by_name=True)


class HistoryManager:
    """Manages conversation history with O(1) size lookups."""

    MINIMUM_HISTORY_LENGTH = 2

    def trim_history(self, history: list[HistoryMessage], user_settings: UserSettings) -> None:
        """Trims history based on character count using O(N) total complexity."""
        if len(history) <= self.MINIMUM_HISTORY_LENGTH:
            return

        current_chars = sum(msg.size for msg in history)
        limit = user_settings.llm_max_context_chars

        while current_chars > limit and len(history) > self.MINIMUM_HISTORY_LENGTH:
            popped = history.pop(1)
            current_chars -= popped.size
            logger.debug(f"Trimmed history: {current_chars}/{limit} chars remaining.")

    def _calculate_content_size(
        self,
        content: str | dict[str, Any] | None,
        tool_name: str | None = None,
        anp_event: bool = False,
    ) -> int:
        """Calculates char count as serialized by BAML."""
        if content is None:
            return 0

        if isinstance(content, dict):
            try:
                base_size = len(json.dumps(content))
            except (TypeError, ValueError):
                logger.warning(
                    "Failed to JSON serialize content for size calculation, using str representation"
                )
                base_size = len(str(content))
        else:
            base_size = len(str(content))

        if anp_event and tool_name:
            # Manual overhead calculation for ANP events
            # Template: <notification from="{{ msg.tool_name }}">{{ msg.content }}</notification>
            base_size += len(f'<notification from="{tool_name}">') + len("</notification>")
        elif tool_name:
            # Manual overhead calculation to match baml_src/main.baml template
            # Template line 102: <observation tool="{{ msg.tool_name }}">{{ msg.content }}</observation>
            # If the template format changes, this calculation must be updated accordingly
            base_size += len(f'<observation tool="{tool_name}">') + len("</observation>")

        return base_size

    def add_user_message(self, history: list[HistoryMessage], content: str) -> None:
        """Add user message to history."""
        size = self._calculate_content_size(content)
        history.append(HistoryMessage(role="user", content=content, _size=size))

    def add_final_response(
        self,
        history: list[HistoryMessage],
        final_response: dict[str, Any],
    ) -> None:
        """Add FinalResponse JSON object to history."""
        size = self._calculate_content_size(final_response)
        history.append(
            HistoryMessage(
                role="assistant", type="final_response", content=final_response, _size=size
            )
        )

    def add_tool_call(
        self,
        history: list[HistoryMessage],
        tool_call: dict[str, Any],
    ) -> None:
        """Add ToolCall JSON object to history."""
        size = self._calculate_content_size(tool_call)
        history.append(
            HistoryMessage(role="assistant", type="tool_call", content=tool_call, _size=size)
        )

    def add_tool_result(
        self,
        history: list[HistoryMessage],
        tool_name: str,
        result: str,
    ) -> None:
        """Add tool execution result to history with BAML tag overhead."""
        size = self._calculate_content_size(result, tool_name=tool_name)
        history.append(
            HistoryMessage(
                role="user",
                content=result,
                tool_result=True,
                tool_name=tool_name,
                _size=size,
            )
        )

    def add_autonomous_observation(
        self, history: list[HistoryMessage], content: str, plugin_slug: str
    ) -> None:
        """Adds a Channel C event to history using the event:prefix."""
        size = self._calculate_content_size(
            content, tool_name=f"event:{plugin_slug}", anp_event=True
        )
        history.append(
            HistoryMessage(
                role="user",
                content=content,
                anp_event=True,
                tool_name=f"event:{plugin_slug}",
                _size=size,
            )
        )

    def add_error_message(self, history: list[HistoryMessage], error_message: str) -> None:
        """Add error message to history for LLM correction."""
        content = f"Error: {error_message} Please provide a valid response."
        size = self._calculate_content_size(content)
        history.append(HistoryMessage(role="assistant", content=content, _size=size))

    def add_action_request(
        self,
        history: list[HistoryMessage],
        action_data: dict[str, Any],
    ) -> None:
        """Add action request to history for HITL approval."""
        # action_data contains: approval_id, tool_name, tool_args, explanation
        size = self._calculate_content_size(action_data)
        history.append(
            HistoryMessage(
                role="assistant",
                type="action_request",
                content=action_data,
                tool_name=action_data.get("tool_name"),
                approval_id=action_data.get("approval_id"),
                action_status="pending",
                _size=size,
            )
        )

    def resolve_action_request(
        self,
        history: list[HistoryMessage],
        approval_id: str,
        status: Literal["approved", "denied"],
    ) -> HistoryMessage | None:
        """
        Find and mark an action_request message as resolved by approval_id.
        Returns the updated message if found, None otherwise.
        """
        for msg in history:
            if msg.type == "action_request" and msg.approval_id == approval_id:
                msg.action_status = status
                return msg
        return None

    @staticmethod
    def to_dict_list(history: list[HistoryMessage]) -> list[dict[str, Any]]:
        """Convert HistoryMessage list to dict list for persistence/BAML."""
        return [msg.model_dump(by_alias=True) for msg in history]

    @staticmethod
    def from_dict_list(history: list[dict[str, Any]]) -> list[HistoryMessage]:
        """Convert dict list to HistoryMessage list from persistence."""
        return [HistoryMessage(**msg) for msg in history]
