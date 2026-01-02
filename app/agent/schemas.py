from typing import Literal

from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """The model's final answer to the user."""

    thought: str = Field(description="Brief internal reasoning")
    answer: str = Field(description="The final message to the user")
    acknowledged_ids: list[str] | None = Field(
        default=None, description="List of notification IDs acknowledged by agent"
    )
    internal_turn: bool | None = Field(
        default=None, description="Whether this was an internal turn (non-UI)"
    )


class ActionRequestResponse(BaseModel):
    """Response when agent needs user approval for external action."""

    type: Literal["action_request"] = "action_request"
    event_id: str = Field(description="ANP event ID for this action")
    explanation: str = Field(description="Agent's user-friendly explanation")
    tool_name: str = Field(description="Internal tool identifier")
    tool_args: dict = Field(description="Tool arguments for execution")
