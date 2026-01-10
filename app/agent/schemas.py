from typing import Literal

from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """The model's final answer to the user."""

    type: Literal["final_response"] = "final_response"
    thought: str = Field(..., description="Brief internal reasoning")
    answer: str = Field(..., description="The final message to the user")
    acknowledged_ids: list[str] | None = Field(default=None)


class ActionRequestResponse(BaseModel):
    """Response when agent needs user approval for external action."""

    type: Literal["action_request"] = "action_request"
    approval_id: str = Field(description="Session-scoped approval identifier")
    explanation: str = Field(description="Agent's user-friendly explanation")
    tool_name: str = Field(description="Internal tool identifier")
    tool_args: dict = Field(description="Tool arguments for execution")
    tool_call_id: str | None = Field(
        default=None,
        description="Tool call ID from LLM for matching results (DeepSeek 1:1 pairing)",
    )
