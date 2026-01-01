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
