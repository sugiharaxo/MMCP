from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """The model's final answer to the user."""

    thought: str = Field(description="Brief internal reasoning")
    answer: str = Field(description="The final message to the user")
