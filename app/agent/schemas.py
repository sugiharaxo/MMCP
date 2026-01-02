from typing import Any, Literal

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


class ReasonedToolCall(BaseModel):
    """
    Tool call with rationale for single-turn reasoned HITL flow.

    Ensures atomic consistency between the agent's logic and the explanation
    shown to the user, preventing Lies-in-the-Loop (LITL) injection attacks.
    """

    rationale: str = Field(description="Concise explanation of why this tool is being used")
    tool_call: Any = Field(description="The specific tool schema (populated dynamically)")


class ActionRequestResponse(BaseModel):
    """Response when agent needs user approval for external action."""

    type: Literal["action_request"] = "action_request"
    approval_id: str = Field(description="Session-scoped approval identifier")
    explanation: str = Field(description="Agent's user-friendly explanation")
    tool_name: str = Field(description="Internal tool identifier")
    tool_args: dict = Field(description="Tool arguments for execution")


def get_reasoned_model(
    available_tools: list[type[BaseModel]],
) -> type[FinalResponse | ReasonedToolCall]:
    """
    Build a Union model for single-turn reasoned tool selection.

    Returns a Union of FinalResponse and ReasonedToolCall, where tool_call
    is a Union of all provided tool types. This ensures the LLM returns
    both the tool selection and rationale in a single atomic call.

    Note: The tool_call field uses Any type annotation, but Instructor will
    validate it against the Union of available tool schemas at runtime.

    Args:
        available_tools: List of tool input schema classes (BaseModel subclasses)

    Returns:
        Union type that can be FinalResponse or ReasonedToolCall with tool_call
        validated against the Union of all available tool schemas
    """
    from functools import reduce

    if not available_tools:
        # If no tools, return Union of FinalResponse and ReasonedToolCall with Any tool_call
        return FinalResponse | ReasonedToolCall

    # Build Union of all tool schemas for runtime validation
    # Instructor will use this to validate the tool_call field
    tool_union = reduce(lambda acc, schema: acc | schema, available_tools)

    # Create a dynamic ReasonedToolCall class with tool_call as the Union type
    # We use create_model to properly set up the Union type annotation
    from pydantic import create_model

    # Create the model with tool_call typed as the Union
    # Pydantic will validate tool_call matches one of the tool schemas
    ReasonedToolCallWithUnion = create_model(
        "ReasonedToolCallWithUnion",
        rationale=(str, Field(description="Concise explanation of why this tool is being used")),
        tool_call=(
            tool_union,
            Field(description="The specific tool schema (Union of available tools)"),
        ),
        __base__=BaseModel,
    )

    # Return Union of FinalResponse and the reasoned tool call
    return FinalResponse | ReasonedToolCallWithUnion
