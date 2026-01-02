from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class FinalResponse(BaseModel):
    """The model's final answer to the user."""

    type: Literal["final_response"] = "final_response"
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

    type: Literal["reasoned_tool_call"] = "reasoned_tool_call"
    rationale: str = Field(description="Concise explanation of why this tool is being used")
    tool_call: Any = Field(description="The specific tool schema (populated dynamically)")


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


class AgentTurn(BaseModel):
    """
    The unified container for every agent turn.

    Instructor will always return this model, ensuring metadata preservation.
    This wrapper pattern fixes the AttributeError when using Union types as
    top-level response_model with instructor.create_with_completion.
    """

    action: FinalResponse | ReasonedToolCall = Field(
        description="The agent's action: either a final response or a reasoned tool call"
    )


def get_reasoned_model(
    available_tools: list[type[BaseModel]],
) -> type[Union[FinalResponse, ReasonedToolCall]]:  # noqa: UP007
    """
    Build a Union model for single-turn reasoned tool selection.

    Returns a Union of FinalResponse and ReasonedToolCall, where tool_call
    is a Union of all provided tool types. This ensures the LLM returns
    both the tool selection and rationale in a single atomic call.

    Note: This Union is wrapped in AgentTurn by get_agent_decision to avoid
    the AttributeError with instructor.create_with_completion.

    Note: We use typing.Union (not | operator) because:
    1. The | operator cannot be used in reduce() lambda expressions
    2. typing.Union is more explicit and compatible with Instructor/Pydantic
    3. Instructor (via Pydantic) handles Union types correctly for discriminated unions

    Args:
        available_tools: List of tool input schema classes (BaseModel subclasses)

    Returns:
        Union type that can be FinalResponse or ReasonedToolCall with tool_call
        validated against the Union of all available tool schemas
    """
    from functools import reduce

    if not available_tools:
        # If no tools, return Union of FinalResponse and ReasonedToolCall with Any tool_call
        return Union[FinalResponse, ReasonedToolCall]  # type: ignore[return-value]  # noqa: UP007

    # Build Union of all tool schemas for runtime validation using typing.Union
    # Instructor will use this to validate the tool_call field
    # Note: Python automatically normalizes nested Unions (Union[Union[A, B], C] -> Union[A, B, C])
    tool_union = reduce(lambda acc, schema: Union[acc, schema], available_tools)  # type: ignore[assignment]  # noqa: UP007

    # Create a dynamic ReasonedToolCall class with tool_call as the Union type
    # We use create_model to properly set up the Union type annotation
    from pydantic import create_model

    # Create the model with tool_call typed as the Union
    # Pydantic will validate tool_call matches one of the tool schemas
    # Use __base__=ReasonedToolCall so isinstance checks work correctly
    ReasonedToolCallWithUnion = create_model(
        "ReasonedToolCallWithUnion",
        type=(
            Literal["reasoned_tool_call"],
            Field(default="reasoned_tool_call", description="Response type discriminator"),
        ),
        rationale=(str, Field(description="Concise explanation of why this tool is being used")),
        tool_call=(
            tool_union,
            Field(description="The specific tool schema (Union of available tools)"),
        ),
        __base__=ReasonedToolCall,
    )

    # Return Union of FinalResponse and the reasoned tool call using typing.Union
    return Union[FinalResponse, ReasonedToolCallWithUnion]  # type: ignore[return-value]  # noqa: UP007
