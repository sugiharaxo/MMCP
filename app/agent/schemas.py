from typing import Any, Literal, Union

from pydantic import BaseModel, Field, create_model


class MMCPAction(BaseModel):
    """
    Base class for all MMCP actions (tools and final responses).

    Provides a rationale field for all actions, enabling flattened Union structure
    and reducing schema nesting for better KV cache efficiency.
    """

    rationale: str = Field(default="", description="Brief internal reasoning for this action")


class FinalResponse(MMCPAction):
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
    Tool call with rationale for single-turn reasoned HITL flow (legacy wrapper).

    NOTE: This is kept for backward compatibility. New flattened structure uses
    MMCPAction base class directly on tool schemas.
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
    The unified container for every agent turn (legacy wrapper).

    NOTE: This wrapper is no longer needed with flattened Union structure.
    Kept for backward compatibility during migration.
    """

    action: FinalResponse | ReasonedToolCall = Field(
        description="The agent's action: either a final response or a reasoned tool call"
    )


def get_reasoned_model(available_tools: list[type[BaseModel]]) -> type:
    """
    Build a flattened Union model for single-turn reasoned tool selection.

    Returns a flattened Union of FinalResponse and all tool schemas (with rationale
    added dynamically). This reduces schema nesting and improves KV cache efficiency.

    Each tool schema is dynamically extended to include the rationale field from
    MMCPAction, enabling a flat Union structure without nested wrappers.
    """
    if not available_tools:
        return FinalResponse  # type: ignore[return-value]

    extended_tools = []
    for tool_schema in available_tools:
        slug = tool_schema.__name__.lower().replace("input", "").replace("metadata", "")
        type_value = f"tool_{slug.strip('_')}"
        ExtendedTool = create_model(
            f"Extended{tool_schema.__name__}",
            __base__=(MMCPAction, tool_schema),
            type=(Literal[type_value], Field(default=type_value)),  # type: ignore
        )
        extended_tools.append(ExtendedTool)

    # Build flattened Union (compatible with Python 3.10)
    union_type: type = FinalResponse
    for extended_tool in extended_tools:
        union_type = Union[union_type, extended_tool]  # type: ignore[assignment]

    return union_type  # type: ignore[return-value]
