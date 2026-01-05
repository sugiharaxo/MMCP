from typing import Literal

from pydantic import BaseModel, Field, create_model


# 1. Static, Clean FinalResponse. No reasoning/protocol pollution.
class FinalResponse(BaseModel):
    """The model's final answer to the user."""

    type: Literal["final_response"] = "final_response"
    thought: str = Field(..., description="Brief internal reasoning")
    answer: str = Field(..., description="The final message to the user")
    acknowledged_ids: list[str] | None = Field(default=None)


# 2. Base for Plugin Tools only (Agentic Notification Protocol reasoning)
class MMCPToolAction(BaseModel):
    """Base for tools requiring reasoning/rationale."""

    rationale: str = Field(default="", description="Internal reasoning for this action")


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


def get_reasoned_model(tools_map: dict[str, type[BaseModel]]) -> list[type[BaseModel]]:
    """
    Returns a list of models.
    FinalResponse remains pure. Plugin tools get rationale injected.

    Args:
        tools_map: A dictionary mapping tool_name (e.g. 'test_external_action')
                  to its input_schema (e.g. TestExternalInput).

    Returns:
        List of Pydantic models including FinalResponse and all extended tool schemas.
        FinalResponse is returned as-is (no rationale injection).
        Plugin tools are extended with MMCPToolAction to include rationale and type fields.
        This list is converted to Union[...] in get_agent_decision() for Instructor.
    """
    # Start with pure FinalResponse
    models = [FinalResponse]

    for tool_name, tool_schema in tools_map.items():
        type_value = f"tool_{tool_name}"

        # Use tool name directly as model name since tool names are guaranteed unique.
        # The model name is only for Python's internal registry, not sent to the LLM.
        # Inject reasoning ONLY into plugin tools
        extended_tool = create_model(
            tool_name,
            __base__=(tool_schema, MMCPToolAction),
            type=(Literal[type_value], Field(default=type_value)),
            __module__=__name__,
        )
        models.append(extended_tool)

    return models  # Return the list for Instructor to unroll
