"""Tests for LLM mode-agnostic functionality across Instructor modes.

NOTE: This test file tests functions from the old app.core.llm module that no longer exist
in the new architecture. The functions get_agent_decision and unwrap_response have been
replaced by the Transport/Intelligence/Orchestrator pattern. These tests are marked as
skipped until they can be updated for the new architecture.
"""

import pytest

# Skip all tests in this file - they test old architecture
pytestmark = pytest.mark.skip(
    reason="Tests old app.core.llm functions that no longer exist in new architecture"
)

from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import instructor
from pydantic import BaseModel

from app.core.config import user_settings

# from app.core.llm import get_agent_decision, unwrap_response  # No longer exists


# Test models for the ReAct loop
class ToolA(BaseModel):
    """Test tool A."""

    type: Literal["tool_a"] = "tool_a"
    val: str


class ToolB(BaseModel):
    """Test tool B."""

    type: Literal["tool_b"] = "tool_b"
    num: int


class FinalResponse(BaseModel):
    """Test final response model."""

    type: Literal["final_response"] = "final_response"
    answer: str


# Modes to test (Generic + Key Provider Modes)
# Note: Some modes may be provider-specific, but we test the universal logic
TESTABLE_MODES = [
    instructor.Mode.TOOLS,
    instructor.Mode.JSON,
    instructor.Mode.MD_JSON,
    instructor.Mode.PARALLEL_TOOLS,
]

# Add ANTHROPIC_TOOLS if available
if hasattr(instructor.Mode, "ANTHROPIC_TOOLS"):
    TESTABLE_MODES.append(instructor.Mode.ANTHROPIC_TOOLS)


class MockModelResponse:
    """Mock LiteLLM ModelResponse object."""

    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage or {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5}

    def __getitem__(self, key):
        """Support dict-like access for hybrid LiteLLM responses."""
        return getattr(self, key, None)


class MockChoice:
    """Mock choice object."""

    def __init__(self, message):
        self.message = message


class MockMessage:
    """Mock message object."""

    def __init__(self, role="assistant", content=None, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


@pytest.fixture
def mock_llm_model(monkeypatch):
    """Set a mock LLM model in user_settings."""
    monkeypatch.setattr(user_settings, "llm_model", "test/openai-gpt-4")
    monkeypatch.setattr(user_settings, "llm_api_key", None)
    monkeypatch.setattr(user_settings, "llm_base_url", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", TESTABLE_MODES)
async def test_get_agent_decision_across_modes(mode):
    """
    Verifies that get_agent_decision captures metadata and unwraps
    correctly across different Instructor parsing modes.
    """
    # Create a generic mock response - Instructor will handle mode-specific formatting
    mock_msg = MockMessage(
        role="assistant",
        content='{"type": "tool_a", "val": "hello"}',
        tool_calls=[
            {
                "id": "test_call_123",
                "type": "function",
                "function": {"name": "ToolA", "arguments": '{"val": "hello"}'},
            }
        ],
    )

    mock_choice = MockChoice(mock_msg)
    mock_completion = MockModelResponse([mock_choice])

    # Mock Instructor client to return our test model
    mock_parsed = ToolA(type="tool_a", val="hello")

    # Patch acompletion to return our mock
    with (
        patch("app.core.llm.acompletion", new_callable=AsyncMock) as mock_aco,
        patch("app.core.llm.get_instructor_mode", return_value=mode),
        patch("app.core.llm.instructor.from_litellm") as mock_from_litellm,
    ):
        mock_aco.return_value = mock_completion

        # Store hook callback to invoke it later
        hook_callback = None

        def register_hook(event, callback):
            nonlocal hook_callback
            if event == "completion:response":
                hook_callback = callback

        async def create_with_hook(*_args, **_kwargs):
            # Invoke the hook with the raw completion before returning parsed
            if hook_callback:
                hook_callback(mock_completion)
            return mock_parsed

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=create_with_hook)

        # Mock hook registration to capture the callback
        mock_client.on = MagicMock(side_effect=register_hook)
        mock_client.off = MagicMock()
        mock_client.chat.completions.on = MagicMock(side_effect=register_hook)
        mock_client.chat.completions.off = MagicMock()

        mock_from_litellm.return_value = mock_client

        # Execute the call
        parsed_obj, raw_meta = await get_agent_decision(
            messages=[{"role": "user", "content": "test"}],
            response_model=[ToolA, FinalResponse],
        )

        # ASSERTIONS
        # 1. Check if metadata was captured by the hook
        assert raw_meta is not None
        assert hasattr(raw_meta, "choices") or "choices" in str(type(raw_meta))

        # 2. Check if unwrapping worked
        assert isinstance(parsed_obj, (ToolA, FinalResponse, BaseModel))

        # 3. Verify the parsed object is correct
        if isinstance(parsed_obj, ToolA):
            assert parsed_obj.val == "hello"


@pytest.mark.asyncio
async def test_get_agent_decision_single_model():
    """Test that get_agent_decision works with a single model (not a list)."""
    mock_msg = MockMessage(role="assistant", content='{"answer": "test response"}')
    mock_choice = MockChoice(mock_msg)
    mock_completion = MockModelResponse([mock_choice])

    mock_parsed = FinalResponse(type="final_response", answer="test response")

    # Patch acompletion to return our mock
    with (
        patch("app.core.llm.acompletion", new_callable=AsyncMock) as mock_aco,
        patch("app.core.llm.get_instructor_mode", return_value=instructor.Mode.JSON),
        patch("app.core.llm.instructor.from_litellm") as mock_from_litellm,
    ):
        mock_aco.return_value = mock_completion

        # Store hook callback to invoke it later
        hook_callback = None

        def register_hook(event, callback):
            nonlocal hook_callback
            if event == "completion:response":
                hook_callback = callback

        async def create_with_hook(*_args, **_kwargs):
            # Invoke the hook with the raw completion before returning parsed
            if hook_callback:
                hook_callback(mock_completion)
            return mock_parsed

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=create_with_hook)
        mock_client.on = MagicMock(side_effect=register_hook)
        mock_client.off = MagicMock()
        mock_client.chat.completions.on = MagicMock(side_effect=register_hook)
        mock_client.chat.completions.off = MagicMock()

        mock_from_litellm.return_value = mock_client

        # Pass single model instead of list
        parsed_obj, raw_meta = await get_agent_decision(
            messages=[{"role": "user", "content": "test"}], response_model=FinalResponse
        )

        assert isinstance(parsed_obj, FinalResponse)
        assert parsed_obj.answer == "test response"
        assert raw_meta is not None


@pytest.mark.asyncio
async def test_get_agent_decision_empty_choices():
    """Test that get_agent_decision raises ValueError for empty choices."""
    # Mock completion with empty choices
    mock_completion = MockModelResponse([])

    with (
        patch("app.core.llm.acompletion", new_callable=AsyncMock) as mock_aco,
        patch("app.core.llm.get_instructor_mode", return_value=instructor.Mode.JSON),
        patch("app.core.llm.instructor.from_litellm") as mock_from_litellm,
    ):
        mock_aco.return_value = mock_completion

        mock_client = MagicMock()
        # Instructor will fail to parse empty response
        mock_client.chat.completions.create = AsyncMock(
            side_effect=ValueError("No valid objects returned")
        )
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()
        mock_from_litellm.return_value = mock_client

        with pytest.raises((ValueError, Exception)):
            await get_agent_decision(
                messages=[{"role": "user", "content": "test"}],
                response_model=[FinalResponse],
            )


@pytest.mark.asyncio
async def test_unwrap_response_with_wrapper():
    """Test unwrap_response correctly unwraps Instructor synthetic wrappers."""

    # Create a mock wrapper like Instructor might create
    class Response(BaseModel):
        """Synthetic Instructor wrapper."""

        tool: ToolA

    wrapped = Response(tool=ToolA(type="tool_a", val="test"))

    # Should unwrap to ToolA
    unwrapped = unwrap_response(wrapped)
    assert isinstance(unwrapped, ToolA)
    assert unwrapped.val == "test"


@pytest.mark.asyncio
async def test_unwrap_response_without_wrapper():
    """Test unwrap_response doesn't unwrap legitimate single-field models."""

    # Create a legitimate single-field model (not an Instructor wrapper)
    class MyCustomTool(BaseModel):
        """Legitimate single-field tool (not an Instructor wrapper)."""

        value: str

    tool = MyCustomTool(value="test")

    # Should NOT unwrap (class name not in wrapper list)
    unwrapped = unwrap_response(tool)
    assert isinstance(unwrapped, MyCustomTool)
    assert unwrapped.value == "test"


@pytest.mark.asyncio
async def test_unwrap_response_with_agent_turn_wrapper():
    """Test unwrap_response handles AgentTurn wrapper."""

    # Create AgentTurn-like wrapper
    class AgentTurn(BaseModel):
        """Synthetic AgentTurn wrapper."""

        action: FinalResponse

    wrapped = AgentTurn(action=FinalResponse(type="final_response", answer="test"))

    # Should unwrap to FinalResponse
    unwrapped = unwrap_response(wrapped)
    assert isinstance(unwrapped, FinalResponse)
    assert unwrapped.answer == "test"


@pytest.mark.asyncio
async def test_unwrap_response_non_model():
    """Test unwrap_response handles non-Pydantic objects."""
    # Should return as-is for non-models
    assert unwrap_response(None) is None
    assert unwrap_response("string") == "string"
    assert unwrap_response(123) == 123
    assert unwrap_response({"key": "value"}) == {"key": "value"}


@pytest.mark.asyncio
async def test_get_agent_decision_empty_model_list():
    """Test that get_agent_decision raises ValueError for empty model list."""
    with pytest.raises(ValueError, match="response_model list cannot be empty"):
        await get_agent_decision(messages=[{"role": "user", "content": "test"}], response_model=[])


@pytest.mark.asyncio
async def test_get_agent_decision_no_llm_model(monkeypatch):
    """Test that get_agent_decision raises ValueError when LLM_MODEL is not set."""
    monkeypatch.setattr(user_settings, "llm_model", None)

    with pytest.raises(ValueError, match="LLM_MODEL must be set"):
        await get_agent_decision(
            messages=[{"role": "user", "content": "test"}], response_model=[FinalResponse]
        )
