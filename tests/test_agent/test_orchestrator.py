"""Tests for agent orchestrator."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import FinalResponse


@pytest.fixture
def mock_loader():
    """Create a mock plugin loader."""
    loader = MagicMock()
    loader.list_tools.return_value = {"test_tool": "A test tool"}
    loader.tools = {}  # Empty tools dict
    loader._schema_to_tool = {}  # Empty schema mapping
    return loader


# Define a test tool input schema for testing
class TestToolInput(BaseModel):
    """Test tool input schema."""

    param: str = Field(..., description="Test parameter")


@pytest.fixture
def orchestrator(mock_loader):
    """Create an orchestrator instance with a mock loader."""
    return AgentOrchestrator(mock_loader)


@pytest.mark.asyncio
async def test_chat_final_response(orchestrator: AgentOrchestrator):
    """Test that chat returns a final response when LLM provides one."""
    response = FinalResponse(thought="I can answer this", answer="Test response")

    with patch("app.agent.orchestrator.get_agent_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = response

        result = await orchestrator.chat("Test message")

        assert result == "Test response"
        assert len(orchestrator.history) >= 2  # System prompt + user message + assistant response


@pytest.mark.asyncio
async def test_chat_tool_call(orchestrator: AgentOrchestrator):
    """Test that chat executes tools when LLM requests them."""
    from app.core.context import MMCPContext

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value={"result": "Tool executed"})
    orchestrator.loader.get_tool_by_schema.return_value = mock_tool

    # First call: tool input schema, Second call: final response
    tool_input = TestToolInput(param="value")
    final_response = FinalResponse(thought="Tool executed successfully", answer="Done")

    with patch("app.agent.orchestrator.get_agent_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [tool_input, final_response]

        result = await orchestrator.chat("Test message")

        assert result == "Done"
        # Verify tool was called with context and params
        args, kwargs = mock_tool.execute.call_args
        assert len(args) == 1
        assert isinstance(args[0], MMCPContext)  # First arg is context
        assert kwargs == {"param": "value"}


@pytest.mark.asyncio
async def test_chat_tool_not_found(orchestrator: AgentOrchestrator):
    """Test that chat handles missing tools gracefully."""
    tool_input = TestToolInput(param="value")
    final_response = FinalResponse(
        thought="Tool missing, responding anyway", answer="Tool not found, but continuing"
    )

    orchestrator.loader.get_tool_by_schema.return_value = None

    with patch("app.agent.orchestrator.get_agent_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [tool_input, final_response]

        result = await orchestrator.chat("Test message")

        assert "Tool not found" in result or "continuing" in result


@pytest.mark.asyncio
async def test_chat_max_steps(orchestrator: AgentOrchestrator):
    """Test that chat stops after max steps."""
    tool_input = TestToolInput(param="value")

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value={"result": "ok"})
    orchestrator.loader.get_tool_by_schema.return_value = mock_tool

    with patch("app.agent.orchestrator.get_agent_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = tool_input  # Always return tool call

        result = await orchestrator.chat("Test message")

        assert "reasoning limit" in result.lower() or "reasoning" in result.lower()


def test_trim_history_character_limit(orchestrator: AgentOrchestrator):
    """Test that history trimming respects character limits."""
    # Set a very low character limit
    with patch("app.agent.orchestrator.settings") as mock_settings:
        mock_settings.llm_max_context_chars = 100

        # Add system prompt
        orchestrator.history.append({"role": "system", "content": "System prompt"})

        # Add many messages that exceed the limit
        for _ in range(10):
            orchestrator.history.append(
                {
                    "role": "user",
                    "content": "A" * 50,  # 50 chars each
                }
            )

        orchestrator._trim_history()

        # Should have trimmed down to system prompt + minimal messages
        total_chars = sum(len(m.get("content", "")) for m in orchestrator.history)
        assert total_chars <= mock_settings.llm_max_context_chars or len(orchestrator.history) <= 2


def test_trim_history_preserves_system_prompt(orchestrator: AgentOrchestrator):
    """Test that system prompt is always preserved."""
    with patch("app.agent.orchestrator.settings") as mock_settings:
        mock_settings.llm_max_context_chars = 10  # Very low limit

        orchestrator.history.append({"role": "system", "content": "System prompt"})
        orchestrator.history.append({"role": "user", "content": "A" * 1000})  # Exceeds limit

        orchestrator._trim_history()

        # System prompt should still be at index 0
        assert orchestrator.history[0]["role"] == "system"
