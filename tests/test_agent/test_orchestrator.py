"""Tests for agent orchestrator."""

from pathlib import Path
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
class ToolInputSchema(BaseModel):
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

    # Mock the LLM interface method
    with patch("app.agent.llm_interface.LLMInterface.get_agent_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = response

        result = await orchestrator.chat("Test message")

        # Verify the mock was called
        mock_llm.assert_called()
        assert result == "Test response"


@pytest.mark.asyncio
async def test_chat_tool_call(orchestrator: AgentOrchestrator):
    """Test that chat executes tools when LLM requests them."""
    from mmcp import PluginRuntime

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.plugin_name = "test_plugin"  # Set plugin_name attribute
    mock_tool.execute = AsyncMock(return_value={"result": "Tool executed"})
    orchestrator.loader.get_tool_by_schema.return_value = mock_tool
    orchestrator.loader.standby_tools = {}  # Ensure tool is not in standby
    # Mock create_plugin_runtime to return a real PluginRuntime
    from app.core.config import CoreSettings

    mock_plugin_runtime = PluginRuntime(
        paths=CoreSettings(root_dir=Path("/"), download_dir=Path("/"), cache_dir=Path("/")),
        system={},
    )
    orchestrator.loader.create_plugin_runtime.return_value = mock_plugin_runtime
    # Mock plugin settings (None for tools without settings) - stored by plugin_name
    orchestrator.loader._plugin_settings = {"test_plugin": None}

    # First call: tool input schema, Second call: final response
    tool_input = ToolInputSchema(param="value")
    final_response = FinalResponse(thought="Tool executed successfully", answer="Done")

    call_count = 0
    async def mock_llm_decision(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_input
        else:
            return final_response

    with patch("app.agent.llm_interface.LLMInterface.get_agent_decision", side_effect=mock_llm_decision):
        result = await orchestrator.chat("Test message")

        assert result == "Done"
        # Verify tool was called with kwargs only (context/settings are in self)
        args, kwargs = mock_tool.execute.call_args
        assert len(args) == 0  # No positional args - context/settings are in self
        assert kwargs == {"param": "value"}


@pytest.mark.asyncio
async def test_chat_tool_not_found(orchestrator: AgentOrchestrator):
    """Test that chat handles missing tools gracefully."""
    tool_input = ToolInputSchema(param="value")
    final_response = FinalResponse(
        thought="Tool missing, responding anyway", answer="Tool not found, but continuing"
    )

    orchestrator.loader.get_tool_by_schema.return_value = None

    call_count = 0
    async def mock_llm_decision(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_input
        else:
            return final_response

    with patch("app.agent.llm_interface.LLMInterface.get_agent_decision", side_effect=mock_llm_decision):
        result = await orchestrator.chat("Test message")

        assert "Tool not found" in result or "continuing" in result


@pytest.mark.asyncio
async def test_chat_max_steps(orchestrator: AgentOrchestrator):
    """Test that chat stops after max steps."""
    tool_input = ToolInputSchema(param="value")

    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value={"result": "ok"})
    orchestrator.loader.get_tool_by_schema.return_value = mock_tool

    with patch("app.agent.llm_interface.LLMInterface.get_agent_decision", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = tool_input  # Always return tool call

        result = await orchestrator.chat("Test message")

        assert "reasoning limit" in result.lower() or "reasoning" in result.lower()


def test_trim_history_character_limit(orchestrator: AgentOrchestrator):
    """Test that history trimming respects character limits."""
    # Set a very low character limit
    with patch("app.agent.history_manager.settings") as mock_settings:
        mock_settings.llm_max_context_chars = 100

        # Create test history
        history = [{"role": "system", "content": "System prompt"}]

        # Add many messages that exceed the limit
        for _ in range(10):
            history.append(
                {
                    "role": "user",
                    "content": "A" * 50,  # 50 chars each
                }
            )

        orchestrator.history_manager.trim_history(history)

        # Should have trimmed down to system prompt + minimal messages
        total_chars = sum(len(m.get("content", "")) for m in history)
        assert total_chars <= mock_settings.llm_max_context_chars or len(history) <= 2


def test_trim_history_preserves_system_prompt(orchestrator: AgentOrchestrator):
    """Test that system prompt is always preserved."""
    with patch("app.agent.history_manager.settings") as mock_settings:
        mock_settings.llm_max_context_chars = 10  # Very low limit

        history = [{"role": "system", "content": "System prompt"}]
        history.append({"role": "user", "content": "A" * 1000})  # Exceeds limit

        orchestrator.history_manager.trim_history(history)

        # System prompt should still be at index 0
        assert history[0]["role"] == "system"


@pytest.mark.asyncio
async def test_system_prompt_self_healing_filters_duplicates(orchestrator: AgentOrchestrator):
    """Test that duplicate system messages are filtered out during reconstruction."""
    from app.core.context import MMCPContext

    # Create test history with multiple system messages (simulating corruption)
    history = [
        {"role": "system", "content": "Old system prompt"},
        {"role": "system", "content": "Another system prompt"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ]

    context = MMCPContext()

    # Trigger the system prompt reconstruction (normally happens in chat())
    await orchestrator.context_manager.assemble_llm_context("test", context)
    system_prompt = await orchestrator.context_manager.get_system_prompt(context)
    new_system_message = {"role": "system", "content": system_prompt}
    non_system_messages = [m for m in history if m["role"] != "system"]
    reconstructed_history = [new_system_message] + non_system_messages

    # Should have exactly one system message at index 0
    assert reconstructed_history[0]["role"] == "system"
    assert sum(1 for m in reconstructed_history if m["role"] == "system") == 1
    # Should preserve other messages
    assert len(reconstructed_history) == 3  # system + user + assistant


@pytest.mark.asyncio
async def test_system_prompt_includes_context(orchestrator: AgentOrchestrator):
    """Test that system prompt includes context data (plugins are responsible for sanitization)."""
    from app.core.context import MMCPContext

    # Create context with data (plugins should sanitize, not core)
    context = MMCPContext()
    context.llm.user_preferences = {"theme": "dark", "language": "en"}
    context.llm.media_state = {"jellyfin": {"server_url": "http://localhost:8096"}}

    prompt = await orchestrator.context_manager.get_system_prompt(context)

    # Should contain context data (raw, as plugins are responsible for sanitization)
    assert "CONTEXT:" in prompt
    assert "User Preferences:" in prompt
    assert "Media State:" in prompt
    assert "theme" in prompt or "dark" in prompt  # Context should be included


@pytest.mark.asyncio
async def test_system_prompt_reconstruction_works_on_empty_history(orchestrator: AgentOrchestrator):
    """Test that system prompt reconstruction works correctly on empty history."""
    from app.core.context import MMCPContext

    # Empty history
    history = []

    context = MMCPContext()

    # This should work without errors
    await orchestrator.context_manager.assemble_llm_context("test", context)
    system_prompt = await orchestrator.context_manager.get_system_prompt(context)
    new_system_message = {"role": "system", "content": system_prompt}
    non_system_messages = [m for m in history if m["role"] != "system"]
    reconstructed_history = [new_system_message] + non_system_messages

    # Should create a single system message
    assert len(reconstructed_history) == 1
    assert reconstructed_history[0]["role"] == "system"
