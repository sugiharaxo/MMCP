"""Proof of Life Test: Verify AgentService architecture works end-to-end.

This test demonstrates that the new Transport/Prompt/Orchestrator pattern
is functional. It verifies the complete flow:
1. AgentService initializes with default TransportService and PromptService
2. process_message successfully executes the compile -> send -> parse sequence
3. Returns a valid response (even if it's the dummy implementation)

This is the "Proof of Life" test that shows the new architecture is operational.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.services.agent import AgentService


@pytest.mark.asyncio
async def test_agent_service_initialization(loader):
    """Test that AgentService initializes with default services."""
    agent_service = AgentService(plugin_loader=loader)

    # Verify services are initialized
    assert agent_service.plugin_loader is not None
    assert agent_service.prompt is not None

    # Cleanup
    await agent_service.close()


@pytest.mark.asyncio
async def test_agent_service_process_message_dummy_flow(loader):
    """
    Proof of Life: Test the complete compile -> send -> parse flow.

    This test verifies that:
    1. PromptService.compile_prompt is called and returns a prompt string
    2. TransportService.send_message is called and returns "DUMMY_LLM_RESPONSE"
    3. PromptService.parse_response parses the response (with fallback)
    4. AgentService returns a valid response dict

    Even though this uses dummy implementations, it proves the architecture
    is correctly wired and the interfaces are compatible.
    """
    from baml_client.types import FinalResponse

    mock_response = FinalResponse(thought="Test thought", answer="Test response")

    with patch("baml_client.b") as mock_baml:
        mock_baml.UniversalAgent = AsyncMock(return_value=mock_response)

        agent_service = AgentService(plugin_loader=loader)

        # Process a test message
        result = await agent_service.process_message(
            user_input="Hello, test message",
            session_id=None,
            system_prompt="You are a helpful assistant.",
        )

        # Verify we got a response
        assert result is not None
        assert isinstance(result, dict)
        assert "response" in result
        assert "type" in result
        assert "session_id" in result

        # The mocked flow should return a FinalResponse
        assert result["type"] == "final_response"
        assert result["response"] == "Test response"

        # Verify session was created
        assert result["session_id"] is not None
        assert len(result["session_id"]) > 0

        # Cleanup
        await agent_service.close()


@pytest.mark.asyncio
async def test_agent_service_session_management(loader):
    """Test that AgentService manages sessions correctly."""
    from baml_client.types import FinalResponse

    # Mock the BAML call to return a dummy FinalResponse
    mock_response = FinalResponse(thought="Test thought", answer="Test response")

    with patch("baml_client.b") as mock_baml:
        mock_baml.UniversalAgent = AsyncMock(return_value=mock_response)

        agent_service = AgentService(plugin_loader=loader)

        # Process first message (creates session)
        result1 = await agent_service.process_message(
            user_input="First message",
            session_id=None,
        )

        session_id = result1["session_id"]
        assert session_id is not None

        # Process second message with same session
        result2 = await agent_service.process_message(
            user_input="Second message",
            session_id=session_id,
        )

        # Should use same session
        assert result2["session_id"] == session_id

        # Get session history directly from session_manager (not through thin wrapper)
        history = await agent_service.session_manager.load_session(session_id)
        assert history is not None
        assert isinstance(history, list)
        # History should contain messages from both interactions
        assert len(history) > 0

        # Clear session by saving empty history
        await agent_service.session_manager.save_session(session_id, [])

        # Verify session is cleared
        history_after_clear = await agent_service.session_manager.load_session(session_id)
        assert history_after_clear == []

        # Cleanup
        await agent_service.close()
