"""
Tests for Agent Integration.

Tests notification injection, ACK tracking, and shared awareness.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.anp.agent_integration import AgentNotificationInjector
from app.anp.event_bus import EventBus
from app.anp.models import EventLedger


class TestAgentNotificationInjector:
    """Test AgentNotificationInjector functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus instance."""
        EventBus._instance = None
        return EventBus()

    @pytest.fixture
    def injector(self, event_bus):
        """Create an AgentNotificationInjector instance."""
        return AgentNotificationInjector(event_bus)

    @pytest.mark.asyncio
    async def test_get_pending_notifications(self, injector, mock_session, monkeypatch):
        """Test querying pending notifications."""
        # Mock database session
        mock_event = MagicMock(spec=EventLedger)
        mock_event.id = "test-id"
        mock_event.content = "Test notification"
        mock_event.routing = {"handler": "agent", "target": "agent"}

        # Mock the execute chain properly
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_event]
        mock_result.scalars.return_value = mock_scalars

        mock_session.execute = AsyncMock(return_value=mock_result)

        # Create a proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        def mock_get_session():
            return mock_cm

        # Replace the get_session function with our mock
        monkeypatch.setattr("app.anp.agent_integration.get_session", mock_get_session)

        events = await injector.get_pending_notifications("test_user")

        # Verify query was made
        mock_session.execute.assert_called_once()
        assert len(events) == 1
        assert events[0] == mock_event

    def test_format_for_system_prompt(self, injector):
        """Test formatting notifications for system prompt."""
        pending = [
            EventLedger(
                id="pending-1",
                content="Action required: process file",
                routing={"handler": "agent"},
                event_metadata={},
            )
        ]

        recent_acks = [
            EventLedger(
                id="ack-1", content="Download completed", routing={"handler": "system"}, event_metadata={}
            )
        ]

        formatted = injector.format_for_system_prompt(pending, recent_acks)

        assert "ACTION REQUIRED:" in formatted
        assert "Action required: process file" in formatted
        assert "INFO:" in formatted
        assert "Download completed" in formatted

    def test_set_last_turn_time(self, injector):
        """Test setting last turn time for shared awareness."""
        from datetime import datetime

        test_time = datetime(2024, 1, 1, 12, 0, 0)
        injector.set_last_turn_time(test_time)

        assert injector._last_turn_time == test_time
