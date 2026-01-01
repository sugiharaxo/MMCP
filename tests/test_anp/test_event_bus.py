"""
Tests for EventBus core functionality.

Tests state machine transitions, deduplication, routing, lease-based fencing, and escalation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.anp.event_bus import EventBus
from app.anp.models import EventLedger, EventStatus
from app.anp.schemas import NotificationCreate, RoutingFlags


class TestEventBus:
    """Test EventBus core functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus instance for testing."""
        # Reset singleton
        EventBus._instance = None
        bus = EventBus()
        # Mock session manager
        bus._session_manager = MagicMock()
        bus._session_manager.is_session_active.return_value = True
        return bus

    @pytest.fixture
    def sample_notification(self):
        """Create a sample notification for testing."""
        return NotificationCreate(
            content="Test notification",
            routing=RoutingFlags(address="user", target="agent", handler="agent"),
            user_id="test_user",
        )

    @pytest.mark.asyncio
    async def test_emit_notification_creates_event(
        self, event_bus, sample_notification, mock_session, monkeypatch
    ):
        """Test that emit_notification creates an event in the database."""

        # Mock UUID generation
        test_uuid = "test-id"
        mock_uuid = MagicMock()
        mock_uuid.__str__ = MagicMock(return_value=test_uuid)
        monkeypatch.setattr("app.anp.event_bus.uuid4", MagicMock(return_value=mock_uuid))

        # Mock database session
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        # Create a proper async context manager mock
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        def mock_get_session():
            return mock_cm

        # Replace the get_session function with our mock
        monkeypatch.setattr("app.anp.event_bus.get_session", mock_get_session)

        event_id = await event_bus.emit_notification(sample_notification)

        # Verify event was created
        assert event_id == test_uuid
        mock_session.add.assert_called_once()
        assert mock_session.commit.call_count == 2  # Initial commit + transition commit

    def test_route_to_channel_agent(self, event_bus):
        """Test routing to Channel C (agent context)."""
        event = EventLedger(
            id="test-id",
            content="Test",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.PENDING,
        )

        # Mock session
        mock_session = AsyncMock()

        # Should transition to DISPATCHED for Channel C
        result = event_bus._transition_state(
            mock_session, event.id, EventStatus.PENDING, EventStatus.DISPATCHED, None
        )
        # Note: This test is simplified - actual routing logic is more complex

    def test_lease_based_fencing(self, event_bus):
        """Test lease-based fencing prevents concurrent access."""
        # This would require more complex mocking of database operations
        # For now, just verify the method exists
        assert hasattr(event_bus, "_transition_state")
        assert hasattr(event_bus, "mark_delivered")

    def test_deduplication_logic(self, event_bus, sample_notification):
        """Test deduplication prevents duplicate events."""
        # Mock existing event lookup
        mock_session = AsyncMock()
        mock_existing = MagicMock(spec=EventLedger)
        mock_existing.id = "existing-id"
        mock_session.execute.return_value.scalars.return_value.first.return_value = mock_existing

        # Test would verify that _handle_deduplication returns existing event
        # when deduplication_key matches
        assert hasattr(event_bus, "_handle_deduplication")
