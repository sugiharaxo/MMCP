"""
Tests for NotificationDispatcher.

Tests unified WebSocket management, notification routing, and ACK handling.
"""

from unittest.mock import AsyncMock

import pytest

from app.anp.models import EventLedger, EventStatus
from app.anp.notification_dispatcher import NotificationDispatcher


class TestNotificationDispatcher:
    """Test NotificationDispatcher functionality."""

    @pytest.fixture
    def dispatcher(self):
        """Create a fresh NotificationDispatcher instance for testing."""
        # Reset singleton
        NotificationDispatcher._instance = None
        dispatcher = NotificationDispatcher()
        # Mock EventBus
        dispatcher._event_bus = AsyncMock()
        return dispatcher

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        return AsyncMock()

    @pytest.fixture
    def sample_event(self):
        """Create a sample notification event."""
        return EventLedger(
            id="test-event-123",
            content="Test notification",
            routing={"address": "session", "target": "user", "handler": "system"},
            status=EventStatus.DISPATCHED,
            user_id="test_user",
            session_id="test_session",
            owner_lease=1,
            event_metadata={},
        )

    @pytest.mark.asyncio
    async def test_register_connection(self, dispatcher, mock_websocket):
        """Test registering a WebSocket connection."""
        user_id = "test_user"

        dispatcher.register_connection(mock_websocket, user_id)

        assert dispatcher._connections[user_id] is mock_websocket
        assert dispatcher.is_user_connected(user_id)

    @pytest.mark.asyncio
    async def test_unregister_connection(self, dispatcher, mock_websocket):
        """Test unregistering a WebSocket connection."""
        user_id = "test_user"

        # Register first
        dispatcher.register_connection(mock_websocket, user_id)
        assert dispatcher.is_user_connected(user_id)

        # Unregister
        dispatcher.unregister_connection(user_id)
        assert not dispatcher.is_user_connected(user_id)

    @pytest.mark.asyncio
    async def test_send_notification_success(self, dispatcher, mock_websocket, sample_event):
        """Test successful notification sending."""
        user_id = sample_event.user_id

        # Register connection
        dispatcher.register_connection(mock_websocket, user_id)

        # Send notification
        success = await dispatcher.send_notification(sample_event)

        assert success
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]

        # Verify message format
        assert call_args["type"] == "notification"
        assert call_args["id"] == sample_event.id
        assert call_args["session_id"] == sample_event.session_id
        assert call_args["content"] == sample_event.content

    @pytest.mark.asyncio
    async def test_send_notification_no_connection(self, dispatcher, sample_event):
        """Test notification sending when no connection exists."""
        success = await dispatcher.send_notification(sample_event)

        assert not success

    @pytest.mark.asyncio
    async def test_send_notification_connection_error(
        self, dispatcher, mock_websocket, sample_event
    ):
        """Test notification sending when connection fails."""
        user_id = sample_event.user_id

        # Register connection that will fail
        mock_websocket.send_json.side_effect = Exception("Connection lost")
        dispatcher.register_connection(mock_websocket, user_id)

        # Send notification
        success = await dispatcher.send_notification(sample_event)

        assert not success
        # Connection should be cleaned up on failure
        assert not dispatcher.is_user_connected(user_id)

    @pytest.mark.asyncio
    async def test_handle_ack_success(self, dispatcher):
        """Test successful ACK processing."""
        from app.anp.schemas import NotificationAck

        user_id = "test_user"
        ack_data = NotificationAck(id="test-event-123", lease_id=1)

        # Mock successful EventBus response
        dispatcher._event_bus.mark_delivered.return_value = True

        success = await dispatcher.handle_ack(ack_data, user_id)

        assert success
        dispatcher._event_bus.mark_delivered.assert_called_once_with("test-event-123", 1)

    @pytest.mark.asyncio
    async def test_handle_ack_failure(self, dispatcher):
        """Test ACK processing failure."""
        from app.anp.schemas import NotificationAck

        user_id = "test_user"
        ack_data = NotificationAck(id="test-event-123", lease_id=1)

        # Mock failed EventBus response
        dispatcher._event_bus.mark_delivered.return_value = False

        success = await dispatcher.handle_ack(ack_data, user_id)

        assert not success
        dispatcher._event_bus.mark_delivered.assert_called_once_with("test-event-123", 1)

    def test_connection_count(self, dispatcher, mock_websocket):
        """Test connection count tracking."""
        assert dispatcher.get_connection_count() == 0

        dispatcher.register_connection(mock_websocket, "user1")
        assert dispatcher.get_connection_count() == 1

        dispatcher.register_connection(mock_websocket, "user2")
        assert dispatcher.get_connection_count() == 2

        dispatcher.unregister_connection("user1")
        assert dispatcher.get_connection_count() == 1
