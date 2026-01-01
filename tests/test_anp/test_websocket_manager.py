"""
Tests for WebSocket Manager.

Tests connection handling, notification delivery, and ACK processing.
"""

from unittest.mock import AsyncMock

import pytest

from app.anp.models import EventLedger
from app.anp.websocket_manager import WebSocketManager


class TestWebSocketManager:
    """Test WebSocketManager functionality."""

    @pytest.fixture
    def ws_manager(self):
        """Create a fresh WebSocketManager instance."""
        # Reset singleton
        WebSocketManager._instance = None
        manager = WebSocketManager()
        return manager

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        ws.send_text = AsyncMock()
        return ws

    def test_register_connection(self, ws_manager, mock_websocket):
        """Test registering WebSocket connections."""
        ws_manager.register_connection(mock_websocket, "user1", "session1")

        assert "user1" in ws_manager._user_connections
        assert "session1" in ws_manager._session_connections
        assert ws_manager._user_connections["user1"] == mock_websocket
        assert ws_manager._session_connections["session1"] == mock_websocket

    def test_unregister_connection(self, ws_manager, mock_websocket):
        """Test unregistering WebSocket connections."""
        ws_manager.register_connection(mock_websocket, "user1", "session1")
        ws_manager.unregister_connection("user1", "session1")

        assert "user1" not in ws_manager._user_connections
        assert "session1" not in ws_manager._session_connections

    @pytest.mark.asyncio
    async def test_send_notification_user_connection(self, ws_manager, mock_websocket):
        """Test sending notification via user connection."""
        ws_manager.register_connection(mock_websocket, "user1")

        event = EventLedger(
            id="test-id",
            content="Test notification",
            routing={"address": "user", "target": "user", "handler": "system"},
            user_id="user1",
        )

        success = await ws_manager.send_notification(event)

        assert success is True
        mock_websocket.send_text.assert_called_once()
        call_args = mock_websocket.send_text.call_args[0][0]
        assert '"type": "notification"' in call_args
        assert '"id": "test-id"' in call_args

    @pytest.mark.asyncio
    async def test_send_notification_no_connection(self, ws_manager):
        """Test sending notification when no connection exists."""
        event = EventLedger(
            id="test-id",
            content="Test notification",
            routing={"address": "user", "target": "user", "handler": "system"},
            user_id="user1",
        )

        success = await ws_manager.send_notification(event)

        assert success is False
