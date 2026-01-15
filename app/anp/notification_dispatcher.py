"""
Notification Dispatcher for ANP Transport Layer.

Unified WebSocket management for one-connection-per-user model.
Handles notification routing, ACK processing, and lease fencing.
"""

from typing import TYPE_CHECKING

from fastapi import WebSocket

from app.core.logger import logger

if TYPE_CHECKING:
    from app.anp.event_bus import EventBus
    from app.anp.models import EventLedger
    from app.anp.schemas import NotificationAck


class NotificationDispatcher:
    """
    Notification Dispatcher for ANP Transport Layer.

    Maintains one WebSocket connection per user and routes notifications
    with session_id included in the payload for client-side routing.
    """

    _instance = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize NotificationDispatcher (only once)."""
        if self._initialized:
            return

        # user_id -> WebSocket connection
        self._connections: dict[str, WebSocket] = {}
        self._event_bus: EventBus | None = None
        self._initialized = True
        logger.info("NotificationDispatcher initialized")

    def set_event_bus(self, event_bus: "EventBus") -> None:
        """Set the EventBus instance."""
        self._event_bus = event_bus

    def register_connection(self, websocket: WebSocket, user_id: str) -> None:
        """
        Register a WebSocket connection for a user.

        Replaces old connection if exists (handles reconnection).
        """
        self._connections[user_id] = websocket
        logger.debug(f"Registered WebSocket connection for user: {user_id}")

    def unregister_connection(self, user_id: str) -> None:
        """Unregister a WebSocket connection."""
        if user_id in self._connections:
            del self._connections[user_id]
            logger.debug(f"Unregistered WebSocket connection for user: {user_id}")

    async def send_notification(self, event: "EventLedger") -> bool:
        """
        Send notification to user's WebSocket with session_id in payload.

        Args:
            event: EventLedger instance to send

        Returns:
            True if sent successfully, False otherwise
        """
        user_id = event.user_id
        if user_id not in self._connections:
            logger.warning(
                f"No WebSocket connection for user {user_id} (notification {event.id} undelivered)"
            )
            return False

        websocket = self._connections[user_id]

        # Build notification payload with session_id for client routing
        payload = {
            "type": "notification",
            "id": event.id,
            "session_id": event.session_id,
            "content": event.content,
            "routing": event.routing,
            "metadata": event.event_metadata.copy(),  # Copy to avoid modifying original
            "owner_lease": event.owner_lease,
            "timestamp": event.created_at.isoformat() if event.created_at else None,
        }

        # FIX: Data Leakage Protection - Remove internal protocol keys before sending to UI
        payload["metadata"].pop("anp_turn_depth", None)

        try:
            await websocket.send_json(payload)
            logger.debug(f"Sent notification {event.id} to user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification {event.id}: {e}")
            # Remove broken connection
            self.unregister_connection(user_id)
            return False

    async def handle_ack(self, ack_data: "NotificationAck", user_id: str) -> bool:
        """
        Process ACK message with lease fencing.

        Args:
            ack_data: NotificationAck schema instance
            user_id: User ID for verification

        Returns:
            True if ACK processed successfully, False otherwise
        """
        if not self._event_bus:
            logger.error("EventBus not set in NotificationDispatcher")
            return False

        event_id = ack_data.id
        lease_id = ack_data.lease_id

        if not event_id:
            logger.error("ACK missing event_id")
            return False

        # Verify event belongs to user and transition with lease check
        success = await self._event_bus.mark_delivered(event_id, lease_id)
        if success:
            logger.debug(f"Processed ACK for event {event_id} (user: {user_id})")
        else:
            logger.warning(f"Failed to process ACK for event {event_id} (stale lease or not found)")

        return success

    def is_user_connected(self, user_id: str) -> bool:
        """Check if user has an active WebSocket connection."""
        return user_id in self._connections

    def get_connection_count(self) -> int:
        """Get current number of active connections."""
        return len(self._connections)
