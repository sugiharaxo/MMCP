"""
WebSocket Manager for ANP Channel A/B delivery.

Handles connection storage, Channel A/B delivery, and ACK handling.
"""

import json
from typing import TYPE_CHECKING

from fastapi import WebSocket
from sqlalchemy import select

from app.anp.models import EventLedger, EventStatus
from app.anp.schemas import NotificationAck
from app.core.database import get_session
from app.core.logger import logger

if TYPE_CHECKING:
    from app.anp.event_bus import EventBus


class WebSocketManager:
    """
    WebSocket Manager for Channel A/B delivery.

    Manages WebSocket connections by user_id and session_id,
    handles notification delivery and ACK processing.
    """

    _instance: "WebSocketManager | None" = None

    def __new__(cls) -> "WebSocketManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize WebSocketManager (only once)."""
        if self._initialized:
            return

        # Connection storage: user_id → WebSocket
        self._user_connections: dict[str, WebSocket] = {}
        # session_id → WebSocket
        self._session_connections: dict[str, WebSocket] = {}
        self._event_bus: EventBus | None = None

        self._initialized = True
        logger.info("WebSocketManager initialized")

    def set_event_bus(self, event_bus: "EventBus") -> None:
        """Set the EventBus instance."""
        self._event_bus = event_bus

    def register_connection(
        self, websocket: WebSocket, user_id: str, session_id: str | None = None
    ) -> None:
        """
        Store WebSocket by user_id and optionally session_id.

        Replaces old connection if exists (handles reconnection).
        """
        self._user_connections[user_id] = websocket
        if session_id:
            self._session_connections[session_id] = websocket
        logger.debug(f"Registered WebSocket: user_id={user_id}, session_id={session_id}")

    def unregister_connection(self, user_id: str, session_id: str | None = None) -> None:
        """Remove WebSocket on disconnect."""
        if user_id in self._user_connections:
            del self._user_connections[user_id]
        if session_id and session_id in self._session_connections:
            del self._session_connections[session_id]
        logger.debug(f"Unregistered WebSocket: user_id={user_id}, session_id={session_id}")

    async def send_notification(self, event: EventLedger) -> bool:
        """
        Route notification to appropriate connection (user or session).

        Args:
            event: EventLedger instance to send

        Returns:
            True if sent successfully, False otherwise
        """
        routing = event.routing
        address = routing.get("address")
        session_id = event.session_id
        user_id = event.user_id

        # Determine which connection to use
        websocket = None
        if address == "session" and session_id and session_id in self._session_connections:
            websocket = self._session_connections[session_id]
        elif user_id in self._user_connections:
            websocket = self._user_connections[user_id]

        if not websocket:
            logger.warning(
                f"No WebSocket connection for notification {event.id} "
                f"(user_id={user_id}, session_id={session_id})"
            )
            return False

        # Build notification payload
        payload = {
            "type": "notification",
            "id": event.id,
            "content": event.content,
            "routing": event.routing,
            "metadata": event.event_metadata,
            "created_at": event.created_at.isoformat() if event.created_at else None,
        }

        try:
            await websocket.send_text(json.dumps(payload))
            logger.debug(f"Sent notification {event.id} via WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to send notification {event.id}: {e}", exc_info=True)
            return False

    async def handle_ack(self, ack_data: NotificationAck, user_id: str) -> bool:
        """
        Process ACK message, transition event to DELIVERED.

        Args:
            ack_data: NotificationAck schema
            user_id: User ID for verification

        Returns:
            True if ACK processed successfully, False otherwise
        """
        if not self._event_bus:
            logger.error("EventBus not set in WebSocketManager")
            return False

        # Verify event belongs to user
        async with get_session() as session:
            stmt = select(EventLedger).where(EventLedger.id == ack_data.id)
            result = await session.execute(stmt)
            event = result.scalar_one_or_none()

            if not event:
                logger.warning(f"Event not found for ACK: {ack_data.id}")
                return False

            if event.user_id != user_id:
                logger.warning(
                    f"ACK user mismatch: event.user_id={event.user_id}, ack.user_id={user_id}"
                )
                return False

            # Mark as delivered
            success = await self._event_bus.mark_delivered(ack_data.id)
            return success

    async def process_delivery_queue(self) -> None:
        """
        Process pending DISPATCHED events for Channel A/B.

        Called periodically or after state transitions to deliver notifications.
        """
        if not self._event_bus:
            return

        async with get_session() as session:
            # Query DISPATCHED events for Channel A/B
            stmt = select(EventLedger).where(EventLedger.status == EventStatus.DISPATCHED)
            result = await session.execute(stmt)
            events = result.scalars().all()

            for event in events:
                routing = event.routing
                handler = routing.get("handler")
                address = routing.get("address")

                # Only process Channel A/B (Handler=SYSTEM)
                if handler == "system" and (address == "user" or address == "session"):
                    await self.send_notification(event)
