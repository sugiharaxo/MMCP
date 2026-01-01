"""
FastAPI routes for ANP notifications.

Provides WebSocket endpoint, ACK endpoint, and notification query endpoint.
"""

import json
from typing import Annotated

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from sqlalchemy import and_, select

from app.anp.event_bus import EventBus
from app.anp.models import EventLedger, EventStatus
from app.anp.schemas import NotificationAck, NotificationResponse
from app.anp.session_manager import SessionManager
from app.anp.websocket_manager import WebSocketManager
from app.core.database import get_session
from app.core.logger import logger

event_bus = EventBus()
websocket_manager = WebSocketManager()
session_manager = SessionManager()

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: Annotated[str | None, Query()] = None,
):
    """
    WebSocket endpoint for Channel A/B delivery.

    Accepts optional session_id query parameter.
    If not provided, generates a new session ID.
    """
    await websocket.accept()

    # Generate session_id if not provided
    if not session_id:
        session_id = session_manager.create_session()
    else:
        session_manager.update_session_activity(session_id)

    user_id = "default"  # Single-user system for now

    # Register connection
    websocket_manager.register_connection(websocket, user_id, session_id)

    try:
        # Send pending notifications
        async with get_session() as session:
            stmt = select(EventLedger).where(
                and_(
                    EventLedger.status == EventStatus.DISPATCHED,
                    EventLedger.user_id == user_id,
                )
            )
            result = await session.execute(stmt)
            events = result.scalars().all()

            for event in events:
                routing = event.routing
                handler = routing.get("handler")
                address = routing.get("address")

                # Only send Channel A/B (Handler=SYSTEM)
                if handler == "system" and (address == "user" or address == "session"):
                    await websocket_manager.send_notification(event)

        # Listen for ACK messages
        while True:
            data = await websocket.receive_text()
            try:
                ack_data = json.loads(data)
                ack = NotificationAck(**ack_data)

                if ack.type == "ack":
                    success = await websocket_manager.handle_ack(ack, user_id)
                    if success:
                        await websocket.send_text(json.dumps({"status": "ok"}))
                    else:
                        await websocket.send_text(
                            json.dumps({"status": "error", "message": "ACK failed"})
                        )
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}", exc_info=True)
                await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session_id={session_id}")
    finally:
        websocket_manager.unregister_connection(user_id, session_id)


@router.post("/ack")
async def ack_endpoint(
    ack: NotificationAck,
):
    """
    HTTP endpoint for ACK (alternative to WebSocket ACK).

    Useful for clients that don't support WebSockets.
    """
    user_id = "default"  # Single-user system for now

    success = await websocket_manager.handle_ack(ack, user_id)

    if success:
        return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "ok", "id": ack.id})
    else:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"status": "error", "message": "ACK failed"},
        )


@router.get("", response_model=list[NotificationResponse])
async def get_notifications(
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    """
    Query notifications (for UI inbox).

    Args:
        status_filter: Optional status filter (pending, dispatched, delivered, etc.)
        limit: Maximum number of notifications to return (1-100)

    Returns:
        List of NotificationResponse
    """
    user_id = "default"  # Single-user system for now

    async with get_session() as session:
        stmt = select(EventLedger).where(EventLedger.user_id == user_id)

        if status_filter:
            try:
                status_enum = EventStatus(status_filter.lower())
                stmt = stmt.where(EventLedger.status == status_enum)
            except ValueError:
                # Invalid status, return empty list
                return []

        stmt = stmt.order_by(EventLedger.created_at.desc()).limit(limit)

        result = await session.execute(stmt)
        events = result.scalars().all()

        # Convert to NotificationResponse
        notifications = []
        for event in events:
            from app.anp.schemas import AgentInstructions, RoutingFlags

            routing = RoutingFlags(**event.routing)
            agent_instructions = (
                AgentInstructions(**event.agent_instructions) if event.agent_instructions else None
            )

            notifications.append(
                NotificationResponse(
                    id=event.id,
                    content=event.content,
                    deduplication_key=event.deduplication_key,
                    metadata=event.event_metadata,
                    routing=routing,
                    agent_instructions=agent_instructions,
                    status=event.status.value,
                    owner_lease=event.owner_lease,
                    created_at=event.created_at,
                    ack_at=event.ack_at,
                    delivery_deadline=event.delivery_deadline,
                    session_id=event.session_id,
                    user_id=event.user_id,
                )
            )

        return notifications
