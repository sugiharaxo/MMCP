"""
FastAPI routes for ANP notifications.

Provides WebSocket endpoint, ACK endpoint, and notification query endpoint.
"""

import json
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from sqlalchemy import and_, select

from app.agent.orchestrator import AgentOrchestrator
from app.anp.event_bus import EventBus
from app.anp.models import EventLedger, EventStatus
from app.anp.notification_dispatcher import NotificationDispatcher
from app.anp.schemas import NotificationAck, NotificationResponse, SessionResponse
from app.core.database import get_session
from app.core.logger import logger
from app.core.session_manager import SessionManager

event_bus = EventBus()
notification_dispatcher = NotificationDispatcher()
session_manager = SessionManager()

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


@router.post("/sessions", response_model=SessionResponse)
async def create_session():
    """
    Create a new chat session.

    Sessions represent persistent chat conversations.
    They exist until explicitly deleted.
    """
    # user_id = "default"  # Single-user system for now - reserved for future use

    session_id = session_manager.create_session()

    # Return session info
    return SessionResponse(id=session_id, created_at=datetime.now(timezone.utc))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a chat session.

    This permanently removes the session and any associated context.
    """
    # user_id = "default"  # Single-user system for now - reserved for future use

    session_manager.delete_session(session_id)
    return {"status": "ok", "message": f"Session {session_id} deleted"}


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions():
    """
    List all active chat sessions for the user.

    Returns sessions that exist (i.e., chats that haven't been deleted).
    """
    # user_id = "default"  # Single-user system for now - reserved for future use

    # For now, we can't easily track creation timestamps for existing sessions
    # In a real implementation, sessions would be stored in the database
    # For this MVP, we'll return session IDs without timestamps
    active_sessions = list(session_manager.active_sessions)

    return [
        SessionResponse(id=session_id, created_at=datetime.now(timezone.utc))
        for session_id in active_sessions
    ]


@router.post("/notifications/{event_id}/approve")
async def approve_action(event_id: str):
    """
    Approve and execute a pending external tool action.

    Retrieves the frozen state, executes the tool, and resumes the agent conversation.
    """
    user_id = "default"  # Single-user system for now

    async with get_session() as session:
        # Verify the event exists and belongs to the user
        stmt = select(EventLedger).where(EventLedger.id == event_id)
        result = await session.execute(stmt)
        event = result.scalar_one_or_none()

        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        if event.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if not event.pending_action_data:
            raise HTTPException(status_code=400, detail="No pending action for this event")

        # Mark event as delivered (tool approved)
        success = await event_bus.mark_delivered(event_id)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to approve action")

        # Use orchestrator to handle the resumption
        try:
            orchestrator = AgentOrchestrator()
            final_response = await orchestrator.resume_action(
                event_id=event_id, session_id=event.session_id, trace_id=f"resumed-{event_id}"
            )

            return {"status": "success", "response": final_response}

        except Exception as e:
            logger.error(f"Failed to resume conversation: {e}")
            raise HTTPException(status_code=500, detail="Failed to resume conversation") from e


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ANP notification delivery.

    Maintains one connection per user for all their chat sessions.
    Notifications include session_id for client-side routing.
    """
    await websocket.accept()

    user_id = "default"  # Single-user system for now

    # Register connection (one per user)
    notification_dispatcher.register_connection(websocket, user_id)

    try:
        # Send pending notifications (SYSTEM handler only)
        async with get_session() as session:
            stmt = select(EventLedger).where(
                and_(
                    EventLedger.status == EventStatus.DISPATCHED,
                    EventLedger.user_id == user_id,
                    EventLedger.routing["handler"].astext == "system",  # Only SYSTEM notifications
                )
            )
            result = await session.execute(stmt)
            events = result.scalars().all()

            for event in events:
                await notification_dispatcher.send_notification(event)

        # Listen for messages (ACKs, etc.)
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                if message.get("type") == "ack":
                    success = await notification_dispatcher.handle_ack(message, user_id)
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
        logger.info("WebSocket disconnected")
    finally:
        notification_dispatcher.unregister_connection(user_id)


@router.post("/ack")
async def ack_endpoint(
    ack: NotificationAck,
):
    """
    HTTP endpoint for ACK (alternative to WebSocket ACK).

    Useful for clients that don't support WebSockets.
    """
    # user_id = "default"  # Single-user system for now - reserved for future use

    # Convert NotificationAck to dict format expected by NotificationDispatcher
    ack_dict = {"event_id": ack.id, "lease_id": getattr(ack, "lease_id", 1)}
    success = await notification_dispatcher.handle_ack(ack_dict, "default")

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
