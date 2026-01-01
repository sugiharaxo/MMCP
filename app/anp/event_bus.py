"""
Event Bus core implementation for ANP.

Handles routing logic, state machine, deduplication, lease-based fencing, and escalation.
"""

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.anp.models import EventLedger, EventStatus
from app.anp.schemas import NotificationCreate
from app.core.database import get_session
from app.core.logger import logger

if TYPE_CHECKING:
    from app.anp.session_manager import SessionManager

# Default TTL for Channel C events (5 minutes)
DEFAULT_TTL_SECONDS = 300


class EventBus:
    """
    Event Bus singleton for ANP notification routing and state management.

    Implements:
    - Channel routing (A/B/C) based on routing flags
    - State machine transitions with lease-based fencing
    - Deduplication via deduplication_key
    - Escalation for expired events
    - Causal serialization (queue system alerts during agent turns)
    """

    _instance: "EventBus | None" = None
    _lock = asyncio.Lock()

    def __new__(cls) -> "EventBus":
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize EventBus (only once)."""
        if self._initialized:
            return

        # Causal serialization: queue for system alerts during agent turns
        self._system_alert_queue: asyncio.Queue[EventLedger] = asyncio.Queue()
        self._agent_turn_lock = asyncio.Lock()
        self._session_manager: SessionManager | None = None

        self._initialized = True
        logger.info("EventBus initialized")

    def set_session_manager(self, session_manager: "SessionManager") -> None:
        """Set the session manager for session validation."""
        self._session_manager = session_manager

    async def emit_notification(self, notification: NotificationCreate) -> str:
        """
        Accept notification from plugins, handle deduplication, route to channels.

        Args:
            notification: NotificationCreate schema from plugin

        Returns:
            Notification ID (UUID string)
        """
        async with get_session() as session:
            # Handle deduplication if key provided
            if notification.deduplication_key:
                existing = await self._handle_deduplication(
                    session, notification.deduplication_key, notification
                )
                if existing:
                    logger.debug(
                        f"Deduplicated notification: {existing.id} "
                        f"(key={notification.deduplication_key})"
                    )
                    return existing.id

            # Create new event
            event = EventLedger(
                id=str(uuid4()),
                content=notification.content,
                deduplication_key=notification.deduplication_key,
                event_metadata=notification.metadata,
                routing=notification.routing.model_dump(),
                agent_instructions=(
                    notification.agent_instructions.model_dump()
                    if notification.agent_instructions
                    else None
                ),
                status=EventStatus.PENDING,
                owner_lease=1,
                delivery_deadline=notification.delivery_deadline,
                session_id=notification.session_id,
                user_id=notification.user_id,
            )

            session.add(event)
            await session.commit()
            await session.refresh(event)

            logger.info(f"Created notification: {event.id} (status={event.status.value})")

            # Route to appropriate channel
            await self._route_to_channel(session, event)

            return event.id

    async def _handle_deduplication(
        self,
        session: AsyncSession,
        deduplication_key: str,
        notification: NotificationCreate,
    ) -> EventLedger | None:
        """
        Handle deduplication: upsert logic for deduplication_key.

        If matching PENDING/DISPATCHED event exists, update it.
        If matching DELIVERED/ESCALATED event exists, create new event.
        """
        # Query for existing event with same key in non-terminal states
        stmt = select(EventLedger).where(
            and_(
                EventLedger.deduplication_key == deduplication_key,
                EventLedger.status.in_([EventStatus.PENDING, EventStatus.DISPATCHED]),
            )
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing event
            existing.content = notification.content
            existing.event_metadata = notification.metadata
            existing.delivery_deadline = notification.delivery_deadline
            # Reset status to PENDING if it was DISPATCHED
            if existing.status == EventStatus.DISPATCHED:
                existing.status = EventStatus.PENDING
            await session.commit()
            await session.refresh(existing)
            return existing

        # No existing event in PENDING/DISPATCHED - create new one
        return None

    async def _route_to_channel(self, session: AsyncSession, event: EventLedger) -> None:
        """
        Route event to appropriate channel (A/B/C) based on routing flags.

        Channel A: Address=USER + Handler=SYSTEM
        Channel B: Address=SESSION + Handler=SYSTEM + Target=USER
        Channel C: Handler=AGENT OR Target=AGENT

        Also handles causal serialization: queues Channel A/B events during agent turns.
        """
        routing = event.routing
        address = routing.get("address")
        target = routing.get("target")
        handler = routing.get("handler")

        # Check if session expired (promote SESSION→USER)
        if (
            address == "session"
            and event.session_id
            and self._session_manager
            and not self._session_manager.is_session_active(event.session_id)
        ):
            logger.debug(f"Session {event.session_id} expired, promoting Address SESSION→USER")
            routing["address"] = "user"
            event.routing = routing
            address = "user"

        # Determine channel
        is_channel_a = address == "user" and handler == "system"
        is_channel_b = address == "session" and handler == "system" and target == "user"
        is_channel_c = handler == "agent" or target == "agent"

        # Causal serialization: if agent turn in progress, queue Channel A/B events
        if (is_channel_a or is_channel_b) and self._agent_turn_lock.locked():
            logger.debug(f"Agent turn in progress, queueing notification {event.id}")
            await self._system_alert_queue.put(event)
            return

        # Route to channel
        if is_channel_a or is_channel_b:
            # Channel A/B: System-controlled delivery (WebSocket)
            await self._transition_state(
                session, event.id, EventStatus.PENDING, EventStatus.DISPATCHED, None
            )
            # WebSocketManager will handle actual delivery
            logger.debug(f"Routed to Channel {'A' if is_channel_a else 'B'}: {event.id}")

        elif is_channel_c:
            # Channel C: Agent context (set TTL if not provided)
            if not event.delivery_deadline:
                event.delivery_deadline = datetime.utcnow() + timedelta(seconds=DEFAULT_TTL_SECONDS)
            await self._transition_state(
                session, event.id, EventStatus.PENDING, EventStatus.DISPATCHED, None
            )
            logger.debug(f"Routed to Channel C: {event.id}")

    async def _transition_state(
        self,
        session: AsyncSession,
        event_id: str,
        from_status: EventStatus,
        to_status: EventStatus,
        expected_lease: int | None,
    ) -> bool:
        """
        Atomic state transition with lease verification (CAS).

        Args:
            session: Database session
            event_id: Event ID
            from_status: Expected current status
            to_status: Target status
            expected_lease: Expected owner_lease (None = no check)

        Returns:
            True if transition succeeded, False otherwise
        """
        # Validate transition (prevent terminal state changes)
        if from_status in [EventStatus.DELIVERED, EventStatus.FAILED]:
            logger.warning(
                f"Cannot transition from terminal state {from_status.value} for event {event_id}"
            )
            return False

        # Build update statement with lease check
        conditions = [
            EventLedger.id == event_id,
            EventLedger.status == from_status,
        ]
        if expected_lease is not None:
            conditions.append(EventLedger.owner_lease == expected_lease)

        stmt = (
            update(EventLedger)
            .where(and_(*conditions))
            .values(status=to_status)
            .execution_options(synchronize_session=False)
        )

        result = await session.execute(stmt)
        await session.commit()

        if result.rowcount == 0:
            logger.warning(
                f"State transition failed for {event_id}: "
                f"expected {from_status.value}→{to_status.value} "
                f"(lease check: {expected_lease})"
            )
            return False

        logger.debug(f"State transition: {event_id} {from_status.value}→{to_status.value}")
        return True

    async def mark_delivered(self, event_id: str, lease_id: int | None = None) -> bool:
        """
        Mark event as DELIVERED (from WebSocket ACK or agent ACK).

        Args:
            event_id: Event ID
            lease_id: Optional lease ID for verification

        Returns:
            True if successful, False otherwise
        """
        async with get_session() as session:
            # Get current event
            stmt = select(EventLedger).where(EventLedger.id == event_id)
            result = await session.execute(stmt)
            event = result.scalar_one_or_none()

            if not event:
                logger.warning(f"Event not found: {event_id}")
                return False

            # Verify lease if provided
            if lease_id is not None and event.owner_lease != lease_id:
                logger.warning(
                    f"Lease mismatch for {event_id}: expected {lease_id}, got {event.owner_lease}"
                )
                return False

            # Transition to DELIVERED
            success = await self._transition_state(
                session, event_id, event.status, EventStatus.DELIVERED, lease_id
            )

            if success:
                # Update ack_at timestamp
                stmt = (
                    update(EventLedger)
                    .where(EventLedger.id == event_id)
                    .values(ack_at=datetime.utcnow())
                )
                await session.execute(stmt)
                await session.commit()
                logger.info(f"Event {event_id} marked as DELIVERED")

            return success

    async def _escalate_event(self, session: AsyncSession, event: EventLedger) -> None:
        """
        Watchdog-triggered escalation: Handler→SYSTEM, Target→USER.

        Increments owner_lease and transitions to ESCALATED, then re-routes to Channel A/B.
        """
        # Increment lease
        new_lease = event.owner_lease + 1

        # Update routing flags
        routing = event.routing.copy()
        routing["handler"] = "system"
        if routing.get("target") == "agent":
            routing["target"] = "user"

        # Transition to LOCKED first (watchdog owns it)
        await self._transition_state(
            session, event.id, event.status, EventStatus.LOCKED, event.owner_lease
        )

        # Update routing and lease
        stmt = (
            update(EventLedger)
            .where(EventLedger.id == event.id)
            .values(routing=routing, owner_lease=new_lease, status=EventStatus.ESCALATED)
        )
        await session.execute(stmt)
        await session.commit()
        await session.refresh(event)

        logger.info(f"Escalated event {event.id} (lease {event.owner_lease}→{new_lease})")

        # Re-route to Channel A/B
        await self._route_to_channel(session, event)

    async def escalate_expired_events(self) -> int:
        """
        Escalate expired events (called by WatchdogService).

        Returns:
            Number of events escalated
        """
        async with get_session() as session:
            # Query expired events in PENDING/DISPATCHED
            now = datetime.utcnow()
            stmt = select(EventLedger).where(
                and_(
                    EventLedger.delivery_deadline < now,
                    EventLedger.status.in_([EventStatus.PENDING, EventStatus.DISPATCHED]),
                )
            )
            result = await session.execute(stmt)
            expired = result.scalars().all()

            count = 0
            for event in expired:
                await self._escalate_event(session, event)
                count += 1

            if count > 0:
                logger.info(f"Escalated {count} expired event(s)")

            return count

    async def flush_system_alert_queue(self) -> None:
        """
        Flush queued system alerts after agent turn completes.

        Sends all queued Channel A/B notifications.
        """
        while not self._system_alert_queue.empty():
            event = await self._system_alert_queue.get()
            async with get_session() as session:
                await self._route_to_channel(session, event)
            logger.debug(f"Flushed queued notification: {event.id}")

    def get_agent_turn_lock(self) -> asyncio.Lock:
        """Get the agent turn lock for causal serialization."""
        return self._agent_turn_lock
