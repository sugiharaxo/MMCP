"""
Agent Integration for ANP Channel C.

Handles pending notification queries, recent user ACK queries (shared awareness),
system prompt formatting, ACK tracking, and internal turn detection.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import and_, or_, select

from app.anp.models import EventLedger, EventStatus
from app.core.database import get_session
from app.core.logger import logger

if TYPE_CHECKING:
    from app.anp.event_bus import EventBus


class AgentNotificationInjector:
    """
    Agent Notification Injector for Channel C.

    Queries pending notifications, formats for system prompt,
    tracks ACKs, and handles internal turn detection.
    """

    def __init__(self, event_bus: "EventBus"):
        """
        Initialize AgentNotificationInjector.

        Args:
            event_bus: EventBus instance
        """
        self.event_bus = event_bus
        self._last_turn_time: datetime | None = None

    async def get_pending_notifications(
        self, user_id: str = "default", limit: int = 10
    ) -> list[EventLedger]:
        """
        Query Channel C events (status PENDING/DISPATCHED).

        Args:
            user_id: User ID to filter by
            limit: Maximum number of notifications to return

        Returns:
            List of EventLedger instances
        """
        async with get_session() as session:
            # Query Channel C events: Handler=AGENT OR Target=AGENT
            stmt = (
                select(EventLedger)
                .where(
                    and_(
                        EventLedger.status.in_([EventStatus.PENDING, EventStatus.DISPATCHED]),
                        EventLedger.user_id == user_id,
                        or_(
                            EventLedger.routing["handler"].as_string() == "agent",
                            EventLedger.routing["target"].as_string() == "agent",
                        ),
                    )
                )
                .order_by(EventLedger.created_at.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            events = result.scalars().all()
            logger.debug(f"Found {len(events)} pending notifications for user {user_id}")
            return list(events)

    async def get_recent_user_acks(
        self, user_id: str = "default", limit: int = 5
    ) -> list[EventLedger]:
        """
        Query Channel A/B events where status=DELIVERED and ack_at > last_turn_time.

        This implements shared awareness: prevents agent from announcing things
        the user already saw.

        Args:
            user_id: User ID to filter by
            limit: Maximum number of notifications to return

        Returns:
            List of EventLedger instances
        """
        if not self._last_turn_time:
            # First turn - no previous turn time
            return []

        async with get_session() as session:
            # Query DELIVERED events from Channel A/B (Handler=SYSTEM)
            # where ack_at > last_turn_time
            stmt = (
                select(EventLedger)
                .where(
                    and_(
                        EventLedger.status == EventStatus.DELIVERED,
                        EventLedger.user_id == user_id,
                        EventLedger.ack_at.isnot(None),
                        EventLedger.ack_at > self._last_turn_time,
                        EventLedger.routing["handler"].as_string() == "system",
                    )
                )
                .order_by(EventLedger.ack_at.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            events = result.scalars().all()
            logger.debug(f"Found {len(events)} recent user ACKs since {self._last_turn_time}")
            return list(events)

    def format_for_system_prompt(
        self, pending: list[EventLedger], recent_acks: list[EventLedger]
    ) -> str:
        """
        Format notifications for agent context with two sections.

        Section 1: "ACTION REQUIRED" (Pending notifications)
        Section 2: "INFO: The user recently saw..." (Recent user ACKs)

        Args:
            pending: List of pending notifications
            recent_acks: List of recent user ACKs

        Returns:
            Formatted string for system prompt
        """
        sections = []

        # Section 1: Action Required (Pending)
        if pending:
            action_items = []
            for event in pending:
                routing = event.routing
                handler = routing.get("handler")
                directive = ""
                if event.agent_instructions and event.agent_instructions.get("directive"):
                    directive = f" ({event.agent_instructions['directive']})"

                # Mark read-only if Handler=SYSTEM
                marker = "[SEEN]" if handler == "system" else ""
                action_items.append(f"- {marker} {event.content}{directive}")

            sections.append("ACTION REQUIRED: You must process these notifications:")
            sections.extend(action_items)

        # Section 2: Info (Recent User ACKs)
        if recent_acks:
            info_items = []
            for event in recent_acks:
                info_items.append(f"- {event.content}")

            sections.append("\nINFO: The user recently saw these notifications (do not announce):")
            sections.extend(info_items)

        return "\n".join(sections) if sections else ""

    async def mark_agent_processed(self, event_ids: list[str], user_id: str = "default") -> int:
        """
        Process agent ACKs from acknowledged_ids, transition state to DELIVERED.

        Renamed from mark_acknowledged to avoid confusion with user ACKs.

        Args:
            event_ids: List of notification IDs to mark as processed
            user_id: User ID for verification

        Returns:
            Number of events successfully marked as processed
        """
        count = 0
        for event_id in event_ids:
            # Verify event belongs to user
            async with get_session() as session:
                stmt = select(EventLedger).where(EventLedger.id == event_id)
                result = await session.execute(stmt)
                event = result.scalar_one_or_none()

                if not event:
                    logger.warning(f"Event not found for agent ACK: {event_id}")
                    continue

                if event.user_id != user_id:
                    logger.warning(
                        f"Agent ACK user mismatch: event.user_id={event.user_id}, "
                        f"ack.user_id={user_id}"
                    )
                    continue

                # Mark as delivered with lease validation
                success = await self.event_bus.mark_delivered(event_id, event.owner_lease)
                if success:
                    count += 1
                    logger.debug(f"Agent processed notification: {event_id}")

        return count

    def set_last_turn_time(self, timestamp: datetime) -> None:
        """Set the last turn time for shared awareness queries."""
        self._last_turn_time = timestamp
        logger.debug(f"Set last turn time: {timestamp}")

    async def promote_external_tool(self, event_id: str | None = None) -> None:
        """
        Auto-promote Target→USER if EXTERNAL tool invoked.

        This is called immediately when an EXTERNAL tool is detected,
        before tool execution (for robustness).

        Args:
            event_id: Optional specific event ID to promote.
                     If None, promotes all pending Channel C events.
        """
        async with get_session() as session:
            if event_id:
                # Promote specific event
                stmt = select(EventLedger).where(EventLedger.id == event_id)
                result = await session.execute(stmt)
                event = result.scalar_one_or_none()
                events = [event] if event else []
            else:
                # Promote all pending Channel C events
                stmt = select(EventLedger).where(
                    and_(
                        EventLedger.status.in_([EventStatus.PENDING, EventStatus.DISPATCHED]),
                        EventLedger.routing["target"].as_string() == "agent",
                    )
                )
                result = await session.execute(stmt)
                events = result.scalars().all()

            for event in events:
                routing = event.routing.copy()
                if routing.get("target") == "agent":
                    routing["target"] = "user"
                    event.routing = routing
                    await session.commit()
                    await session.refresh(event)
                    logger.info(f"Promoted event {event.id} Target→USER (EXTERNAL tool)")

                    # Re-route to Channel A/B
                    await self.event_bus._route_to_channel(session, event)
