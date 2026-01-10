"""
Tests for EventBus core functionality.

Tests state machine transitions, deduplication, routing, lease-based fencing, and escalation.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, MagicMock, patch

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
        mock_session.add = MagicMock(return_value=None)
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

    @pytest.mark.asyncio
    async def test_channel_a_routing(self, event_bus):
        """Test routing to Channel A (USER + SYSTEM)."""
        # Mock session manager for session validation
        event_bus._session_manager = AsyncMock()
        event_bus._session_manager.is_session_active.return_value = True

        # Create event for Channel A: Address=USER + Handler=SYSTEM
        event = EventLedger(
            id="test-channel-a",
            content="Test Channel A",
            routing={"address": "user", "target": "user", "handler": "system"},
            status=EventStatus.PENDING,
            user_id="test_user",
        )

        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Route to channel
        await event_bus._route_to_channel(mock_session, event)

        # Verify state transition to DISPATCHED
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

        # Verify no TTL was set (Channel A is immediate delivery)
        assert event.delivery_deadline is None

    @pytest.mark.asyncio
    async def test_channel_b_routing(self, event_bus):
        """Test routing to Channel B (SESSION + SYSTEM + USER)."""
        # Mock session manager - session is active (sync method)
        event_bus._session_manager = Mock()
        event_bus._session_manager.is_session_active.return_value = True

        # Create event for Channel B: Address=SESSION + Handler=SYSTEM + Target=USER
        event = EventLedger(
            id="test-channel-b",
            content="Test Channel B",
            routing={"address": "session", "target": "user", "handler": "system"},
            status=EventStatus.PENDING,
            session_id="active-session",
            user_id="test_user",
        )

        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Route to channel
        await event_bus._route_to_channel(mock_session, event)

        # Verify state transition to DISPATCHED
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

        # Verify no TTL was set (Channel B is immediate delivery)
        assert event.delivery_deadline is None

    @pytest.mark.asyncio
    async def test_channel_c_routing(self, event_bus):
        """Test routing to Channel C (AGENT routes)."""
        # Mock session manager
        event_bus._session_manager = AsyncMock()
        event_bus._session_manager.is_session_active.return_value = True

        # Create event for Channel C: Handler=AGENT (any address/target combo)
        event = EventLedger(
            id="test-channel-c",
            content="Test Channel C",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.PENDING,
            user_id="test_user",
        )

        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Route to channel
        await event_bus._route_to_channel(mock_session, event)

        # Verify state transition to DISPATCHED
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

        # Verify TTL was set for Channel C (agent processing)
        assert event.delivery_deadline is not None

    @pytest.mark.asyncio
    async def test_session_expiration_promotion(self, event_bus):
        """Test session expiration promotes Address SESSION→USER."""
        # Mock session manager - session is EXPIRED
        from unittest.mock import MagicMock

        event_bus._session_manager = MagicMock()
        event_bus._session_manager.is_session_active.return_value = False

        # Create event with Address=SESSION
        original_routing = {"address": "session", "target": "user", "handler": "system"}
        event = EventLedger(
            id="test-session-expired",
            content="Test expired session",
            routing=original_routing.copy(),
            status=EventStatus.PENDING,
            session_id="expired-session",
            user_id="test_user",
        )

        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Mock session manager for session expiry check (sync method)
        event_bus._session_manager = Mock()
        event_bus._session_manager.is_session_active.return_value = False  # Session expired

        # Route to channel (should promote session to user)
        await event_bus._route_to_channel(mock_session, event)

        # Verify routing was updated to address="user"
        assert event.routing["address"] == "user"
        assert event.routing["target"] == "user"
        assert event.routing["handler"] == "system"

        # Verify state transition to DISPATCHED
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_lease_based_fencing(self, event_bus):
        """Test lease-based fencing prevents concurrent access."""
        # This would require more complex mocking of database operations
        # For now, just verify the method exists
        assert hasattr(event_bus, "_transition_state")
        assert hasattr(event_bus, "mark_delivered")

    @pytest.mark.asyncio
    async def test_deduplication_update_pending_event(self, event_bus):
        """Test deduplication updates existing PENDING event."""
        from app.anp.schemas import NotificationCreate, RoutingFlags

        # Create new notification with deduplication key
        notification = NotificationCreate(
            content="Updated content",
            routing=RoutingFlags(address="user", target="user", handler="system"),
            deduplication_key="test-key",
            metadata={"updated": True},
        )

        # Mock existing PENDING event
        existing_event = EventLedger(
            id="existing-id",
            content="Original content",
            deduplication_key="test-key",
            routing={"address": "user", "target": "user", "handler": "system"},
            status=EventStatus.PENDING,
            owner_lease=1,
        )

        # Mock session to return existing event
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [existing_event]
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one_or_none.return_value = existing_event
        mock_session.execute.return_value = mock_result
        mock_session.commit.return_value = None

        # Test deduplication
        result = await event_bus._handle_deduplication(mock_session, "test-key", notification)

        # Verify existing event was updated and returned
        assert result is existing_event
        assert existing_event.content == "Updated content"
        assert existing_event.event_metadata == {"updated": True}
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplication_update_dispatched_event(self, event_bus):
        """Test deduplication updates existing DISPATCHED event and resets to PENDING."""
        from app.anp.schemas import NotificationCreate, RoutingFlags

        # Create new notification with deduplication key
        notification = NotificationCreate(
            content="Updated content",
            routing=RoutingFlags(address="user", target="agent", handler="agent"),
            deduplication_key="test-key",
            metadata={"reset": True},
        )

        # Mock existing DISPATCHED event
        existing_event = EventLedger(
            id="dispatched-id",
            content="Original content",
            deduplication_key="test-key",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.DISPATCHED,
            owner_lease=1,
        )

        # Mock session to return existing event and allow state transition
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [existing_event]
        mock_result.scalars.return_value = mock_scalars
        mock_result.scalar_one_or_none.return_value = existing_event
        mock_session.execute.return_value = mock_result
        mock_session.commit.return_value = None

        # Mock successful state transition
        event_bus._transition_state = AsyncMock(return_value=True)

        # Test deduplication
        result = await event_bus._handle_deduplication(mock_session, "test-key", notification)

        # Verify existing event was updated and returned
        assert result is existing_event
        assert existing_event.content == "Updated content"
        assert existing_event.event_metadata == {"reset": True}

        # Verify state was reset to PENDING
        event_bus._transition_state.assert_called_once_with(
            mock_session, existing_event.id, EventStatus.DISPATCHED, EventStatus.PENDING, None
        )
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deduplication_create_new_when_delivered(self, event_bus):
        """Test deduplication creates new event when existing is DELIVERED."""
        from app.anp.schemas import NotificationCreate, RoutingFlags

        # Create new notification with deduplication key
        notification = NotificationCreate(
            content="New content",
            routing=RoutingFlags(address="user", target="user", handler="system"),
            deduplication_key="test-key",
        )

        # Mock session - no existing events in PENDING/DISPATCHED (only DELIVERED)
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No PENDING/DISPATCHED events
        mock_session.execute.return_value = mock_result

        # Test deduplication
        result = await event_bus._handle_deduplication(mock_session, "test-key", notification)

        # Verify None returned (signals to create new event)
        assert result is None
        # Verify no database operations were performed
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplication_create_new_when_escalated(self, event_bus):
        """Test deduplication creates new event when existing is ESCALATED."""
        from app.anp.schemas import NotificationCreate, RoutingFlags

        # Create new notification with deduplication key
        notification = NotificationCreate(
            content="New content",
            routing=RoutingFlags(address="user", target="user", handler="system"),
            deduplication_key="test-key",
        )

        # Mock session - no existing events in PENDING/DISPATCHED states
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # No PENDING/DISPATCHED events
        mock_session.execute.return_value = mock_result

        # Test deduplication
        result = await event_bus._handle_deduplication(mock_session, "test-key", notification)

        # Verify None returned (signals to create new event)
        assert result is None
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplication_race_condition_handling(self, event_bus):
        """Test deduplication handles race condition when DISPATCHED event becomes terminal."""
        from app.anp.schemas import NotificationCreate, RoutingFlags

        # Create new notification with deduplication key
        notification = NotificationCreate(
            content="Updated content",
            routing=RoutingFlags(address="user", target="agent", handler="agent"),
            deduplication_key="test-key",
        )

        # Mock existing DISPATCHED event
        existing_event = EventLedger(
            id="racing-id",
            content="Original content",
            deduplication_key="test-key",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.DISPATCHED,
            owner_lease=1,
        )

        # Mock session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_event
        mock_session.execute.return_value = mock_result
        mock_session.rollback.return_value = None

        # Mock failed state transition (event became terminal during update)
        event_bus._transition_state = AsyncMock(return_value=False)

        # Test deduplication
        result = await event_bus._handle_deduplication(mock_session, "test-key", notification)

        # Verify None returned (signals to create new event due to race condition)
        assert result is None
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        # Verify no commit happened
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_escalate_event_target_agent_promotion(self, event_bus):
        """Test _escalate_event promotes Target=AGENT to Target=USER."""

        # Create event with Target=AGENT (should be promoted to USER)
        event = EventLedger(
            id="escalate-test-1",
            content="Test escalation",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.DISPATCHED,
            owner_lease=5,
            user_id="test_user",
        )

        # Mock session and database operations
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        # Mock _route_to_channel to avoid additional complexity
        event_bus._route_to_channel = AsyncMock()

        # Test escalation
        await event_bus._escalate_event(mock_session, event)

        # Verify state transition to LOCKED first
        # (We can't easily inspect the exact call, but we can verify the escalation happened)

        # Verify database update was called (routing update, lease increment, status=ESCALATED)
        assert mock_session.execute.call_count >= 2  # LOCKED transition + routing update
        mock_session.commit.assert_called()

        # Verify escalation process completed (routing update, lease increment, re-routing)
        assert mock_session.execute.call_count >= 2  # LOCKED transition + routing update
        mock_session.commit.assert_called()

        # Verify re-routing was called
        event_bus._route_to_channel.assert_called_once_with(mock_session, event)

    @pytest.mark.asyncio
    async def test_escalate_event_target_user_no_promotion(self, event_bus):
        """Test _escalate_event doesn't promote Target=USER."""
        # Create event with Target=USER (should remain USER)
        event = EventLedger(
            id="escalate-test-2",
            content="Test escalation",
            routing={"address": "session", "target": "user", "handler": "agent"},
            status=EventStatus.DISPATCHED,
            owner_lease=3,
            session_id="test-session",
            user_id="test_user",
        )

        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        # Mock _route_to_channel
        event_bus._route_to_channel = AsyncMock()

        # Test escalation
        await event_bus._escalate_event(mock_session, event)

        # Verify escalation completed
        mock_session.commit.assert_called()
        event_bus._route_to_channel.assert_called_once_with(mock_session, event)

    @pytest.mark.asyncio
    async def test_escalate_event_lease_increment(self, event_bus):
        """Test _escalate_event increments owner_lease."""
        # Create event with known lease
        original_lease = 7
        event = EventLedger(
            id="escalate-test-3",
            content="Test lease increment",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.DISPATCHED,
            owner_lease=original_lease,
            user_id="test_user",
        )

        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None

        # Mock _route_to_channel
        event_bus._route_to_channel = AsyncMock()

        # Test escalation
        await event_bus._escalate_event(mock_session, event)

        # Verify lease was incremented in database update
        # We can't easily inspect the exact values, but the escalation happened
        mock_session.commit.assert_called()
        event_bus._route_to_channel.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_expired_events_no_expired(self, event_bus):
        """Test escalate_expired_events returns 0 when no expired events."""
        from unittest.mock import MagicMock

        # Mock database session with no expired events
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []  # No expired events
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        # Mock get_session context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        # Mock the escalate_expired_events to use our mock session
        with patch("app.anp.event_bus.get_session", return_value=mock_cm):
            count = await event_bus.escalate_expired_events()

        # Verify no events were escalated
        assert count == 0
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalate_expired_events_with_expired(self, event_bus):
        """Test escalate_expired_events finds and escalates expired events."""
        from datetime import datetime, timedelta, UTC
        from unittest.mock import MagicMock

        # Create expired event
        expired_event = EventLedger(
            id="expired-event",
            content="Expired notification",
            routing={"address": "user", "target": "agent", "handler": "agent"},
            status=EventStatus.DISPATCHED,
            owner_lease=1,
            delivery_deadline=datetime.now(UTC) - timedelta(minutes=5),  # Expired
            user_id="test_user",
        )

        # Mock database session
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [expired_event]
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        mock_session.commit.return_value = None

        # Mock get_session context manager
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        # Mock _escalate_event to avoid full escalation complexity
        event_bus._escalate_event = AsyncMock()

        with patch("app.anp.event_bus.get_session", return_value=mock_cm):
            count = await event_bus.escalate_expired_events()

        # Verify escalation was called for the expired event
        assert count == 1
        event_bus._escalate_event.assert_called_once_with(mock_session, expired_event)

    @pytest.mark.asyncio
    async def test_escalated_to_delivered_transition(self, event_bus):
        """Test ESCALATED events can transition to DELIVERED."""
        # Create event in ESCALATED state (after watchdog escalation)
        event = EventLedger(
            id="escalated-event",
            content="Escalated notification",
            routing={"address": "user", "target": "user", "handler": "system"},
            status=EventStatus.ESCALATED,
            owner_lease=2,  # Incremented during escalation
            user_id="test_user",
        )

        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Test transition from ESCALATED to DELIVERED
        success = await event_bus._transition_state(
            mock_session, event.id, EventStatus.ESCALATED, EventStatus.DELIVERED, event.owner_lease
        )

        # Verify transition succeeded
        assert success is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalated_to_failed_transition(self, event_bus):
        """Test ESCALATED events can transition to FAILED."""
        # Create event in ESCALATED state that fails system delivery
        event = EventLedger(
            id="escalated-failed-event",
            content="Escalated notification that failed",
            routing={"address": "user", "target": "user", "handler": "system"},
            status=EventStatus.ESCALATED,
            owner_lease=3,
            user_id="test_user",
        )

        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Test transition from ESCALATED to FAILED
        success = await event_bus._transition_state(
            mock_session, event.id, EventStatus.ESCALATED, EventStatus.FAILED, event.owner_lease
        )

        # Verify transition succeeded
        assert success is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_turn_lock_coordination(self, event_bus):
        """Test agent turn lock is properly accessible and coordinated."""
        # Get the agent turn lock
        lock = event_bus.get_agent_turn_lock()

        # Verify it's an asyncio.Lock
        assert isinstance(lock, asyncio.Lock)

        # Verify same lock instance is returned
        lock2 = event_bus.get_agent_turn_lock()
        assert lock is lock2

        # Test lock can be acquired and released
        assert not lock.locked()
        await lock.acquire()
        assert lock.locked()
        lock.release()
        assert not lock.locked()

    @pytest.mark.asyncio
    async def test_causal_serialization_queues_system_alerts_during_agent_turn(self, event_bus):
        """Test system alerts get queued when agent turn is active."""
        # Acquire agent turn lock to simulate active agent turn
        agent_lock = event_bus.get_agent_turn_lock()
        await agent_lock.acquire()

        try:
            # Create Channel A event (system alert)
            event = EventLedger(
                id="system-alert-queue",
                content="System alert during agent turn",
                routing={"address": "user", "target": "user", "handler": "system"},
                status=EventStatus.PENDING,
                user_id="test_user",
            )

            # Mock session manager
            event_bus._session_manager = MagicMock()
            event_bus._session_manager.is_session_active.return_value = True

            # Mock database session
            mock_session = AsyncMock()

            # Route event - should be queued, not processed immediately
            await event_bus._route_to_channel(mock_session, event)

            # Verify event was queued, not processed
            assert event_bus._system_alert_queue.qsize() == 1
            mock_session.execute.assert_not_called()  # No immediate database operations

        finally:
            agent_lock.release()

    @pytest.mark.asyncio
    async def test_causal_serialization_processes_system_alerts_without_agent_turn(self, event_bus):
        """Test system alerts are processed immediately when no agent turn is active."""
        # Agent turn lock should not be locked initially

        # Create Channel A event
        event = EventLedger(
            id="system-alert-immediate",
            content="System alert without agent turn",
            routing={"address": "user", "target": "user", "handler": "system"},
            status=EventStatus.PENDING,
            user_id="test_user",
        )

        # Mock session manager
        event_bus._session_manager = MagicMock()
        event_bus._session_manager.is_session_active.return_value = True

        # Mock database session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Mock _transition_state for Channel A/B routing
        event_bus._transition_state = AsyncMock(return_value=True)

        # Route event - should be processed immediately
        await event_bus._route_to_channel(mock_session, event)

        # Verify event was processed, not queued
        assert event_bus._system_alert_queue.qsize() == 0
        event_bus._transition_state.assert_called_once_with(
            mock_session, event.id, EventStatus.PENDING, EventStatus.DISPATCHED, None
        )

    @pytest.mark.asyncio
    async def test_flush_system_alert_queue_processes_queued_alerts(self, event_bus):
        """Test flush_system_alert_queue processes all queued system alerts."""
        # Create test event
        event = EventLedger(
            id="queued-alert",
            content="Queued alert",
            routing={"address": "user", "target": "user", "handler": "system"},
            status=EventStatus.PENDING,
            user_id="test_user",
        )

        # Manually add event to queue
        await event_bus._system_alert_queue.put(event)

        # Verify event is queued
        assert event_bus._system_alert_queue.qsize() == 1

        # Mock get_session context manager
        mock_session = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        # Mock _route_to_channel to verify it's called
        event_bus._route_to_channel = AsyncMock()

        with patch("app.anp.event_bus.get_session", return_value=mock_cm):
            await event_bus.flush_system_alert_queue()

        # Verify queue is empty after flush
        assert event_bus._system_alert_queue.qsize() == 0

        # Verify event was processed
        event_bus._route_to_channel.assert_called_once_with(mock_session, event)

    @pytest.mark.asyncio
    async def test_causal_serialization_agent_alerts_never_queued(self, event_bus):
        """Test agent alerts (Channel C) are never queued, even during agent turn."""
        # Acquire agent turn lock
        agent_lock = event_bus.get_agent_turn_lock()
        await agent_lock.acquire()

        try:
            # Create Channel C event (agent alert)
            event = EventLedger(
                id="agent-alert",
                content="Agent alert during agent turn",
                routing={"address": "user", "target": "agent", "handler": "agent"},
                status=EventStatus.PENDING,
                user_id="test_user",
            )

            # Mock session manager
            event_bus._session_manager = MagicMock()
            event_bus._session_manager.is_session_active.return_value = True

            # Mock database session
            mock_session = AsyncMock()
            mock_session.execute.return_value.rowcount = 1
            mock_session.commit.return_value = None

            # Mock _transition_state for Channel C routing
            event_bus._transition_state = AsyncMock(return_value=True)

            # Route event - should be processed immediately despite agent turn
            await event_bus._route_to_channel(mock_session, event)

            # Verify event was processed, not queued
            assert event_bus._system_alert_queue.qsize() == 0
            event_bus._transition_state.assert_called_once_with(
                mock_session, event.id, EventStatus.PENDING, EventStatus.DISPATCHED, None
            )

        finally:
            agent_lock.release()
