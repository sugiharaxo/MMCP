"""
Tests for Lease-Based Fencing.

Tests race condition prevention and atomic state transitions.
"""

from unittest.mock import AsyncMock

import pytest

from app.anp.event_bus import EventBus
from app.anp.models import EventStatus


class TestLeaseFencing:
    """Test lease-based fencing functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus instance."""
        EventBus._instance = None
        return EventBus()

    @pytest.mark.asyncio
    async def test_transition_state_atomic_successful(self, event_bus):
        """Test successful atomic state transitions."""
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Test successful transition without lease check
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.PENDING, EventStatus.DISPATCHED, None
        )

        # Verify transition succeeded
        assert result is True

        # Verify database execute was called
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_state_atomic_with_lease_success(self, event_bus):
        """Test successful atomic state transitions with lease verification."""
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1
        mock_session.commit.return_value = None

        # Test successful transition with lease check
        expected_lease = 5
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.PENDING, EventStatus.DISPATCHED, expected_lease
        )

        # Verify transition succeeded
        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_state_atomic_wrong_current_state(self, event_bus):
        """Test that transitions fail when current state doesn't match."""
        # Mock session - simulate no rows updated (wrong current state)
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 0
        mock_session.commit.return_value = None

        # Test transition with wrong current state
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.PENDING, EventStatus.DISPATCHED, None
        )

        # Verify transition failed
        assert result is False
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_state_atomic_stale_lease(self, event_bus):
        """Test that transitions fail with stale lease."""
        # Mock session - simulate no rows updated (stale lease)
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 0
        mock_session.commit.return_value = None

        # Test transition with stale lease
        expected_lease = 5
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.PENDING, EventStatus.DISPATCHED, expected_lease
        )

        # Verify transition failed
        assert result is False
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_prevent_terminal_state_transition_delivered(self, event_bus):
        """Test that DELIVERED (terminal state) cannot be changed."""
        # Mock session - should not be called for terminal states
        mock_session = AsyncMock()

        # Try to transition from DELIVERED (terminal state)
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.DELIVERED, EventStatus.PENDING, None
        )

        # Should return False immediately without database operations
        assert result is False
        # Verify no database operation was attempted
        mock_session.execute.assert_not_called()
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_prevent_terminal_state_transition_failed(self, event_bus):
        """Test that FAILED (terminal state) cannot be changed."""
        # Mock session - should not be called for terminal states
        mock_session = AsyncMock()

        # Try to transition from FAILED (terminal state)
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.FAILED, EventStatus.PENDING, None
        )

        # Should return False immediately without database operations
        assert result is False
        # Verify no database operation was attempted
        mock_session.execute.assert_not_called()
        mock_session.commit.assert_not_called()
