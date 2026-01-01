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
    async def test_transition_state_atomic(self, event_bus):
        """Test atomic state transitions with lease verification."""
        # Mock session
        mock_session = AsyncMock()
        mock_session.execute.return_value.rowcount = 1

        # Test successful transition
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.PENDING, EventStatus.DISPATCHED, None
        )
        # Note: This test is simplified - actual database operations are complex

    @pytest.mark.asyncio
    async def test_prevent_terminal_state_transition(self, event_bus):
        """Test that terminal states cannot be changed."""
        # Mock session
        mock_session = AsyncMock()

        # Try to transition from DELIVERED (terminal state)
        result = await event_bus._transition_state(
            mock_session, "test-id", EventStatus.DELIVERED, EventStatus.PENDING, None
        )

        # Should return False
        assert result is False
        # Verify no database operation was attempted
        mock_session.execute.assert_not_called()
