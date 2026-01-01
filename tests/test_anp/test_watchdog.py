"""
Tests for Watchdog Service.

Tests TTL monitoring and expired event escalation.
"""

from unittest.mock import AsyncMock

import pytest

from app.anp.event_bus import EventBus
from app.anp.watchdog import WatchdogService


class TestWatchdogService:
    """Test WatchdogService functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh EventBus instance."""
        EventBus._instance = None
        bus = EventBus()
        bus.escalate_expired_events = AsyncMock(return_value=5)
        return bus

    @pytest.fixture
    def watchdog(self, event_bus):
        """Create a WatchdogService instance."""
        return WatchdogService(event_bus)

    @pytest.mark.asyncio
    async def test_check_expired_events(self, watchdog):
        """Test checking for expired events."""
        await watchdog.check_expired_events()

        # Verify escalate_expired_events was called
        watchdog.event_bus.escalate_expired_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_stop(self, watchdog):
        """Test starting and stopping the watchdog service."""
        # Test start
        await watchdog.start(interval_seconds=1)
        assert watchdog._running is True
        assert watchdog.scheduler is not None

        # Test stop
        await watchdog.stop()
        assert watchdog._running is False
        assert watchdog.scheduler is None
