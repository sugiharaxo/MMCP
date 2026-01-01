"""
Watchdog Service for ANP TTL monitoring.

Uses APScheduler to monitor expired events and trigger escalation.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.logger import logger

if True:  # TYPE_CHECKING
    from app.anp.event_bus import EventBus


class WatchdogService:
    """
    Watchdog Service for TTL monitoring and escalation.

    Uses APScheduler (in-process memory store) to periodically check
    for expired events and escalate them.
    """

    def __init__(self, event_bus: "EventBus"):
        """
        Initialize WatchdogService.

        Args:
            event_bus: EventBus instance
        """
        self.event_bus = event_bus
        self.scheduler: AsyncIOScheduler | None = None
        self._running = False

    async def start(self, interval_seconds: int = 10) -> None:
        """
        Start the watchdog scheduler.

        Args:
            interval_seconds: Check interval in seconds (default: 10)
        """
        if self._running:
            logger.warning("WatchdogService already running")
            return

        self.scheduler = AsyncIOScheduler()
        self.scheduler.add_job(
            self.check_expired_events,
            trigger=IntervalTrigger(seconds=interval_seconds),
            id="anp_watchdog",
            name="ANP Watchdog - Check Expired Events",
        )
        self.scheduler.start()
        self._running = True
        logger.info(f"WatchdogService started (interval: {interval_seconds}s)")

    async def stop(self) -> None:
        """Stop the watchdog scheduler."""
        if not self._running:
            return

        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            self.scheduler = None

        self._running = False
        logger.info("WatchdogService stopped")

    async def check_expired_events(self) -> None:
        """
        Check for expired events and escalate them.

        Called periodically by APScheduler.
        """
        try:
            count = await self.event_bus.escalate_expired_events()
            if count > 0:
                logger.info(f"Watchdog escalated {count} expired event(s)")
        except Exception as e:
            logger.error(f"Watchdog check failed: {e}", exc_info=True)
