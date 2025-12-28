"""
Health Monitor - Circuit Breaker for Context Providers.

Lightweight in-memory health tracking. No database writes for health checks.
Tracks provider state and implements circuit breaker pattern.
"""

import time
from enum import Enum
from typing import Any

from app.core.config import settings
from app.core.logger import logger


class ProviderState(str, Enum):
    """State of a context provider."""

    AVAILABLE = "available"
    BLOCKED = "blocked"


class ProviderHealth:
    """Health state for a single provider."""

    def __init__(self):
        self.state: ProviderState = ProviderState.AVAILABLE
        self.failure_count: int = 0
        self.last_failure_time: float | None = None

    def record_success(self):
        """Reset failure tracking on successful execution."""
        self.state = ProviderState.AVAILABLE
        self.failure_count = 0
        self.last_failure_time = None

    def record_failure(self):
        """Record a failure and update state."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= settings.context_failure_threshold:
            self.state = ProviderState.BLOCKED
            logger.warning(
                f"Context provider circuit breaker tripped after {self.failure_count} failures. "
                f"Will retry after {settings.context_recovery_wait_minutes} minutes."
            )

    def is_available(self) -> bool:
        """
        Check if provider is available (not blocked or recovery time has passed).

        Returns:
            True if provider can be called, False if blocked.
        """
        if self.state == ProviderState.AVAILABLE:
            return True

        # Check if recovery time has passed
        if self.last_failure_time is None:
            return True

        recovery_seconds = settings.context_recovery_wait_minutes * 60
        time_since_failure = time.time() - self.last_failure_time

        if time_since_failure >= recovery_seconds:
            # Recovery period passed - reset to available
            logger.info("Context provider recovery period passed. Resetting to available state.")
            self.record_success()
            return True

        return False


class HealthMonitor:
    """
    Circuit breaker for context providers.

    Tracks health state of each provider in memory.
    Implements failure threshold and recovery wait logic.
    """

    def __init__(self):
        self._providers: dict[str, ProviderHealth] = {}

    def _get_provider_health(self, provider_key: str) -> ProviderHealth:
        """Get or create health state for a provider."""
        if provider_key not in self._providers:
            self._providers[provider_key] = ProviderHealth()
        return self._providers[provider_key]

    def is_available(self, provider_key: str) -> bool:
        """
        Check if a provider is available (not circuit-broken).

        Args:
            provider_key: The context_key of the provider.

        Returns:
            True if provider can be called, False if blocked.
        """
        health = self._get_provider_health(provider_key)
        return health.is_available()

    def record_success(self, provider_key: str):
        """
        Record a successful execution.

        Args:
            provider_key: The context_key of the provider.
        """
        health = self._get_provider_health(provider_key)
        health.record_success()

    def record_failure(self, provider_key: str):
        """
        Record a failed execution.

        Args:
            provider_key: The context_key of the provider.
        """
        health = self._get_provider_health(provider_key)
        health.record_failure()

    def get_status(self) -> dict[str, Any]:
        """
        Get health status of all providers.

        Returns:
            Dictionary mapping provider_key to health status.
        """
        return {
            key: {
                "state": health.state.value,
                "failure_count": health.failure_count,
                "last_failure_time": health.last_failure_time,
                "is_available": health.is_available(),
            }
            for key, health in self._providers.items()
        }
