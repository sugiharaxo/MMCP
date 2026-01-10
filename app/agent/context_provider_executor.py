"""Context Provider Executor - handles safe execution of context providers."""

import asyncio
from collections.abc import Coroutine
from typing import Any, Protocol, runtime_checkable

from app.core.config import user_settings
from app.core.context import MMCPContext
from app.core.errors import map_provider_error
from app.core.health import HealthMonitor
from app.core.logger import logger
from app.core.utils.pruner import ContextPruner


@runtime_checkable
class MMCPContextProvider(Protocol):
    context_key: str

    async def provide_context(self) -> Any: ...


class ContextProviderExecutor:
    """Executes context providers with timeout, error handling, and health monitoring."""

    def __init__(self, health_monitor: HealthMonitor | None = None):
        self.health = health_monitor

    async def execute_provider(
        self, provider: MMCPContextProvider, user_input: str | None, context: MMCPContext
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Safely execute a context provider with timeout and error handling.

        Args:
            provider: The MMCPContextProvider instance.
            user_input: The user's query for eligibility checking.
            context: The MMCPContext for trace logging.

        Returns:
            Tuple of (provider_key, data_dict or None if failed).
        """
        provider_key = provider.context_key
        # Normalize input once at the entry point
        effective_input = user_input or ""

        try:
            # 1. Resolve Eligibility
            eligible_coro = self._get_eligibility_coro(provider, effective_input)
            eligible = await asyncio.wait_for(
                eligible_coro, timeout=user_settings.context_per_provider_timeout_ms / 1000.0
            )

            if not eligible:
                logger.debug(f"Provider '{provider_key}' not eligible for input.")
                return provider_key, None

            # 2. Execute Provider
            result = await asyncio.wait_for(
                provider.provide_context(),
                timeout=user_settings.context_per_provider_timeout_ms / 1000.0,
            )

            data = (
                ContextPruner.truncate_provider_data(result.data, provider_key)
                if result.data
                else None
            )

            if self.health:
                self.health.record_success(provider_key)

            return provider_key, data

        except Exception as e:
            if self.health:
                self.health.record_failure(provider_key)

            provider_error = map_provider_error(e)
            logger.warning(
                f"Context provider '{provider_key}' failed: {provider_error.message}",
                extra={"trace_id": context.runtime.trace_id, "provider_key": provider_key},
            )
            return provider_key, None

    def _get_eligibility_coro(self, provider: Any, input_str: str) -> Coroutine:
        """Handles the duck-typing for sync/async eligibility checks."""
        if hasattr(provider, "is_eligible_async"):
            return provider.is_eligible_async(input_str)

        if hasattr(provider, "is_eligible"):
            res = provider.is_eligible(input_str)
            if asyncio.iscoroutine(res):
                return res

            # Wrap sync result in a coroutine
            async def _sync_wrapper():
                return res

            return _sync_wrapper()

        async def _default_true():
            return True

        return _default_true()
