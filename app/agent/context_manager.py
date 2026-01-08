"""Context Manager - handles context assembly and system prompt generation."""

import asyncio
import platform
from typing import Any

from app.anp.agent_integration import AgentNotificationInjector
from app.core.config import user_settings
from app.core.context import MMCPContext
from app.core.errors import map_provider_error
from app.core.health import HealthMonitor
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.core.utils.pruner import ContextPruner


class ContextManager:
    """Manages context assembly and system prompt generation for LLM awareness."""

    def __init__(
        self,
        loader: PluginLoader,
        notification_injector: AgentNotificationInjector,
        health: HealthMonitor | None = None,
    ):
        self.loader = loader
        self.notification_injector = notification_injector
        self.health = health

    async def assemble_llm_context(self, user_input: str | None, context: MMCPContext) -> None:
        """
        Assemble LLM context by running all eligible context providers in parallel.

        This is the "ReAct preparation phase" - fetches dynamic state before the loop begins.
        Implements global timeout, per-provider timeouts, and circuit breaker protection.

        Args:
            user_input: The user's query (for eligibility filtering). Can be None for action resumption.
            context: The MMCPContext to update with provider data.
        """
        if not self.loader.context_providers:
            logger.debug("No context providers registered")
            return

        # Filter by eligibility and health
        active_providers = []
        for provider in self.loader.context_providers.values():
            # If health monitor exists, check it; otherwise, assume available
            if self.health and not self.health.is_available(provider.context_key):
                continue
            active_providers.append(provider)

        if not active_providers:
            logger.debug("No eligible context providers available")
            return

        logger.info(
            f"Assembling context from {len(active_providers)} provider(s) "
            f"(global timeout: {user_settings.context_global_timeout_ms}ms)"
        )

        # Run all providers in parallel with global timeout
        # Pass empty string if user_input is None (for action resumption)
        effective_input = user_input if user_input is not None else ""
        tasks = [self._safe_execute_provider(p, effective_input, context) for p in active_providers]

        try:
            global_timeout_seconds = user_settings.context_global_timeout_ms / 1000.0
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=global_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Context assembly timed out after {global_timeout_seconds}s "
                f"(global timeout exceeded)"
            )
            return

        # Update LLM context provider data with successful results
        # Note: _safe_execute_provider wraps all exceptions, so results will be tuples, not Exceptions
        for result in results:
            # Defensive check (shouldn't happen, but safe programming)
            if isinstance(result, Exception):
                logger.error(f"Context provider raised exception: {result}", exc_info=True)
                continue

            provider_key, data = result
            if data is not None:
                context.llm.context_provider_data[provider_key] = data
                logger.debug(f"Updated context_provider_data with data from '{provider_key}'")
            else:
                logger.debug(f"Provider '{provider_key}' returned no data (skipped or failed)")

    async def _safe_execute_provider(
        self, provider, user_input: str | None, context: MMCPContext
    ) -> tuple[str, dict[str, Any] | None]:
        """
        Safely execute a context provider with timeout and error handling.

        Args:
            provider: The MMCPContextProvider instance.
            user_input: The user's query for eligibility checking.

        Returns:
            Tuple of (provider_key, data_dict or None if failed).
        """
        provider_key = provider.context_key

        try:
            # Check eligibility first (async)
            # Handle None user_input gracefully (for action resumption)
            effective_input = user_input if user_input is not None else ""
            if hasattr(provider, "is_eligible_async"):
                eligible = await asyncio.wait_for(
                    provider.is_eligible_async(effective_input),
                    timeout=user_settings.context_per_provider_timeout_ms / 1000.0,
                )
            elif hasattr(provider, "is_eligible"):
                # Check if it's a coroutine function and await if so
                if asyncio.iscoroutinefunction(provider.is_eligible):
                    eligible = await asyncio.wait_for(
                        provider.is_eligible(effective_input),
                        timeout=user_settings.context_per_provider_timeout_ms / 1000.0,
                    )
                else:
                    eligible = provider.is_eligible(effective_input)
            else:
                eligible = True  # Default to eligible if no eligibility check

            if not eligible:
                logger.debug(f"Provider '{provider_key}' not eligible for input: {effective_input}")
                return provider_key, None

            # Execute provider with timeout
            result = await asyncio.wait_for(
                provider.provide_context(),
                timeout=user_settings.context_per_provider_timeout_ms / 1000.0,
            )

            # Extract data from ContextResponse
            data = result.data
            # TTL is available in ContextResponse but not currently used (future enhancement)
            # Could implement caching based on result.ttl if needed

            # Prune data if it's too large
            if data:
                data = ContextPruner.truncate_provider_data(data, provider_key)

            # Record successful execution for health monitoring
            if self.health:
                self.health.record_success(provider_key)

            return provider_key, data

        except Exception as e:
            # Record failed execution for health monitoring
            if self.health:
                self.health.record_failure(provider_key)

            # Map provider errors for consistent logging
            provider_error = map_provider_error(e)
            logger.warning(
                f"Context provider '{provider_key}' failed (will be skipped): {provider_error.message}",
                extra={"trace_id": context.runtime.trace_id, "provider_key": provider_key},
            )
            return provider_key, None



