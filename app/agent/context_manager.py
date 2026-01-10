"""Context Manager - handles context assembly and system prompt generation."""

import asyncio

from app.agent.context_provider_executor import ContextProviderExecutor
from app.anp.agent_integration import AgentNotificationInjector
from app.core.config import user_settings
from app.core.context import MMCPContext
from app.core.health import HealthMonitor
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader


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
        self.executor = ContextProviderExecutor(health)

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
        # Executor handles None -> "" conversion; passing raw user_input is cleaner
        tasks = [self.executor.execute_provider(p, user_input, context) for p in active_providers]

        try:
            global_timeout_seconds = user_settings.context_global_timeout_ms / 1000.0
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=global_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                f"Context assembly timed out after {global_timeout_seconds}s "
                f"(global timeout exceeded)"
            )
            return

        # Update LLM context provider data with successful results
        # Note: executor.execute_provider wraps all exceptions, so results will be tuples, not Exceptions
        for result in results:
            # executor.execute_provider always returns tuple[str, dict | None], never Exception
            # This check is defensive but should never trigger if execute_provider is correct
            if not isinstance(result, tuple) or len(result) != 2:
                logger.error(
                    f"Critical System Error: Unexpected result type {type(result)}", exc_info=True
                )
                continue

            provider_key, data = result
            if data is not None:
                context.llm.context_provider_data[provider_key] = data
                logger.debug(f"Updated context_provider_data with data from '{provider_key}'")
            else:
                logger.debug(f"Provider '{provider_key}' returned no data (skipped or failed)")
