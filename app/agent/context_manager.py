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

        # Update LLM media state with successful results
        # Note: _safe_execute_provider wraps all exceptions, so results will be tuples, not Exceptions
        for result in results:
            # Defensive check (shouldn't happen, but safe programming)
            if isinstance(result, Exception):
                logger.error(f"Context provider raised exception: {result}", exc_info=True)
                continue

            provider_key, data = result
            if data is not None:
                context.llm.media_state[provider_key] = data
                logger.debug(f"Updated media_state with data from '{provider_key}'")
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

    async def get_static_instructions(self) -> str:
        """
        Generate static system instructions that never change during a session.

        This includes identity, global rules, and system configuration.
        Instructor will automatically append the tool schema to this message,
        making it cache-safe for KV cache optimization.

        Returns:
            Static prompt string that remains constant throughout the session.
        """
        host_os = platform.system()
        return f"""You are MMCP (Modular Media Control Plane), an intelligent media assistant.
You help users manage their media library, search for metadata, and handle downloads.

IDENTITY:
- Be concise and helpful.
- Use tools to fetch data before making recommendations.
- If a tool fails, explain why to the user and try a different approach.

SYSTEM:
- OS: {host_os}
- Use tools when you need specific information or actions.
- When you have enough information to answer the user, provide a FinalResponse with your answer.

(Instructor will append the available tool schemas here automatically.)"""

    async def get_dynamic_state(self, context: MMCPContext | None = None) -> str:
        """
        Generate dynamic state that changes every turn.

        This includes current time, media state, notifications, and standby alerts.
        This content is sent as a separate system message to allow KV cache
        persistence of the static instructions.

        Args:
            context: MMCPContext with current state

        Returns:
            Dynamic state string that changes each turn.
        """
        from datetime import datetime, timezone

        # Get current time
        current_time = datetime.now(timezone.utc).isoformat()

        # Get the sanitized LLM payload for context-aware prompts
        state_parts = [f"TIME: {current_time}"]

        if context:
            state = context.get_llm_payload()
            if state.get("user_preferences"):
                state_parts.append(f"User Preferences: {state['user_preferences']}")
            if state.get("media_state"):
                state_parts.append(f"Media State: {state['media_state']}")

        # Add standby alerts for system awareness
        standby_alerts = self._get_standby_alerts()
        if standby_alerts:
            state_parts.append(standby_alerts)

        # Inject notifications (pending + recent user ACKs)
        user_id = "default"  # Single-user system for now
        pending = await self.notification_injector.get_pending_notifications(user_id)
        recent_acks = await self.notification_injector.get_recent_user_acks(user_id)
        notification_section = self.notification_injector.format_for_system_prompt(
            pending, recent_acks
        )
        if notification_section:
            state_parts.append(notification_section)

        return "\n".join(state_parts)

    def _get_standby_alerts(self) -> str:
        """
        Generate standby alerts for system awareness.

        Only reports plugins that FAILED to load, providing the 'Proprioception'.
        This keeps the context lean while enabling self-resolution.
        """
        standby = self.loader.standby_tools
        if not standby:
            return ""

        alerts = ["DISABLED CAPABILITIES (SYSTEM ERRORS):"]
        for name, tool in standby.items():
            # Get plugin name for error lookup (settings are stored at plugin level)
            plugin_name = tool.plugin_name
            reason = self.loader._plugin_config_errors.get(plugin_name, "Unknown system error")
            alerts.append(f"- {name}: {reason}")

        return "\n".join(alerts)
