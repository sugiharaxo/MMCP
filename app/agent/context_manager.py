"""Context Manager - handles context assembly and system prompt generation."""

import asyncio
from typing import Any

from app.anp.agent_integration import AgentNotificationInjector
from app.core.config import settings
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

    async def assemble_llm_context(self, user_input: str, context: MMCPContext) -> None:
        """
        Assemble LLM context by running all eligible context providers in parallel.

        This is the "ReAct preparation phase" - fetches dynamic state before the loop begins.
        Implements global timeout, per-provider timeouts, and circuit breaker protection.

        Args:
            user_input: The user's query (for eligibility filtering).
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
            f"(global timeout: {settings.context_global_timeout_ms}ms)"
        )

        # Run all providers in parallel with global timeout
        tasks = [self._safe_execute_provider(p, user_input, context) for p in active_providers]

        try:
            global_timeout_seconds = settings.context_global_timeout_ms / 1000.0
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
        self, provider, user_input: str, context: MMCPContext
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
            if hasattr(provider, "is_eligible_async"):
                eligible = await asyncio.wait_for(
                    provider.is_eligible_async(user_input),
                    timeout=settings.context_per_provider_timeout_ms / 1000.0,
                )
            elif hasattr(provider, "is_eligible"):
                # Check if it's a coroutine function and await if so
                if asyncio.iscoroutinefunction(provider.is_eligible):
                    eligible = await asyncio.wait_for(
                        provider.is_eligible(user_input),
                        timeout=settings.context_per_provider_timeout_ms / 1000.0,
                    )
                else:
                    eligible = provider.is_eligible(user_input)
            else:
                eligible = True  # Default to eligible if no eligibility check

            if not eligible:
                logger.debug(f"Provider '{provider_key}' not eligible for input: {user_input}")
                return provider_key, None

            # Execute provider with timeout
            result = await asyncio.wait_for(
                provider.provide_context(context),
                timeout=settings.context_per_provider_timeout_ms / 1000.0,
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

    async def get_system_prompt(self, context: MMCPContext | None = None) -> str:
        """
        Generate system prompt with dynamic tool descriptions and context-aware state.

        Simplified prompt focusing on "when to use tools" rather than formatting rules.
        The Union response_model handles the "how" of formatting automatically.

        Injects dynamic state from context (user preferences, media state) to make
        the LLM aware of current server state.

        Also injects relevant context for agent awareness.
        """
        if not self.loader:
            tool_desc = "No tools are currently available."
        else:
            tool_descriptions = []
            for idx, (name, tool) in enumerate(self.loader.tools.items(), start=1):
                desc = f"{idx}. {name}: {tool.description}"

                # Add clean argument descriptions instead of raw JSON schema
                if hasattr(tool, "input_schema"):
                    schema = tool.input_schema
                    schema_json = schema.model_json_schema()
                    properties = schema_json.get("properties", {})
                    required = schema_json.get("required", [])

                    if properties:
                        arg_list = []
                        for prop_name, prop_info in properties.items():
                            prop_type = prop_info.get("type", "any")
                            is_required = prop_name in required
                            prop_desc = prop_info.get("description", "")
                            default = prop_info.get("default", None)

                            if is_required:
                                arg_list.append(
                                    f"  - {prop_name} ({prop_type}): {prop_desc} [REQUIRED]"
                                )
                            else:
                                default_str = f", default: {default}" if default is not None else ""
                                arg_list.append(
                                    f"  - {prop_name} ({prop_type}): {prop_desc} [optional{default_str}]"
                                )

                        if arg_list:
                            desc += "\n   Args:\n" + "\n".join(arg_list)

                tool_descriptions.append(desc)

            tool_desc = "\n".join(tool_descriptions)

        # Get the sanitized LLM payload for context-aware prompts
        context_section = ""
        if context:
            state = context.get_llm_payload()
            # Build human-readable state description
            state_parts = []
            if state.get("user_preferences"):
                state_parts.append(f"User Preferences: {state['user_preferences']}")
            if state.get("media_state"):
                state_parts.append(f"Media State: {state['media_state']}")
            if state_parts:
                state_desc = "\n".join(state_parts)
                context_section = f"CONTEXT:\n{state_desc}"

        # Add standby alerts for system awareness
        standby_alerts = self._get_standby_alerts()

        # Inject notifications (pending + recent user ACKs)
        user_id = "default"  # Single-user system for now
        pending = await self.notification_injector.get_pending_notifications(user_id)
        recent_acks = await self.notification_injector.get_recent_user_acks(user_id)
        notification_section = self.notification_injector.format_for_system_prompt(
            pending, recent_acks
        )

        return f"""You are MMCP (Modular Media Control Plane), an intelligent media assistant.
You help users manage their media library, search for metadata, and handle downloads.

IDENTITY:
- Be concise and helpful.
- Use tools to fetch data before making recommendations.
- If a tool fails, explain why to the user and try a different approach.

{context_section}

{standby_alerts}

{notification_section}

AVAILABLE TOOLS:
{tool_desc}

Use tools when you need specific information or actions. When you have enough information to answer the user, provide a FinalResponse with your answer."""

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
