"""
MMCP Plugin Interfaces - Public Protocol Definitions

This module defines the public interfaces for MMCP plugins using typing.Protocols.
Protocols allow structural typing - plugins don't need to inherit from internal classes,
they just need to match the expected interface shape.

This prevents internal implementation details from leaking through inheritance.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .schemas import ContextResponse, PluginRuntime


@runtime_checkable
class Tool(Protocol):
    """
    Protocol for MMCP Tools.

    Tools are the core execution units that the LLM can call to perform actions.
    They must have strict typing via Pydantic for the Agent loop.

    Philosophy:
    - strict typing via Pydantic (for the Agent)
    - strict return types (for the API)
    - async execution (for performance)
    """

    @property
    def name(self) -> str:
        """
        The internal name of the tool (e.g., 'search_youtube').
        Must be unique across all plugins.
        """
        ...

    @property
    def description(self) -> str:
        """
        A clear, concise description of what this tool does.
        The LLM reads this to decide if it should call this tool.
        """
        ...

    @property
    def version(self) -> str:
        """
        The version of this tool (e.g., '1.0.0').
        Used for status reporting and API responses.
        """
        ...

    @property
    def input_schema(self) -> type[BaseModel]:
        """
        The Pydantic model defining the arguments this tool accepts.
        """
        ...

    @property
    def settings_model(self) -> type[BaseModel] | None:
        """
        Optional Pydantic BaseModel defining this tool's configuration requirements.

        **Note:** Settings are defined at the plugin level, not the tool level. All tools
        within a plugin share the same settings instance. If the parent plugin has a
        `settings_model`, tools will receive those settings via `self.settings`. If None,
        the tool has no configuration requirements and the settings parameter will be None
        in `is_available()` and `execute()`.

        The loader discovers and validates settings from namespaced environment variables
        using the pattern: `MMCP_PLUGIN_{PLUGIN_SLUG}_{FIELD_NAME}`.

        Example:
            class MySettings(BaseModel):
                api_key: SecretStr
                language: str = "en-US"

            class MyPlugin(Plugin):
                settings_model = MySettings  # Define at plugin level

            class MyTool(Tool):
                # Tools automatically inherit settings from parent plugin
                ...
        """
        ...

    def is_available(self, settings: BaseModel | None, runtime: PluginRuntime) -> bool:
        """
        Check if this tool is available given its settings and system context.

        Called in Phase 2 after all plugins are discovered, ensuring complete context.
        Can check both private settings (e.g., API keys) and global context (e.g., other tools).

        Args:
            settings: Validated settings instance (BaseModel) for this tool, or None if no settings_model defined
            runtime: Complete PluginRuntime with access to all system state

        Returns:
            True if the tool can be used, False otherwise.
        """
        ...

    async def execute(
        self, runtime: PluginRuntime, settings: BaseModel | None, **kwargs: Any
    ) -> Any:
        """
        The actual logic.

        Philosophy: Tools never fetch context themselves — context is injected.

        Args:
            runtime: PluginRuntime with safe access to system state
            settings: Validated settings instance (BaseModel) for this tool, or None if no settings_model defined
            **kwargs: valid data matching input_schema.

        Returns:
            A string, dict, or Pydantic model containing the result.
        """
        ...


@runtime_checkable
class ContextProvider(Protocol):
    """
    Protocol for Context Providers.

    Context providers fetch dynamic state (e.g., Jellyfin library status, Plex server info)
    that should be available to the LLM before the ReAct loop begins.

    Philosophy:
    - Lightweight eligibility checks (no heavy I/O)
    - Async execution with timeouts
    - Circuit breaker protection via health monitor
    - Truncation to prevent context bloat
    """

    @property
    def context_key(self) -> str:
        """
        The key in the JSON object (e.g., 'jellyfin', 'plex').

        This key will be used in context.llm.media_state[context_key] = data.
        Must be unique across all context providers.
        """
        ...

    async def provide_context(self) -> ContextResponse:
        """
        Fetch the dynamic context data.

        Returns:
            ContextResponse with context data (will be truncated if too large).
        """
        ...

    async def is_eligible(self, _query: str) -> bool:
        """
        Lightweight check if this provider should run for the given query.

        Override this to implement query-based filtering (e.g., only run
        Jellyfin provider if query mentions "jellyfin" or "library").

        Args:
            query: The user's input query.

        Returns:
            True if this provider should execute, False otherwise.
        """
        return True
