"""
MMCP Base Classes - Strict Inheritance Plugin Model

All plugins must inherit from Plugin. Tools and Providers are nested classes
that are automatically discovered via __init_subclass__ magic.
"""

import abc
import logging
from typing import Any, ClassVar

from pydantic import BaseModel as Settings


class Tool(abc.ABC):
    """Base for all Tools. Inheriting provides managed logging and settings."""

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[Settings]]
    version: ClassVar[str] = "1.0.0"
    settings_model: ClassVar[type[Settings] | None] = None

    def __init_subclass__(cls, **kwargs):
        """
        Primary validation gate for the MMCP Tool Protocol.

        Validates required class metadata (name, description, input_schema) at class
        definition time, ensuring tools are structurally correct before they enter
        the plugin registry. Raises TypeError if validation fails.
        """
        super().__init_subclass__(**kwargs)
        # Skip validation for the abstract base itself
        if cls.__name__ == "Tool":
            return

        # Simple, fast validation at definition time
        errors = []
        if not isinstance(getattr(cls, "name", None), str):
            errors.append("name (str)")
        if not isinstance(getattr(cls, "description", None), str):
            errors.append("description (str)")
        input_schema = getattr(cls, "input_schema", None)
        if not (isinstance(input_schema, type) and issubclass(input_schema, Settings)):
            errors.append("input_schema (BaseModel subclass)")

        if errors:
            raise TypeError(
                f"Tool '{cls.__name__}' missing or invalid metadata: {', '.join(errors)}"
            )

    def __init__(self, settings: Settings | None, runtime: Any, plugin_name: str):
        self.settings = settings
        # Extract paths and system from PluginRuntime for flat API
        self.paths = runtime.paths
        self.system = runtime.system
        self.plugin_name = plugin_name  # Store plugin name for settings lookup
        self.logger = logging.getLogger(f"mmcp.{plugin_name}.{type(self).name}")

    def is_available(self, _settings: Settings | None, _context: Any) -> bool:
        """Check if tool is available. Override in subclasses."""
        return True

    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool's logic.
        Use self.paths, self.system, and self.settings to access managed resources and configuration.
        """
        raise NotImplementedError


class ContextProvider(abc.ABC):
    """Base for all Context Providers."""

    context_key: ClassVar[str]

    def __init_subclass__(cls, **kwargs):
        """
        Primary validation gate for the MMCP Provider Protocol.

        Validates required class metadata (context_key) at class definition time,
        ensuring providers are structurally correct before they enter the plugin
        registry. Raises TypeError if validation fails.
        """
        super().__init_subclass__(**kwargs)
        # Skip validation for the abstract base itself
        if cls.__name__ == "Provider":
            return

        # Simple, fast validation at definition time
        if not isinstance(getattr(cls, "context_key", None), str):
            raise TypeError(
                f"Provider '{cls.__name__}' missing or invalid metadata: context_key (str)"
            )

    def __init__(self, settings: Settings | None, runtime: Any, plugin_name: str):
        self.settings = settings
        self.paths = runtime.paths
        self.system = runtime.system
        self.plugin_name = plugin_name
        self.logger = logging.getLogger(f"mmcp.{plugin_name}.{type(self).context_key}")

    async def is_eligible(self, _query: str) -> bool:
        """Check if provider should run for the given query. Override in subclasses."""
        return True

    async def provide_context(self) -> Any:
        raise NotImplementedError


class Plugin(abc.ABC):
    """The mandatory base class for all MMCP Plugins."""

    name: ClassVar[str]
    version: ClassVar[str]
    settings_model: ClassVar[type[Settings] | None] = None

    def __init_subclass__(cls, **kwargs):
        """
        Primary validation gate for the MMCP Plugin Protocol.

        Validates required class metadata (name, version) at class definition time,
        ensuring plugins are structurally correct before they enter the plugin registry.
        Also discovers nested Tool and Provider classes for automatic registration.
        Raises TypeError if validation fails.
        """
        super().__init_subclass__(**kwargs)
        # Skip validation for the abstract base itself
        if cls.__name__ == "Plugin":
            return

        # Simple, fast validation at definition time
        errors = []
        if not isinstance(getattr(cls, "name", None), str):
            errors.append("name (str)")
        if not isinstance(getattr(cls, "version", None), str):
            errors.append("version (str)")

        if errors:
            raise TypeError(
                f"Plugin '{cls.__name__}' missing or invalid metadata: {', '.join(errors)}"
            )

        # Pattern A: Magic discovery of nested Tool/Provider classes
        cls._tool_classes = [
            v
            for v in cls.__dict__.values()
            if isinstance(v, type) and issubclass(v, Tool) and v is not Tool
        ]
        cls._provider_classes = [
            v
            for v in cls.__dict__.values()
            if isinstance(v, type) and issubclass(v, ContextProvider) and v is not ContextProvider
        ]

    def get_tools(self, settings: Settings | None, context: Any) -> list[Tool]:
        tools = []
        for T in getattr(self, "_tool_classes", []):
            # Validation happens at definition time via __init_subclass__
            try:
                tools.append(T(settings, context, self.name))
            except Exception as e:
                logging.getLogger("mmcp").error(
                    f"Failed to instantiate tool {T.__name__} in plugin {self.name}: {e}. "
                    f"Skipping tool."
                )
                continue
        return tools

    def get_providers(self, settings: Settings | None, runtime: Any) -> list[ContextProvider]:
        providers = []
        for P in getattr(self, "_provider_classes", []):
            # Validation happens at definition time via __init_subclass__
            try:
                providers.append(P(settings, runtime, self.name))
            except Exception as e:
                logging.getLogger("mmcp").error(
                    f"Failed to instantiate provider {P.__name__} in plugin {self.name}: {e}. "
                    f"Skipping provider."
                )
                continue
        return providers

    async def is_available(self, _settings: Settings | None, _context: Any) -> bool:
        return True
