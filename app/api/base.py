"""
MMCP Base Classes - Strict Inheritance Plugin Model

All plugins must inherit from Plugin. Tools and Providers are nested classes
that are automatically discovered via __init_subclass__ magic.
"""

import abc
import logging
from typing import Any

from pydantic import BaseModel as Settings


class Tool(abc.ABC):
    """Base for all Tools. Inheriting provides managed logging and settings."""

    name: str
    description: str
    input_schema: type[Settings]
    version: str = "1.0.0"
    settings_model: type[Settings] | None = None

    def __init__(self, settings: Settings | None, context: Any, plugin_name: str):
        self.settings = settings
        self.context = context
        self.plugin_name = plugin_name  # Store plugin name for settings lookup
        self.logger = logging.getLogger(f"mmcp.{plugin_name}.{self.name}")

    def is_available(self, _settings: Settings | None, _context: Any) -> bool:
        """Check if tool is available. Override in subclasses."""
        return True

    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool's logic.
        Use self.context and self.settings to access managed services.
        """
        raise NotImplementedError


class Provider(abc.ABC):
    """Base for all Context Providers."""

    context_key: str

    def __init__(self, settings: Settings | None):
        self.settings = settings

    async def is_eligible(self, _query: str) -> bool:
        """Check if provider should run for the given query. Override in subclasses."""
        return True

    async def provide_context(self, context: Any) -> Any:
        raise NotImplementedError


class Plugin(abc.ABC):
    """The mandatory base class for all MMCP Plugins."""

    name: str
    version: str
    settings_model: type[Settings] | None = None

    def __init_subclass__(cls):
        # Pattern A: Magic discovery of nested Tool/Provider classes
        cls._tool_classes = [
            v
            for v in cls.__dict__.values()
            if isinstance(v, type) and issubclass(v, Tool) and v is not Tool
        ]
        cls._provider_classes = [
            v
            for v in cls.__dict__.values()
            if isinstance(v, type) and issubclass(v, Provider) and v is not Provider
        ]

    def get_tools(self, settings: Settings | None, context: Any) -> list[Tool]:
        tools = []
        for T in getattr(self, "_tool_classes", []):
            # Validate Tool class has required metadata before instantiation
            required = ["name", "description", "input_schema"]
            missing = [
                attr for attr in required if not hasattr(T, attr) or getattr(T, attr, None) is None
            ]
            if missing:
                logging.getLogger("mmcp").error(
                    f"Tool class {T.__name__} in plugin {self.name} is missing required metadata: {missing}. "
                    f"Skipping tool."
                )
                continue
            try:
                tools.append(T(settings, context, self.name))
            except AttributeError as e:
                logging.getLogger("mmcp").error(
                    f"Failed to instantiate tool {T.__name__} in plugin {self.name}: {e}. "
                    f"Skipping tool."
                )
                continue
        return tools

    def get_providers(self, settings: Settings | None) -> list[Provider]:
        providers = []
        for P in getattr(self, "_provider_classes", []):
            # Validate Provider class has required metadata before instantiation
            if not hasattr(P, "context_key") or getattr(P, "context_key", None) is None:
                logging.getLogger("mmcp").error(
                    f"Provider class {P.__name__} in plugin {self.name} is missing required 'context_key' metadata. "
                    f"Skipping provider."
                )
                continue
            try:
                providers.append(P(settings))
            except AttributeError as e:
                logging.getLogger("mmcp").error(
                    f"Failed to instantiate provider {P.__name__} in plugin {self.name}: {e}. "
                    f"Skipping provider."
                )
                continue
        return providers

    async def is_available(self, _settings: Settings | None, _context: Any) -> bool:
        return True
