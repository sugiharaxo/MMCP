"""
MMCP Base Classes - Strict Inheritance Plugin Model

All plugins must inherit from Plugin. Tools and Providers are nested classes
that are automatically discovered via __init_subclass__ magic.
"""

import logging
from typing import Any

from pydantic import BaseModel as Settings


class Tool:
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


class Provider:
    """Base for all Context Providers."""

    context_key: str

    def __init__(self, settings: Settings | None):
        self.settings = settings

    async def is_eligible(self, _query: str) -> bool:
        """Check if provider should run for the given query. Override in subclasses."""
        return True

    async def provide_context(self, context: Any) -> Any:
        raise NotImplementedError


class Plugin:
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
        return [T(settings, context, self.name) for T in getattr(self, "_tool_classes", [])]

    def get_providers(self, settings: Settings | None) -> list[Provider]:
        return [P(settings) for P in getattr(self, "_provider_classes", [])]

    async def is_available(self, _settings: Settings | None, _context: Any) -> bool:
        return True
