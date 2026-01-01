"""
MMCP Plugin Schemas - Public Pydantic Models

This module contains all the Pydantic models that plugins use for I/O.
These are lightweight schemas that define the contract between plugins and core.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.anp.schemas import NotificationCreate
from app.core.config import CoreSettings


class ContextResponse(BaseModel):
    """
    Response model for context providers.

    Handles TTL and metadata internally while keeping the LLM's data clean.
    """

    data: dict[str, Any] = Field(description="The context data to inject into LLM media_state")
    ttl: int = Field(default=300, description="Time-to-live in seconds (default: 5 minutes)")
    provider_name: str = Field(description="Name of the provider for logging and health tracking")


class PluginStatus(BaseModel):
    """
    Standardized status report for any MMCP plugin.

    Ensures all plugins report status using the same structure, enabling
    consistent UI/API responses and LLM visibility into plugin health.

    Uses Pydantic's from_attributes=True to automatically map Tool Protocol
    properties (name, version, description) to this model, eliminating
    the need for plugins to implement get_status_info().
    """

    model_config = ConfigDict(from_attributes=True)

    service_name: str = Field(
        alias="name",
        serialization_alias="service_name",
        description="Human-readable name of the service/plugin",
    )
    is_available: bool = Field(description="Whether the plugin is currently available")
    version: str = Field(description="Plugin version")
    description: str = Field(description="Description of the plugin")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Plugin-specific additional status data"
    )


class PluginRuntime(BaseModel):
    """
    Safe facade for plugin access to system runtime state.

    This replaces passing the full PluginLoader to plugins. It provides only
    the subset of functionality that plugins are allowed to use, preventing
    plugins from loading/unloading other plugins or accessing internal state.

    Philosophy: Sandboxed access - plugins get what they need, nothing more.
    """

    # Safe paths configuration (read-only subset of settings)
    paths: CoreSettings

    # System information (read-only)
    system: dict[str, Any]

    async def emit_notification(self, notification: "NotificationCreate") -> str:
        """
        Emit a notification via ANP Event Bus.

        Plugins can call this to send notifications to users or agents.
        Access via self.system.emit_notification() in tools/providers.

        Args:
            notification: NotificationCreate schema

        Returns:
            Notification ID (UUID string)
        """
        from app.anp.event_bus import EventBus

        event_bus = EventBus()
        return await event_bus.emit_notification(notification)
