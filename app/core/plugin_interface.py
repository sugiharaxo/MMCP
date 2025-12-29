from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.core.plugin_loader import PluginLoader

# Export PluginLoader for plugin developers to use in type hints
__all__ = ["ContextResponse", "PluginLoader"]


class ContextResponse(BaseModel):
    """
    Response model for context providers.

    Handles TTL and metadata internally while keeping the LLM's data clean.
    """

    data: dict[str, Any] = Field(description="The context data to inject into LLM media_state")
    ttl: int = Field(default=300, description="Time-to-live in seconds (default: 5 minutes)")
    provider_name: str = Field(description="Name of the provider for logging and health tracking")
