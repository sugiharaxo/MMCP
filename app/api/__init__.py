"""
MMCP SDK - Public API for Plugin Development

This package provides the official SDK for developing plugins for the Modular Media Control Plane.
It contains all the base classes and models needed to build MMCP plugins.

Usage:
    from mmcp import Plugin, Tool, ContextProvider, PluginRuntime, ContextResponse

For more information, see: https://github.com/your-repo/mmcp/docs/sdk.md
"""

from .base import ContextProvider, Plugin, Tool
from .schemas import ContextResponse, PluginRuntime, PluginStatus

__all__ = [
    "Plugin",
    "Tool",
    "ContextProvider",
    "ContextResponse",
    "PluginRuntime",
    "PluginStatus",
]
