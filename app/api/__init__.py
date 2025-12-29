"""
MMCP SDK - Public API for Plugin Development

This package provides the official SDK for developing plugins for the Modular Media Control Plane.
It contains all the interfaces, protocols, and models needed to build MMCP plugins.

Usage:
    from mmcp import Tool, ContextProvider, PluginContext, ContextResponse

For more information, see: https://github.com/your-repo/mmcp/docs/sdk.md
"""

from .interfaces import ContextProvider, Tool
from .schemas import ContextResponse, PluginContext, PluginStatus

__all__ = [
    "Tool",
    "ContextProvider",
    "ContextResponse",
    "PluginContext",
    "PluginStatus",
]
