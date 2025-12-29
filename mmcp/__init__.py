"""
MMCP - Modular Media Control Plane SDK

The official SDK for developing plugins for the Modular Media Control Plane.

Example:
    from mmcp import Tool, ContextProvider, PluginContext

    class MyTool:
        async def execute(self, context: PluginContext, **kwargs):
            return {"result": "Hello from my tool!"}
"""

from app.api import ContextProvider, ContextResponse, PluginContext, PluginStatus, Tool

__all__ = [
    "Tool",
    "ContextProvider",
    "ContextResponse",
    "PluginContext",
    "PluginStatus",
]
