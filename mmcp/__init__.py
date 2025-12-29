"""
MMCP - Modular Media Control Plane SDK

The official SDK for developing plugins for the Modular Media Control Plane.

Example:
    from mmcp import Plugin, Tool, Provider, PluginContext

    class MyPlugin(Plugin):
        name = "my_plugin"
        version = "1.0.0"

        class MyTool(Tool):
            name = "my_tool"
            description = "Does something"
            input_schema = MyInputSchema

            async def execute(self, context, settings, **kwargs):
                return {"result": "Hello from my tool!"}
"""

from app.api import ContextResponse, Plugin, PluginContext, PluginStatus, Provider, Tool

__all__ = [
    "Plugin",
    "Tool",
    "Provider",
    "ContextResponse",
    "PluginContext",
    "PluginStatus",
]
