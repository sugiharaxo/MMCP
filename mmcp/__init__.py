"""
MMCP - Modular Media Control Plane SDK

The official SDK for developing plugins for the Modular Media Control Plane.

Example:
    from mmcp import Plugin, Tool, ContextProvider, PluginRuntime
    from pydantic import BaseModel

    class MyInputSchema(BaseModel):
        message: str

    class MyPlugin(Plugin):
        name = "my_plugin"
        version = "1.0.0"

        class MyTool(Tool):
            name = "my_tool"
            description = "Does something"
            input_schema = MyInputSchema

            async def execute(self, **kwargs):
                return {"result": "Hello from my tool!"}
"""

from app.api import ContextProvider, ContextResponse, Plugin, PluginRuntime, PluginStatus, Tool

__all__ = [
    "Plugin",
    "Tool",
    "ContextProvider",
    "ContextResponse",
    "PluginRuntime",
    "PluginStatus",
]
