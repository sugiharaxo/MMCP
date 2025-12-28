import importlib
import inspect
import pkgutil
from pathlib import Path

from pydantic import BaseModel

from app.core.logger import logger
from app.core.plugin_interface import MMCPTool


class PluginLoader:
    """
    Responsible for discovering and instantiating tools from the 'plugins' directory.
    """

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.tools: dict[str, MMCPTool] = {}
        # Reverse mapping: input_schema class -> tool instance
        self._schema_to_tool: dict[type[BaseModel], MMCPTool] = {}

    def load_plugins(self):
        """
        Scans the plugin directory and loads all valid MMCPTool subclasses.
        """
        logger.info(f"Scanning for plugins in: {self.plugin_dir}")

        if not self.plugin_dir.exists():
            logger.warning("Plugin directory not found. Creating it.")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)

        # Walk through the directory packages
        for module_info in pkgutil.iter_modules([str(self.plugin_dir)]):
            if module_info.ispkg:
                # We expect plugins to be folders: plugins/my_tool/
                self._load_single_plugin(module_info.name)

    def _load_single_plugin(self, module_name: str):
        try:
            # Dynamic import: app.plugins.my_tool
            # Note: We assume the folder is in the python path or relative
            full_module_name = f"app.plugins.{module_name}"
            module = importlib.import_module(full_module_name)

            # Inspect the module for classes inheriting from MMCPTool
            found_tool = False
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, MMCPTool) and obj is not MMCPTool:
                    tool_instance = obj()

                    # Check if tool is available before loading
                    if not tool_instance.is_available():
                        logger.warning(
                            f"Skipping Tool: {tool_instance.name} (not available - check configuration)"
                        )
                        continue

                    self.tools[tool_instance.name] = tool_instance

                    # Build reverse mapping: input_schema -> tool instance
                    if hasattr(tool_instance, "input_schema") and tool_instance.input_schema:
                        self._schema_to_tool[tool_instance.input_schema] = tool_instance

                    logger.info(f"Loaded Tool: {tool_instance.name} (v{tool_instance.version})")
                    found_tool = True

            if not found_tool:
                logger.debug(f"No MMCPTool subclass found in {module_name}")

        except Exception as e:
            # Graceful failure: Log the error but keep the server running
            logger.error(f"Failed to load plugin '{module_name}': {e}")

    def get_tool(self, name: str) -> MMCPTool | None:
        return self.tools.get(name)

    def get_tool_by_schema(self, schema_class: type[BaseModel]) -> MMCPTool | None:
        """
        Get tool instance by its input_schema Pydantic model class.

        Used to map LLM tool call responses (which are instances of tool input schemas)
        back to the tool instance that should execute them.

        Args:
            schema_class: The Pydantic model class (e.g., TMDbMetadataInput)

        Returns:
            The MMCPTool instance that uses this schema, or None if not found
        """
        return self._schema_to_tool.get(schema_class)

    def list_tools(self) -> dict[str, str]:
        """Returns a dict of {name: description} for the Agent."""
        return {name: tool.description for name, tool in self.tools.items()}
