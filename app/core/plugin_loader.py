import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from app.api.schemas import PluginContext, PluginStatus
from app.core.config import settings
from app.api.schemas import PluginContext, PluginStatus
from app.core.config import settings
from app.core.logger import logger
from mmcp import ContextProvider, Tool

if TYPE_CHECKING:
    pass
from mmcp import ContextProvider, Tool

if TYPE_CHECKING:
    pass


class PluginLoader:
    """
    Responsible for discovering and instantiating tools from the 'plugins' directory.
    """

    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.tools: dict[str, Tool] = {}
        self.tools: dict[str, Tool] = {}
        # Reverse mapping: input_schema class -> tool instance
        self._schema_to_tool: dict[type[BaseModel], Tool] = {}
        self._schema_to_tool: dict[type[BaseModel], Tool] = {}
        # Context providers: context_key -> provider instance
        self.context_providers: dict[str, ContextProvider] = {}

    def create_plugin_context(self) -> PluginContext:
        """
        Create a safe PluginContext facade for plugins to use.
        """
    def create_plugin_context(self) -> PluginContext:
        """
        Create a safe PluginContext facade for plugins to use.
        """
        import os

        plugin_config = {
            "root_dir": str(settings.root_dir),
            "download_dir": str(settings.download_dir),
            "cache_dir": str(settings.cache_dir),
            # Include API keys from environment
            "TMDB_API_KEY": os.getenv("TMDB_API_KEY"),
        }

        server_info = {
            "version": "0.1.0",
            "environment": "development",
        }

        return PluginContext(
            config=plugin_config,
            server_info=server_info,
        )

    def load_plugins(self):
        """
        Scans the plugin directory and loads all valid Tool and ContextProvider protocol implementations.
        Scans the plugin directory and loads all valid Tool and ContextProvider protocol implementations.
        """
        logger.info(f"Scanning for plugins in: {self.plugin_dir}")

        if not self.plugin_dir.exists():
            logger.warning("Plugin directory not found. Creating it.")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)

        for module_info in pkgutil.iter_modules([str(self.plugin_dir)]):
            if module_info.ispkg:
                self._load_single_plugin(module_info.name)

    def _load_single_plugin(self, module_name: str):
        try:
            full_module_name = f"app.plugins.{module_name}"
            module = importlib.import_module(full_module_name)

            found_tool = False
            found_context_provider = False

            # Single loop: iterate once, check for both Tool and ContextProvider
            found_context_provider = False

            # Single loop: iterate once, check for both Tool and ContextProvider
            for _, obj in inspect.getmembers(module):
                if not (inspect.isclass(obj) and not inspect.isabstract(obj)):
                    continue

                try:
                    instance = obj()
                if not (inspect.isclass(obj) and not inspect.isabstract(obj)):
                    continue

                try:
                    instance = obj()

                    # 1. Handle Tools
                    if isinstance(instance, Tool):
                        # Check if tool is available before loading
                        if not instance.is_available():
                            logger.warning(
                                f"Skipping Tool: {instance.name} (not available - check configuration)"
                            )
                            continue
                    # 1. Handle Tools
                    if isinstance(instance, Tool):
                        # Check if tool is available before loading
                        if not instance.is_available():
                            logger.warning(
                                f"Skipping Tool: {instance.name} (not available - check configuration)"
                            )
                            continue

                        self.tools[instance.name] = instance
                        self.tools[instance.name] = instance

                        if hasattr(instance, "input_schema") and instance.input_schema:
                            self._schema_to_tool[instance.input_schema] = instance
                        if hasattr(instance, "input_schema") and instance.input_schema:
                            self._schema_to_tool[instance.input_schema] = instance

                        # Log tool version (avoid async call during sync loading)
                        # Status generation is async and should be called via get_tool_status()
                        # during runtime, not during plugin loading
                        logger.info(f"Loaded Tool: {instance.name} (v{instance.version})")
                        found_tool = True
                        # Log tool version (avoid async call during sync loading)
                        # Status generation is async and should be called via get_tool_status()
                        # during runtime, not during plugin loading
                        logger.info(f"Loaded Tool: {instance.name} (v{instance.version})")
                        found_tool = True

                    # 2. Handle Providers (using elif since an object shouldn't be both)
                    elif isinstance(instance, ContextProvider):
                        context_key = instance.context_key

                        if context_key in self.context_providers:
                            logger.error(
                                f"Duplicate context_key '{context_key}' found in {module_name}. "
                                f"Skipping provider."
                            )
                            continue
                    # 2. Handle Providers (using elif since an object shouldn't be both)
                    elif isinstance(instance, ContextProvider):
                        context_key = instance.context_key

                        if context_key in self.context_providers:
                            logger.error(
                                f"Duplicate context_key '{context_key}' found in {module_name}. "
                                f"Skipping provider."
                            )
                            continue

                        self.context_providers[context_key] = instance
                        logger.info(f"Loaded Context Provider: {context_key}")
                        found_context_provider = True
                        self.context_providers[context_key] = instance
                        logger.info(f"Loaded Context Provider: {context_key}")
                        found_context_provider = True

                except (TypeError, Exception):
                    # TypeError happens if __init__ requires args (likely not a plugin)
                    # Exception catches anything else
                    continue

                except (TypeError, Exception):
                    # TypeError happens if __init__ requires args (likely not a plugin)
                    # Exception catches anything else
                    continue

            if not found_tool and not found_context_provider:
                logger.debug(
                    f"No Tool or ContextProvider protocol implementations found in {module_name}"
                )
                logger.debug(
                    f"No Tool or ContextProvider protocol implementations found in {module_name}"
                )

        except Exception as e:
            # Graceful failure: Log the error but keep the server running
            logger.error(f"Failed to load plugin '{module_name}': {e}")

    def get_tool(self, name: str) -> Tool | None:
    def get_tool(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def get_tool_by_schema(self, schema_class: type[BaseModel]) -> Tool | None:
    def get_tool_by_schema(self, schema_class: type[BaseModel]) -> Tool | None:
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

    async def get_tool_status(self, tool: Tool) -> PluginStatus:
        """
        Generate PluginStatus from a Tool instance using the Observer-Loader pattern.

        This centralizes status generation in the Core, eliminating the need for
        plugins to implement get_status_info(). Uses Pydantic's model_validate
        to automatically map Tool Protocol properties (name, version, description)
        to the PluginStatus model.

        Supports both sync and async is_available() methods to handle I/O-bound
        availability checks (e.g., API key validation, service reachability).

        Args:
            tool: The Tool instance to generate status for

        Returns:
            PluginStatus model with all tool metadata and availability status
        """
        # 1. Resolve the dynamic boolean (Handle sync/async)
        # is_available is a method, not a property, so we must call it
        if inspect.iscoroutinefunction(tool.is_available):
            available = await tool.is_available()
        else:
            available = tool.is_available()

        # 2. Build the status object from a dict
        # We use model_validate with a dict to bridge the gap between
        # the Tool's properties and the dynamic 'is_available' value.
        # The 'name' field will be mapped to 'service_name' via the alias.
        status = PluginStatus.model_validate(
            {
                "name": tool.name,  # Mapped to service_name via alias
                "version": tool.version,
                "description": tool.description,
                "is_available": available,
            }
        )

        # Handle 'extra' info if the tool supports it via optional method
        if hasattr(tool, "get_extra_info"):
            try:
                # Support both sync and async extra info methods
                if inspect.iscoroutinefunction(tool.get_extra_info):
                    status.extra = await tool.get_extra_info()
                else:
                    status.extra = tool.get_extra_info()
            except Exception as e:
                logger.warning(f"Failed to get extra info from {tool.name}: {e}")

        return status

    async def get_plugin_statuses(self) -> dict[str, PluginStatus]:
    async def get_tool_status(self, tool: Tool) -> PluginStatus:
        """
        Generate PluginStatus from a Tool instance using the Observer-Loader pattern.

        This centralizes status generation in the Core, eliminating the need for
        plugins to implement get_status_info(). Uses Pydantic's model_validate
        to automatically map Tool Protocol properties (name, version, description)
        to the PluginStatus model.

        Supports both sync and async is_available() methods to handle I/O-bound
        availability checks (e.g., API key validation, service reachability).

        Args:
            tool: The Tool instance to generate status for

        Returns:
            PluginStatus model with all tool metadata and availability status
        """
        # 1. Resolve the dynamic boolean (Handle sync/async)
        # is_available is a method, not a property, so we must call it
        if inspect.iscoroutinefunction(tool.is_available):
            available = await tool.is_available()
        else:
            available = tool.is_available()

        # 2. Build the status object from a dict
        # We use model_validate with a dict to bridge the gap between
        # the Tool's properties and the dynamic 'is_available' value.
        # The 'name' field will be mapped to 'service_name' via the alias.
        status = PluginStatus.model_validate(
            {
                "name": tool.name,  # Mapped to service_name via alias
                "version": tool.version,
                "description": tool.description,
                "is_available": available,
            }
        )

        # Handle 'extra' info if the tool supports it via optional method
        if hasattr(tool, "get_extra_info"):
            try:
                # Support both sync and async extra info methods
                if inspect.iscoroutinefunction(tool.get_extra_info):
                    status.extra = await tool.get_extra_info()
                else:
                    status.extra = tool.get_extra_info()
            except Exception as e:
                logger.warning(f"Failed to get extra info from {tool.name}: {e}")

        return status

    async def get_plugin_statuses(self) -> dict[str, PluginStatus]:
        """
        Aggregate status information from all loaded plugins.

        Uses centralized status generation (Observer-Loader pattern) where
        the Core observes the plugin and generates status, rather than
        the plugin reporting its own status.

        This method is async to support async is_available() checks that may
        perform I/O operations (e.g., validating API keys, checking service reachability).
        Uses centralized status generation (Observer-Loader pattern) where
        the Core observes the plugin and generates status, rather than
        the plugin reporting its own status.

        This method is async to support async is_available() checks that may
        perform I/O operations (e.g., validating API keys, checking service reachability).
        """
        statuses = {}
        for name, tool in self.tools.items():
            statuses[name] = await self.get_tool_status(tool)
            statuses[name] = await self.get_tool_status(tool)
        return statuses

    async def gather_context(self, query: str) -> dict[str, Any]:
        """
        Gather context from all eligible context providers.
        """
        context = self.create_plugin_context()
        gathered_context = {}

        for provider in self.context_providers.values():
            try:
                if await provider.is_eligible(query):
                    response = await provider.provide_context(context)
                    gathered_context[provider.context_key] = response.data
            except Exception as e:
                logger.warning(f"Failed to gather context from {provider.context_key}: {e}")
                continue

        return gathered_context

    async def gather_context(self, query: str) -> dict[str, Any]:
        """
        Gather context from all eligible context providers.
        """
        context = self.create_plugin_context()
        gathered_context = {}

        for provider in self.context_providers.values():
            try:
                if await provider.is_eligible(query):
                    response = await provider.provide_context(context)
                    gathered_context[provider.context_key] = response.data
            except Exception as e:
                logger.warning(f"Failed to gather context from {provider.context_key}: {e}")
                continue

        return gathered_context
