import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from app.api.base import Plugin, Provider, Tool
from app.api.schemas import PluginContext, PluginStatus
from app.core.config import CoreSettings, settings
from app.core.logger import logger
from app.core.settings_manager import SettingsManager

if TYPE_CHECKING:
    pass


class PluginLoader:
    """
    Responsible for discovering and instantiating tools from the 'plugins' directory.
    """

    def __init__(self, plugin_dir: Path, settings_manager: SettingsManager | None = None):
        self.plugin_dir = plugin_dir
        self.settings_manager = settings_manager or SettingsManager()
        # Plugin registry: plugin name -> Plugin instance
        self.plugins: dict[str, Plugin] = {}
        self.tools: dict[str, Tool] = {}
        # Standby tools: plugins that are discovered but not configured
        self.standby_tools: dict[str, Tool] = {}
        # Reverse mapping: input_schema class -> tool instance
        self._schema_to_tool: dict[type[BaseModel], Tool] = {}
        # Context providers: context_key -> provider instance
        self.context_providers: dict[str, Provider] = {}
        # Plugin settings: plugin name -> validated BaseModel instance (or None if no settings)
        self._plugin_settings: dict[str, BaseModel | None] = {}
        # Plugin config errors: plugin name -> agent-friendly error message
        self._plugin_config_errors: dict[str, str] = {}
        # Locked fields per plugin: plugin name -> set of locked field names
        self._locked_fields: dict[str, set[str]] = {}

    @staticmethod
    def _slugify_plugin_name(name: str) -> str:
        """
        Convert plugin name to deterministic ENV prefix slug.

        Prevents collisions between similar names (e.g., 'TMDB-Lookup' vs 'TMDB_Lookup').
        Normalizes non-alphanumeric characters to underscores, uppercases, and collapses.

        Examples:
            'TMDB-Lookup' -> 'TMDB_LOOKUP'
            'tmdb_lookup_metadata' -> 'TMDB_LOOKUP_METADATA'
            'my-plugin_v2' -> 'MY_PLUGIN_V2'

        Args:
            name: Plugin name (e.g., from tool.name property)

        Returns:
            Deterministic uppercase slug for environment variable prefix
        """
        # Replace any non-alphanumeric character with underscore
        slug = re.sub(r"[^a-zA-Z0-9]", "_", name)
        # Uppercase
        slug = slug.upper()
        # Collapse multiple underscores and strip leading/trailing
        slug = re.sub(r"_+", "_", slug).strip("_")
        return slug

    def create_plugin_context(self) -> PluginContext:
        """
        Create a safe PluginContext facade for plugins to use.

        Returns CoreSettings instance instead of dict for type-safe access.
        """
        # Instantiate CoreSettings from settings instance
        core_config = CoreSettings(
            root_dir=settings.root_dir,
            download_dir=settings.download_dir,
            cache_dir=settings.cache_dir,
        )

        server_info = {
            "version": "0.1.0",
            "environment": "development",
        }

        return PluginContext(
            config=core_config,
            server_info=server_info,
        )

    async def load_plugins(self):
        """
        Scans the plugin directory and loads all Plugin subclasses.

        Uses two-phase loading to prevent circularity:
        - Phase 1: Discover all Plugin classes, instantiate, load settings (config validation only)
        - Phase 2: After all plugins discovered, extract tools/providers and check availability

        Plugins that fail validation or availability check go into standby_tools instead of being dropped.
        """
        logger.info(f"Scanning for plugins in: {self.plugin_dir}")

        if not self.plugin_dir.exists():
            logger.warning("Plugin directory not found. Creating it.")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Discovery and Config Validation (No Context)
        discovered_plugins: list[Plugin] = []
        for module_info in pkgutil.iter_modules([str(self.plugin_dir)]):
            if module_info.ispkg:
                plugin = await self._load_single_plugin(module_info.name)
                if plugin:
                    discovered_plugins.append(plugin)

        # Phase 2: Functional Availability Check (With Complete Context)
        # Create complete PluginContext after all plugins are discovered
        plugin_context = self.create_plugin_context()

        for plugin in discovered_plugins:
            # Store plugin instance in registry
            self.plugins[plugin.name] = plugin

            # Load plugin-level settings (returns settings and locked_fields)
            plugin_settings, locked_fields = await self._load_plugin_settings_for_plugin(plugin)

            # Store plugin-level settings and locked fields (all tools in a plugin share these)
            self._plugin_settings[plugin.name] = plugin_settings
            self._locked_fields[plugin.name] = locked_fields

            # Check plugin availability
            try:
                if inspect.iscoroutinefunction(plugin.is_available):
                    plugin_available = await plugin.is_available(plugin_settings, plugin_context)
                else:
                    plugin_available = plugin.is_available(plugin_settings, plugin_context)
            except Exception as e:
                logger.warning(f"Error checking availability for plugin {plugin.name}: {e}")
                plugin_available = False

            if not plugin_available:
                logger.info(f"Plugin {plugin.name} is not available, skipping tools/providers")
                continue

            # 1. Register Tools (The Plugin has already injected settings/context/logger)

            for tool in plugin.get_tools(plugin_settings, plugin_context):
                # Validation happens in Plugin.get_tools() before instantiation
                # Check tool availability
                try:
                    if inspect.iscoroutinefunction(tool.is_available):
                        available = await tool.is_available(tool.settings, plugin_context)
                    else:
                        available = tool.is_available(tool.settings, plugin_context)
                except Exception as e:
                    logger.warning(f"Error checking availability for {tool.name}: {e}")
                    available = False

                if available:
                    self.tools[tool.name] = tool
                    if hasattr(tool, "input_schema") and tool.input_schema:
                        self._schema_to_tool[tool.input_schema] = tool
                    logger.info(f"Loaded Tool: {tool.name} (v{tool.version})")
                else:
                    self.standby_tools[tool.name] = tool
                    config_error = self._plugin_config_errors.get(plugin.name)
                    reason = f" - {config_error}" if config_error else ""
                    logger.info(f"Tool {tool.name} on standby (not available{reason})")

            # Extract and register providers from this plugin
            providers = plugin.get_providers(plugin_settings)
            for provider in providers:
                if provider.context_key in self.context_providers:
                    logger.error(
                        f"Duplicate context_key '{provider.context_key}' found in {plugin.name}. "
                        f"Skipping provider."
                    )
                    continue

                self.context_providers[provider.context_key] = provider
                logger.info(f"Loaded Context Provider: {provider.context_key}")

    async def _load_single_plugin(self, module_name: str) -> Plugin | None:
        """
        Phase 1: Discover Plugin subclass, instantiate it.

        Returns:
            Plugin instance if found, None otherwise
        """
        try:
            full_module_name = f"app.plugins.{module_name}"
            module = importlib.import_module(full_module_name)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                # 1. Must inherit from Plugin
                # 2. Must not BE the Plugin base class
                # 3. Must be defined in the plugin file (not imported)
                # 4. Must not be abstract (only concrete implementations become active plugins)
                if (
                    issubclass(obj, Plugin)
                    and obj is not Plugin
                    and obj.__module__ == module.__name__
                    and not inspect.isabstract(obj)
                ):
                    # Validation happens at definition time via __init_subclass__
                    # If validation failed, the import would have raised TypeError
                    instance = obj()  # Instantiate the Plugin class
                    return instance

            logger.debug(f"No Plugin subclass found in {module_name}")
            return None

        except TypeError as e:
            # Catch definition-time validation errors from __init_subclass__
            logger.error(f"[PluginLoader] Definition Error in {module_name}: {e}")
            return None
        except Exception as e:
            # Graceful failure: Log the error but keep the server running
            logger.error(f"Failed to load plugin '{module_name}': {e}")
            return None

    async def _load_plugin_settings_for_plugin(
        self, plugin_instance: Plugin
    ) -> tuple[BaseModel | None, set[str]]:
        """
        Load and validate plugin-level settings using SettingsManager.

        Args:
            plugin_instance: Plugin instance with optional settings_model attribute

        Returns:
            Tuple of (validated BaseModel instance or None, set of locked field names)
        """
        # Return None if no settings_model (explicit optional)
        if not hasattr(plugin_instance, "settings_model") or plugin_instance.settings_model is None:
            return None, set()

        settings_model = plugin_instance.settings_model

        # Derive plugin slug using deterministic slugification
        plugin_slug = self._slugify_plugin_name(plugin_instance.name)

        # Use SettingsManager to get settings (merges defaults/db/env)
        settings, locked_fields, validation_error = await self.settings_manager.get_plugin_settings(
            plugin_slug, settings_model
        )

        # If validation failed, translate the ValidationError into actionable environment variables
        if validation_error is not None:
            env_prefix = f"MMCP_PLUGIN_{plugin_slug}_"
            error_details = []

            for error in validation_error.errors():
                loc = error.get("loc", [])
                if isinstance(loc, (list, tuple)) and len(loc) > 0:
                    field_path = ".".join(str(part) for part in loc)
                    env_field = field_path.replace(".", "__").upper()
                    env_name = f"{env_prefix}{env_field}"

                    error_type = error.get("type", "validation_error")
                    error_msg = error.get("msg", "invalid value")

                    if error_type == "missing":
                        error_details.append(f"{env_name} (required)")
                    elif error_type in ("string_type", "int_type", "bool_type"):
                        error_details.append(
                            f"{env_name} (wrong type, expected {error_type.replace('_type', '')})"
                        )
                    else:
                        error_details.append(f"{env_name} ({error_msg})")

            if error_details:
                error_report = f"Configuration failed: {', '.join(error_details)}"
            else:
                error_report = "Configuration validation failed (unknown error)"

            self._plugin_config_errors[plugin_instance.name] = error_report
            return None, locked_fields

        return settings, locked_fields

    def get_tool(self, name: str) -> Tool | None:
        """Get tool from active tools or standby tools."""
        return self.tools.get(name) or self.standby_tools.get(name)

    def get_tool_by_schema(self, schema_class: type[BaseModel]) -> Tool | None:
        """
        Get tool instance by its input_schema Pydantic model class.

        Used to map LLM tool call responses (which are instances of tool input schemas)
        back to the tool instance that should execute them.

        Checks active tools first, then standby tools.

        Args:
            schema_class: The Pydantic model class (e.g., TMDbMetadataInput)

        Returns:
            The Tool instance that uses this schema, or None if not found
        """
        # Check active tools first
        tool = self._schema_to_tool.get(schema_class)
        if tool:
            return tool

        # Check standby tools
        for standby_tool in self.standby_tools.values():
            if hasattr(standby_tool, "input_schema") and standby_tool.input_schema == schema_class:
                return standby_tool

        return None

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
        # Retrieve settings by plugin name (all tools in a plugin share the same settings)
        plugin_name = tool.plugin_name
        settings = self._plugin_settings.get(plugin_name, None)
        plugin_context = self.create_plugin_context()

        # 1. Resolve the dynamic boolean (Handle sync/async)
        # is_available now receives settings and context
        try:
            if inspect.iscoroutinefunction(tool.is_available):
                available = await tool.is_available(settings, plugin_context)
            else:
                available = tool.is_available(settings, plugin_context)
        except Exception as e:
            logger.warning(f"Error checking availability for {tool.name}: {e}")
            available = False

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

        # Include config error in extra field for agent visibility (by plugin name)
        config_error = self._plugin_config_errors.get(plugin_name)
        if config_error:
            status.extra["config_error"] = config_error

        # Handle 'extra' info if the tool supports it via optional method
        if hasattr(tool, "get_extra_info"):
            try:
                # Support both sync and async extra info methods
                if inspect.iscoroutinefunction(tool.get_extra_info):
                    extra_info = await tool.get_extra_info()
                else:
                    extra_info = tool.get_extra_info()
                # Merge with existing extra (preserving config_error if present)
                status.extra.update(extra_info)
            except Exception as e:
                logger.warning(f"Failed to get extra info from {tool.name}: {e}")

        return status

    async def get_plugin_statuses(self) -> dict[str, PluginStatus]:
        """
        Aggregate status information from all loaded and standby plugins.

        Uses centralized status generation (Observer-Loader pattern) where
        the Core observes the plugin and generates status, rather than
        the plugin reporting its own status.

        This method is async to support async is_available() checks that may
        perform I/O operations (e.g., validating API keys, checking service reachability).
        """
        statuses = {}
        # Include active tools
        for name, tool in self.tools.items():
            statuses[name] = await self.get_tool_status(tool)
        # Include standby tools
        for name, tool in self.standby_tools.items():
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
