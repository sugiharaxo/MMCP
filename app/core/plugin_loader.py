import ast
import contextlib
import importlib.util
import inspect
import re
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from app.api.base import ContextProvider, Plugin, Tool
from app.api.schemas import PluginRuntime, PluginStatus
from app.core.config import CoreSettings, user_settings
from app.core.logger import logger
from app.core.settings_manager import SettingsManager

if TYPE_CHECKING:
    pass


class PluginLoader:
    """
    Responsible for discovering and instantiating tools from the 'plugins' directory.
    """

    # Class-level lock for thread-safe sys.path modifications
    _sys_path_lock = threading.RLock()  # Reentrant lock for nested contexts

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
        self.context_providers: dict[str, ContextProvider] = {}
        # Plugin settings: plugin name -> validated BaseModel instance (or None if no settings)
        self._plugin_settings: dict[str, BaseModel | None] = {}
        # Plugin config errors: plugin name -> agent-friendly error message
        self._plugin_config_errors: dict[str, str] = {}
        # Locked fields per plugin: plugin name -> set of locked field names
        self._locked_fields: dict[str, set[str]] = {}

    @contextlib.contextmanager
    def _plugin_import_context(self):
        """
        Thread-safe context manager for temporary sys.path modification.

        Uses RLock to allow nested contexts (e.g., if a plugin imports another plugin).
        Ensures plugins can import local modules during discovery without
        permanently polluting global sys.path state.

        Note: While sys.path is cleaned up after use, imported modules remain
        cached in sys.modules for the application lifetime. This is fine for
        normal operation but prevents hot-reloading of plugins without
        additional cleanup of sys.modules entries.
        """
        plugin_path = str(self.plugin_dir)

        with self._sys_path_lock:
            added = False
            try:
                if plugin_path not in sys.path:
                    sys.path.insert(0, plugin_path)  # Higher priority
                    added = True
                yield
            finally:
                # Thread-safe cleanup - only remove if we added it
                if added and plugin_path in sys.path:
                    sys.path.remove(plugin_path)

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

    def create_plugin_runtime(self) -> PluginRuntime:
        """
        Create a safe PluginRuntime facade for plugins to use.

        Returns CoreSettings instance instead of dict for type-safe access.
        """
        # Instantiate CoreSettings from settings instance
        core_paths = CoreSettings(
            root_dir=user_settings.root_dir,
            download_dir=user_settings.download_dir,
            cache_dir=user_settings.cache_dir,
        )

        system_info = {
            "version": "0.1.0",
            "environment": "development",
        }

        return PluginRuntime(
            paths=core_paths,
            system=system_info,
        )

    def _get_module_name(self, file_path: Path) -> str:
        """Converts a file path to a unique dot-notated module name."""
        # Get relative path from project root (plugin_dir.parent)
        relative_path = file_path.relative_to(self.plugin_dir.parent)
        # Result: "plugins.metadata.tmdb" or "plugins.foundation"
        return ".".join(relative_path.with_suffix("").parts)

    def _is_plugin_file(self, file_path: Path) -> bool:
        """AST scan: Returns True if the file defines a class inheriting from 'Plugin'."""
        try:
            tree = ast.parse(file_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # We look for the literal name "Plugin" in the bases
                if isinstance(node, ast.ClassDef) and any(
                    (isinstance(base, ast.Name) and base.id == "Plugin")
                    or (isinstance(base, ast.Attribute) and base.attr == "Plugin")
                    for base in node.bases
                ):
                    return True
        except SyntaxError:
            logger.warning(f"Skipping plugin {file_path.name}: Syntax Error in file.")
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
        return False

    def _discover_plugins(self) -> list[type[Plugin]]:
        """
        Phase 1: Discover all Plugin classes using AST scanning.

        Returns:
            List of Plugin class types (not instances yet)
        """
        discovered_plugins: list[type[Plugin]] = []

        # Use context manager to scope sys.path modification
        with self._plugin_import_context():
            # 1. Recursively find all .py files
            for py_file in self.plugin_dir.rglob("*.py"):
                # 2. Skip private files and standard boilerplate
                if py_file.name.startswith("_") or py_file.name == "__init__.py":
                    continue

                # 3. Use AST to filter out helper files
                if not self._is_plugin_file(py_file):
                    continue

                # 4. Import the file as a module
                module_name = self._get_module_name(py_file)
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is not None and spec.loader is not None:
                        module = importlib.util.module_from_spec(spec)
                        # Note: This caches the module in sys.modules permanently.
                        # Good for performance, but prevents hot-reloading without
                        # explicit sys.modules cleanup.
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)

                        # 5. Extract the Plugin class(es)
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, Plugin)
                                and attr is not Plugin
                                and not inspect.isabstract(attr)
                            ):
                                discovered_plugins.append(attr)
                                logger.info(
                                    f"Discovered Plugin: {attr.name} v{attr.version} ({module_name})"
                                )

                except Exception as e:
                    logger.error(f"Failed to load plugin module {module_name}: {e}")

        return discovered_plugins

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
        discovered_plugin_classes = self._discover_plugins()
        discovered_plugins: list[Plugin] = []
        for plugin_class in discovered_plugin_classes:
            try:
                # Validation happens at definition time via __init_subclass__
                # If validation failed, the import would have raised TypeError
                instance = plugin_class()  # Instantiate the Plugin class
                discovered_plugins.append(instance)
            except TypeError as e:
                # Catch definition-time validation errors from __init_subclass__
                logger.error(f"[PluginLoader] Definition Error in {plugin_class.__name__}: {e}")
            except Exception as e:
                logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")

        # Phase 2: Functional Availability Check (With Complete Context)
        # Create complete PluginRuntime after all plugins are discovered
        plugin_runtime = self.create_plugin_runtime()

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
                    plugin_available = await plugin.is_available(plugin_settings, plugin_runtime)
                else:
                    plugin_available = plugin.is_available(plugin_settings, plugin_runtime)
            except Exception as e:
                logger.warning(f"Error checking availability for plugin {plugin.name}: {e}")
                plugin_available = False

            if not plugin_available:
                logger.info(f"Plugin {plugin.name} is not available, skipping tools/providers")
                continue

            # 1. Register Tools (The Plugin has already injected settings/context/logger)

            for tool in plugin.get_tools(plugin_settings, plugin_runtime):
                # Validation happens in Plugin.get_tools() before instantiation
                # Check tool availability
                try:
                    if inspect.iscoroutinefunction(tool.is_available):
                        available = await tool.is_available(tool.settings, plugin_runtime)
                    else:
                        available = tool.is_available(tool.settings, plugin_runtime)
                except Exception as e:
                    logger.warning(f"Error checking availability for {tool.name}: {e}")
                    available = False

                # Fail fast on duplicate tool names - tool names must be globally unique
                if tool.name in self.tools:
                    existing_tool = self.tools[tool.name]
                    raise ValueError(
                        f"Duplicate tool name '{tool.name}' detected. "
                        f"Tool from plugin '{tool.plugin_name}' conflicts with existing tool "
                        f"from plugin '{existing_tool.plugin_name}'. "
                        f"Tool names must be globally unique across all plugins."
                    )
                if tool.name in self.standby_tools:
                    existing_tool = self.standby_tools[tool.name]
                    raise ValueError(
                        f"Duplicate tool name '{tool.name}' detected. "
                        f"Tool from plugin '{tool.plugin_name}' conflicts with existing standby tool "
                        f"from plugin '{existing_tool.plugin_name}'. "
                        f"Tool names must be globally unique across all plugins."
                    )

                if available:
                    self.tools[tool.name] = tool
                    logger.info(f"Loaded Tool: {tool.name} (v{tool.version})")
                else:
                    self.standby_tools[tool.name] = tool
                    config_error = self._plugin_config_errors.get(plugin.name)
                    reason = f" - {config_error}" if config_error else ""
                    logger.info(f"Tool {tool.name} on standby (not available{reason})")

            # Extract and register providers from this plugin
            providers = plugin.get_providers(plugin_settings, plugin_runtime)
            for provider in providers:
                if provider.context_key in self.context_providers:
                    logger.error(
                        f"Duplicate context_key '{provider.context_key}' found in {plugin.name}. "
                        f"Skipping provider."
                    )
                    continue

                self.context_providers[provider.context_key] = provider
                logger.info(f"Loaded Context Provider: {provider.context_key}")

        # Registry dump: list all successfully registered tools
        if self.tools:
            tool_names = sorted(self.tools.keys())
            logger.info(f"Plugin discovery complete. Registered tools: {', '.join(tool_names)}")
        else:
            logger.info("Plugin discovery complete. No tools registered.")

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
        plugin_runtime = self.create_plugin_runtime()

        # 1. Resolve the dynamic boolean (Handle sync/async)
        # is_available now receives settings and context
        try:
            if inspect.iscoroutinefunction(tool.is_available):
                available = await tool.is_available(settings, plugin_runtime)
            else:
                available = tool.is_available(settings, plugin_runtime)
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
        gathered_context = {}

        for provider in self.context_providers.values():
            try:
                if await provider.is_eligible(query):
                    response = await provider.provide_context()
                    gathered_context[provider.context_key] = response.data
            except Exception as e:
                logger.warning(f"Failed to gather context from {provider.context_key}: {e}")
                continue

        return gathered_context
