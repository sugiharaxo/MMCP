import importlib
import inspect
import os
import pkgutil
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from app.api.schemas import PluginContext, PluginStatus
from app.core.config import CoreSettings, settings
from app.core.logger import logger
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
        # Reverse mapping: input_schema class -> tool instance
        self._schema_to_tool: dict[type[BaseModel], Tool] = {}
        # Context providers: context_key -> provider instance
        self.context_providers: dict[str, ContextProvider] = {}
        # Plugin settings: tool name -> validated BaseModel instance (or None if no settings)
        self._plugin_settings: dict[str, BaseModel | None] = {}
        # Plugin config errors: tool name -> agent-friendly error message
        self._plugin_config_errors: dict[str, str] = {}

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

    def _load_plugin_settings(self, plugin_instance: Tool) -> BaseModel | None:
        """
        Load and validate plugin settings from namespaced environment variables.

        Uses manual env var loading (not BaseSettings) for loader-controlled prefixing.
        Supports nested configuration via double underscore (__) delimiter.

        Pattern: MMCP_PLUGIN_{SLUG}_{FIELD_NAME} or MMCP_PLUGIN_{SLUG}_{NESTED__FIELD}

        Args:
            plugin_instance: Tool instance with optional settings_model attribute

        Returns:
            Validated BaseModel instance, or None if no model/validation fails
        """
        # Return None if no settings_model (explicit optional)
        if not hasattr(plugin_instance, "settings_model") or plugin_instance.settings_model is None:
            return None

        settings_model = plugin_instance.settings_model

        # Derive plugin slug using deterministic slugification
        plugin_slug = self._slugify_plugin_name(plugin_instance.name)
        env_prefix = f"MMCP_PLUGIN_{plugin_slug}_"

        # Manual env var loading (not BaseSettings) for prefix control
        # Use case_sensitive=False for cross-platform compatibility (Windows env vars are case-insensitive)
        env_data: dict[str, Any] = {}
        prefix_lower = env_prefix.lower()

        for env_key, env_value in os.environ.items():
            env_key_lower = env_key.lower()
            # Case-insensitive prefix matching
            if env_key_lower.startswith(prefix_lower):
                # Strip prefix to get field name
                field_name_with_nesting = env_key_lower[len(prefix_lower) :]
                # Handle nested configuration: double underscore (__) maps to nested model
                # Example: MMCP_PLUGIN_TMDB_RETRY_POLICY__MAX_RETRIES -> retry_policy.max_retries
                if "__" in field_name_with_nesting:
                    # Split on double underscore for nesting
                    parts = field_name_with_nesting.split("__", 1)
                    nested_key = parts[0]
                    nested_field = parts[1]
                    # Build nested dict structure
                    if nested_key not in env_data:
                        env_data[nested_key] = {}
                    env_data[nested_key][nested_field] = env_value
                else:
                    # Simple field (no nesting)
                    env_data[field_name_with_nesting] = env_value

        # Instantiate plugin's BaseModel directly (not BaseSettings)
        try:
            return settings_model(**env_data)
        except ValidationError as e:
            # Generate agent-friendly error message instead of raw Pydantic trace
            missing_fields = []
            for error in e.errors():
                if error["type"] == "missing":
                    field_name = error.get("loc", ("unknown",))[-1]
                    # Reconstruct expected env var name for clarity
                    if "__" in str(field_name):
                        # Nested field
                        nested_parts = str(field_name).split("__")
                        expected_env = (
                            f"{env_prefix}{nested_parts[0].upper()}__{nested_parts[1].upper()}"
                        )
                    else:
                        expected_env = f"{env_prefix}{str(field_name).upper()}"
                    missing_fields.append(expected_env)

            if missing_fields:
                error_msg = f"Missing required environment variable(s): {', '.join(missing_fields)}"
            else:
                # Fallback for other validation errors
                error_summary = "; ".join(
                    [
                        f"{err.get('loc', ('unknown',))[-1]}: {err.get('msg', 'validation error')}"
                        for err in e.errors()[:3]
                    ]
                )
                error_msg = f"Configuration validation failed: {error_summary}"
                if len(e.errors()) > 3:
                    error_msg += f" (and {len(e.errors()) - 3} more)"

            logger.warning(
                f"Plugin '{plugin_instance.name}' configuration validation failed: {error_msg}"
            )
            # Store error for agent visibility
            self._plugin_config_errors[plugin_instance.name] = error_msg
            # Return None if validation fails (tool will receive None in is_available/execute)
            return None

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

    def load_plugins(self):
        """
        Scans the plugin directory and loads all valid Tool and ContextProvider protocol implementations.

        Uses two-phase loading to prevent circularity:
        - Phase 1: Discover all plugins, instantiate, load settings (config validation only)
        - Phase 2: After all plugins discovered, call is_available(settings, context) with complete context
        """
        logger.info(f"Scanning for plugins in: {self.plugin_dir}")

        if not self.plugin_dir.exists():
            logger.warning("Plugin directory not found. Creating it.")
            self.plugin_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Discovery and Config Validation (No Context)
        discovered_tools: list[Tool] = []
        for module_info in pkgutil.iter_modules([str(self.plugin_dir)]):
            if module_info.ispkg:
                tool = self._load_single_plugin(module_info.name)
                if tool:
                    discovered_tools.append(tool)

        # Phase 2: Functional Availability Check (With Complete Context)
        # Create complete PluginContext after all plugins are discovered
        plugin_context = self.create_plugin_context()

        for tool in discovered_tools:
            # Retrieve settings loaded in Phase 1
            settings = self._plugin_settings.get(tool.name, None)

            # Check availability with complete context (prevents circularity)
            # Protocol specifies sync is_available(), but handle gracefully if async
            try:
                if inspect.iscoroutinefunction(tool.is_available):
                    logger.warning(
                        f"Tool {tool.name} has async is_available() but protocol specifies sync. "
                        f"Treating as unavailable."
                    )
                    available = False
                else:
                    available = tool.is_available(settings, plugin_context)
            except Exception as e:
                logger.warning(f"Error checking availability for {tool.name}: {e}")
                available = False

            if available:
                self.tools[tool.name] = tool
                if hasattr(tool, "input_schema") and tool.input_schema:
                    self._schema_to_tool[tool.input_schema] = tool
                logger.info(f"Loaded Tool: {tool.name} (v{tool.version})")
            else:
                config_error = self._plugin_config_errors.get(tool.name)
                reason = f" - {config_error}" if config_error else ""
                logger.warning(f"Skipping Tool: {tool.name} (not available{reason})")

    def _load_single_plugin(self, module_name: str) -> Tool | None:
        """
        Phase 1: Discover plugin, instantiate, and load settings (config validation only).

        Does NOT call is_available() here - that's deferred to Phase 2 after all plugins are discovered.

        Returns:
            Tool instance if found, None otherwise
        """
        try:
            full_module_name = f"app.plugins.{module_name}"
            module = importlib.import_module(full_module_name)

            found_tool = False
            found_context_provider = False
            tool_instance: Tool | None = None

            # Single loop: iterate once, check for both Tool and ContextProvider
            for _, obj in inspect.getmembers(module):
                if not (inspect.isclass(obj) and not inspect.isabstract(obj)):
                    continue

                try:
                    instance = obj()

                    # 1. Handle Tools
                    if isinstance(instance, Tool):
                        # Phase 1: Load settings (config validation only)
                        settings = self._load_plugin_settings(instance)
                        self._plugin_settings[instance.name] = settings

                        # Store tool instance for Phase 2 availability check
                        tool_instance = instance
                        found_tool = True
                        # Do NOT call is_available() here - deferred to Phase 2

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

                except (TypeError, Exception):
                    # TypeError happens if __init__ requires args (likely not a plugin)
                    # Exception catches anything else
                    continue

            if not found_tool and not found_context_provider:
                logger.debug(
                    f"No Tool or ContextProvider protocol implementations found in {module_name}"
                )

            return tool_instance

        except Exception as e:
            # Graceful failure: Log the error but keep the server running
            logger.error(f"Failed to load plugin '{module_name}': {e}")
            return None

    def get_tool(self, name: str) -> Tool | None:
        return self.tools.get(name)

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
        # Retrieve settings loaded in Phase 1
        settings = self._plugin_settings.get(tool.name, None)
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

        # Include config error in extra field for agent visibility
        config_error = self._plugin_config_errors.get(tool.name)
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
        Aggregate status information from all loaded plugins.

        Uses centralized status generation (Observer-Loader pattern) where
        the Core observes the plugin and generates status, rather than
        the plugin reporting its own status.

        This method is async to support async is_available() checks that may
        perform I/O operations (e.g., validating API keys, checking service reachability).
        """
        statuses = {}
        for name, tool in self.tools.items():
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
