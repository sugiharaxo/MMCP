"""Tests for plugin loader."""

from pathlib import Path

import pytest

from app.core.plugin_loader import PluginLoader


def test_loader_initialization(plugin_dir: Path):
    """Test that PluginLoader initializes correctly."""
    loader = PluginLoader(plugin_dir)
    assert loader.plugin_dir == plugin_dir
    assert loader.tools == {}


@pytest.mark.asyncio
async def test_load_plugins_empty_directory(loader: PluginLoader):
    """Test loading plugins from an empty directory."""
    await loader.load_plugins()
    assert loader.tools == {}


@pytest.mark.asyncio
async def test_load_plugins_with_valid_plugin(plugin_dir: Path):
    """Test loading a valid plugin from a Python file."""

    # Create a plugin file in the test directory
    plugin_content = """
from pydantic import BaseModel
from app.api.base import Plugin, Tool

class TestPlugin(Plugin):
    name = "test_plugin"
    version = "1.0.0"
    settings_model = None

    class TestTool(Tool):
        name = "test_tool"
        description = "A test tool"
        version = "1.0.0"
        input_schema = BaseModel

        def is_available(self, settings, context) -> bool:
            return True

        async def execute(self, **kwargs) -> dict:
            return {"result": "test"}
"""

    # Create the plugin file
    plugin_file = plugin_dir / "test_plugin.py"
    plugin_file.write_text(plugin_content)

    loader = PluginLoader(plugin_dir)
    await loader.load_plugins()

    # Check that tool was loaded
    assert "test_tool" in loader.list_tools()
    tool = loader.get_tool("test_tool")
    assert tool is not None
    assert tool.name == "test_tool"
    assert tool.plugin_name == "test_plugin"  # Verify plugin_name is set


def test_list_tools(loader: PluginLoader, mock_tool):
    """Test listing available tools."""
    # Manually add a tool (simulating loaded plugin)
    loader.tools["test_tool"] = mock_tool

    tools = loader.list_tools()
    assert "test_tool" in tools
    assert tools["test_tool"] == mock_tool.description


def test_get_tool(loader: PluginLoader, mock_tool):
    """Test retrieving a tool by name."""
    loader.tools["test_tool"] = mock_tool

    retrieved = loader.get_tool("test_tool")
    assert retrieved == mock_tool

    # Test non-existent tool
    assert loader.get_tool("nonexistent") is None


@pytest.mark.asyncio
async def test_get_tool_unavailable(plugin_dir: Path):
    """Test that unavailable tools go to standby_tools instead of being dropped."""

    # Create a plugin file with an unavailable tool
    plugin_content = """
from pydantic import BaseModel
from app.api.base import Plugin, Tool

class UnavailablePlugin(Plugin):
    name = "unavailable_plugin"
    version = "1.0.0"
    settings_model = None

    class UnavailableTool(Tool):
        name = "unavailable_tool"
        description = "An unavailable tool"
        version = "1.0.0"
        input_schema = BaseModel

        def is_available(self, settings, context) -> bool:
            return False  # Tool is not available

        async def execute(self, **kwargs) -> dict:
            return {}
"""

    # Create the plugin file
    plugin_file = plugin_dir / "unavailable_plugin.py"
    plugin_file.write_text(plugin_content)

    loader = PluginLoader(plugin_dir)
    await loader.load_plugins()

    # Tool should be in standby_tools, not active tools
    assert "unavailable_tool" not in loader.list_tools()
    assert "unavailable_tool" in loader.standby_tools
    tool = loader.get_tool("unavailable_tool")  # Should still be retrievable
    assert tool is not None
    assert tool.name == "unavailable_tool"
    assert tool.plugin_name == "unavailable_plugin"  # Verify plugin_name is set


@pytest.mark.asyncio
async def test_get_tool_status_sync(loader: PluginLoader, mock_tool):
    """Test get_tool_status with sync is_available()."""

    loader.tools["mock_tool"] = mock_tool
    # Set up plugin settings (loaded in Phase 1) - None for tools without settings
    # Settings are stored by plugin_name, not tool name
    loader._plugin_settings[mock_tool.plugin_name] = None

    status = await loader.get_tool_status(mock_tool)

    assert status.service_name == "mock_tool"
    assert status.version == "1.0.0"
    assert status.description == "A mock tool for testing"
    assert status.is_available is True
    assert status.extra == {}


@pytest.mark.asyncio
async def test_get_tool_status_async():
    """Test get_tool_status with async is_available() (protocol specifies sync, but test edge case)."""
    from pydantic import BaseModel

    from app.core.plugin_loader import PluginLoader

    class AsyncTool:
        settings_model = None  # No settings required

        @property
        def name(self) -> str:
            return "async_tool"

        @property
        def description(self) -> str:
            return "An async tool"

        @property
        def version(self) -> str:
            return "2.0.0"

        @property
        def input_schema(self):
            return BaseModel

        # Note: Protocol specifies sync, but test edge case where someone implements async
        async def is_available(self, _settings, _context) -> bool:
            # Simulate async I/O check
            import asyncio

            await asyncio.sleep(0.01)
            return True

        def get_extra_info(self) -> dict:
            return {"async": True, "test": "data"}

        async def execute(self, _context, _settings, **_kwargs) -> dict:
            return {"result": "async"}

    loader = PluginLoader(Path("/tmp"))
    tool = AsyncTool()
    tool.plugin_name = "async_plugin"  # Set plugin_name attribute
    loader.tools["async_tool"] = tool
    # Set up plugin settings by plugin_name
    loader._plugin_settings["async_plugin"] = None

    status = await loader.get_tool_status(tool)

    assert status.service_name == "async_tool"
    assert status.version == "2.0.0"
    assert status.description == "An async tool"
    assert status.is_available is True
    assert status.extra == {"async": True, "test": "data"}


@pytest.mark.asyncio
async def test_get_plugin_statuses(loader: PluginLoader, mock_tool):
    """Test get_plugin_statuses aggregates status from all tools."""
    loader.tools["mock_tool"] = mock_tool

    statuses = await loader.get_plugin_statuses()

    assert "mock_tool" in statuses
    assert statuses["mock_tool"].service_name == "mock_tool"
    assert statuses["mock_tool"].is_available is True


@pytest.mark.asyncio
async def test_multiple_loaders_no_conflicts(tmp_path):
    """Test that multiple PluginLoader instances don't interfere with each other."""
    import sys

    # Create two separate plugin directories
    plugin_dir1 = tmp_path / "plugins1"
    plugin_dir2 = tmp_path / "plugins2"
    plugin_dir1.mkdir()
    plugin_dir2.mkdir()

    # Create different plugins in each directory
    plugin1_file = plugin_dir1 / "plugin1.py"
    plugin1_file.write_text("""
from pydantic import BaseModel
from app.api.base import Plugin, Tool

class Tool1Input(BaseModel):
    pass

class Plugin1(Plugin):
    name = "plugin1"
    version = "1.0.0"
    settings_model = None

    class Tool1(Tool):
        name = "tool1"
        description = "Tool from plugin1"
        input_schema = Tool1Input
        version = "1.0.0"

        def is_available(self, settings=None, runtime=None):
            return True

        async def execute(self):
            return {"source": "plugin1"}
""")

    plugin2_file = plugin_dir2 / "plugin2.py"
    plugin2_file.write_text("""
from pydantic import BaseModel
from app.api.base import Plugin, Tool

class Tool2Input(BaseModel):
    pass

class Plugin2(Plugin):
    name = "plugin2"
    version = "1.0.0"
    settings_model = None

    class Tool2(Tool):
        name = "tool2"
        description = "Tool from plugin2"
        input_schema = Tool2Input
        version = "1.0.0"

        def is_available(self, settings=None, runtime=None):
            return True

        async def execute(self):
            return {"source": "plugin2"}
""")

    # Create and load multiple loaders
    initial_path = sys.path.copy()
    loader1 = PluginLoader(plugin_dir1)
    loader2 = PluginLoader(plugin_dir2)

    await loader1.load_plugins()
    await loader2.load_plugins()

    # Verify sys.path is clean after multiple loaders
    assert sys.path == initial_path, f"sys.path was polluted: {set(sys.path) - set(initial_path)}"

    # Verify isolation - each loader only has its own plugins
    assert "plugin1" in loader1.plugins
    assert "tool1" in loader1.tools
    assert "plugin2" not in loader1.plugins
    assert "tool2" not in loader1.tools

    assert "plugin2" in loader2.plugins
    assert "tool2" in loader2.tools
    assert "plugin1" not in loader2.plugins
    assert "tool1" not in loader2.tools


@pytest.mark.asyncio
async def test_exception_handling_cleans_sys_path(tmp_path):
    """Test that plugin load failures don't leave sys.path polluted."""
    import sys

    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    # Create a plugin with syntax error
    bad_plugin_file = plugin_dir / "bad_plugin.py"
    bad_plugin_file.write_text("""
from app.api.base import Plugin, Tool

class BadPlugin(Plugin):
    name = "bad_plugin"
    version = "1.0.0"
    settings_model = None

    # Syntax error - missing closing paren
    def broken_method(self
        return "incomplete syntax"
""")

    # Create a good plugin
    good_plugin_file = plugin_dir / "good_plugin.py"
    good_plugin_file.write_text("""
from pydantic import BaseModel
from app.api.base import Plugin, Tool

class GoodToolInput(BaseModel):
    pass

class GoodPlugin(Plugin):
    name = "good_plugin"
    version = "1.0.0"
    settings_model = None

    class GoodTool(Tool):
        name = "good_tool"
        description = "Good tool"
        input_schema = GoodToolInput
        version = "1.0.0"

        def is_available(self, settings=None, runtime=None):
            return True

        async def execute(self):
            return {"status": "good"}
""")

    # Record sys.path before loading
    initial_path = set(sys.path)

    loader = PluginLoader(plugin_dir)
    await loader.load_plugins()

    # Verify sys.path is clean after loading (even with failures)
    final_path = set(sys.path)
    assert initial_path == final_path, f"sys.path was polluted: {final_path - initial_path}"

    # Verify the good plugin still loaded despite the bad one
    assert "good_plugin" in loader.plugins
    assert "good_tool" in loader.tools

    # Verify the bad plugin didn't load
    assert "bad_plugin" not in loader.plugins


@pytest.mark.asyncio
async def test_plugin_with_local_imports(tmp_path):
    """Test that plugins can import local helper modules."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()

    # Create a helper module
    helper_file = plugin_dir / "test_helper.py"
    helper_file.write_text("""
def get_helper_data():
    return "helper_data"

HELPER_CONSTANT = "constant_value"
""")

    # Create a plugin that imports the helper
    plugin_file = plugin_dir / "plugin_with_helper.py"
    plugin_file.write_text("""
from pydantic import BaseModel
from app.api.base import Plugin, Tool
from test_helper import get_helper_data, HELPER_CONSTANT

class HelperToolInput(BaseModel):
    pass

class HelperPlugin(Plugin):
    name = "helper_plugin"
    version = "1.0.0"
    settings_model = None

    class HelperTool(Tool):
        name = "helper_tool"
        description = "Tool that uses helper module"
        input_schema = HelperToolInput
        version = "1.0.0"

        def is_available(self, settings=None, runtime=None):
            return True

        async def execute(self):
            return {
                "data": get_helper_data(),
                "constant": HELPER_CONSTANT
            }
""")

    loader = PluginLoader(plugin_dir)
    await loader.load_plugins()

    # Verify the plugin was loaded and can access helper
    assert "helper_plugin" in loader.plugins
    assert len(loader.tools) == 1

    tool = loader.tools["helper_tool"]
    result = await tool.execute()
    assert result == {"data": "helper_data", "constant": "constant_value"}
