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
    """Test loading a valid plugin."""
    from unittest.mock import MagicMock, patch

    from pydantic import BaseModel

    # Create a proper TestTool class that implements Tool protocol
    class TestTool:
        settings_model = None  # No settings required

        @property
        def name(self) -> str:
            return "test_tool"

        @property
        def description(self) -> str:
            return "A test tool"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def input_schema(self):
            return BaseModel

        def is_available(self, _settings, _context) -> bool:
            return True

        async def execute(self, _context, _settings, **_kwargs) -> dict:
            return {"result": "test"}

    # Create a mock module with the TestTool class
    import types

    mock_module = types.ModuleType("test_plugin")
    mock_module.TestTool = TestTool

    loader = PluginLoader(plugin_dir)

    # Mock the import_module and pkgutil to return our test plugin
    with (
        patch("app.core.plugin_loader.importlib.import_module", return_value=mock_module),
        patch("app.core.plugin_loader.pkgutil.iter_modules") as mock_iter,
    ):
        mock_iter.return_value = [MagicMock(name="test_plugin", ispkg=True)]
        await loader.load_plugins()

    # Check that tool was loaded
    assert "test_tool" in loader.list_tools()
    tool = loader.get_tool("test_tool")
    assert tool is not None
    assert tool.name == "test_tool"


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
    from unittest.mock import MagicMock, patch

    from pydantic import BaseModel

    # Create a proper UnavailableTool class that implements Tool protocol
    class UnavailableTool:
        settings_model = None  # No settings required

        @property
        def name(self) -> str:
            return "unavailable_tool"

        @property
        def description(self) -> str:
            return "An unavailable tool"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def input_schema(self):
            return BaseModel

        def is_available(self, _settings, _context) -> bool:
            return False  # Tool is not available

        async def execute(self, _context, _settings, **_kwargs) -> dict:
            return {}

    # Create a mock module with the UnavailableTool class
    import types

    mock_module = types.ModuleType("test_plugin")
    mock_module.UnavailableTool = UnavailableTool

    loader = PluginLoader(plugin_dir)

    # Mock the import_module and pkgutil to return our test plugin
    with (
        patch("app.core.plugin_loader.importlib.import_module", return_value=mock_module),
        patch("app.core.plugin_loader.pkgutil.iter_modules") as mock_iter,
    ):
        mock_iter.return_value = [MagicMock(name="unavailable_plugin", ispkg=True)]
        await loader.load_plugins()

    # Tool should be in standby_tools, not active tools
    assert "unavailable_tool" not in loader.list_tools()
    assert "unavailable_tool" in loader.standby_tools
    tool = loader.get_tool("unavailable_tool")  # Should still be retrievable
    assert tool is not None
    assert tool.name == "unavailable_tool"


@pytest.mark.asyncio
async def test_get_tool_status_sync(loader: PluginLoader, mock_tool):
    """Test get_tool_status with sync is_available()."""

    loader.tools["mock_tool"] = mock_tool
    # Set up plugin settings (loaded in Phase 1) - None for tools without settings
    loader._plugin_settings["mock_tool"] = None

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
    loader.tools["async_tool"] = tool

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
