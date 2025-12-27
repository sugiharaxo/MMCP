"""Tests for plugin loader."""

from pathlib import Path

from app.core.plugin_loader import PluginLoader


def test_loader_initialization(plugin_dir: Path):
    """Test that PluginLoader initializes correctly."""
    loader = PluginLoader(plugin_dir)
    assert loader.plugin_dir == plugin_dir
    assert loader.tools == {}


def test_load_plugins_empty_directory(loader: PluginLoader):
    """Test loading plugins from an empty directory."""
    loader.load_plugins()
    assert loader.tools == {}


def test_load_plugins_with_valid_plugin(plugin_dir: Path):
    """Test loading a valid plugin."""
    from unittest.mock import MagicMock, patch

    from pydantic import BaseModel

    from app.core.plugin_interface import MMCPTool

    # Create a proper TestTool class that inherits from MMCPTool
    class TestTool(MMCPTool):
        @property
        def name(self) -> str:
            return "test_tool"

        @property
        def description(self) -> str:
            return "A test tool"

        @property
        def input_schema(self):
            return BaseModel

        def is_available(self) -> bool:
            return True

        async def execute(self, **_kwargs) -> dict:
            return {"result": "test"}

    # Create a mock module with the TestTool class
    mock_module = MagicMock()
    mock_module.TestTool = TestTool

    loader = PluginLoader(plugin_dir)

    # Mock the import_module and pkgutil to return our test plugin
    with (
        patch("app.core.plugin_loader.importlib.import_module", return_value=mock_module),
        patch("app.core.plugin_loader.pkgutil.iter_modules") as mock_iter,
    ):
        mock_iter.return_value = [MagicMock(name="test_plugin", ispkg=True)]
        loader.load_plugins()

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


def test_get_tool_unavailable(plugin_dir: Path):
    """Test that unavailable tools are not loaded."""
    from unittest.mock import MagicMock, patch

    from pydantic import BaseModel

    from app.core.plugin_interface import MMCPTool

    # Create a proper UnavailableTool class that inherits from MMCPTool
    class UnavailableTool(MMCPTool):
        @property
        def name(self) -> str:
            return "unavailable_tool"

        @property
        def description(self) -> str:
            return "An unavailable tool"

        @property
        def input_schema(self):
            return BaseModel

        def is_available(self) -> bool:
            return False  # Tool is not available

        async def execute(self, **_kwargs) -> dict:
            return {}

    # Create a mock module with the UnavailableTool class
    mock_module = MagicMock()
    mock_module.UnavailableTool = UnavailableTool

    loader = PluginLoader(plugin_dir)

    # Mock the import_module and pkgutil to return our test plugin
    with (
        patch("app.core.plugin_loader.importlib.import_module", return_value=mock_module),
        patch("app.core.plugin_loader.pkgutil.iter_modules") as mock_iter,
    ):
        mock_iter.return_value = [MagicMock(name="unavailable_plugin", ispkg=True)]
        loader.load_plugins()

    # Tool should not be loaded because it's unavailable
    assert "unavailable_tool" not in loader.list_tools()
