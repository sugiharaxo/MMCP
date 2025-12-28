"""Shared pytest fixtures for MMCP tests."""

from pathlib import Path

import pytest

from app.core.plugin_loader import PluginLoader


@pytest.fixture
def plugin_dir(tmp_path: Path) -> Path:
    """Create a temporary plugin directory for testing."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    return plugin_dir


@pytest.fixture
def loader(plugin_dir: Path) -> PluginLoader:
    """Create a PluginLoader instance with a temporary plugin directory."""
    return PluginLoader(plugin_dir)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    from app.agent.schemas import FinalResponse

    return FinalResponse(thought="Test thought", answer="Test response")


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    from pydantic import BaseModel, Field

    class MockInput(BaseModel):
        test_param: str = Field(..., description="Test parameter")

    class MockTool:
        @property
        def name(self) -> str:
            return "mock_tool"

        @property
        def description(self) -> str:
            return "A mock tool for testing"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def input_schema(self) -> type[BaseModel]:
            return MockInput

        def is_available(self) -> bool:
            return True

        async def execute(self, _, test_param: str) -> dict:
            return {"result": f"Processed: {test_param}"}

    return MockTool()


@pytest.fixture
def env_override(monkeypatch):
    """Helper fixture to override environment variables."""

    def _override(key: str, value: str | None):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    return _override
