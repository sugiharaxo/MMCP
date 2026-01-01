"""Shared pytest fixtures for MMCP tests."""

from pathlib import Path
from unittest.mock import AsyncMock

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

    class MockSettings(BaseModel):
        """Mock settings model (optional - demonstrates Null Object Pattern)."""

        api_key: str = "test_key"

    class MockTool:
        # Optional settings_model (Null Object Pattern if None)
        settings_model = MockSettings

        def __init__(self):
            # Set plugin_name attribute (required by Tool base class)
            self.plugin_name = "mock_plugin"

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

        def is_available(self, _settings: BaseModel, _context) -> bool:
            """New signature: receives settings and context."""
            return True

        async def execute(self, test_param: str) -> dict:
            """Tool execution signature: kwargs only (context/settings are in self)."""
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


@pytest.fixture(scope="session", autouse=True)
async def init_test_database():
    """Initialize database for all tests that need it."""
    # Use in-memory SQLite for tests
    from app.core.config import settings

    # Override database path to use in-memory database
    original_db_path = settings.data_dir / "mmcp.db"
    test_db_url = "sqlite+aiosqlite:///:memory:"

    # Monkey patch the database URL
    import app.core.database

    original_init_database = app.core.database.init_database

    async def mock_init_database():
        """Mock init_database to use in-memory database."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from app.core.database import Base

        global _engine, _session_maker
        app.core.database._engine = create_async_engine(
            test_db_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )
        app.core.database._session_maker = async_sessionmaker(
            app.core.database._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables
        async with app.core.database._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    app.core.database.init_database = mock_init_database

    try:
        await mock_init_database()
        yield
    finally:
        # Cleanup
        if hasattr(app.core.database, "_engine") and app.core.database._engine:
            await app.core.database._engine.dispose()
        app.core.database.init_database = original_init_database


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    return AsyncMock()
