"""
Database layer for MMCP using SQLAlchemy Async.

Provides async engine, session maker, and base model for plugin settings persistence.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ANP models are registered with Base when imported elsewhere
from app.core.config import user_settings
from app.core.logger import logger


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class PluginSetting(Base):
    """
    Plugin settings storage model.

    Simple key-value store per plugin slug.
    """

    __tablename__ = "plugin_settings"

    plugin_slug: Mapped[str] = mapped_column(String(100), primary_key=True, index=True)
    key: Mapped[str] = mapped_column(String(100), primary_key=True, index=True)
    value: Mapped[str] = mapped_column(String(1000), nullable=False)

    def __repr__(self) -> str:
        return f"<PluginSetting(plugin_slug={self.plugin_slug!r}, key={self.key!r})>"


# Global database engine and session maker
_engine = None
_session_maker = None


def get_db_path() -> Path:
    """Get the database file path."""
    return user_settings.data_dir / "mmcp.db"


async def init_database() -> None:
    """
    Initialize the database engine and create tables.

    Must be called in async context (e.g., FastAPI lifespan).
    Sets file permissions to 0o600 on non-Windows systems.
    """
    global _engine, _session_maker

    db_path = get_db_path()
    # SQLite async connection string
    db_url = f"sqlite+aiosqlite:///{db_path}"

    # Create async engine
    _engine = create_async_engine(
        db_url,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,
        connect_args={"check_same_thread": False},  # Required for SQLite async
    )

    # Create session maker
    _session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Set file permissions on non-Windows systems
    if os.name != "nt":
        try:
            os.chmod(db_path, 0o600)
            logger.info(f"Set database file permissions to 0o600: {db_path}")
        except OSError as e:
            logger.warning(f"Failed to set database permissions: {e}")

    logger.info(f"Database initialized: {db_path}")


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    Get an async database session.

    Must be used as async context manager:
        async with get_session() as session:
            ...
    """
    if _session_maker is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    async with _session_maker() as session:
        yield session


async def close_database() -> None:
    """Close the database engine."""
    global _engine, _session_maker
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_maker = None
        logger.info("Database connection closed")
