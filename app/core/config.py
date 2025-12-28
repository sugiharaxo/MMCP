"""
Configuration management using pydantic-settings.

Handles environment variables, defines ROOT_DIR for file operations,
and provides typed configuration for plugins and the agent.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get the project root directory (3 levels up from this file)."""
    return Path(__file__).parent.parent.parent


def _get_plugin_dir() -> Path:
    """Get the default plugin directory."""
    return Path(__file__).parent.parent / "plugins"


def _get_download_dir() -> Path:
    """Get the default download directory."""
    return _get_project_root() / "downloads"


def _get_cache_dir() -> Path:
    """Get the default cache directory."""
    return _get_project_root() / "cache"


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Loads from .env file and environment variables.
    Environment variables take precedence over .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Root directory for all file operations (downloads, cache, etc.)
    # Defaults to project root if not set
    root_dir: Path = Field(
        default_factory=_get_project_root,
        description="Root directory for file operations",
    )

    # Agent/LLM Configuration
    # LiteLLM standard format: provider/model (e.g., openai/gpt-4o, ollama/llama3)
    llm_model: str = Field(
        default="gemini/gemini-flash-latest",
        description="LLM model in LiteLLM format: provider/model (e.g., 'openai/gpt-4o' or 'ollama/llama3')",
    )
    llm_api_key: str | None = Field(
        default=None,
        description="API key for LLM provider (not needed for Ollama)",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Base URL for LLM API (only needed for Ollama/LocalAI, e.g., 'http://localhost:11434')",
    )

    # Context Management (Character-based)
    # Defaulting to 100k chars (~50k tokens), which is safe for most models.
    llm_max_context_chars: int = Field(
        default=200_000,
        description="Maximum character count for conversation history",
    )

    # Plugin Configuration
    plugin_dir: Path = Field(
        default_factory=_get_plugin_dir,
        description="Directory containing plugin modules",
    )

    # Media/Download Configuration
    download_dir: Path = Field(
        default_factory=_get_download_dir,
        description="Directory for downloaded media files",
    )
    cache_dir: Path = Field(
        default_factory=_get_cache_dir,
        description="Directory for cache files",
    )

    # Context Provider Configuration
    # Heuristics for context provider execution and health monitoring
    context_global_timeout_ms: int = Field(
        default=800,
        description="Global timeout in milliseconds for all context providers combined",
    )
    context_per_provider_timeout_ms: int = Field(
        default=300,
        description="Per-provider timeout in milliseconds for individual context fetches",
    )
    context_max_chars_per_provider: int = Field(
        default=2000,
        description="Maximum characters per provider response (truncated if exceeded)",
    )
    context_max_string_length: int = Field(
        default=200,
        description="Maximum length for individual strings in provider data during truncation",
    )
    context_max_list_items: int = Field(
        default=5,
        description="Maximum number of items in lists during truncation",
    )
    context_failure_threshold: int = Field(
        default=3,
        description="Number of consecutive failures before circuit breaker trips",
    )
    context_recovery_wait_minutes: int = Field(
        default=5,
        description="Minutes to wait before retrying a circuit-broken provider",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        for d in [self.download_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # Normalize paths (resolve symlinks, make absolute)
        self.root_dir = self.root_dir.resolve()
        self.plugin_dir = self.plugin_dir.resolve()
        self.download_dir = self.download_dir.resolve()
        self.cache_dir = self.cache_dir.resolve()


# Global settings instance
# Import this in other modules: `from app.core.config import settings`
settings = Settings()
