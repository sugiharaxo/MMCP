"""
Configuration management using pydantic-settings.

Handles environment variables, defines ROOT_DIR for file operations,
and provides typed configuration for plugins and the agent.
"""

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    """Get the project root directory (3 levels up from this file)."""
    return Path(__file__).parent.parent.parent


def _get_plugin_dir() -> Path:
    """Get the default plugin directory."""
    return _get_project_root() / "plugins"


def _get_download_dir() -> Path:
    """Get the default download directory."""
    return _get_project_root() / "downloads"


def _get_cache_dir() -> Path:
    """Get the default cache directory."""
    return _get_project_root() / "cache"


def _get_data_dir() -> Path:
    """Get the default data directory for database and persistent storage."""
    return _get_project_root() / "data"


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
    data_dir: Path = Field(
        default_factory=_get_data_dir,
        description="Directory for database and persistent storage",
    )

    # Authentication Configuration
    require_auth: bool = Field(
        default=False,
        description="Require authentication for settings endpoints (uses .admin_token file)",
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
        default=10000,
        description="Maximum characters per provider response (truncated if exceeded)",
    )
    context_max_string_length: int = Field(
        default=500,
        description="Maximum length for individual strings in provider data during truncation",
    )
    context_max_list_items: int = Field(
        default=10,
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
        for d in [self.download_dir, self.cache_dir, self.data_dir]:
            d.mkdir(parents=True, exist_ok=True)
        # Normalize paths (resolve symlinks, make absolute)
        self.root_dir = self.root_dir.resolve()
        self.plugin_dir = self.plugin_dir.resolve()
        self.download_dir = self.download_dir.resolve()
        self.cache_dir = self.cache_dir.resolve()
        self.data_dir = self.data_dir.resolve()


class ReasoningProfile(BaseModel):
    """
    Profile for internal system steering (reasoning, tool selection, deterministic logic).

    Low temperature and token count for focused, consistent decision-making.
    """

    temperature: float = Field(default=0.0, description="Temperature for reasoning calls")
    max_tokens: int = Field(default=300, description="Maximum tokens for reasoning responses")


class DialogueProfile(BaseModel):
    """
    Profile for external human communication (helpfulness, natural language, summaries).

    Higher temperature and token count for natural, conversational responses.
    """

    temperature: float = Field(default=0.7, description="Temperature for dialogue calls")
    max_tokens: int = Field(default=2048, description="Maximum tokens for dialogue responses")


class LLMProfile(BaseModel):
    """
    Complete LLM profile containing both reasoning and dialogue configurations.

    Used for model-specific overrides. Falls back to default_profile if not specified.
    """

    reasoning: ReasoningProfile = Field(
        default_factory=ReasoningProfile, description="Reasoning profile configuration"
    )
    dialogue: DialogueProfile = Field(
        default_factory=DialogueProfile, description="Dialogue profile configuration"
    )


class CoreSettings(BaseModel):
    """
    Core settings shared across all plugins.

    Extracted from Settings for type-safe plugin access.
    Plugins receive this via PluginRuntime.paths, accessible as self.paths in tools.
    """

    root_dir: Path = Field(description="Root directory for file operations")
    download_dir: Path = Field(description="Directory for downloaded media files")
    cache_dir: Path = Field(description="Directory for cache files")


# Global settings instance
# Import this in other modules: `from app.core.config import settings`
settings = Settings()

# Default LLM profile (used as fallback)
default_profile = LLMProfile()

# Model-specific profile overrides
# Example: model_profiles["gemini/gemini-flash-latest"] = LLMProfile(...)
model_profiles: dict[str, LLMProfile] = {}
