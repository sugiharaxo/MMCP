"""
Configuration management using pydantic-settings.

Handles environment variables, defines ROOT_DIR for file operations,
and provides typed configuration for plugins and the agent.

Architecture:
- UserSettings: User-tunable settings (from .env or DB) with UI metadata
- Internal Settings: Logic constants loaded from settings.json (not user-configurable)
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
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


def _load_internal_settings() -> dict[str, Any]:
    """
    Load internal logic constants from settings.json.

    These are parameters the user should never see or modify.
    The Machine's Manual - immutable at runtime.
    """
    settings_file = Path(__file__).parent / "settings.json"
    if not settings_file.exists():
        # Return safe defaults if file doesn't exist
        return {
            "react_loop": {"backoff_base": 2, "max_llm_retries": 2, "instructor_max_retries": 2},
            "event_bus": {"default_ttl_seconds": 300},
            "logging": {"max_bytes": 5242880, "backup_count": 3},
            "notifications": {"pending_limit": 10, "recent_acks_limit": 5},
            "watchdog": {"interval_seconds": 10},
        }
    with open(settings_file, encoding="utf-8") as f:
        return json.load(f)


# Load internal settings once at module import
internal_settings = _load_internal_settings()


def parse_llm_model_string(model_string: str) -> tuple[str, str]:
    """
    Parse any-llm format string (provider:model) into BAML provider and model.

    Examples:
        "openai:gpt-4o" -> ("openai", "gpt-4o")
        "ollama:llama3" -> ("openai-generic", "llama3")  # Ollama uses openai-generic
        "anthropic:claude-3-5-sonnet" -> ("anthropic", "claude-3-5-sonnet")

    Returns:
        Tuple of (provider, model_name)
    """
    if ":" not in model_string:
        raise ValueError(f"Invalid model string format: {model_string}. Expected 'provider:model'")

    provider, model = model_string.split(":", 1)

    # Handle special cases for BAML providers
    if provider.lower() == "ollama":
        # Ollama uses openai-generic provider in BAML
        provider = "openai-generic"

    return provider, model


class HitlDecision(str, Enum):
    """Enum for Human-in-the-Loop (HITL) decision values."""

    APPROVE = "approve"
    DENY = "deny"


class UserSettings(BaseSettings):
    """
    User-tunable settings (The Dashboard).

    Loads from .env file, environment variables, or database overrides.
    Environment variables take precedence over .env file values.

    Fields marked with json_schema_extra={"ui_advanced": True} are hidden
    behind an "Advanced" toggle in the UI.
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
    # any-llm universal routing format: provider:model (e.g., openai:gpt-4o, ollama:llama3)
    llm_model: str = Field(
        default="openai:gpt-4o-mini",
        description="LLM model in BAML format: provider:model (e.g., 'openai:gpt-4o' or 'ollama:llama3'). Parsed by parse_llm_model_string() for BAML ClientRegistry.",
    )

    @field_validator("llm_model")
    @classmethod
    def validate_llm_model_format(cls, v: str) -> str:
        """Validate llm_model format using parse_llm_model_string."""
        try:
            parse_llm_model_string(v)
        except ValueError as e:
            raise ValueError(f"Invalid llm_model format: {e}") from e
        return v

    llm_api_key: str | None = Field(
        default=None,
        description="API key for LLM provider (not needed for Ollama)",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Custom base URL for LLM API (optional). Useful for self-hosted instances, API proxies, or custom endpoints. Examples: Ollama='http://localhost:11434', OpenAI proxy='https://your-proxy.com/v1'",
    )
    instructor_mode: str = Field(
        default="tool_call",
        description="Instructor mode: 'tool_call' (native tool calling), 'json' (JSON output), or 'markdown_json' (MD+JSON output)",
    )
    models_requiring_system_merge: tuple[str, ...] = Field(
        default=("gemma",),
        description="Model name keywords (case-insensitive) that require system instruction merging. Configure via comma-separated string in env: MMCP_MODELS_REQUIRING_SYSTEM_MERGE='gemma,custom-model'",
        json_schema_extra={"ui_advanced": True},
    )

    @field_validator("models_requiring_system_merge", mode="before")
    @classmethod
    def parse_models_requiring_system_merge(cls, v: Any) -> tuple[str, ...]:
        """Parse comma-separated string or list into immutable tuple."""
        if isinstance(v, str):
            return tuple(keyword.strip() for keyword in v.split(",") if keyword.strip())
        if isinstance(v, (list, tuple)):
            return tuple(str(item).strip() for item in v if str(item).strip())
        return v if isinstance(v, tuple) else (v,)

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

    # --- ADVANCED SETTINGS ---
    # ReAct Loop Configuration
    react_max_steps: int = Field(
        default=5,
        description="Maximum ReAct loop iterations before giving up",
        json_schema_extra={"ui_advanced": True},
    )
    react_max_llm_retries: int = Field(
        default=2,
        description="Maximum retry attempts for LLM validation errors",
        json_schema_extra={"ui_advanced": True},
    )
    react_max_rate_limit_retries: int = Field(
        default=3,
        description="Maximum retry attempts for rate limit errors",
        json_schema_extra={"ui_advanced": True},
    )
    tool_execution_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout in seconds for tool execution",
        json_schema_extra={"ui_advanced": True},
    )
    session_lock_timeout_seconds: float = Field(
        default=300.0,
        description="Timeout in seconds for acquiring session locks (5 minutes default)",
        json_schema_extra={"ui_advanced": True},
    )
    tool_circuit_breaker_threshold: int = Field(
        default=3,
        description="Number of consecutive failures before tool circuit breaker trips",
        json_schema_extra={"ui_advanced": True},
    )

    # Orchestrator Configuration
    max_resume_depth: int = Field(
        default=10,
        description="Maximum recursion depth for action resumption",
        json_schema_extra={"ui_advanced": True},
    )

    # Context Provider Configuration
    # Heuristics for context provider execution and health monitoring
    context_global_timeout_ms: int = Field(
        default=800,
        description="Global timeout in milliseconds for all context providers combined",
        json_schema_extra={"ui_advanced": True},
    )
    context_per_provider_timeout_ms: int = Field(
        default=300,
        description="Per-provider timeout in milliseconds for individual context fetches",
        json_schema_extra={"ui_advanced": True},
    )
    context_max_chars_per_provider: int = Field(
        default=10000,
        description="Maximum characters per provider response (truncated if exceeded)",
        json_schema_extra={"ui_advanced": True},
    )
    context_max_string_length: int = Field(
        default=500,
        description="Maximum length for individual strings in provider data during truncation",
        json_schema_extra={"ui_advanced": True},
    )
    context_max_list_items: int = Field(
        default=10,
        description="Maximum number of items in lists during truncation",
        json_schema_extra={"ui_advanced": True},
    )
    context_failure_threshold: int = Field(
        default=3,
        description="Number of consecutive failures before circuit breaker trips",
        json_schema_extra={"ui_advanced": True},
    )
    context_recovery_wait_minutes: int = Field(
        default=5,
        description="Minutes to wait before retrying a circuit-broken provider",
        json_schema_extra={"ui_advanced": True},
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


# Load .env file into os.environ so both UserSettings AND SettingsManager can access it
# This ensures plugin settings can also read from .env files
env_file = _get_project_root() / ".env"
if env_file.exists():
    load_dotenv(env_file, override=False)  # override=False: actual env vars take precedence

# Global user settings instance
# Import this in other modules: `from app.core.config import user_settings`
user_settings = UserSettings()

# Default LLM profile (used as fallback)
default_profile = LLMProfile()

# Model-specific profile overrides
# Example: model_profiles["gemini/gemini-flash-latest"] = LLMProfile(...)
model_profiles: dict[str, LLMProfile] = {}
