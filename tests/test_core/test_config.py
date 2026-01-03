"""Tests for configuration management."""

from pathlib import Path

from app.core.config import UserSettings


def test_default_paths():
    """Test that default paths are resolved correctly."""
    settings = UserSettings()

    assert isinstance(settings.root_dir, Path)
    assert isinstance(settings.plugin_dir, Path)
    assert isinstance(settings.download_dir, Path)
    assert isinstance(settings.cache_dir, Path)

    # All paths should be absolute
    assert settings.root_dir.is_absolute()
    assert settings.plugin_dir.is_absolute()
    assert settings.download_dir.is_absolute()
    assert settings.cache_dir.is_absolute()


def test_directories_created(tmp_path: Path, monkeypatch):
    """Test that download and cache directories are created automatically."""
    monkeypatch.setenv("DOWNLOAD_DIR", str(tmp_path / "downloads"))
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    settings = UserSettings()

    assert settings.download_dir.exists()
    assert settings.cache_dir.exists()


def test_llm_config_from_env(monkeypatch):
    """Test that LLM configuration can be loaded from environment."""
    monkeypatch.setenv("LLM_MODEL", "gemini/flash")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434")

    settings = UserSettings()

    assert settings.llm_model == "gemini/flash"
    assert settings.llm_api_key == "sk-test-key"
    assert settings.llm_base_url == "http://localhost:11434"
