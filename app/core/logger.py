"""MMCP Logger - Cross-platform, self-cleaning logging utility."""

import logging
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import internal_settings


class MMCPLogger:
    """
    Cross-platform logging utility that follows MMCP's lightweight philosophy.

    Features:
    - Zero external dependencies (stdlib only)
    - Cross-platform log directory handling
    - Self-cleaning with size-based rotation
    - Structured logging for observability
    """

    def __init__(self, name: str = "mmcp"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self._setup_handlers()

    def _get_log_dir(self) -> Path:
        """Cross-platform log directory discovery."""
        if platform.system() == "Windows":
            # Windows: %LOCALAPPDATA%\mmcp\logs
            base_dir = Path.home() / "AppData/Local/mmcp"
        else:
            # Linux/macOS: Following XDG Base Directory Specification
            # ~/.local/state/mmcp/logs (user data) or /var/log/mmcp (system)
            base_dir = Path.home() / ".local/state/mmcp"

        log_dir = base_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _setup_handlers(self):
        """Set up console and rotating file handlers."""
        # Prevent double logging if handlers already exist
        if self.logger.handlers:
            return

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # 1. Console Handler (Standard Output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

        # 2. Rotating File Handler (The "Self-Cleaner")
        try:
            log_file = self._get_log_dir() / "server.log"
            max_bytes = internal_settings["logging"]["max_bytes"]
            backup_count = internal_settings["logging"]["backup_count"]
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)  # File gets more detail
            self.logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            # If file logging fails, continue with console only
            # This ensures the app doesn't crash on log directory issues
            self.logger.warning(f"Could not set up file logging: {e}")
            self.logger.info("Continuing with console logging only")


# Global instance - MMCP applications should use this logger
logger = MMCPLogger().logger
