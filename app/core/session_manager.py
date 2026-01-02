"""
Session Manager - MMCP Core Infrastructure.

Session tracking infrastructure that supports ANP protocol by providing
session lifecycle management and validation. Not part of ANP spec itself,
but enables ANP's Address=SESSION functionality.
"""

from datetime import datetime, timedelta
from typing import ClassVar
from uuid import uuid4

from app.core.logger import logger


class SessionManager:
    """
    Session tracking infrastructure for MMCP.

    Provides session lifecycle management to support ANP's Address=SESSION
    functionality. Tracks active sessions and their last activity timestamps
    to enable session expiration promotion in the ANP protocol.

    For now, single-user system (user_id = "default").
    Future: Extend to multi-user when auth is added.
    """

    _instance: ClassVar["SessionManager | None"] = None

    def __new__(cls) -> "SessionManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize SessionManager (only once)."""
        if self._initialized:
            return

        # session_id → last_activity timestamp
        self.active_sessions: dict[str, datetime] = {}
        self._initialized = True
        logger.info("SessionManager initialized")

    def create_session(self) -> str:
        """
        Generate new session ID and store timestamp.

        Returns:
            New session ID (UUID string)
        """
        session_id = str(uuid4())
        self.active_sessions[session_id] = datetime.utcnow()
        logger.debug(f"Created session: {session_id}")
        return session_id

    def is_session_active(self, session_id: str, threshold_hours: int = 24) -> bool:
        """
        Check if session exists and is recent.

        Args:
            session_id: Session ID to check
            threshold_hours: Hours before session expires (default: 24)

        Returns:
            True if session is active, False otherwise
        """
        if session_id not in self.active_sessions:
            return False

        last_activity = self.active_sessions[session_id]
        age = datetime.utcnow() - last_activity

        if age > timedelta(hours=threshold_hours):
            # Session expired, remove it
            del self.active_sessions[session_id]
            logger.debug(f"Session expired: {session_id}")
            return False

        return True

    def update_session_activity(self, session_id: str) -> None:
        """Update last activity timestamp for session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id] = datetime.utcnow()

    def expire_old_sessions(self, threshold_hours: int = 24) -> int:
        """
        Cleanup sessions older than threshold.

        Args:
            threshold_hours: Hours before session expires (default: 24)

        Returns:
            Number of sessions expired
        """
        now = datetime.utcnow()
        expired = [
            sid
            for sid, last_activity in self.active_sessions.items()
            if (now - last_activity) > timedelta(hours=threshold_hours)
        ]

        for sid in expired:
            del self.active_sessions[sid]

        if expired:
            logger.debug(f"Expired {len(expired)} old session(s)")

        return len(expired)
