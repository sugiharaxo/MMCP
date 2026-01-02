"""
Session Manager - MMCP Core Infrastructure.

Session tracking infrastructure that supports ANP protocol by providing
session lifecycle management and validation. Not part of ANP spec itself,
but enables ANP's Address=SESSION functionality.

Sessions represent open chat conversations. They exist until explicitly
deleted when the conversation ends.
"""

from typing import ClassVar
from uuid import uuid4

from app.core.logger import logger


class SessionManager:
    """
    Session tracking infrastructure for MMCP.

    Manages chat session lifecycles for ANP's Address=SESSION functionality.
    Sessions represent open conversations and exist until explicitly deleted.

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

        # Set of active session IDs (representing open chats)
        self.active_sessions: set[str] = set()
        self._initialized = True
        logger.info("SessionManager initialized")

    def create_session(self) -> str:
        """
        Create a new chat session.

        Returns:
            New session ID (UUID string)
        """
        session_id = str(uuid4())
        self.active_sessions.add(session_id)
        logger.debug(f"Created session: {session_id}")
        return session_id

    def is_session_active(self, session_id: str) -> bool:
        """
        Check if session exists (i.e., chat is still open).

        Args:
            session_id: Session ID to check

        Returns:
            True if session exists, False otherwise
        """
        return session_id in self.active_sessions

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session (called when chat closes).

        Args:
            session_id: Session ID to delete
        """
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
            logger.debug(f"Deleted session: {session_id}")
        else:
            logger.warning(f"Attempted to delete non-existent session: {session_id}")
