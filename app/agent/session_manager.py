"""Session Management - handles concurrency control and persistence."""

import asyncio
from typing import Any
from weakref import WeakValueDictionary

from sqlalchemy import select

from app.agent.history_manager import HistoryManager, HistoryMessage
from app.anp.models import ChatSession
from app.core.database import get_session
from app.core.logger import logger


class SessionLockManager:
    """Manages per-session locks using WeakValueDictionary for automatic cleanup."""

    def __init__(self):
        # Maps session_id -> Lock. Entry vanishes when no request is using the lock.
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a specific session. Synchronous for atomicity."""
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock


class SessionManager:
    """Handles session persistence and state management."""

    def __init__(self):
        self.lock_manager = SessionLockManager()

    async def load_session(self, session_id: str) -> list[HistoryMessage]:
        """
        Load conversation history from database for session resumption.

        Args:
            session_id: Session ID to load

        Returns:
            List of HistoryMessage objects (empty list if session doesn't exist)
        """
        async with get_session() as session:
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await session.execute(stmt)
            chat_session = result.scalar_one_or_none()

            if chat_session:
                # Defensive check: handle None history from database
                if chat_session.history is not None:
                    history_dict = chat_session.history.copy()
                else:
                    history_dict = []
                    logger.warning(f"Session {session_id} had None history, initializing empty")
                logger.debug(f"Loaded session {session_id} with {len(history_dict)} messages")
                return HistoryManager.from_dict_list(history_dict)
            else:
                # Initialize new session
                logger.debug(f"Initialized new session {session_id}")
                return []

    async def load_pending_action(self, session_id: str) -> dict[str, Any] | None:
        """
        Load pending action from database for HITL resumption.

        Args:
            session_id: Session ID to load

        Returns:
            Pending action dictionary or None if not found
        """
        async with get_session() as session:
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await session.execute(stmt)
            chat_session = result.scalar_one_or_none()

            if chat_session and chat_session.pending_action:
                logger.debug(
                    f"Loaded pending action for session {session_id}: "
                    f"{chat_session.pending_action.get('approval_id')}"
                )
                return chat_session.pending_action.copy()
            return None

    async def save_session(self, session_id: str, history: list[HistoryMessage]) -> None:
        """
        Save conversation history to database.

        Args:
            session_id: Session ID to save
            history: The history list to save (HistoryMessage objects)
        """
        # Normalize to dict list for persistence
        history_dict = HistoryManager.to_dict_list(history)

        async with get_session() as session:
            # Try to update existing session
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await session.execute(stmt)
            chat_session = result.scalar_one_or_none()

            if chat_session:
                # Update existing
                chat_session.history = history_dict.copy()
            else:
                # Create new
                chat_session = ChatSession(id=session_id, history=history_dict.copy())
                session.add(chat_session)

            await session.commit()
            logger.debug(f"Saved session {session_id} with {len(history)} messages")

    async def save_session_with_pending_action(
        self, session_id: str, history: list[HistoryMessage], pending_action: dict[str, Any]
    ) -> None:
        """Save session with pending action for HITL."""
        # Normalize to dict list for persistence
        history_dict = HistoryManager.to_dict_list(history)

        async with get_session() as session:
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await session.execute(stmt)
            chat_session = result.scalar_one_or_none()

            if chat_session:
                chat_session.history = history_dict.copy()
                chat_session.pending_action = pending_action
            else:
                chat_session = ChatSession(
                    id=session_id, history=history_dict.copy(), pending_action=pending_action
                )
                session.add(chat_session)

            await session.commit()
            logger.debug(
                f"Saved session {session_id} with pending action: {pending_action.get('approval_id')}"
            )
