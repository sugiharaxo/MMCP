"""
Sessions API routes for session management.

Provides paginated session listing with cursor-based pagination.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.anp.models import ChatSession
from app.core.database import get_session

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


class SessionListItem(BaseModel):
    """Schema for session list item."""

    id: str = Field(description="Session ID")
    title: str = Field(description="Title (first 50 chars of first user message)")
    updated_at: datetime = Field(description="Last update timestamp")


@router.get("", response_model=list[SessionListItem])
async def list_sessions(
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query()] = None,
):
    """
    List chat sessions with cursor-based pagination.

    Args:
        limit: Maximum number of sessions to return (1-100, default: 50)
        cursor: ISO timestamp cursor for pagination (exclusive). If provided,
                returns sessions updated before this timestamp.

    Returns:
        List of SessionListItem with id, title, and updated_at
    """
    async with get_session() as session:
        stmt = select(ChatSession).order_by(ChatSession.updated_at.desc())

        # Apply cursor if provided
        if cursor:
            try:
                cursor_dt = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
                stmt = stmt.where(ChatSession.updated_at < cursor_dt)
            except ValueError:
                # Invalid cursor format, return empty list
                return []

        stmt = stmt.limit(limit)

        result = await session.execute(stmt)
        chat_sessions = result.scalars().all()

        # Convert to response format
        sessions = []
        for chat_session in chat_sessions:
            # Use custom title if set, otherwise extract from first user message
            title = chat_session.title
            if not title and chat_session.history:
                for msg in chat_session.history:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            title = content[:50]
                        elif isinstance(content, dict):
                            # Handle dict content (e.g., FinalResponse)
                            title = str(content)[:50]
                        break
            if not title:
                title = "New Chat"

            sessions.append(
                SessionListItem(
                    id=chat_session.id,
                    title=title,
                    updated_at=chat_session.updated_at,
                )
            )

        return sessions


class SessionUpdateRequest(BaseModel):
    """Schema for updating a session."""

    title: str = Field(description="New title for the session", max_length=255)


@router.patch("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_session(session_id: str, request: SessionUpdateRequest):
    """
    Update a session's title.

    Args:
        session_id: Session ID to update
        request: Update request with new title

    Raises:
        HTTPException: 404 if session not found
    """
    async with get_session() as session:
        stmt = select(ChatSession).where(ChatSession.id == session_id)
        result = await session.execute(stmt)
        chat_session = result.scalar_one_or_none()

        if not chat_session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        # Preserve the updated_at timestamp when only renaming (don't update it)
        current_updated_at = chat_session.updated_at
        chat_session.title = request.title
        chat_session.updated_at = current_updated_at
        await session.commit()


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """
    Delete a session.

    Args:
        session_id: Session ID to delete

    Raises:
        HTTPException: 404 if session not found
    """
    async with get_session() as session:
        # Check if session exists first
        stmt = select(ChatSession).where(ChatSession.id == session_id)
        result = await session.execute(stmt)
        chat_session = result.scalar_one_or_none()

        if not chat_session:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

        await session.delete(chat_session)
        await session.commit()
