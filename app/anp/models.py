"""
Database models for ANP Event Ledger.

Defines the EventLedger SQLAlchemy model with all ANP fields, indexes, and state machine enums.
"""

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EventStatus(str, Enum):
    """ANP state machine status enum."""

    PENDING = "pending"
    DISPATCHED = "dispatched"
    LOCKED = "locked"
    DELIVERED = "delivered"
    ESCALATED = "escalated"
    FAILED = "failed"


class EventLedger(Base):
    """
    Event Ledger model for ANP notifications.

    Stores all notification events with routing flags, state machine status,
    and lease-based ownership fencing.
    """

    __tablename__ = "event_ledger"

    # Primary key
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # Content fields
    content: Mapped[str] = mapped_column(Text, nullable=False)
    deduplication_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    event_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=lambda: {})

    # Routing flags (stored as JSON)
    routing: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Agent instructions (stored as JSON)
    agent_instructions: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Pending action data for HITL (stored as JSON)
    pending_action_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # State machine
    status: Mapped[EventStatus] = mapped_column(
        String(20), nullable=False, default=EventStatus.PENDING, index=True
    )

    # Lease-based fencing
    owner_lease: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    ack_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    delivery_deadline: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    # Address fields
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, default="default", index=True)

    def __repr__(self) -> str:
        return (
            f"<EventLedger(id={self.id!r}, status={self.status.value}, "
            f"user_id={self.user_id!r}, session_id={self.session_id!r})>"
        )


class ChatSession(Base):
    """
    Chat session persistence for stateful conversations.

    Stores conversation history and state for HITL interruptions.
    """

    __tablename__ = "chat_sessions"

    # Primary key (same as session_id)
    id: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Custom title (nullable, falls back to first user message if not set)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Conversation history (stored as JSON)
    history: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Pending action for conversational HITL (stored as JSON)
    pending_action: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id!r}, history_length={len(self.history)})>"


# Composite indexes for performance
Index("ix_event_ledger_user_status", EventLedger.user_id, EventLedger.status)
