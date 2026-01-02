"""
Pydantic schemas for ANP notifications.

Defines RoutingFlags, AgentInstructions, NotificationCreate, NotificationResponse,
and enums for type safety and validation.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Enums for routing flags
Address = Literal["session", "user"]
Target = Literal["user", "agent"]
Handler = Literal["system", "agent"]


class RoutingFlags(BaseModel):
    """Routing flags that determine channel and delivery behavior."""

    address: Address = Field(description="Spatial boundary: session or user")
    target: Target = Field(description="Initial receiver: user or agent")
    handler: Handler = Field(description="Processing responsibility: system or agent")


class AgentInstructions(BaseModel):
    """Optional instructions for the agent when handling notifications."""

    directive: str | None = Field(default=None, description="Optional directive string for the LLM")
    tone_hint: str | None = Field(
        default=None, description="Optional tone hint (e.g., 'concise', 'friendly')"
    )


class NotificationCreate(BaseModel):
    """Schema for creating a notification (used by plugins)."""

    content: str = Field(description="Human-readable notification content")
    deduplication_key: str | None = Field(
        default=None, description="Optional key for deduplication"
    )
    metadata: dict | None = Field(default=None, description="Optional metadata dictionary")
    routing: RoutingFlags = Field(description="Routing flags for channel determination")
    agent_instructions: AgentInstructions | None = Field(
        default=None, description="Optional agent instructions"
    )
    delivery_deadline: datetime | None = Field(
        default=None, description="Optional delivery deadline (for TTL monitoring)"
    )
    session_id: str | None = Field(
        default=None, description="Optional session ID (for Address=SESSION)"
    )
    user_id: str = Field(default="default", description="User ID (default: 'default')")


class NotificationResponse(BaseModel):
    """Schema for API responses (notification details)."""

    id: str
    content: str
    deduplication_key: str | None
    metadata: dict | None
    routing: RoutingFlags
    agent_instructions: AgentInstructions | None
    status: str
    owner_lease: int
    created_at: datetime
    ack_at: datetime | None
    delivery_deadline: datetime | None
    session_id: str | None
    user_id: str


class NotificationAck(BaseModel):
    """Schema for WebSocket ACK messages."""

    type: Literal["ack"] = "ack"
    id: str = Field(description="Notification ID to acknowledge")


class SessionCreate(BaseModel):
    """Schema for creating a new chat session."""

    pass  # No fields needed for basic session creation


class SessionResponse(BaseModel):
    """Schema for session information."""

    id: str = Field(description="Session ID")
    created_at: datetime = Field(description="Session creation timestamp")
