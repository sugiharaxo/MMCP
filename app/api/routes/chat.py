"""
Chat API routes for AG-UI.

Provides the main chat endpoint that handles user messages and HITL interruptions.
"""

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import HitlDecision
from app.core.errors import StaleApprovalError
from app.services.agent import AgentService


class ChatRequest(BaseModel):
    """Request model for chat messages."""

    message: str
    session_id: str | None = None


class ActionResponse(BaseModel):
    """
    Request to respond to a pending action (HITL approval/denial).

    TEMPORARY LIMITATION: session_id is currently required (non-nullable).
    TODO: Make session_id nullable to support global-scope actions (ANP Address=USER).

    Context: Per ANP spec Section 2.1, actions can be escalated to global scope when
    a session-bound action's session expires. Global actions have session_id=null and
    are stored in EventLedger.pending_action_data rather than ChatSession.pending_action.

    Required changes:
    1. Change session_id: str | None = None
    2. Update resume_action_stream() to handle null session_id
    3. Query EventLedger by approval_id when session_id is null
    4. Use EventLedger.pending_action_data for global actions instead of ChatSession
    """

    session_id: str  # TODO: Make nullable (str | None) for global-scope actions
    approval_id: str
    decision: HitlDecision


router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_agent_service(request: Request) -> AgentService:
    """Dependency to get AgentService instance from app state."""
    return request.app.state.agent_service


@router.post("/chat")
async def chat_endpoint(
    chat_request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
):
    """
    Main chat endpoint for AG-UI with streaming support.

    Processes user messages and streams BAML chunks as NDJSON for real-time UI updates.
    """

    if not chat_request.message:
        raise HTTPException(status_code=400, detail="Message is required")

    from app.core.logger import logger

    async def event_generator():
        """Generator that yields BAML chunks as NDJSON."""
        try:
            logger.info(
                f"Chat request received (streaming): message='{chat_request.message[:50]}...', "
                f"session_id={chat_request.session_id}"
            )
            async for chunk in agent_service.process_message_stream(
                chat_request.message, chat_request.session_id
            ):
                # Yield each chunk as a JSON line (NDJSON format)
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            logger.error(f"Chat streaming error: {e}", exc_info=True)
            # Yield error as final chunk
            error_chunk = {
                "response": f"Chat processing failed: {str(e)}",
                "type": "error",
                "session_id": chat_request.session_id or "unknown",
            }
            yield json.dumps(error_chunk) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.post("/chat/respond")
async def respond_to_action(
    action_response: ActionResponse, agent_service: AgentService = Depends(get_agent_service)
):
    """
    Respond to a pending external tool action in chat with streaming support.

    Approves or denies the action and continues the conversation with streaming
    for real-time UI updates.

    TEMPORARY LIMITATION: Only supports session-scoped actions.
    TODO: Add support for global-scope actions (session_id=null):
    - Query EventLedger by approval_id when session_id is null
    - Use EventLedger.pending_action_data instead of ChatSession.pending_action
    - Resume agent execution in global context (no session history)

    See: docs/specs/anp-v1.0.md Section 2.1 (Address escalation)
    """
    from app.core.logger import logger

    async def event_generator():
        """Generator that yields BAML chunks as NDJSON."""
        try:
            logger.info(
                f"Action response received (streaming): session_id={action_response.session_id}, "
                f"approval_id={action_response.approval_id}, decision={action_response.decision}"
            )
            async for chunk in agent_service.resume_action_stream(
                session_id=action_response.session_id,
                approval_id=action_response.approval_id,
                decision=action_response.decision,
            ):
                # Yield each chunk as a JSON line (NDJSON format)
                yield json.dumps(chunk) + "\n"
        except StaleApprovalError as e:
            logger.error(f"Action response error: {e}", exc_info=True)
            error_chunk = {
                "response": str(e),
                "type": "error",
                "session_id": action_response.session_id,
            }
            yield json.dumps(error_chunk) + "\n"
        except Exception as e:
            logger.error(f"Action response error: {e}", exc_info=True)
            error_chunk = {
                "response": f"Action response failed: {str(e)}",
                "type": "error",
                "session_id": action_response.session_id,
            }
            yield json.dumps(error_chunk) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


def _extract_message_content(msg) -> str:
    """Extract displayable content from a HistoryMessage."""
    if isinstance(msg.content, str):
        return msg.content
    elif isinstance(msg.content, dict):
        if msg.type == "final_response":
            return msg.content.get("answer", str(msg.content))
        elif msg.type == "tool_call":
            # Use correct field names from BAML Pydantic models
            tool_name = msg.content.get("tool_name", "unknown")
            args = msg.content.get("args", {})
            return f"Tool call: {tool_name}({json.dumps(args)})"
        else:
            # Fallback for unexpected dict content
            return f"[Structured content: {json.dumps(msg.content)}]"
    else:
        return str(msg.content)


def should_show_in_ui(msg) -> bool:
    """
    Determine if a message should be shown in the UI.

    Rule: Only show messages intended for the Human (User bubbles, Agent bubbles, System Pills).
    Do NOT show ToolResults (Observations) or raw ToolCalls - these are for Agent's internal context only.
    """
    # Hide tool results - these are observations for the agent, not user-facing
    if msg.tool_result:
        return False

    # Hide raw tool calls - these are internal agent reasoning, not user-facing
    # (EXTERNAL tool calls become action_requests, INTERNAL tool calls become system pills via notifications)
    if msg.type == "tool_call":
        return False

    # Show everything else (user messages, assistant final responses, system messages, ANP notifications)
    return True


def _determine_sender(msg) -> str:
    """Determine the sender type for frontend display."""
    if msg.role == "system":
        return "system"
    elif msg.role == "user":
        return "user"
    else:  # assistant
        return "agent"


def _determine_handler(msg) -> str:
    """Determine the handler type for frontend display."""
    return "system" if (msg.anp_event or msg.tool_result) else "agent"


@router.get("/chat/history")
async def get_chat_history(
    session_id: Annotated[str, Query(description="Session ID to fetch history for")],
    agent_service: AgentService = Depends(get_agent_service),
):
    """
    Get chat history for a session.

    Returns the conversation history formatted for frontend display.
    """
    from app.core.logger import logger

    try:
        history = await agent_service.session_manager.load_session(session_id)

        # Convert HistoryMessage to frontend format
        # Filter out tool results and raw tool calls - these are for agent's internal context only
        messages = []
        for msg in history:
            # Skip messages that should not be shown in UI (tool results, raw tool calls)
            if not should_show_in_ui(msg):
                continue

            content = _extract_message_content(msg)
            sender = _determine_sender(msg)
            handler = _determine_handler(msg)

            message_data = {
                "id": str(uuid.uuid4()),  # Generate ID for frontend
                "sender": sender,
                "content": content,
                "handler": handler,
            }

            # Include action_request data if this is an action request
            if msg.type == "action_request" and isinstance(msg.content, dict):
                message_data["type"] = "action_request"
                message_data["approval_id"] = msg.approval_id
                message_data["tool_name"] = msg.content.get("tool_name")
                message_data["tool_args"] = msg.content.get("tool_args", {})
                message_data["explanation"] = msg.content.get("explanation", "")
                message_data["action_status"] = msg.action_status or "pending"

            messages.append(message_data)

        logger.debug(f"Fetched history for session {session_id}: {len(messages)} messages")
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Failed to fetch history for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}") from e
