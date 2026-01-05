"""
Chat API routes for AG-UI.

Provides the main chat endpoint that handles user messages and HITL interruptions.
"""

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.agent.schemas import ActionRequestResponse
from app.core.errors import StaleApprovalError
from app.services.agent import AgentService


class ChatRequest(BaseModel):
    """Request model for chat messages."""

    message: str
    session_id: str | None = None


class Decision(str, Enum):
    """Decision for pending action."""

    APPROVE = "approve"
    DENY = "deny"


class ActionResponse(BaseModel):
    """Request to respond to a pending action."""

    session_id: str
    approval_id: str
    decision: Decision


router = APIRouter(prefix="/api/v1", tags=["chat"])


def get_agent_service(request: Request) -> AgentService:
    """Dependency to get AgentService instance from app state."""
    return request.app.state.agent_service


@router.post("/chat")
async def chat_endpoint(
    chat_request: ChatRequest, agent_service: AgentService = Depends(get_agent_service)
):
    """
    Main chat endpoint for AG-UI.

    Processes user messages and returns either regular responses or action_request interruptions.
    """

    if not chat_request.message:
        raise HTTPException(status_code=400, detail="Message is required")

    from app.core.logger import logger

    try:
        # Use the agent service instance from app state
        logger.info(
            f"Chat request received: message='{chat_request.message[:50]}...', "
            f"session_id={chat_request.session_id}"
        )
        response = await agent_service.process_message(
            chat_request.message, chat_request.session_id
        )

        # Check if this is an ActionRequestResponse (HITL interruption)
        if isinstance(response, ActionRequestResponse):
            # This is an interruption - return as JSON
            result = response.model_dump()
            logger.info(
                f"Returning ActionRequestResponse: event_id={result.get('event_id')}, "
                f"type={result.get('type')}"
            )
            # Use JSONResponse explicitly to ensure proper serialization
            return JSONResponse(
                status_code=status.HTTP_200_OK, content=result, media_type="application/json"
            )

        # Regular string response - return as JSON with expected format
        result = {"response": str(response), "type": "regular"}
        logger.info(
            f"Returning regular response: response length={len(str(response))}, "
            f"session_id={chat_request.session_id}"
        )
        # Use JSONResponse explicitly to ensure proper serialization
        return JSONResponse(
            status_code=status.HTTP_200_OK, content=result, media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}") from e


@router.post("/chat/respond")
async def respond_to_action(
    action_response: ActionResponse, agent_service: AgentService = Depends(get_agent_service)
):
    """
    Respond to a pending external tool action in chat.

    Approves or denies the action and continues the conversation.
    """
    from app.core.logger import logger

    try:
        logger.info(
            f"Action response received: session_id={action_response.session_id}, "
            f"approval_id={action_response.approval_id}, decision={action_response.decision}"
        )
        was_approved = action_response.decision == Decision.APPROVE
        # TODO: Implement resume_action in AgentService
        response = f"Action {'approved' if was_approved else 'denied'} - not yet implemented in new architecture"
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"response": str(response), "type": "regular"},
            media_type="application/json",
        )
    except StaleApprovalError as e:
        logger.error(f"Action response error: {e}", exc_info=True)
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Action response error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Action response failed: {str(e)}") from e
