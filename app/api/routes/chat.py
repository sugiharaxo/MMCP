"""
Chat API routes for AG-UI.

Provides the main chat endpoint that handles user messages and HITL interruptions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agent.orchestrator import AgentOrchestrator
from app.agent.schemas import ActionRequestResponse


class ChatRequest(BaseModel):
    """Request model for chat messages."""

    message: str
    session_id: str | None = None


router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint for AG-UI.

    Processes user messages and returns either regular responses or action_request interruptions.
    """

    if not chat_request.message:
        raise HTTPException(status_code=400, detail="Message is required")

    try:
        # Create orchestrator and process the message
        orchestrator = AgentOrchestrator()
        response = await orchestrator.chat(chat_request.message, chat_request.session_id)

        # Check if this is an ActionRequestResponse (HITL interruption)
        if isinstance(response, ActionRequestResponse):
            # This is an interruption - return the action request data
            return response.model_dump()

        # Regular string response
        return {"response": response, "type": "regular"}

    except Exception as e:
        from app.core.logger import logger

        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}") from e
