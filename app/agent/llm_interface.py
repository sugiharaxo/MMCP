"""LLM Interface - unified interface for all LLM operations."""

from typing import Any

import litellm

from app.core.config import settings
from app.core.llm import get_agent_decision
from app.core.logger import logger


class LLMInterface:
    """Unified interface for LLM operations."""

    async def generate_hitl_explanation(
        self, messages: list[dict[str, Any]], trace_id: str | None = None
    ) -> str:
        """
        Generate a user-friendly explanation for HITL (Human-in-the-Loop) tool approvals.

        When the agent needs user approval for an external tool, this generates
        a clear, contextual explanation of what the tool will do.

        Args:
            messages: Conversation messages including specialized system prompt
            trace_id: Optional trace ID for logging

        Returns:
            User-friendly explanation text
        """
        from app.core.errors import map_provider_error

        logger.info(
            f"Generating text response (trace_id={trace_id})",
            extra={"trace_id": trace_id} if trace_id else {},
        )

        try:
            # Use raw litellm completion for simple text generation (not structured)
            response = await litellm.acompletion(
                messages=messages,
                model=settings.llm_model,
                temperature=0.7,
                max_tokens=500,
            )

            return response.choices[0].message.content

        except Exception as e:
            # Map and log provider errors consistently with project patterns
            provider_error = map_provider_error(e, trace_id)
            logger.error(
                f"Text generation failed (trace_id={trace_id}): {provider_error.message}",
                exc_info=True,
                extra={"trace_id": trace_id} if trace_id else {},
            )
            # Return a fallback message
            return "I encountered an issue generating a response. Please try again."

    async def get_agent_decision(
        self, messages: list[dict[str, Any]], response_model: Any, trace_id: str | None = None
    ) -> Any:
        """
        Get structured decision from LLM.

        Args:
            messages: Conversation messages
            response_model: Pydantic model for structured output
            trace_id: Optional trace ID for logging

        Returns:
            Structured response instance
        """
        logger.info(
            f"Getting agent decision (trace_id={trace_id})",
            extra={"trace_id": trace_id} if trace_id else {},
        )

        return await get_agent_decision(messages, response_model=response_model)
