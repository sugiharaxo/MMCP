"""LLM Interface - unified interface for all LLM operations."""

from typing import Any

from app.core.config import default_profile, model_profiles, user_settings
from app.core.llm import generate_dialogue, get_agent_decision
from app.core.logger import logger


class LLMInterface:
    """Unified interface for LLM operations."""

    async def get_reasoned_decision(
        self,
        messages: list[dict[str, Any]],
        response_model: Any,
        trace_id: str | None = None,
    ) -> tuple[Any, Any]:
        """
        Get structured decision from LLM using ReasoningProfile.

        This is the first turn in the single-turn reasoned flow, using
        low temperature and token count for focused, consistent decision-making.

        Args:
            messages: Conversation messages
            response_model: Pydantic model for structured output (typically Union[FinalResponse, flattened tool schemas])
            trace_id: Optional trace ID for logging

        Returns:
            Tuple of (parsed_object, raw_completion) where:
            - parsed_object: Structured response instance (FinalResponse or flattened tool schema)
            - raw_completion: Raw completion object with tool_calls metadata
        """
        logger.info(
            f"Getting reasoned decision (trace_id={trace_id})",
            extra={"trace_id": trace_id} if trace_id else {},
        )

        # Get profile for this model (or use default)
        profile = model_profiles.get(user_settings.llm_model, default_profile)

        # Use reasoning profile settings
        temperature = profile.reasoning.temperature if hasattr(profile, "reasoning") else None
        max_tokens = profile.reasoning.max_tokens if hasattr(profile, "reasoning") else None

        return await get_agent_decision(
            messages,
            response_model=response_model,
            trace_id=trace_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_dialogue_response(
        self, messages: list[dict[str, Any]], trace_id: str | None = None
    ) -> str:
        """
        Generate a conversational dialogue response using DialogueProfile.

        Used for generating rich, natural language responses to users after
        the reasoning phase has completed. This is the second turn in the flow
        when a FinalResponse is returned from reasoning.

        Args:
            messages: Conversation messages
            trace_id: Optional trace ID for logging

        Returns:
            Generated dialogue text
        """
        logger.info(
            f"Generating dialogue response (trace_id={trace_id})",
            extra={"trace_id": trace_id} if trace_id else {},
        )

        # Get profile for this model (or use default)
        profile = model_profiles.get(user_settings.llm_model, default_profile)

        # Use dialogue profile settings
        temperature = profile.dialogue.temperature if hasattr(profile, "dialogue") else None
        max_tokens = profile.dialogue.max_tokens if hasattr(profile, "dialogue") else None

        return await generate_dialogue(
            messages, trace_id=trace_id, temperature=temperature, max_tokens=max_tokens
        )

    async def get_agent_decision(
        self, messages: list[dict[str, Any]], response_model: Any, trace_id: str | None = None
    ) -> tuple[Any, Any]:
        """
        Get structured decision from LLM (legacy method, maintained for compatibility).

        For new code, use get_reasoned_decision() instead.

        Args:
            messages: Conversation messages
            response_model: Pydantic model for structured output
            trace_id: Optional trace ID for logging

        Returns:
            Tuple of (parsed_object, raw_completion)
        """
        logger.info(
            f"Getting agent decision (trace_id={trace_id})",
            extra={"trace_id": trace_id} if trace_id else {},
        )

        return await get_agent_decision(messages, response_model=response_model, trace_id=trace_id)
