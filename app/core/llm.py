"""LLM interface using LiteLLM and Instructor for structured outputs."""

import logging
from typing import Any

import instructor
import litellm
from litellm import acompletion

from app.core.config import settings
from app.core.logger import logger

# Hard-suppress LiteLLM verbose output to prevent "Provider List" spam
litellm.set_verbose = False
litellm.suppress_debug_info = True
litellm.drop_params = True  # Prevents "unsupported param" warnings
litellm.add_known_names = False  # Stops the "Did you mean...?" provider list logic
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)


def get_instructor_mode(model_name: str | None = None) -> instructor.Mode:
    """
    Get Instructor mode from config profile, with fallback to auto-detection.

    Args:
        model_name: Optional model identifier (uses settings.llm_model if not provided)

    Returns:
        Instructor Mode enum value
    """
    from app.core.config import default_profile, model_profiles, settings

    # Use provided model_name or fallback to settings
    model = model_name or settings.llm_model

    # Get profile for this model (or use default)
    profile = model_profiles.get(model, default_profile)

    # Check if profile has explicit instructor_mode
    if hasattr(profile, "reasoning") and hasattr(profile.reasoning, "instructor_mode"):
        mode_str = profile.reasoning.instructor_mode.lower()
        mode_map = {
            "tool_call": instructor.Mode.TOOLS,
            "json": instructor.Mode.JSON,
            "markdown_json": instructor.Mode.MD_JSON,
        }
        if mode_str in mode_map:
            return mode_map[mode_str]

    # Fallback to auto-detection based on model name
    model_lower = model.lower()
    if any(
        provider in model_lower for provider in ["gemini", "gpt-", "claude", "azure", "deepseek"]
    ):
        return instructor.Mode.TOOLS
    return instructor.Mode.JSON  # Fallback for Ollama/Local models


async def get_agent_decision(
    messages: list[dict],
    response_model: type[Any],
    trace_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[Any, Any]:
    """
    Sends the conversation history to the LLM and gets a structured decision.

    Uses Instructor to enforce that the LLM returns a response matching the provided
    response_model (typically a Union of FinalResponse and tool input schemas).

    This function wraps the Union in AgentTurn internally to fix the AttributeError
    with instructor.create_with_completion, then unwraps it before returning.

    Uses LiteLLM standard naming: 'provider/model' format (e.g., 'openai/gpt-4o' or 'ollama/llama3').

    Instructor's max_retries handles validation errors automatically by retrying the LLM call
    with the validation error fed back to the model.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        response_model: Pydantic model or Union type that the LLM should return.
        trace_id: Optional trace ID for logging and observability.
        temperature: Optional temperature override (uses profile default if not provided).
        max_tokens: Optional max_tokens override (uses profile default if not provided).

    Returns:
        Tuple of (parsed_object, raw_completion) where:
        - parsed_object: Instance of response_model (FinalResponse or a tool input schema)
        - raw_completion: Raw completion object from Instructor with tool_calls metadata

    Raises:
        ValueError: If LLM_MODEL is not set or if response is empty/invalid.
        Exception: Any LiteLLM or Instructor exceptions (will be mapped by orchestrator)
    """
    from typing import Union as TypingUnion
    from typing import get_origin

    from app.agent.schemas import AgentTurn
    from app.core.config import default_profile, model_profiles

    if not settings.llm_model:
        raise ValueError("LLM_MODEL must be set (e.g., 'openai/gpt-4o' or 'ollama/llama3')")

    # Get profile for this model (or use default)
    profile = model_profiles.get(settings.llm_model, default_profile)

    # Determine instructor mode dynamically per-call
    instructor_mode = get_instructor_mode(settings.llm_model)

    # Create client dynamically based on mode
    client = instructor.from_litellm(acompletion, mode=instructor_mode)

    # Wrap Union types in AgentTurn to fix AttributeError with instructor
    # Check if response_model is a Union type
    origin = get_origin(response_model)
    needs_unwrap = False
    if origin is not None and origin is TypingUnion:
        # Wrap the Union in AgentTurn using create_model
        from pydantic import create_model

        wrapped_model = create_model(
            "AgentTurnWrapper",
            action=(response_model, ...),
            __base__=AgentTurn,
        )
        actual_response_model = wrapped_model
        needs_unwrap = True
    else:
        # Not a Union, use as-is
        actual_response_model = response_model

    # Build kwargs: only pass what is explicitly provided
    kwargs = {
        "model": settings.llm_model,
        "messages": messages,
        "response_model": actual_response_model,
        "max_retries": 2,  # Instructor will retry on validation errors
        "num_retries": 0,  # Disable LiteLLM's built-in retries - we handle them in orchestrator
    }

    # Apply temperature (explicit override > profile > None)
    if temperature is not None:
        kwargs["temperature"] = temperature
    elif hasattr(profile, "reasoning") and hasattr(profile.reasoning, "temperature"):
        kwargs["temperature"] = profile.reasoning.temperature

    # Apply max_tokens (explicit override > profile > None)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    elif hasattr(profile, "reasoning") and hasattr(profile.reasoning, "max_tokens"):
        kwargs["max_tokens"] = profile.reasoning.max_tokens

    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key

    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url

    try:
        # Use create_with_completion to preserve tool_calls metadata
        parsed_object, raw_completion = await client.chat.completions.create_with_completion(
            **kwargs
        )

        # Validate response to prevent async hangs
        if raw_completion is None:
            raise ValueError("LLM returned empty response")

        if not hasattr(raw_completion, "choices") or not raw_completion.choices:
            raise ValueError("LLM response has no choices")

        if not raw_completion.choices[0].message:
            raise ValueError("LLM response message is empty")

        # Unwrap AgentTurn if we wrapped it
        if needs_unwrap and isinstance(parsed_object, AgentTurn):
            parsed_object = parsed_object.action

        return parsed_object, raw_completion
    except Exception as e:
        # Log the raw error with full context
        logger.error(
            f"LLM call failed (trace_id={trace_id}): {type(e).__name__}: {e}",
            exc_info=True,
            extra={
                "model": settings.llm_model,
                "message_count": len(messages),
                "trace_id": trace_id,
            }
            if trace_id
            else {"model": settings.llm_model, "message_count": len(messages)},
        )
        # Re-raise to be handled by orchestrator's error mapping
        raise


async def generate_dialogue(
    messages: list[dict],
    trace_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Generate a conversational dialogue response using the DialogueProfile.

    Used for generating rich, natural language responses to users after
    the reasoning phase has completed.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        trace_id: Optional trace ID for logging and observability.
        temperature: Optional temperature override (uses dialogue profile default if not provided).
        max_tokens: Optional max_tokens override (uses dialogue profile default if not provided).

    Returns:
        Generated dialogue text.

    Raises:
        ValueError: If LLM_MODEL is not set.
        Exception: Any LiteLLM exceptions (will be mapped by orchestrator)
    """
    from app.core.config import default_profile, model_profiles

    if not settings.llm_model:
        raise ValueError("LLM_MODEL must be set (e.g., 'openai/gpt-4o' or 'ollama/llama3')")

    # Get profile for this model (or use default)
    profile = model_profiles.get(settings.llm_model, default_profile)

    # Build kwargs
    kwargs = {
        "model": settings.llm_model,
        "messages": messages,
    }

    # Apply temperature (explicit override > profile > default)
    if temperature is not None:
        kwargs["temperature"] = temperature
    elif hasattr(profile, "dialogue") and hasattr(profile.dialogue, "temperature"):
        kwargs["temperature"] = profile.dialogue.temperature
    else:
        kwargs["temperature"] = 0.7  # Safe default

    # Apply max_tokens (explicit override > profile > default)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    elif hasattr(profile, "dialogue") and hasattr(profile.dialogue, "max_tokens"):
        kwargs["max_tokens"] = profile.dialogue.max_tokens
    else:
        kwargs["max_tokens"] = 2048  # Safe default

    if settings.llm_api_key:
        kwargs["api_key"] = settings.llm_api_key

    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url

    try:
        # Use raw litellm completion for dialogue generation (not structured)
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        # Log the raw error with full context
        logger.error(
            f"Dialogue generation failed (trace_id={trace_id}): {type(e).__name__}: {e}",
            exc_info=True,
            extra={
                "model": settings.llm_model,
                "message_count": len(messages),
                "trace_id": trace_id,
            }
            if trace_id
            else {"model": settings.llm_model, "message_count": len(messages)},
        )
        # Re-raise to be handled by orchestrator's error mapping
        raise
