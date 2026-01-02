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


def get_instructor_mode(model_name: str) -> instructor.Mode:
    """
    Auto-detect Instructor mode based on model provider.

    Mode.TOOLS for cloud models (Gemini, GPT, Claude, Azure) - uses native tool calling API.
    Mode.JSON for local models (Ollama) - uses JSON output in message content.

    Args:
        model_name: Model identifier in LiteLLM format (e.g., 'gemini/gemini-1.5-flash')

    Returns:
        Instructor Mode enum value
    """
    model_lower = model_name.lower()
    if any(provider in model_lower for provider in ["gemini", "gpt-", "claude", "azure"]):
        return instructor.Mode.TOOLS
    return instructor.Mode.JSON  # Fallback for Ollama/Local models


# Patch LiteLLM async completion with Instructor for structured outputs
# from_litellm returns an AsyncInstructor client object
# Mode is auto-detected based on the configured model
client = instructor.from_litellm(acompletion, mode=get_instructor_mode(settings.llm_model))


async def get_agent_decision(
    messages: list[dict],
    response_model: type[Any],
    trace_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """
    Sends the conversation history to the LLM and gets a structured decision.

    Uses Instructor to enforce that the LLM returns a response matching the provided
    response_model (typically a Union of FinalResponse and tool input schemas).

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
        Instance of response_model (FinalResponse or a tool input schema).

    Raises:
        ValueError: If LLM_MODEL is not set.
        Exception: Any LiteLLM or Instructor exceptions (will be mapped by orchestrator)
    """
    from app.core.config import default_profile, model_profiles

    if not settings.llm_model:
        raise ValueError("LLM_MODEL must be set (e.g., 'openai/gpt-4o' or 'ollama/llama3')")

    # Get profile for this model (or use default)
    profile = model_profiles.get(settings.llm_model, default_profile)

    # Build kwargs: only pass what is explicitly provided
    kwargs = {
        "model": settings.llm_model,
        "messages": messages,
        "response_model": response_model,
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
        # AsyncInstructor client uses OpenAI-compatible interface: client.chat.completions.create()
        return await client.chat.completions.create(**kwargs)
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
