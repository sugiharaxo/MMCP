"""LLM interface using LiteLLM and Instructor for structured outputs."""

import logging
from typing import Any

import instructor
import litellm
from litellm import acompletion

from app.core.config import settings

logger = logging.getLogger("mmcp.llm")

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


async def get_agent_decision(messages: list[dict], response_model: type[Any]) -> Any:
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

    Returns:
        Instance of response_model (FinalResponse or a tool input schema).

    Raises:
        ValueError: If LLM_MODEL is not set.
        Exception: Any LiteLLM or Instructor exceptions (will be mapped by orchestrator)
    """
    if not settings.llm_model:
        raise ValueError("LLM_MODEL must be set (e.g., 'openai/gpt-4o' or 'ollama/llama3')")

    # Build kwargs: only pass what is explicitly provided
    kwargs = {
        "model": settings.llm_model,
        "messages": messages,
        "response_model": response_model,
        "max_retries": 2,  # Instructor will retry on validation errors
        "num_retries": 0,  # Disable LiteLLM's built-in retries - we handle them in orchestrator
    }

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
            f"LLM call failed: {type(e).__name__}: {e}",
            exc_info=True,
            extra={"model": settings.llm_model, "message_count": len(messages)},
        )
        # Re-raise to be handled by orchestrator's error mapping
        raise
