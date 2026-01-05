"""LLM interface using LiteLLM and Instructor for structured outputs."""

import logging
from typing import Any

import instructor
import litellm
from litellm import acompletion
from pydantic import BaseModel

from app.core.config import UserSettings, internal_settings, user_settings
from app.core.logger import logger

# Hard-suppress LiteLLM verbose output to prevent "Provider List" spam
litellm.set_verbose = False
litellm.suppress_debug_info = True
litellm.drop_params = True  # Prevents "unsupported param" warnings
litellm.add_known_names = False  # Stops the "Did you mean...?" provider list logic
logging.getLogger("LiteLLM").setLevel(logging.INFO)


def unwrap_response(obj: Any) -> Any:
    """
    Safely unwraps Instructor's synthetic 'Response' or 'AgentTurn' wrappers.

    In Mode.TOOLS, Instructor returns the selected tool/model directly without wrapping.
    In Mode.JSON, Instructor may wrap the Union in a synthetic 'Response' object with a single field.
    This function extracts the actual tool/model from the wrapper when needed, but only for
    known Instructor wrapper class names to prevent accidental unwrapping of legitimate
    single-field user models.
    """
    if obj is None or not isinstance(obj, BaseModel):
        return obj

    # Access model_fields from the class (not instance) to avoid Pydantic V2.11+ deprecation
    fields = list(obj.__class__.model_fields.keys())
    # Instructor wrappers usually have exactly one field and generic names
    if len(fields) == 1:
        class_name = obj.__class__.__name__
        if class_name in ("Response", "AgentTurn", "dynamic_response"):
            return getattr(obj, fields[0])
    return obj


def get_instructor_mode(user_settings: UserSettings) -> instructor.Mode:
    """
    Returns the user's configured mode.
    The settings system handles the defaults/fallbacks at the DB/Env level.
    """
    mode_str = user_settings.instructor_mode.lower()
    mode_map = {
        "tool_call": instructor.Mode.TOOLS,
        "tools": instructor.Mode.TOOLS,
        "json": instructor.Mode.JSON,
        "markdown_json": instructor.Mode.MD_JSON,
        "md_json": instructor.Mode.MD_JSON,
    }
    if mode_str in mode_map:
        return mode_map[mode_str]

    # Safe fallback for invalid values
    return instructor.Mode.JSON


async def get_agent_decision(
    messages: list[dict],
    response_model: type[Any] | list[type[Any]],
    trace_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[Any, Any]:
    """
    Executes a structured LLM call and returns both the parsed object and raw metadata.

    This function handles the conversion of multiple tool schemas into a single Union
    and uses Instructor's event system to capture LiteLLM completion metadata (like usage
    and tool_call_ids) without triggering common library-level attribute errors.

    Args:
        messages: Conversation history in OpenAI-style format.
        response_model: A single Pydantic model or a list of models for the LLM to choose from.
        trace_id: Optional UUID for cross-service log correlation.
        temperature: LLM temperature (overrides profile default).
        max_tokens: LLM token limit (overrides profile default).

    Returns:
        tuple[parsed_object, raw_completion]:
            - parsed_object: The validated Pydantic instance selected by the LLM.
            - raw_completion: The full LiteLLM ModelResponse object.

    Raises:
        ValueError: For configuration or transport errors.
        InstructorRetryException: On maximum validation failures.
    """
    from typing import Union

    from app.core.config import default_profile, model_profiles

    if not user_settings.llm_model:
        raise ValueError("LLM_MODEL must be set (e.g., 'openai/gpt-4o' or 'ollama/llama3')")

    # Get profile for this model (or use default)
    profile = model_profiles.get(user_settings.llm_model, default_profile)

    # Use the User's preferred mode (mode-agnostic - works for all Instructor modes)
    instructor_mode = get_instructor_mode(user_settings)
    client = instructor.from_litellm(acompletion, mode=instructor_mode)

    # Convert list to direct Union (DO NOT use Iterable)
    # In Mode.TOOLS, Union is treated as the toolset - model picks one, Instructor returns that single object
    # Instructor automatically converts Union members to native function calling format
    if isinstance(response_model, list):
        if not response_model:
            raise ValueError("response_model list cannot be empty")
        elif len(response_model) == 1:
            actual_response_model = response_model[0]
        else:
            # Programmatically create Union[ToolA, ToolB, FinalResponse]
            # Use Union.__getitem__ for proper dynamic Union creation (Python 3.10+)
            actual_response_model = Union.__getitem__(tuple(response_model))
    else:
        actual_response_model = response_model

    # Build kwargs: only pass what is explicitly provided
    kwargs = {
        "model": user_settings.llm_model,
        "messages": messages,
        "response_model": actual_response_model,
        "max_retries": internal_settings["react_loop"]["instructor_max_retries"],
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

    if user_settings.llm_api_key:
        kwargs["api_key"] = user_settings.llm_api_key

    if user_settings.llm_base_url:
        kwargs["base_url"] = user_settings.llm_base_url

    # Hook-based metadata capture (avoids AttributeError with Union types)
    raw_completion = None

    def capture_metadata(completion: Any) -> None:
        """Hook: Captures the raw LiteLLM completion before Instructor parses it."""
        nonlocal raw_completion
        raw_completion = completion

    # Register the hook to capture raw completion
    # Hooks are required for metadata capture - fail fast if unavailable
    if hasattr(client, "on"):
        try:
            client.on("completion:response", capture_metadata)
        except (AttributeError, TypeError) as e:
            raise ValueError(
                "Instructor hooks not available: required for metadata capture. "
                f"Client does not support 'on' method: {e}"
            ) from e
    elif hasattr(client.chat.completions, "on"):
        try:
            client.chat.completions.on("completion:response", capture_metadata)
        except (AttributeError, TypeError) as e:
            raise ValueError(
                "Instructor hooks not available: required for metadata capture. "
                f"Client.chat.completions does not support 'on' method: {e}"
            ) from e
    else:
        raise ValueError(
            "Instructor hooks not available: required for metadata capture. "
            "Client does not support hook registration via 'on' method."
        )

    try:
        # Use create() instead of create_with_completion() to avoid AttributeError
        # The hook captures the raw completion metadata
        parsed_output = await client.chat.completions.create(**kwargs)

        # Unwrap in case JSON mode added a synthetic wrapper
        parsed_object = unwrap_response(parsed_output)

        if parsed_object is None:
            raise ValueError("LLM returned no valid objects")

        # Verify hook captured metadata (should always succeed if hook registration succeeded)
        if raw_completion is None or not hasattr(raw_completion, "choices"):
            raise ValueError(
                "Transport Error: Hook did not capture raw completion metadata. "
                "This indicates an Instructor internal error."
            )

        return parsed_object, raw_completion
    except Exception as e:
        # Log the actual error for debugging
        logger.error(
            f"LLM call failed (trace_id={trace_id}): {type(e).__name__}: {e}",
            exc_info=True,
            extra={
                "model": user_settings.llm_model,
                "message_count": len(messages),
                "trace_id": trace_id,
            }
            if trace_id
            else {"model": user_settings.llm_model, "message_count": len(messages)},
        )
        # Re-raise to be handled by orchestrator's error mapping
        raise
    finally:
        # Always clean up hooks to prevent memory leaks in long-running servers
        try:
            if hasattr(client, "off"):
                client.off("completion:response", capture_metadata)
            elif hasattr(client.chat.completions, "off"):
                client.chat.completions.off("completion:response", capture_metadata)
        except (AttributeError, TypeError):
            pass  # Hook cleanup not available, continue


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

    if not user_settings.llm_model:
        raise ValueError("LLM_MODEL must be set (e.g., 'openai/gpt-4o' or 'ollama/llama3')")

    # Get profile for this model (or use default)
    profile = model_profiles.get(user_settings.llm_model, default_profile)

    # Build kwargs
    kwargs = {
        "model": user_settings.llm_model,
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

    if user_settings.llm_api_key:
        kwargs["api_key"] = user_settings.llm_api_key

    if user_settings.llm_base_url:
        kwargs["base_url"] = user_settings.llm_base_url

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
                "model": user_settings.llm_model,
                "message_count": len(messages),
                "trace_id": trace_id,
            }
            if trace_id
            else {"model": user_settings.llm_model, "message_count": len(messages)},
        )
        # Re-raise to be handled by orchestrator's error mapping
        raise


def safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """
    Hardened retrieval for LiteLLM's hybrid response objects.

    LiteLLM ModelResponse is a hybrid dict/object. Depending on the provider
    (e.g. DeepSeek via OpenRouter), attribute access can fail where key access succeeds.
    This function prioritizes key access for better compatibility with raw dict responses.

    Args:
        obj: The object to access (may be dict, object, or hybrid)
        attr: The attribute/key name to retrieve
        default: Default value if attribute/key not found

    Returns:
        The attribute value, dict value, or default
    """
    if obj is None:
        return default

    try:
        # Prioritize key access for LiteLLM/OpenRouter responses
        # Many providers return raw dicts that fail hasattr() but succeed with .get()
        if isinstance(obj, dict):
            return obj.get(attr, default)
        # Fallback to attribute access for object-like responses
        return getattr(obj, attr, default)
    except Exception:
        return default
