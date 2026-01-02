"""
Centralized error handling for MMCP.

Provides exception hierarchy, error mapping utilities, and structured error responses.
Follows "Log Deep, Report Shallow" principle.
"""

import uuid

from pydantic import BaseModel

from app.core.logger import logger

# --- Exception Hierarchy ---


class MMCPError(Exception):
    """Base exception for all MMCP errors."""

    def __init__(self, message: str, retryable: bool = False, trace_id: str | None = None):
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.trace_id = trace_id or str(uuid.uuid4())


class ProviderError(MMCPError):
    """Issues with LiteLLM/Model providers (Authentication, Rate Limits, Context Window)."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        provider: str | None = None,
        trace_id: str | None = None,
    ):
        super().__init__(message, retryable=retryable, trace_id=trace_id)
        self.provider = provider


class ToolError(MMCPError):
    """Failures inside a plugin/tool (File not found, FFmpeg crash, etc.)."""

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        retryable: bool = False,
        trace_id: str | None = None,
    ):
        super().__init__(message, retryable=retryable, trace_id=trace_id)
        self.tool_name = tool_name


class AgentLogicError(MMCPError):
    """Failures in the ReAct loop (LLM returned invalid JSON, max iterations reached)."""

    def __init__(
        self,
        message: str,
        retryable: bool = False,
        trace_id: str | None = None,
    ):
        super().__init__(message, retryable=retryable, trace_id=trace_id)


class ConfigurationError(MMCPError):
    """Missing .env variables or invalid paths."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        trace_id: str | None = None,
    ):
        super().__init__(message, retryable=False, trace_id=trace_id)
        self.config_key = config_key


class StaleApprovalError(MMCPError):
    """Approval ID mismatch or stale approval attempt."""

    def __init__(
        self,
        message: str = "Approval ID mismatch - action may have been overwritten",
        trace_id: str | None = None,
    ):
        super().__init__(message, retryable=False, trace_id=trace_id)


# --- Error Mapping Utilities ---


def map_provider_error(e: Exception, trace_id: str | None = None) -> MMCPError:
    """
    Map third-party provider errors (LiteLLM, etc.) to MMCP errors.

    Args:
        e: The exception to map
        trace_id: Optional trace ID for correlation

    Returns:
        Mapped MMCPError
    """
    error_str = str(e).lower()
    error_type = type(e).__name__

    # LiteLLM provider configuration errors
    if "llm provider not provided" in error_str or "provider not provided" in error_str:
        logger.error(f"Provider configuration error: {e}", exc_info=True)
        return ConfigurationError(
            "Model provider misconfigured. Check LLM_MODEL format (e.g., 'gemini/gemini-flash-latest')",
            config_key="LLM_MODEL",
            trace_id=trace_id,
        )

    # Rate limiting
    if "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str:
        logger.warning(f"Rate limit hit: {e}")
        return ProviderError(
            "AI service rate limit reached. Please try again shortly.",
            retryable=True,
            trace_id=trace_id,
        )

    # Authentication errors
    if (
        "api key" in error_str
        or "authentication" in error_str
        or "401" in error_str
        or "403" in error_str
    ):
        logger.error(f"Authentication error: {e}", exc_info=True)
        return ProviderError(
            "AI service authentication failed. Check API key configuration.",
            retryable=False,
            trace_id=trace_id,
        )

    # Context window / token limit
    # Match actual context window errors, not method names containing "context"
    if (
        "context window" in error_str
        or "context length" in error_str
        or "maximum context" in error_str
        or "token limit" in error_str
        or "token count" in error_str
        or ("exceeds" in error_str
            and ("token" in error_str or "length" in error_str))
        or "too many tokens" in error_str
        or "input too long" in error_str
    ):
        logger.warning(f"Context/token limit: {e}")
        return ProviderError(
            "Request exceeds context window. Try a shorter conversation.",
            retryable=False,
            trace_id=trace_id,
        )

    # Timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        logger.warning(f"Timeout error: {e}")
        return ProviderError(
            "AI service request timed out. Please try again.",
            retryable=True,
            trace_id=trace_id,
        )

    # Generic provider error
    logger.error(f"Unmapped provider error ({error_type}): {e}", exc_info=True)
    return ProviderError(
        f"AI service error: {error_type}",
        retryable=True,
        trace_id=trace_id,
    )


def map_tool_error(
    e: Exception, tool_name: str | None = None, trace_id: str | None = None
) -> ToolError:
    """
    Map tool execution errors to ToolError.

    Args:
        e: The exception from tool execution
        tool_name: Name of the tool that failed
        trace_id: Optional trace ID for correlation

    Returns:
        ToolError with sanitized message
    """
    error_type = type(e).__name__
    error_str = str(e)

    # File not found errors
    if "file not found" in error_str.lower() or "no such file" in error_str.lower():
        logger.warning(f"Tool {tool_name} file error: {e}")
        return ToolError(
            f"File not found: {error_str}",
            tool_name=tool_name,
            retryable=False,
            trace_id=trace_id,
        )

    # Permission errors
    if "permission" in error_str.lower() or "access denied" in error_str.lower():
        logger.warning(f"Tool {tool_name} permission error: {e}")
        return ToolError(
            f"Permission denied: {error_str}",
            tool_name=tool_name,
            retryable=False,
            trace_id=trace_id,
        )

    # Generic tool error (sanitize to avoid exposing internal details)
    logger.error(f"Tool {tool_name} error ({error_type}): {e}", exc_info=True)
    return ToolError(
        f"Tool execution failed: {error_type}",
        tool_name=tool_name,
        retryable=False,
        trace_id=trace_id,
    )


def map_agent_error(e: Exception, trace_id: str | None = None) -> AgentLogicError:
    """
    Map agent logic errors (validation, parsing, etc.).

    Args:
        e: The exception from agent logic
        trace_id: Optional trace ID for correlation

    Returns:
        AgentLogicError
    """
    error_type = type(e).__name__
    error_str = str(e).lower()

    # Instructor validation errors
    if "validation" in error_str or "pydantic" in error_str or "instructor" in error_str:
        logger.warning(f"Agent validation error: {e}")
        return AgentLogicError(
            "LLM returned invalid response format. Retrying...",
            retryable=True,
            trace_id=trace_id,
        )

    # JSON parsing errors
    if "json" in error_str or "parse" in error_str:
        logger.warning(f"Agent parsing error: {e}")
        return AgentLogicError(
            "Failed to parse LLM response. Retrying...",
            retryable=True,
            trace_id=trace_id,
        )

    # Generic agent error
    logger.error(f"Agent logic error ({error_type}): {e}", exc_info=True)
    return AgentLogicError(
        f"Agent processing error: {error_type}",
        retryable=False,
        trace_id=trace_id,
    )


# --- Structured Error Response Models ---


class ErrorDetail(BaseModel):
    """Structured error information for API responses."""

    code: str
    message: str
    tool_name: str | None = None
    retryable: bool = False
    trace_id: str


class AgentResponse(BaseModel):
    """Standardized agent response with success/error structure."""

    success: bool
    data: str | None = None  # The agent's response message
    error: ErrorDetail | None = None
    trace_id: str


def error_to_detail(error: MMCPError) -> ErrorDetail:
    """
    Convert an MMCPError to an ErrorDetail for API response.

    Args:
        error: The MMCPError instance

    Returns:
        ErrorDetail Pydantic model
    """
    # Determine error code from exception type
    if isinstance(error, ConfigurationError):
        code = "ERR_CONFIG"
    elif isinstance(error, ProviderError):
        code = "ERR_PROVIDER"
    elif isinstance(error, ToolError):
        code = "ERR_TOOL"
    elif isinstance(error, AgentLogicError):
        code = "ERR_AGENT_LOGIC"
    else:
        code = "ERR_UNKNOWN"

    return ErrorDetail(
        code=code,
        message=error.message,
        tool_name=getattr(error, "tool_name", None),
        retryable=error.retryable,
        trace_id=error.trace_id,
    )
