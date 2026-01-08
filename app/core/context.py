"""MMCP Context - injected context for all tool executions."""

import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from app.core.config import user_settings
from app.core.logger import logger


class LLMContext(BaseModel):
    """
    Information the LLM uses to make decisions.

    This should only contain context provider data and tool availability.
    Internal metrics are excluded to prevent context pollution.
    """

    context_provider_data: dict[str, Any] = Field(default_factory=dict)
    available_tools: dict[str, str] = Field(default_factory=dict)  # name -> description


class StaticContext(BaseModel):
    """Static context that doesn't change during execution."""

    config: dict[str, Any] = Field(default_factory=dict)
    server_info: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        """Initialize with current settings after model creation."""
        self.config.update(
            {
                "llm_max_context_chars": user_settings.llm_max_context_chars,
                "llm_model": user_settings.llm_model,
                "llm_base_url": user_settings.llm_base_url,
                # Only core system config here - plugins inject their own config
            }
        )
        self.server_info.update(
            {
                "version": "0.1.0",
                "environment": "development",  # Could be expanded later
            }
        )


class RuntimeContext(BaseModel):
    """Runtime context that changes during execution."""

    trace_id: str | None = None
    tool_failure_counts: dict[str, int] = Field(default_factory=dict)
    step_count: int = 0
    conversation_id: str | None = None


class ContextMetrics(BaseModel):
    """Basic metrics for observability (NOT sent to LLM)."""

    start_time: float = Field(default_factory=time.time)
    tool_execution_count: int = 0
    llm_call_count: int = 0


class MMCPContext(BaseModel):
    """
    Injected context for all tool executions.

    Architecture: Separates LLM-visible context from internal runtime state.
    Tools receive full context (trusted code), LLM receives sanitized subset.
    """

    # LLM-visible context (user data, conversation state)
    llm: LLMContext = Field(default_factory=LLMContext)

    # Internal system state (not sent to LLM)
    static: StaticContext = Field(default_factory=StaticContext)
    runtime: RuntimeContext
    metrics: ContextMetrics = Field(default_factory=ContextMetrics)

    # Approval state for HITL workflow
    is_approved: SkipJsonSchema[bool] = Field(default=False, exclude=True)

    def __init__(self, trace_id: str | None = None, **data):
        super().__init__(runtime=RuntimeContext(trace_id=trace_id), **data)

    def get_context_provider_data(self) -> dict[str, Any]:
        """
        Returns context provider data that should be visible to the LLM.

        This excludes available_tools (already output in TOOLS section) to prevent
        token waste and confusion. Only includes context_provider_data from context providers.
        """
        return self.llm.context_provider_data

    def increment_tool_failures(self, tool_name: str) -> int:
        """Increment failure count for a tool and return new count."""
        count = self.runtime.tool_failure_counts.get(tool_name, 0) + 1
        self.runtime.tool_failure_counts[tool_name] = count
        return count

    def increment_step(self) -> int:
        """Increment step count and return new count."""
        self.runtime.step_count += 1
        return self.runtime.step_count

    def increment_tool_execution(self) -> int:
        """Increment tool execution count and return new count."""
        self.metrics.tool_execution_count += 1
        return self.metrics.tool_execution_count

    def increment_llm_call(self) -> int:
        """Increment LLM call count and return new count."""
        self.metrics.llm_call_count += 1
        return self.metrics.llm_call_count

    def get_tool_failure_count(self, tool_name: str) -> int:
        """Get failure count for a specific tool."""
        return self.runtime.tool_failure_counts.get(tool_name, 0)

    def is_tool_circuit_breaker_tripped(self, tool_name: str, threshold: int | None = None) -> bool:
        """Check if circuit breaker should trip for a tool."""
        if threshold is None:
            threshold = user_settings.tool_circuit_breaker_threshold
        return self.get_tool_failure_count(tool_name) >= threshold

    def is_max_steps_reached(self, max_steps: int | None = None) -> bool:
        """Check if maximum reasoning steps reached."""
        if max_steps is None:
            max_steps = user_settings.react_max_steps
        return self.runtime.step_count >= max_steps

    def set_available_tools(self, tools: dict[str, str]):
        """Update the LLM context with available tools."""
        self.llm.available_tools = tools

    def inject_plugin_config(self, config_key: str, value: Any) -> None:
        """
        Allow plugins to inject their config into context.

        Plugins call this during execution to ensure their config is available.
        This keeps plugin-specific config out of core initialization.
        """
        self.static.config[config_key] = value

    def inspect(self) -> dict[str, Any]:
        """
        Return a snapshot of current context state for observability.

        This provides a single point to inspect conversation health,
        performance metrics, and failure patterns.
        """
        elapsed_time = time.time() - self.metrics.start_time
        total_failures = sum(self.runtime.tool_failure_counts.values())

        return {
            "trace_id": self.runtime.trace_id,
            "conversation_id": self.runtime.conversation_id,
            "elapsed_seconds": round(elapsed_time, 2),
            "step_count": self.runtime.step_count,
            "llm_calls": self.metrics.llm_call_count,
            "tool_executions": self.metrics.tool_execution_count,
            "total_tool_failures": total_failures,
            "tool_failure_breakdown": dict(self.runtime.tool_failure_counts),
            "circuit_breaker_status": {
                tool_name: self.is_tool_circuit_breaker_tripped(tool_name)
                for tool_name in self.runtime.tool_failure_counts
            },
        }

    def log_inspection(self):
        """Log a structured inspection of the current context state."""
        inspection = self.inspect()
        logger.info(
            f"Conversation complete: {inspection['step_count']} steps, "
            f"{inspection['llm_calls']} LLM calls, "
            f"{inspection['tool_executions']} tool executions, "
            f"{inspection['total_tool_failures']} failures in {inspection['elapsed_seconds']}s",
            extra={"trace_id": inspection["trace_id"], "inspection": inspection},
        )
