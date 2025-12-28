import abc
from typing import Any, Union

from pydantic import BaseModel, Field

from app.core.context import MMCPContext


class MMCPTool(abc.ABC):
    """
    Abstract Base Class for all MMCP tools.

    Philosophy:
    - strict typing via Pydantic (for the Agent)
    - strict return types (for the API)
    - async execution (for performance)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        The internal name of the tool (e.g., 'search_youtube').
        Must be unique.
        """
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        A clear, concise description of what this tool does.
        The LLM reads this to decide if it should call this tool.
        """
        pass

    @property
    @abc.abstractmethod
    def input_schema(self) -> type[BaseModel]:
        """
        The Pydantic model defining the arguments this tool accepts.
        """
        pass

    @property
    def version(self) -> str:
        """Optional versioning for the tool."""
        return "1.0.0"

    def get_status_info(self) -> dict[str, Any]:
        """
        Return plugin status information for UI/API display.

        This allows plugins to provide their own status information
        without core code knowing about specific plugin details.
        """
        return {
            "service_name": self.name.replace("_", " ").title(),
            "configured": self.is_available(),
            "description": getattr(self, "status_description", None),
        }

    def is_available(self) -> bool:
        """
        Check if this tool is available (e.g., required API keys are configured).

        Override this method to check configuration requirements.
        If False, the plugin loader will skip loading this tool.

        Returns:
            True if the tool can be used, False otherwise.
        """
        return True

    @abc.abstractmethod
    async def execute(self, context: MMCPContext, **kwargs: Any) -> Any:
        """
        The actual logic.

        Philosophy: Tools never fetch context themselves — context is injected.

        Args:
            context: MMCPContext with config, runtime state, and metrics
            **kwargs: valid data matching input_schema.

        Returns:
            A string, dict, or Pydantic model containing the result.
        """
        pass


class ContextResponse(BaseModel):
    """
    Response model for context providers.

    Handles TTL and metadata internally while keeping the LLM's data clean.
    """

    data: dict[str, Any] = Field(description="The context data to inject into LLM media_state")
    ttl: int = Field(default=300, description="Time-to-live in seconds (default: 5 minutes)")
    provider_name: str = Field(description="Name of the provider for logging and health tracking")


class MMCPContextProvider(abc.ABC):
    """
    Abstract Base Class for context providers.

    Context providers fetch dynamic state (e.g., Jellyfin library status, Plex server info)
    that should be available to the LLM before the ReAct loop begins.

    Philosophy:
    - Lightweight eligibility checks (no heavy I/O)
    - Async execution with timeouts
    - Circuit breaker protection via health monitor
    - Truncation to prevent context bloat
    """

    @property
    @abc.abstractmethod
    def context_key(self) -> str:
        """
        The key in the JSON object (e.g., 'jellyfin', 'plex').

        This key will be used in context.llm.media_state[context_key] = data.
        Must be unique across all context providers.
        """
        pass

    @abc.abstractmethod
    async def provide_context(self) -> Union[dict[str, Any], "ContextResponse"]:
        """
        Fetch the dynamic context data.

        This method can return either:
        - A dictionary (for backward compatibility)
        - A ContextResponse (recommended) with TTL and metadata

        The dictionary/ContextResponse.data will be injected into
        the LLM's media_state under the context_key.

        Returns:
            Dictionary or ContextResponse of context data (will be truncated if too large).
        """
        pass

    async def is_eligible(self, _query: str) -> bool:
        """
        Lightweight check if this provider should run for the given query.

        Override this to implement query-based filtering (e.g., only run
        Jellyfin provider if query mentions "jellyfin" or "library").

        Args:
            query: The user's input query.

        Returns:
            True if this provider should execute, False otherwise.
        """
        return True
