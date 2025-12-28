import abc
from typing import Any

from pydantic import BaseModel

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
