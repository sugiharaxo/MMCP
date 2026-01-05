"""Transport Layer: Pure I/O wrapper implementing LLMTransport protocol.

This layer has zero knowledge of Pydantic, Tools, or BAML. It is a "dumb pipe"
that takes a raw string (prompt) and returns a raw string (response).

Currently implements a dummy service that returns hardcoded responses.
The any-llm integration will be added in a future phase.
"""

from typing import Any

from app.core.logger import logger


class TransportService:
    """Stateless transport service implementing LLMTransport protocol.

    Dummy implementation: Returns hardcoded "DUMMY_LLM_RESPONSE" for all requests.
    The any-llm client is imported lazily inside the method (not at module level).
    """

    def __init__(self) -> None:
        """Initialize the transport service."""
        self._client = None

    async def _get_client(self) -> Any:
        """Lazy initialization of any-llm client (not used in dummy mode)."""
        if self._client is None:
            try:
                # Using any-llm for universal model support
                # Lazy import: only when actually needed (not in dummy mode)
                import any_llm  # noqa: F401

                # In dummy mode, we don't actually initialize the client
                logger.debug("any-llm import available (dummy mode active)")
                self._client = "dummy_client"
            except ImportError:
                logger.debug("any-llm package not installed (dummy mode active)")
                self._client = "dummy_client"

        return self._client

    async def send_message(self, prompt: str, **kwargs: Any) -> str:
        """
        Send a raw text prompt and receive a raw text response.

        Dummy implementation: Returns hardcoded response.
        This is the only interface - pure text in, pure text out.
        No knowledge of structured data, tools, or parsing.

        Args:
            prompt: Raw text prompt to send to the LLM
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
                (unused in dummy mode, kept for protocol compliance)

        Returns:
            Raw text response from the LLM (hardcoded "DUMMY_LLM_RESPONSE")
        """
        logger.debug(f"Dummy transport: received prompt (length: {len(prompt)})")
        # Dummy implementation: return hardcoded response
        # kwargs are accepted for protocol compliance but unused in dummy mode
        _ = kwargs  # Suppress unused argument warning
        return "DUMMY_LLM_RESPONSE"

    async def close(self) -> None:
        """Clean up any resources."""
        if self._client:
            self._client = None
