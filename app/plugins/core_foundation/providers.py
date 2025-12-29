"""
Foundation Context Providers for MMCP.

Provides essential situational awareness including time, platform, and service status.
"""

import platform
from datetime import datetime

from app.api.base import Plugin, Provider
from app.api.schemas import ContextResponse, PluginContext


class Foundation(Plugin):
    """
    Foundation Plugin - Core Context Providers.

    Provides essential situational awareness for the MMCP agent including:
    - Identity: Current time and platform information
    """

    name = "foundation"
    version = "1.0.0"
    settings_model = None  # No settings required

    class Identity(Provider):  # Nesting = Auto-discovery (Pattern A)
        """
        Identity Context Provider.

        Provides current time and platform information for temporal reasoning.
        This ensures the agent isn't "time-blind" and can answer questions like
        "How much time do I have before midnight?" or "What was the date two days ago?"
        """

        context_key = "identity"

        async def provide_context(self, context: PluginContext) -> ContextResponse:
            """
            Return current time and platform information.

            Args:
                context: Safe PluginContext facade with system access.

            Returns:
                ContextResponse with current timestamp and platform details.
            """
            now = datetime.now()

            return ContextResponse(
                data={
                    "current_time": now.strftime("%A, %b %d, %Y %I:%M %p"),
                    "timezone": now.astimezone().tzname(),
                    "host_os": platform.system(),
                },
                ttl=0,
                provider_name="Identity",
            )
