"""
Foundation Context Providers for MMCP.

Provides essential situational awareness including time, platform, and service status.
"""

import platform
from datetime import datetime

from mmcp import ContextResponse, PluginContext


class IdentityProvider:
    """
    Identity Context Provider.

    Provides current time and platform information for temporal reasoning.
    This ensures the agent isn't "time-blind" and can answer questions like
    "How much time do I have before midnight?" or "What was the date two days ago?"
    """

    @property
    def context_key(self) -> str:
        return "identity"

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
                # "host_platform": platform.platform(),
            },
            ttl=0,
            provider_name="IdentityProvider",
        )
