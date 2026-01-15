from pydantic import BaseModel

from app.api.base import Plugin, Tool


class TestExternalInput(BaseModel):
    message: str = "Hello from external tool!"


class TestHITLPlugin(Plugin):
    name = "test_hitl"
    version = "1.0.0"
    settings_model = None

    class TestExternalAction(Tool):
        name = "test_external_action"
        description = "Perform a test external action (requires user approval)"
        version = "1.0.0"
        input_schema = TestExternalInput
        settings_model = None
        classification = "EXTERNAL"  # Mark as EXTERNAL to trigger HITL approval

        def is_available(self, _settings, _runtime) -> bool:
            return True

        async def execute(self, **kwargs) -> str:
            message = kwargs.get("message", "Hello from external tool!")
            return f"External action completed: {message}"

    def get_tools(self, settings, runtime):
        # Tool will be instantiated by the loader with proper settings/runtime injection
        return [self.TestExternalAction(settings, runtime, self.name)]

    def get_providers(self, settings, runtime):  # noqa
        return []
