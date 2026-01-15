import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.session_manager import SessionManager as AgentSessionManager
from app.anp.agent_integration import AgentNotificationInjector
from app.anp.event_bus import EventBus
from app.anp.notification_dispatcher import NotificationDispatcher
from app.anp.watchdog import WatchdogService
from app.api.routes import chat as chat_routes
from app.api.routes import notifications as notifications_routes
from app.api.routes import sessions as sessions_routes
from app.api.routes import settings as settings_routes
from app.core.auth import ensure_admin_token
from app.core.config import user_settings
from app.core.database import close_database, init_database
from app.core.errors import (
    AgentLogicError,
    AgentResponse,
    ConfigurationError,
    ErrorDetail,
    MMCPError,
    ProviderError,
    ToolError,
    error_to_detail,
)
from app.core.health import HealthMonitor
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.core.session_manager import SessionManager
from app.services.agent import AgentService
from app.services.prompt import PromptService
from app.services.type_mapper import TypeMapper

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "app" / "static"
STATIC_DIR.mkdir(exist_ok=True, parents=True)

loader = PluginLoader(user_settings.plugin_dir)
health_monitor = HealthMonitor()
event_bus = EventBus()
notification_dispatcher = NotificationDispatcher()
watchdog_service = WatchdogService(event_bus)
session_manager = SessionManager()
event_bus.set_session_manager(session_manager)
notification_dispatcher.set_event_bus(event_bus)
event_bus.set_notification_dispatcher(notification_dispatcher)

notification_injector = AgentNotificationInjector(event_bus)

# Initialize required dependencies for AgentService
type_mapper = TypeMapper()
prompt_service = PromptService(type_mapper=type_mapper)
agent_session_manager = AgentSessionManager()

# Initialize AgentService with all dependencies
agent_service = AgentService(
    plugin_loader=loader,
    user_settings=user_settings,
    notification_injector=notification_injector,
    prompt=prompt_service,
    session_manager=agent_session_manager,
    event_bus=event_bus,
    health_monitor=health_monitor,
)

# Set up the Mediator pattern: EventBus triggers AgentService
event_bus.set_agent_service(agent_service)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Initialize database
    await init_database()

    # Generate admin token if auth is required
    if user_settings.require_auth:
        ensure_admin_token()

    # Load plugins (now async)
    await loader.load_plugins()

    # Attach components to app state for dependency injection
    _app.state.loader = loader
    _app.state.health_monitor = health_monitor
    _app.state.agent_service = agent_service
    _app.state.event_bus = event_bus
    _app.state.notification_dispatcher = notification_dispatcher
    _app.state.session_manager = session_manager

    # Start ANP watchdog service
    await watchdog_service.start()

    yield

    # Shutdown: Stop ANP services, close agent service, and close database
    await watchdog_service.stop()
    await agent_service.close()
    await close_database()


# 1. Initialize FastAPI with lifespan
app = FastAPI(
    title="MMCP Core",
    description="Modular Media Control Plane API",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Exception Handlers ---


@app.exception_handler(MMCPError)
async def mmcp_error_handler(_request: Request, exc: MMCPError) -> JSONResponse:
    """
    Centralized handler for MMCP errors.

    Returns structured error responses with appropriate HTTP status codes.
    Fatal errors (config, auth) return 500, retryable errors return 503.
    """
    error_detail = error_to_detail(exc)

    # Determine status code based on error type
    if isinstance(exc, ConfigurationError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR  # Server misconfiguration
    elif isinstance(exc, ProviderError) and not exc.retryable:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR  # Fatal provider error
    elif isinstance(exc, ProviderError) and exc.retryable:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE  # Retryable provider error
    elif isinstance(exc, ToolError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR  # Tool execution failed
    elif isinstance(exc, AgentLogicError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR  # Agent logic error
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    response = AgentResponse(
        success=False,
        error=error_detail,
        trace_id=exc.trace_id,
    )

    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors from FastAPI request parsing.

    Maps to AgentLogicError since it's a client-side validation issue.
    """
    trace_id = str(uuid.uuid4())
    error_detail = ErrorDetail(
        code="ERR_VALIDATION",
        message=f"Invalid request format: {exc.errors()}",
        retryable=False,
        trace_id=trace_id,
    )

    response = AgentResponse(
        success=False,
        error=error_detail,
        trace_id=trace_id,
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=response.model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unexpected exceptions.

    Logs the full error and returns a sanitized response.
    """
    trace_id = str(uuid.uuid4())
    error_detail = ErrorDetail(
        code="ERR_UNKNOWN",
        message="An unexpected error occurred. Please check logs for details.",
        retryable=False,
        trace_id=trace_id,
    )

    # Log the full error with trace_id
    logger.error(
        f"Unhandled exception (trace_id={trace_id}): {exc}",
        exc_info=True,
        extra={"trace_id": trace_id},
    )

    response = AgentResponse(
        success=False,
        error=error_detail,
        trace_id=trace_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(),
    )


# --- API Endpoints ---

app.include_router(chat_routes.router)
app.include_router(settings_routes.router)
app.include_router(notifications_routes.router)
app.include_router(sessions_routes.router)


@app.get("/api/v1/status")
async def get_status():
    """Health check endpoint with generic plugin status."""
    plugin_statuses = await loader.get_plugin_statuses()

    return {
        "status": "online",
        "llm_configured": bool(user_settings.llm_api_key or user_settings.llm_base_url),
        "plugins": plugin_statuses,  # Generic plugin status dictionary
        "loaded_plugins": list(loader.list_tools().keys()),
    }


@app.get("/api/v1/tools")
async def get_tools():
    """List all currently enabled tools and their descriptions."""
    return loader.list_tools()


# Chat endpoint is now handled by app/api/routes/chat.py router


# --- UI / Static Files ---

# Mount static files at root (for SolidJS SPA)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="web")


if __name__ == "__main__":
    # Run the server
    # Use 127.0.0.1 for local development (works on Windows)
    # For network access, use 0.0.0.0 but access via your machine's IP
    print("Starting MMCP on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
