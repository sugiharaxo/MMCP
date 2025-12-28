import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.orchestrator import AgentOrchestrator
from app.core.config import settings
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
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader

# 2. Setup Paths
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

# 3. Initialize Plugin Loader
loader = PluginLoader(settings.plugin_dir)

# 4. Initialize Agent Orchestrator
orchestrator = AgentOrchestrator(loader)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Load plugins when the server starts
    loader.load_plugins()
    yield
    # Shutdown: Cleanup (if needed in the future)


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


@app.get("/api/v1/status")
async def get_status():
    """Health check endpoint with generic plugin status."""
    plugin_statuses = loader.get_plugin_statuses()

    return {
        "status": "online",
        "llm_configured": bool(settings.llm_api_key or settings.llm_base_url),
        "plugins": plugin_statuses,  # Generic plugin status dictionary
        "loaded_plugins": list(loader.list_tools().keys()),
    }


@app.get("/api/v1/tools")
async def get_tools():
    """List all currently enabled tools and their descriptions."""
    return loader.list_tools()


@app.post("/api/v1/chat")
async def chat_endpoint(message: str = Body(..., embed=True)) -> AgentResponse:
    """
    The functional gateway to the agent.

    Sends a user message to the agent orchestrator and returns a structured response.
    Errors are handled by exception handlers and returned as structured AgentResponse.

    Args:
        message: The user's message
        request: FastAPI request object (for trace_id extraction if needed)

    Returns:
        AgentResponse with success/error structure
    """
    # Generate trace_id for this request
    trace_id = str(uuid.uuid4())

    try:
        # Orchestrator handles errors internally and feeds them back to LLM when possible
        # Fatal errors (config, auth) will be raised as MMCPError and caught by exception handler
        response_text = await orchestrator.chat(message, trace_id=trace_id)

        return AgentResponse(
            success=True,
            data=response_text,
            trace_id=trace_id,
        )
    except MMCPError:
        # Re-raise MMCPErrors to be handled by exception handler
        raise
    except Exception as e:
        # Unexpected error - wrap and re-raise
        # This should rarely happen as orchestrator handles most errors
        logger.error(
            f"Unexpected error in chat endpoint (trace_id={trace_id}): {e}",
            exc_info=True,
            extra={"trace_id": trace_id},
        )
        raise


# --- UI / Static Files ---

# Mount static files (for CSS/JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def get_ui():
    """Serves the Vanilla JS Web Interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "MMCP is running, but index.html was not found in static/"}
    return FileResponse(index_path)


if __name__ == "__main__":
    # Run the server
    # Use 127.0.0.1 for local development (works on Windows)
    # For network access, use 0.0.0.0 but access via your machine's IP
    print("Starting MMCP on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
