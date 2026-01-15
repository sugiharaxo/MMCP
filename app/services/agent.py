"""Agent Service: The Orchestrator coordinating Prompt layer with BAML.

This layer manages:
- State management for the ReAct loop
- Coordination with Prompt layer (BAML end-to-end: prompt → transport → SAP)
- The main while loop and HITL (Human-in-the-loop) approvals
- Context assembly from context providers

Session Management Architecture:
- Uses app.agent.session_manager.SessionManager (aliased as AgentSessionManager)
  for conversation history persistence (database operations)
- This is separate from app.core.session_manager.SessionManager which tracks
  active session IDs (in-memory) for ANP protocol
- The agent session manager handles: load_session, save_session, pending_action
- Lock management is handled by SessionLockManager (also in app.agent.session_manager)

Implements AgentOrchestrator protocol with dependency injection for Prompt (BAML).
Transport layer removed - now handled by BAML's ClientRegistry with HTTP Keep-Alive caching.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from app.agent.context_manager import ContextManager
from app.agent.history_manager import HistoryManager, HistoryMessage
from app.agent.react_loop import ReActLoop
from app.agent.session_manager import SessionLockManager
from app.agent.session_manager import SessionManager as AgentSessionManager
from app.anp.agent_integration import AgentNotificationInjector
from app.anp.event_bus import EventBus
from app.anp.models import EventLedger
from app.anp.schemas import NotificationCreate
from app.core.config import HitlDecision, UserSettings
from app.core.context import MMCPContext
from app.core.errors import StaleApprovalError
from app.core.health import HealthMonitor
from app.core.interfaces import LLMPrompt
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.services.prompt import ToolInfo
from baml_client.stream_types import (
    FinalResponse as StreamFinalResponse,
)
from baml_client.stream_types import (
    ToolCall as StreamToolCall,
)

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class AgentService:
    """Main agent orchestrator implementing AgentOrchestrator protocol.

    Pure orchestrator: Coordinates the ReAct loop with Prompt layer (BAML end-to-end).
    Uses dependency injection to accept any implementation of LLMPrompt.
    Transport removed - BAML handles prompt → transport → SAP with ClientRegistry caching.
    """

    def __init__(
        self,
        plugin_loader: PluginLoader,
        user_settings: UserSettings,
        notification_injector: AgentNotificationInjector,
        prompt: LLMPrompt,
        session_manager: AgentSessionManager,
        event_bus: EventBus,
        health_monitor: HealthMonitor | None = None,
    ):
        """
        Initialize the agent service with dependency injection.

        Args:
            plugin_loader: PluginLoader with discovered tools
            user_settings: UserSettings configuration
            notification_injector: AgentNotificationInjector for ANP notifications
            prompt: LLMPrompt implementation
            session_manager: AgentSessionManager for database persistence
            event_bus: EventBus for ANP event routing
            health_monitor: HealthMonitor for context provider circuit breakers (optional)
        """
        self.plugin_loader = plugin_loader
        self.user_settings = user_settings
        self.prompt = prompt
        self.session_manager = session_manager
        self.event_bus = event_bus

        # Cache tool infos by tool configuration to avoid rebuilding on every request
        # Key: tuple of (tool_name, schema_name, classification) tuples
        # Note: Cache automatically invalidates when tool configuration changes (cache key mismatch).
        # Assumes tools are static at runtime - hot reloading not supported.
        self._tool_infos_cache: dict[tuple[tuple[str, str, str], ...], list[ToolInfo]] = {}

        # Initialize managers
        self.history_manager = HistoryManager()
        self.lock_manager = SessionLockManager()

        # Initialize ReAct loop handler
        self.react_loop = ReActLoop(
            prompt=self.prompt,
            session_manager=self.session_manager,
            user_settings=self.user_settings,
            plugin_loader=self.plugin_loader,
            history_manager=self.history_manager,
        )

        self.context_manager = ContextManager(
            loader=plugin_loader,
            notification_injector=notification_injector,
            health=health_monitor,
        )

    def _get_cached_tool_infos(self) -> list[ToolInfo]:
        """
        Get cached tool infos, rebuilding only if tool configuration has changed.

        Cache automatically invalidates when the cache key changes (tool configuration mismatch).
        Uses same cache key strategy as TypeMapper for consistency.
        """
        # Create cache key from current tool configuration
        cache_key = tuple(
            (tool.name, tool.input_schema.__name__, getattr(tool, "classification", "EXTERNAL"))
            for tool in sorted(self.plugin_loader.tools.values(), key=lambda t: t.name)
            if hasattr(tool, "input_schema") and tool.input_schema
        )

        # Return cached result if available
        if cache_key in self._tool_infos_cache:
            return self._tool_infos_cache[cache_key]

        # Build and cache new tool infos
        tool_infos = self.plugin_loader.build_tool_infos()
        self._tool_infos_cache[cache_key] = tool_infos
        return tool_infos

    async def _execute_tool(self, tool: Any, tool_args: dict[str, Any]) -> Any:
        """Execute a tool by delegating to the ReActLoop."""
        return await self.react_loop.execute_tool(tool, tool_args)

    def _stringify_tool_result(self, tool_result: Any) -> str:
        """Stringify tool result by delegating to the ReActLoop."""
        return self.react_loop.stringify_tool_result(tool_result)

    async def _handle_llm_error(
        self, llm_error: Exception, history: list[HistoryMessage], session_id: str
    ) -> dict[str, Any]:
        """Handle LLM error and emit ANP notification for UI display."""
        # Emit ANP notification for UI display
        from app.anp.event_bus import EventBus
        from app.anp.schemas import NotificationCreate, RoutingFlags

        try:
            await EventBus().emit_notification(
                NotificationCreate(
                    content=f"Agent Error: {llm_error!s}",
                    routing=RoutingFlags(address="session", target="user", handler="system"),
                    session_id=session_id,
                    metadata={"error_type": type(llm_error).__name__},
                )
            )
        except (ConnectionError, TimeoutError, RuntimeError) as emit_error:
            # Don't let notification emission errors break the main flow
            logger.warning(f"Failed to emit error notification: {emit_error!s}")

        # Delegate to ReActLoop for standard error response
        return self.react_loop.handle_llm_error(llm_error, history, session_id)

    async def _run_under_session_lock(
        self, session_id: str, coro: Callable[[], Awaitable[T]]
    ) -> T | dict[str, Any]:
        """Execute agent logic under a session lock with timeout handling."""
        lock = self.lock_manager.get_lock(session_id)
        try:
            async with asyncio.timeout(self.user_settings.session_lock_timeout_seconds):
                async with lock:
                    return await coro()
        except TimeoutError:
            logger.warning(f"Session lock timeout: {session_id}")
            return {
                "response": "The session is currently busy processing another request.",
                "type": "error",
                "session_id": session_id,
            }

    async def _load_or_create_session(
        self, session_id: str | None
    ) -> tuple[str, list[HistoryMessage]]:
        """
        Load existing session or create new one with server-generated ID.

        Args:
            session_id: Existing session ID to resume, or None to create new

        Returns:
            Tuple of (session_id, conversation_history)

        Raises:
            ValueError: If session_id format is invalid
        """
        if session_id is None:
            # Create new session with server-generated ID
            session_id = str(uuid.uuid4())
            history = []  # New sessions start empty
        else:
            # Validate existing session ID format
            try:
                uuid.UUID(session_id)  # Validates UUID format
            except ValueError as err:
                raise ValueError(f"Invalid session ID format: {session_id}") from err

            # Load history from database (already converted to HistoryMessage)
            history = await self.session_manager.load_session(session_id)

        return session_id, history

    async def _prepare_turn(
        self, user_input: str, session_id: str | None
    ) -> tuple[str, list[HistoryMessage], dict[str, Any], list[ToolInfo], MMCPContext]:
        """
        Prepare a turn by loading session, assembling context, and fetching tools.

        This is the deterministic setup phase that doesn't require a session lock.
        All context assembly and tool discovery happens here.

        Args:
            user_input: The user's message
            session_id: Optional session identifier

        Returns:
            Tuple of (session_id, history, context_data, tool_infos, context)
        """
        session_id, history = await self._load_or_create_session(session_id)
        context = MMCPContext(trace_id=str(uuid.uuid4()))
        await self.context_manager.assemble_llm_context(user_input, context)
        return (
            session_id,
            history,
            context.get_context_provider_data(),
            self._get_cached_tool_infos(),
            context,
        )

    async def _execute_agent_turn(
        self,
        user_input: str | None,
        session_id: str,
        history: list[HistoryMessage],
        context_data: dict[str, Any],
        tool_infos: list[ToolInfo],
        stream_callback: Callable[[StreamFinalResponse | StreamToolCall], None] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a single agent turn: LLM call + ReAct loop.

        This is the core execution logic that runs under a session lock.
        Handles both streaming and non-streaming modes.

        Args:
            user_input: The user's message (None for resume after tool execution)
            session_id: Session identifier
            history: Conversation history
            context_data: Context data from context providers
            tool_infos: List of available tools
            stream_callback: Optional callback for streaming raw Pydantic models (FinalResponse | ToolCall)

        Note:
            The is_autonomous flag is handled at the streaming layer (_stream_task_results),
            not in this method, since formatting happens there.

        Returns:
            Response dict with result and metadata
        """
        input_length = len(user_input) if user_input else 0
        logger.debug(f"Executing agent turn for session {session_id} (length: {input_length})")

        # Phase 2: LLM Decision (Probabilistic)
        # Call BAML end-to-end: prompt → transport → SAP
        # Note: to_dict_list() creates a copy for API transport only;
        # the original 'history' list reference is preserved for mutations
        try:
            parsed_response = await self.prompt.call_llm(
                tool_infos=tool_infos,
                context_data=context_data,
                user_input=user_input or "",  # Pass user input directly for runtime context display
                history=HistoryManager.to_dict_list(history),  # Convert only for API transport
                user_settings=self.user_settings,  # Pass settings for ClientRegistry
                stream_callback=stream_callback,
            )
        except Exception as llm_error:
            error_result = await self._handle_llm_error(llm_error, history, session_id)
            await self.session_manager.save_session(session_id, history)
            return error_result

        # Add user input to history now that LLM has processed it (only if provided)
        if user_input:
            self.history_manager.add_user_message(history, user_input)
        else:
            # Resume mode: Ensure history has proper user context for BAML
            # Don't add UI-facing text to history - only raw ToolCall and ToolResult objects should be there
            pass

        # Phase 3: ReAct Loop - Continue until FinalResponse
        max_steps = self.user_settings.react_max_steps
        result = await self.react_loop.execute(
            parsed_response,
            context_data,
            history,
            session_id,
            max_steps,
            tool_infos,
            stream_callback=stream_callback,
        )

        # Save session after turn completes
        await self.session_manager.save_session(session_id, history)
        return result

    async def process_message(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message through the agent loop (non-streaming).

        Full ReAct flow with context assembly and history management.
        Used for internal turns and CLI interactions.

        Args:
            user_input: The user's message
            session_id: Optional session identifier

        Returns:
            Response dict with result and metadata
        """
        # Phase 1: Prepare turn (no lock needed)
        session_id, history, context_data, tool_infos, context = await self._prepare_turn(
            user_input, session_id
        )

        # Phase 2-3: Execute under lock
        async def _process_under_lock() -> dict[str, Any]:
            return await self._execute_agent_turn(
                user_input, session_id, history, context_data, tool_infos, stream_callback=None
            )

        try:
            return await self._run_under_session_lock(session_id, _process_under_lock)
        except Exception as e:
            logger.error(
                f"Agent processing failed for session {session_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    async def process_message_stream(
        self,
        user_input: str,
        session_id: str | None = None,
    ):
        """
        Process a user message through the agent loop with streaming support.

        Yields BAML chunks as they arrive for real-time UI updates.
        The first LLM call streams chunks; subsequent ReAct loop iterations
        are non-streaming for simplicity.

        Lock safety: Uses try/finally to ensure session is saved and lock is released
        even if the client disconnects mid-stream (GeneratorExit).

        Args:
            user_input: The user's message
            session_id: Optional session identifier

        Yields:
            dict: Chunk data with 'response' field for typewriter effect
        """

        # Phase 1: Prepare turn (no lock needed)
        session_id, history, context_data, tool_infos, context = await self._prepare_turn(
            user_input, session_id
        )

        lock = self.lock_manager.get_lock(session_id)
        # Queue to bridge the background thread/generator and the stream
        # Type-safe: holds raw Pydantic models from BAML
        queue: asyncio.Queue[StreamFinalResponse | StreamToolCall] = asyncio.Queue()

        try:
            async with asyncio.timeout(self.user_settings.session_lock_timeout_seconds):
                async with lock:
                    logger.debug(
                        f"Processing message (streaming) for session {session_id} (length: {len(user_input)})"
                    )

                    # Callback puts raw Pydantic models into the queue
                    def stream_callback(chunk: StreamFinalResponse | StreamToolCall) -> None:
                        queue.put_nowait(chunk)

                    # Run the agent turn in the background
                    agent_task = asyncio.create_task(
                        self._execute_agent_turn(
                            user_input,
                            session_id,
                            history,
                            context_data,
                            tool_infos,
                            stream_callback=stream_callback,
                        )
                    )

                    # Stream chunks using shared helper
                    async for chunk in self._stream_task_results(
                        agent_task, queue, session_id, is_autonomous=False
                    ):
                        yield chunk

        except (GeneratorExit, asyncio.CancelledError):
            # Client disconnected or task cancelled - ensure cleanup
            agent_task.cancel()
            try:
                # Wait for cancellation with timeout to prevent hanging session locks
                await asyncio.wait_for(agent_task, timeout=2.0)
            except TimeoutError:
                logger.warning(f"Agent task cancellation timed out for session {session_id}")
            logger.info(f"Stream client disconnected for session {session_id}")
            # Lock will be released by context manager
            # Session will be saved in finally block
            raise
        except TimeoutError:
            logger.warning(f"Session lock timeout: {session_id}")
            yield {
                "response": "The session is currently busy processing another request.",
                "type": "error",
                "session_id": session_id,
            }
        finally:
            # Ensure session is saved even if client disconnects
            # Note: _execute_agent_turn already saves, but we ensure it here for safety
            try:
                await self.session_manager.save_session(session_id, history)
            except Exception as save_error:
                logger.warning(
                    f"Failed to save session {session_id} after stream disconnect: {save_error}"
                )

    def _format_stream_chunk(
        self,
        chunk: StreamFinalResponse | StreamToolCall,
        session_id: str,
        is_autonomous: bool = False,
    ) -> dict[str, Any] | None:
        """
        Format a BAML streaming chunk into the expected front-end response format.

        Stream chunks are the LLM's reasoning process - they should NOT create action_requests.
        Only the ReAct loop's final result creates action_requests (with approval_id).

        Logic Table:
        - StreamFinalResponse → type: "regular" (Typewriter text)
        - StreamToolCall (INTERNAL, is_autonomous=True) → None (Silent in background)
        - StreamToolCall (INTERNAL, is_autonomous=False) → type: "notification" (System Pill)
        - StreamToolCall (EXTERNAL) → Filtered out (ReAct loop handles as action_request)
        - Final result from ReAct loop → Creates action_request (handled separately)

        Args:
            chunk: Raw Pydantic model from BAML stream (FinalResponse | ToolCall)
            session_id: Current session ID
            is_autonomous: Whether this is an autonomous turn (affects INTERNAL tool handling)

        Returns:
            Formatted chunk dict for front-end, or None if chunk should be filtered out
        """
        if isinstance(chunk, StreamFinalResponse):
            # FinalResponse: Always stream as regular typewriter text
            return {
                "response": chunk.answer or "",
                "type": "regular",
                "session_id": session_id,
            }

        elif isinstance(chunk, StreamToolCall):
            # Get tool classification to determine routing
            tool = self.plugin_loader.get_tool(chunk.tool_name)
            if tool is None:
                # Tool not found - filter out
                return None

            tool_classification = getattr(tool, "classification", "EXTERNAL")

            if tool_classification == "EXTERNAL":
                # EXTERNAL tools are handled by ReAct loop as action_requests
                # Filter out stream chunks - let ReAct loop handle it
                return None

            # INTERNAL tool handling
            if tool_classification == "INTERNAL":
                if is_autonomous:
                    # Silent in background - don't show system pill for autonomous tasks
                    return None
                else:
                    # User-facing INTERNAL tool - show as progress pill (not a notification)
                    return {
                        "type": "tool_use",
                        "tool_name": chunk.tool_name,
                        "content": f"Agent used {chunk.tool_name}",
                        "session_id": session_id,
                    }

        return None

    async def _stream_task_results(
        self,
        agent_task: asyncio.Task,
        queue: asyncio.Queue[StreamFinalResponse | StreamToolCall],
        session_id: str,
        is_autonomous: bool,
    ):
        """
        Private generator that consumes the queue and yields formatted chunks.

        Handles the boilerplate of checking task status, waiting for chunks,
        draining remaining chunks, and yielding the final result.

        Args:
            agent_task: The asyncio.Task running the agent turn
            queue: Queue containing raw Pydantic models from BAML stream
            session_id: Current session ID for chunk formatting
            is_autonomous: Whether this is an autonomous turn (affects INTERNAL tool handling)

        Yields:
            dict: Formatted chunk data for front-end
        """
        # Track which tools we've already announced to avoid duplicate progress pills
        yielded_tool_names = set()

        # Stream loop: Yield chunks as they arrive
        while not agent_task.done():
            try:
                # Wait for a chunk with a timeout to check task status
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
                formatted_chunk = self._format_stream_chunk(chunk, session_id, is_autonomous)

                # Deduplicate tool_use messages
                if formatted_chunk and formatted_chunk.get("type") == "tool_use":
                    tool_name = formatted_chunk.get("tool_name")
                    if tool_name in yielded_tool_names:
                        continue  # Skip duplicate tool announcements
                    yielded_tool_names.add(tool_name)

                if formatted_chunk:
                    yield formatted_chunk
            except TimeoutError:
                # No chunk yet, loop again to check if task is done
                continue

        # Task is done. Get the final result
        final_result = await agent_task

        # Drain any remaining chunks in the queue
        while not queue.empty():
            chunk = queue.get_nowait()
            formatted_chunk = self._format_stream_chunk(chunk, session_id, is_autonomous)

            # Deduplicate tool_use messages in drain phase too
            if formatted_chunk and formatted_chunk.get("type") == "tool_use":
                tool_name = formatted_chunk.get("tool_name")
                if tool_name in yielded_tool_names:
                    continue  # Skip duplicate tool announcements
                yielded_tool_names.add(tool_name)

            if formatted_chunk:
                yield formatted_chunk

        # Yield final result (this is the ONLY source of action_requests)
        if isinstance(final_result, dict):
            yield final_result

    async def resume_action(
        self,
        session_id: str,
        approval_id: str,
        decision: HitlDecision,
    ) -> dict[str, Any]:
        """
        Resume processing after a HITL action approval/denial.

        Loads the pending action, executes the tool if approved, and continues the ReAct loop.

        TEMPORARY LIMITATION: Only supports session-scoped actions.
        TODO: Add support for global-scope actions (session_id=None):
        - Add session_id: str | None parameter
        - When session_id is None, query EventLedger by approval_id
        - Load pending_action_data from EventLedger instead of ChatSession
        - Resume execution in global context (no session history, Address=USER)
        - Store result in EventLedger or user-level state instead of ChatSession

        Args:
            session_id: Session identifier (required, but will support None for global actions)
            approval_id: Approval identifier for the pending action
            decision: HitlDecision enum value (APPROVE or DENY)

        Returns:
            Response dict with result from continued agent loop
        """
        was_approved = decision == HitlDecision.APPROVE

        logger.info(
            f"Resuming action for session {session_id}: approval_id={approval_id}, "
            f"decision={decision}"
        )

        # Load pending action from database
        pending_action = await self.session_manager.load_pending_action(session_id)
        if not pending_action or pending_action.get("approval_id") != approval_id:
            raise StaleApprovalError(
                f"Pending action with approval_id {approval_id} not found or mismatch"
            )

        # Load session history (already converted to HistoryMessage)
        history = await self.session_manager.load_session(session_id)

        async def _resume_under_lock() -> dict[str, Any]:
            tool_name = pending_action["tool_name"]
            tool_args = pending_action["tool_args"]

            if not was_approved:
                # User denied: Add denial message and return
                denial_msg = f"User denied execution of tool '{tool_name}'"
                self.history_manager.add_error_message(history, denial_msg)
                await self.session_manager.save_session(session_id, history)
                return {
                    "response": denial_msg,
                    "type": "final_response",
                    "session_id": session_id,
                }

            # User approved: Execute the tool and continue the ReAct loop
            tool = self.plugin_loader.get_tool(tool_name)
            if tool is None:
                error_msg = f"Tool '{tool_name}' not found (may have been unloaded)"
                self.history_manager.add_error_message(history, error_msg)
                await self.session_manager.save_session(session_id, history)
                return {
                    "response": error_msg,
                    "type": "error",
                    "session_id": session_id,
                }

            try:
                # Execute the approved tool
                tool_result = await self._execute_tool(tool, tool_args)
                tool_result_str = self._stringify_tool_result(tool_result)

                # Log tool result with observation tags for developer visibility
                logger.info(f'<observation tool="{tool_name}">{tool_result_str}</observation>')

                # Add tool result to history
                self.history_manager.add_tool_result(history, tool_name, tool_result_str)

                # Clear pending action
                await self.session_manager.save_session(session_id, history)

                # Continue ReAct loop: Get context, compile prompt, send to LLM, parse
                trace_id = str(uuid.uuid4())
                context = MMCPContext(trace_id=trace_id)

                # Assemble context data for BAML
                await self.context_manager.assemble_llm_context(None, context)
                context_data = context.get_context_provider_data()

                # Trim history if needed
                self.history_manager.trim_history(history, self.user_settings)

                # Continue the ReAct loop after tool execution
                max_steps = self.user_settings.react_max_steps
                # Get cached tool infos for resume
                resume_tool_infos = self._get_cached_tool_infos()
                return await self.react_loop.resume_after_tool_execution(
                    context_data, history, session_id, max_steps, resume_tool_infos
                )

            except Exception as tool_error:
                logger.error(f"Tool execution failed after approval: {tool_error}", exc_info=True)
                error_msg = f"Tool execution failed: {str(tool_error)}"
                self.history_manager.add_error_message(history, error_msg)
                await self.session_manager.save_session(session_id, history)
                return {
                    "response": error_msg,
                    "type": "error",
                    "session_id": session_id,
                }

        return await self._run_under_session_lock(session_id, _resume_under_lock)

    async def resume_action_stream(
        self,
        session_id: str,
        approval_id: str,
        decision: HitlDecision,
    ):
        """
        Resume processing after a HITL action approval/denial with streaming support.

        Loads the pending action, executes the tool if approved, and continues the ReAct loop
        with streaming for real-time UI updates.

        TEMPORARY LIMITATION: Only supports session-scoped actions.
        TODO: Add support for global-scope actions (session_id=None):
        - Accept session_id: str | None parameter
        - When session_id is None, query EventLedger by approval_id for global actions
        - Load pending_action_data from EventLedger instead of ChatSession
        - Initialize empty history for global context (no session history)
        - Use global lock mechanism or approval_id-based locking instead of session lock
        - Store execution result in EventLedger or user-level state

        Args:
            session_id: Session identifier (required, but will support None for global actions)
            approval_id: Approval identifier for the pending action
            decision: HitlDecision enum value (APPROVE or DENY)

        Yields:
            dict: Chunk data with 'response' field for typewriter effect
        """
        was_approved = decision == HitlDecision.APPROVE

        logger.info(
            f"Resuming action (streaming) for session {session_id}: approval_id={approval_id}, "
            f"decision={decision}"
        )

        # Load pending action from database
        pending_action = await self.session_manager.load_pending_action(session_id)
        if not pending_action or pending_action.get("approval_id") != approval_id:
            raise StaleApprovalError(
                f"Pending action with approval_id {approval_id} not found or mismatch"
            )

        # Load session history (already converted to HistoryMessage)
        history = await self.session_manager.load_session(session_id)

        lock = self.lock_manager.get_lock(session_id)
        # Type-safe: holds raw Pydantic models from BAML
        queue: asyncio.Queue[StreamFinalResponse | StreamToolCall] = asyncio.Queue()

        try:
            async with asyncio.timeout(self.user_settings.session_lock_timeout_seconds):
                async with lock:
                    tool_name = pending_action["tool_name"]
                    tool_args = pending_action["tool_args"]

                    if not was_approved:
                        # User denied: Add denial message and return
                        denial_msg = f"User denied execution of tool '{tool_name}'"
                        self.history_manager.add_error_message(history, denial_msg)
                        await self.session_manager.save_session(session_id, history)
                        yield {
                            "response": denial_msg,
                            "type": "final_response",
                            "session_id": session_id,
                        }
                        return

                    # User approved: Execute the tool and continue the ReAct loop
                    tool = self.plugin_loader.get_tool(tool_name)
                    if tool is None:
                        error_msg = f"Tool '{tool_name}' not found (may have been unloaded)"
                        self.history_manager.add_error_message(history, error_msg)
                        await self.session_manager.save_session(session_id, history)
                        yield {
                            "response": error_msg,
                            "type": "error",
                            "session_id": session_id,
                        }
                        return

                    try:
                        # Execute the approved tool
                        tool_result = await self._execute_tool(tool, tool_args)
                        tool_result_str = self._stringify_tool_result(tool_result)

                        # Log tool result with observation tags for developer visibility
                        logger.debug(
                            f'<observation tool="{tool_name}">{tool_result_str}</observation>'
                        )

                        # Add tool result to history
                        self.history_manager.add_tool_result(history, tool_name, tool_result_str)

                        # Clear pending action
                        await self.session_manager.save_session(session_id, history)

                        # Continue ReAct loop: Get context, compile prompt, send to LLM, parse
                        trace_id = str(uuid.uuid4())
                        context = MMCPContext(trace_id=trace_id)

                        # Assemble context data for BAML
                        await self.context_manager.assemble_llm_context(None, context)
                        context_data = context.get_context_provider_data()

                        # Trim history if needed
                        self.history_manager.trim_history(history, self.user_settings)

                        # Get cached tool infos for resume
                        resume_tool_infos = self._get_cached_tool_infos()

                        # Callback puts raw Pydantic models into the queue
                        def stream_callback(chunk: StreamFinalResponse | StreamToolCall) -> None:
                            queue.put_nowait(chunk)

                        # Delegate to _execute_agent_turn (no new user input, tool result is in history)
                        # is_autonomous=False because this is a user-initiated action response
                        agent_task = asyncio.create_task(
                            self._execute_agent_turn(
                                user_input=None,  # Signal to use current history instead of new input
                                session_id=session_id,
                                history=history,
                                context_data=context_data,
                                tool_infos=resume_tool_infos,
                                stream_callback=stream_callback,
                            )
                        )

                        # Stream chunks using shared helper
                        async for chunk in self._stream_task_results(
                            agent_task, queue, session_id, is_autonomous=False
                        ):
                            yield chunk

                    except Exception as tool_error:
                        logger.error(
                            f"Tool execution failed after approval: {tool_error}", exc_info=True
                        )
                        error_msg = f"Tool execution failed: {str(tool_error)}"
                        self.history_manager.add_error_message(history, error_msg)
                        await self.session_manager.save_session(session_id, history)
                        yield {
                            "response": error_msg,
                            "type": "error",
                            "session_id": session_id,
                        }

        except (GeneratorExit, asyncio.CancelledError):
            logger.info(f"Resume action stream client disconnected for session {session_id}")
            raise
        except TimeoutError:
            logger.warning(f"Session lock timeout: {session_id}")
            yield {
                "response": "The session is currently busy processing another request.",
                "type": "error",
                "session_id": session_id,
            }
        finally:
            # Ensure session is saved even if client disconnects
            try:
                await self.session_manager.save_session(session_id, history)
            except Exception as save_error:
                logger.warning(
                    f"Failed to save session {session_id} after resume stream disconnect: {save_error}"
                )

    async def process_autonomous_turn(self, event: EventLedger) -> None:
        """
        Process an autonomous agent turn triggered by an ANP event.

        This implements Channel C (agent autonomous turns) where the agent
        can respond to system events without user interaction.

        Args:
            event: EventLedger instance triggering the autonomous turn
        """
        # 1. Prepare turn
        session_id, history, context_data, tool_infos, context = await self._prepare_turn(
            user_input=event.content, session_id=event.session_id
        )

        # 2. Inject Observation
        plugin_slug = event.event_metadata.get("plugin_slug", "system")
        self.history_manager.add_autonomous_observation(history, event.content, plugin_slug)

        # 3. Execute ReAct loop
        async with self.lock_manager.get_lock(session_id):
            result = await self._execute_agent_turn(
                session_id=session_id,
                history=history,
                context_data=context_data,
                tool_infos=tool_infos,
                user_input="",  # Empty for autonomous turns (observation already in history)
            )

            # 4. Narration (Channel C -> Channel A/B)
            if result.get("response"):
                # Create the Narration Notification
                narration = NotificationCreate(
                    content=result["response"],
                    session_id=session_id,
                    user_id=event.user_id,  # FIX: Propagate original user
                    routing={"address": "session", "target": "user", "handler": "system"},
                    metadata={
                        "anp_turn_depth": event.event_metadata.get("anp_turn_depth", 0) + 1,
                        "agent_narration": True,
                    },
                )
                # Just emit it. The EventBus handles the rest.
                await self.event_bus.emit_notification(narration)

    async def close(self) -> None:
        """Clean up all resources."""
        # Close prompt service
        await self.prompt.close()

        # Note: session_manager, lock_manager, and history_manager use automatic cleanup
        # (database sessions with context managers, WeakValueDictionary for locks)

        logger.info("Agent service shut down")
