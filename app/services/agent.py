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
from typing import Any, TypeVar

from app.agent.context_manager import ContextManager
from app.agent.history_manager import HistoryManager, HistoryMessage
from app.agent.react_loop import ReActLoop
from app.agent.session_manager import SessionLockManager
from app.agent.session_manager import SessionManager as AgentSessionManager
from app.anp.agent_integration import AgentNotificationInjector
from app.core.config import HitlDecision, UserSettings
from app.core.context import MMCPContext
from app.core.errors import StaleApprovalError
from app.core.health import HealthMonitor
from app.core.interfaces import LLMPrompt
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader
from app.services.prompt import ToolInfo

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
            health_monitor: HealthMonitor for context provider circuit breakers (optional)
        """
        self.plugin_loader = plugin_loader
        self.user_settings = user_settings
        self.prompt = prompt
        self.session_manager = session_manager

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
                    content=f"🤖 Agent Error: {str(llm_error)}",
                    routing=RoutingFlags(address="session", target="user", handler="system"),
                    session_id=session_id,
                    metadata={"error_type": type(llm_error).__name__},
                )
            )
        except Exception as emit_error:
            # Don't let notification emission errors break the main flow
            logger.warning(f"Failed to emit error notification: {emit_error}")

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
        Load session from database or create new one.

        Returns:
            Tuple of (session_id, conversation_history)
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Load history from database (already converted to HistoryMessage)
        history = await self.session_manager.load_session(session_id)
        return session_id, history

    async def process_message(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a user message through the agent loop.

        Full ReAct flow with context assembly and history management:
        1. Load session from database
        2. Assemble context from context providers
        3. Reconstruct history
        4. BAML.call_llm: End-to-end prompt → transport → SAP with ClientRegistry
        5. Route FinalResponse or ToolCall
        6. Update history and save to database

        Args:
            user_input: The user's message
            session_id: Optional session identifier
            turn_instructions: Optional temporary instructions for this specific turn
            **kwargs: Additional parameters (passed to BAML, may be ignored)

        Returns:
            Response dict with result and metadata
        """
        # Get session lock for concurrency control
        session_id, base_history = await self._load_or_create_session(session_id)

        async def _process_under_lock() -> dict[str, Any]:
            logger.debug(f"Processing message for session {session_id} (length: {len(user_input)})")

            # Create MMCPContext for this turn
            trace_id = str(uuid.uuid4())
            context = MMCPContext(trace_id=trace_id)

            # Phase 1: Context Assembly (Deterministic)
            await self.context_manager.assemble_llm_context(user_input, context)

            # Get context data from providers
            context_data = context.get_context_provider_data()

            # Use base_history directly
            history = base_history

            tool_infos = self._get_cached_tool_infos()

            # Phase 2: LLM Decision (Probabilistic)
            # Call BAML end-to-end: prompt → transport → SAP
            # Note: to_dict_list() creates a copy for API transport only;
            # the original 'history' list reference is preserved for mutations
            try:
                parsed_response = await self.prompt.call_llm(
                    tool_infos=tool_infos,
                    context_data=context_data,
                    user_input=user_input,  # Pass user input directly for runtime context display
                    history=HistoryManager.to_dict_list(history),  # Convert only for API transport
                    user_settings=self.user_settings,  # Pass settings for ClientRegistry
                )
            except Exception as llm_error:
                error_result = self._handle_llm_error(llm_error, history, session_id)
                await self.session_manager.save_session(session_id, history)
                return error_result

            # Add user input to history now that LLM has processed it
            self.history_manager.add_user_message(history, user_input)

            # Phase 3: ReAct Loop - Continue until FinalResponse
            max_steps = self.user_settings.react_max_steps
            return await self.react_loop.execute(
                parsed_response, context_data, history, session_id, max_steps, tool_infos
            )

        try:
            return await self._run_under_session_lock(session_id, _process_under_lock)
        except Exception as e:
            logger.error(
                f"Agent processing failed for session {session_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    async def resume_action(
        self,
        session_id: str,
        approval_id: str,
        decision: HitlDecision,
    ) -> dict[str, Any]:
        """
        Resume processing after a HITL action approval/denial.

        Loads the pending action, executes the tool if approved, and continues the ReAct loop.

        Args:
            session_id: Session identifier
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

    async def close(self) -> None:
        """Clean up all resources."""
        # Close prompt service
        await self.prompt.close()

        # Note: session_manager, lock_manager, and history_manager use automatic cleanup
        # (database sessions with context managers, WeakValueDictionary for locks)

        logger.info("Agent service shut down")
