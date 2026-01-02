"""Agent Orchestrator - manages the ReAct loop and conversation context."""

import asyncio
import json
import uuid
from datetime import timezone

from sqlalchemy import select

from app.agent.context_manager import ContextManager
from app.agent.history_manager import HistoryManager
from app.agent.llm_interface import LLMInterface
from app.agent.react_loop import ReActLoop
from app.agent.schemas import ActionRequestResponse
from app.agent.session_manager import SessionManager

# ANP imports (EventBus for agent turn lock, AgentNotificationInjector for async notifications)
from app.anp.agent_integration import AgentNotificationInjector
from app.anp.event_bus import EventBus
from app.core.context import MMCPContext
from app.core.health import HealthMonitor
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader


class AgentOrchestrator:
    """
    Orchestrates the ReAct loop for the agent.

    Manages conversation flow, tool execution, and HITL interruptions for external tools.
    Uses ANP only for asynchronous notifications, not for synchronous chat HITL.
    """

    def __init__(self, loader: PluginLoader, health: HealthMonitor | None = None):
        """
        Initialize the orchestrator.

        Args:
            loader: PluginLoader instance with loaded tools.
            health: HealthMonitor instance (singleton from app state). If None, creates a new one
                   (for backward compatibility, but not recommended for production).
        """
        self.loader = loader
        self.health = health or HealthMonitor()

        # Initialize components
        self.session_manager = SessionManager()
        self.react_loop = ReActLoop(loader, self._emit_status)
        self.llm_interface = LLMInterface()
        self.history_manager = HistoryManager()

        # ANP components
        self.event_bus = EventBus()
        self.notification_injector = AgentNotificationInjector(self.event_bus)
        self.context_manager = ContextManager(loader, self.notification_injector, self.health)

    async def _emit_status(
        self,
        message: str,
        trace_id: str | None = None,
        status_type: str = "info",  # noqa: UP007
    ):
        """
        Emit status messages for visibility and future extensibility.

        Logs structured messages for observability. UI components should subscribe
        to log streams rather than scraping console output.

        Args:
            message: The status message to emit
            trace_id: Optional trace ID for log correlation
            status_type: Type of status update ('tool_start', 'tool_end', 'thought', 'info')
                        Used by frontend to filter and display appropriate updates
        """
        extra = {"is_status_update": True, "status_type": status_type}
        if trace_id:
            extra["trace_id"] = trace_id
        logger.info(message, extra=extra)

    async def resume_action(
        self, approval_id: str, session_id: str, was_approved: bool, trace_id: str | None = None
    ) -> str:
        """
        Resume a paused conversation by approving/denying an external action.
        """
        from app.core.errors import StaleApprovalError

        # Use session lock for concurrency control
        async with self.session_manager.lock_manager.get_lock(session_id):
            # Load session and validate pending action
            from app.core.database import get_session

            async with get_session() as db_session:
                from app.anp.models import ChatSession

                stmt = select(ChatSession).where(ChatSession.id == session_id)
                result = await db_session.execute(stmt)
                chat_session = result.scalar_one_or_none()

                if not chat_session or not chat_session.pending_action:
                    raise ValueError(f"No pending action found for session {session_id}")

                if chat_session.pending_action.get("approval_id") != approval_id:
                    raise StaleApprovalError()

                # Update history based on approval/denial
                history = chat_session.history.copy()
                if not was_approved:
                    history.append(
                        {
                            "role": "user",
                            "content": "I reject this action. Please try something else.",
                        }
                    )
                else:
                    # Execute approved action
                    pending_action = chat_session.pending_action
                    tool_name = pending_action["tool_name"]
                    tool_args = pending_action["tool_args"]
                    tool_call_id = f"resumed-{approval_id}"

                    # 1. RECONSTRUCT THE ASSISTANT INTENT
                    # This tells the LLM: "You previously decided to call this tool"
                    history.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_args)
                                        if isinstance(tool_args, dict)
                                        else str(tool_args),
                                    },
                                }
                            ],
                        }
                    )

                    # 2. EXECUTE AND ADD RESULT
                    tool = self.loader.get_tool(tool_name)
                    if not tool:
                        raise ValueError(f"Tool '{tool_name}' not found")

                    context = MMCPContext(trace_id=trace_id or f"resumed-{approval_id}")
                    context.set_available_tools(self.loader.list_tools())

                    # Use react loop's safe_tool_call for consistency
                    # Note: Tool is already approved, so classification logging happens in safe_tool_call
                    result, _ = await self.react_loop.safe_tool_call(
                        tool, tool_name, tool_args, context
                    )

                    history.append(
                        {
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call_id,
                        }
                    )

                # Clear pending action and save
                chat_session.history = history.copy()
                chat_session.pending_action = None
                await db_session.commit()

            # Continue conversation
            # Recursively call chat with empty input to let LLM process the tool result
            return await self.chat(
                user_input="",
                session_id=session_id,
                skip_session_load=False,  # Reload the updated history we just committed
                trace_id=trace_id,
            )

    async def chat(
        self,
        user_input: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        skip_session_load: bool = False,
    ) -> str | ActionRequestResponse:
        """
        Main chat entry point. Coordinates session management, context preparation,
        and delegates to the ReAct loop for processing.

        TODO(ANP): Internal Turn Detection - ANP Spec Section 6.1
        Currently, this method doesn't distinguish between regular user-initiated turns
        and internal turns triggered by ANP notifications (Channel C with Target=AGENT).
        Internal turns should:
        1. Be detected when triggered by ANP notifications (Target=AGENT + Handler=AGENT)
        2. Allow agent to reason autonomously without user-visible output
        3. Automatically escalate to user-visible delivery when EXTERNAL tools are invoked
        4. Set FinalResponse.internal_turn=True to mark the turn as internal

        See: docs/specs/anp-v1.0.md Section 6.1 "The Internal Turn Protection"
        """
        trace_id = trace_id or str(uuid.uuid4())

        # Use session lock if session_id provided
        session_lock = (
            self.session_manager.lock_manager.get_lock(session_id) if session_id else asyncio.Lock()
        )

        async with session_lock:
            # Load session history
            history = (
                await self.session_manager.load_session(session_id)
                if (session_id and not skip_session_load)
                else []
            )

            # Prepare context
            context = MMCPContext(trace_id=trace_id)
            context.set_available_tools(self.loader.list_tools())
            await self.context_manager.assemble_llm_context(user_input, context)

            # Set last turn time for notifications
            from datetime import datetime

            self.notification_injector.set_last_turn_time(datetime.now(timezone.utc))

            # Global agent turn lock for causal serialization
            async with self.event_bus.get_agent_turn_lock():
                # Delegate to ReAct loop with system prompt
                system_prompt = await self.context_manager.get_system_prompt(context)

                # UNPACK TUPLE: Result and Updated History
                result, updated_history = await self.react_loop.execute_turn(
                    user_input, history, context, system_prompt, self.notification_injector
                )

                # Handle Session Persistence
                if session_id:
                    if isinstance(result, str):
                        # Standard response: Save updated history
                        # Strip the System Prompt (index 0) before saving to DB
                        db_history = (
                            updated_history[1:]
                            if updated_history and updated_history[0]["role"] == "system"
                            else updated_history
                        )
                        await self.session_manager.save_session(session_id, db_history)

                    elif isinstance(result, ActionRequestResponse):
                        # HITL Request: Save session with pending action
                        # Construct pending payload
                        pending_data = {
                            "approval_id": result.approval_id,
                            "tool_name": result.tool_name,
                            "tool_args": result.tool_args,
                            "context": context.model_dump()
                            if hasattr(context, "model_dump")
                            else {},
                        }

                        # Strip system prompt for DB save
                        db_history = (
                            updated_history[1:]
                            if updated_history and updated_history[0]["role"] == "system"
                            else updated_history
                        )

                        await self.session_manager.save_session_with_pending_action(
                            session_id, db_history, pending_data
                        )

                # Flush notification queue
                await self.event_bus.flush_system_alert_queue()

                return result
