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

import uuid
from typing import Any

from app.agent.context_manager import ContextManager
from app.agent.history_manager import HistoryManager
from app.agent.session_manager import SessionLockManager
from app.agent.session_manager import SessionManager as AgentSessionManager
from app.anp.agent_integration import AgentNotificationInjector
from app.core.config import user_settings
from app.core.context import MMCPContext
from app.core.errors import AgentLogicError, ProviderError
from app.core.health import HealthMonitor
from app.core.interfaces import LLMPrompt
from app.core.logger import logger
from app.core.plugin_loader import PluginLoader


class AgentService:
    """Main agent orchestrator implementing AgentOrchestrator protocol.

    Pure orchestrator: Coordinates the ReAct loop with Prompt layer (BAML end-to-end).
    Uses dependency injection to accept any implementation of LLMPrompt.
    Transport removed - BAML handles prompt → transport → SAP with ClientRegistry caching.
    """

    def __init__(
        self,
        plugin_loader: PluginLoader,
        prompt: LLMPrompt | None = None,
        health_monitor: HealthMonitor | None = None,
        notification_injector: AgentNotificationInjector | None = None,
        session_manager: AgentSessionManager | None = None,
    ):
        """
        Initialize the agent service with dependency injection.

        Args:
            plugin_loader: PluginLoader with discovered tools
            prompt: LLMPrompt implementation (defaults to PromptService)
            health_monitor: HealthMonitor for context provider circuit breakers (optional)
            notification_injector: AgentNotificationInjector for ANP notifications (optional)
            session_manager: SessionManager for database persistence (optional, creates default)
        """
        self.plugin_loader = plugin_loader

        # Dependency injection: Use provided implementations or create defaults

        if prompt is None:
            from app.services.prompt import PromptService

            self.prompt: LLMPrompt = PromptService()
        else:
            self.prompt = prompt

        # Initialize managers
        self.history_manager = HistoryManager()
        self.session_manager = session_manager or AgentSessionManager()
        self.lock_manager = SessionLockManager()

        # Cache TypeBuilder for tool schema compilation (schemas are static per session)
        # This avoids rebuilding BAML DSL on every ReAct step
        self._type_builder_cache = None

        # ContextManager requires notification_injector
        if notification_injector is None:
            # Create a dummy one if not provided (for backward compatibility)
            logger.warning("No notification_injector provided, context assembly will be limited")
            # We'll create a minimal one - but this should be provided in production
            from app.anp.agent_integration import AgentNotificationInjector
            from app.anp.event_bus import EventBus

            dummy_event_bus = EventBus()
            notification_injector = AgentNotificationInjector(dummy_event_bus)

        self.context_manager = ContextManager(
            loader=plugin_loader,
            notification_injector=notification_injector,
            health=health_monitor,
        )

    def _build_tool_infos(self) -> list:
        """
        Build complete tool metadata from plugin loader.

        Returns:
            List of ToolInfo objects with name, description, schema, and classification.
        """
        from app.services.prompt import ToolInfo

        tool_infos = []
        for tool in self.plugin_loader.tools.values():
            if hasattr(tool, "input_schema") and tool.input_schema:
                tool_infos.append(
                    ToolInfo(
                        name=tool.name,
                        description=tool.description,
                        schema=tool.input_schema,
                        classification=getattr(tool, "classification", "EXTERNAL"),
                    )
                )
        return tool_infos

    async def _load_or_create_session(
        self, session_id: str | None
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Load session from database or create new one.

        Returns:
            Tuple of (session_id, conversation_history)
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        # Load history from database
        history = await self.session_manager.load_session(session_id)
        return session_id, history

    async def process_message(
        self,
        user_input: str,
        session_id: str | None = None,
        _turn_instructions: str | None = None,  # Part of protocol interface, currently unused
        **_kwargs: Any,  # Part of protocol interface, currently unused
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
        lock = self.lock_manager.get_lock(session_id)

        async with lock:
            logger.info(f"Processing message for session {session_id} (length: {len(user_input)})")

            try:
                # Create MMCPContext for this turn
                trace_id = str(uuid.uuid4())
                context = MMCPContext(trace_id=trace_id)

                # Phase 1: Context Assembly (Deterministic)
                await self.context_manager.assemble_llm_context(user_input, context)

                # Get context data from providers
                context_data = context.get_context_provider_data()

                # Use base_history directly
                history = base_history

                # Add user input to history at the start (following old pattern)
                # This ensures user message is in history, not passed separately
                self.history_manager.add_user_message(history, user_input)

                # Get complete tool metadata from plugin loader
                tool_infos = self._build_tool_infos()

                # Phase 2: LLM Decision (Probabilistic)
                # Call BAML end-to-end: prompt → transport → SAP
                try:
                    parsed_response = await self.prompt.call_llm(
                        tool_infos=tool_infos,
                        context_data=context_data,
                        user_input=None,  # User input is in history, no new input for this turn
                        history=history,
                        user_settings=user_settings,  # Pass settings for ClientRegistry
                    )
                except (AgentLogicError, ProviderError) as llm_error:
                    # These are already mapped and have proper retryable flags
                    logger.error(
                        f"LLM call failed for session {session_id}: {llm_error}", exc_info=True
                    )
                    error_msg = f"LLM call failed: {llm_error.message}"
                    self.history_manager.add_error_message(history, error_msg)
                    await self.session_manager.save_session(session_id, history)
                    return {
                        "response": error_msg,
                        "type": "error",
                        "session_id": session_id,
                        "retryable": llm_error.retryable,  # Add retryable flag
                    }
                except Exception as llm_error:
                    # Fallback for unmapped errors
                    logger.error(
                        f"Unexpected LLM error for session {session_id}: {llm_error}", exc_info=True
                    )
                    error_msg = f"LLM call failed: {str(llm_error)}"
                    self.history_manager.add_error_message(history, error_msg)
                    await self.session_manager.save_session(session_id, history)
                    return {
                        "response": error_msg,
                        "type": "error",
                        "session_id": session_id,
                    }

                # User message is already in history (added above)

                # Phase 3: ReAct Loop - Continue until FinalResponse
                max_steps = user_settings.react_max_steps
                step = 0

                while step < max_steps:
                    step += 1
                    logger.debug(f"ReAct loop step {step}/{max_steps} for session {session_id}")

                    # Import BAML types for type checking
                    from baml_client.types import FinalResponse

                    # Route based on parsed response type
                    if isinstance(parsed_response, FinalResponse):
                        # Final answer: save full JSON object to history
                        logger.debug(
                            f"Received final response for session {session_id} at step {step}"
                        )
                        # Save full JSON object (thought + answer) for model feedback
                        final_response_dict = parsed_response.model_dump()
                        self.history_manager.add_final_response(history, final_response_dict)

                        # Save updated history to database
                        await self.session_manager.save_session(session_id, history)

                        return {
                            "response": parsed_response.answer,
                            "type": "final_response",
                            "session_id": session_id,
                            "thought": parsed_response.thought,
                            "steps": step,
                            "content": final_response_dict,  # Include full JSON for UI
                        }
                    elif hasattr(parsed_response, "tool_name") and hasattr(parsed_response, "args"):
                        # Tool call: execute tool and loop back to LLM with result
                        tool_name = parsed_response.tool_name
                        tool_args = getattr(parsed_response, "args", {})

                        explanation = parsed_response.thought

                        logger.info(f"Agent Thought: {explanation}")
                        logger.debug(f"Received tool call for session {session_id}: {tool_name}")

                        # Save full JSON object (thought + tool_name + args) for model feedback
                        tool_call_dict = parsed_response.model_dump()
                        self.history_manager.add_tool_call(history, tool_call_dict)

                        # Generate tool_call_id for HITL tracking (if needed)
                        tool_call_id = str(uuid.uuid4())

                        # Look up the tool in PluginLoader
                        tool = self.plugin_loader.get_tool(tool_name)
                        if tool is None:
                            logger.warning(f"Tool '{tool_name}' not found in plugin loader")
                            error_msg = f"Tool '{tool_name}' not found or not available"
                            self.history_manager.add_error_message(history, error_msg)
                            await self.session_manager.save_session(session_id, history)
                            return {
                                "response": error_msg,
                                "type": "error",
                                "session_id": session_id,
                            }

                        # HITL / ANP Logic: Extract rationale for EXTERNAL tools
                        # Check if this tool is EXTERNAL (requires rationale)
                        tool_classification = getattr(tool, "classification", "EXTERNAL")
                        hitl_rationale = None

                        # Ensure tool_args is a dict (not already validated)
                        if not isinstance(tool_args, dict):
                            tool_args = dict(tool_args) if hasattr(tool_args, "__dict__") else {}

                        if tool_classification == "EXTERNAL":
                            # Extract the rationale (it should be there because BAML enforced it)
                            # Use .pop() so we remove it from arguments passed to the actual python function
                            hitl_rationale = tool_args.pop("rationale", None)

                            # Fallback: If LLM missed it (rare with BAML), use the thought
                            if not hitl_rationale:
                                raise ValueError(
                                    f"Protocol Violation: External tool '{tool_name}' requires a 'rationale' field "
                                    "explaining the action to the user. Please retry with the rationale provided."
                                )

                            logger.info(f"HITL Rationale for '{tool_name}': {hitl_rationale}")

                            # If EXTERNAL, require user approval before execution
                            from app.agent.schemas import ActionRequestResponse

                            approval_id = str(uuid.uuid4())

                            # Create action request response using rationale (not thought)
                            action_request = ActionRequestResponse(
                                approval_id=approval_id,
                                explanation=hitl_rationale,  # Use rationale for user-facing message
                                tool_name=tool_name,
                                tool_args=tool_args,  # Clean args (rationale already popped)
                                tool_call_id=tool_call_id,
                            )

                            # Save pending action to session
                            pending_action = {
                                "approval_id": approval_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,  # Clean args
                                "tool_call_id": tool_call_id,
                                "explanation": hitl_rationale,  # Use rationale
                            }
                            await self.session_manager.save_session_with_pending_action(
                                session_id, history, pending_action
                            )

                            logger.info(
                                f"EXTERNAL tool '{tool_name}' requires approval (approval_id: {approval_id})"
                            )

                            return {
                                "response": action_request.explanation,
                                "explanation": action_request.explanation,
                                "type": "action_request",
                                "session_id": session_id,
                                "approval_id": approval_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,  # Clean args
                                "content": tool_call_dict,  # Include full JSON for UI
                            }

                        # INTERNAL tool: Execute immediately
                        # tool_args is already clean (no rationale field for INTERNAL tools)
                        try:
                            # Validate tool_args against tool's input_schema
                            # This validation will pass because 'rationale' is gone (for EXTERNAL)
                            # or never existed (for INTERNAL)
                            if hasattr(tool, "input_schema") and tool.input_schema:
                                validated_args = tool.input_schema.model_validate(tool_args)
                                tool_result = await tool.execute(**validated_args.model_dump())
                            else:
                                tool_result = await tool.execute(**tool_args)

                            # Convert tool result to string for history
                            if isinstance(tool_result, dict):
                                import json

                                tool_result_str = json.dumps(tool_result, indent=2)
                            else:
                                tool_result_str = str(tool_result)

                            # Add tool result to history (as user message for next LLM turn)
                            self.history_manager.add_tool_result(
                                history, tool_name, tool_result_str
                            )

                            # Save updated history to database
                            await self.session_manager.save_session(session_id, history)

                            # Loop back: Compile prompt with tool result, send to LLM, parse response
                            # Trim history if needed before next iteration
                            self.history_manager.trim_history(history)

                            # Re-compile prompt with updated history (includes tool result)
                            # Rebuild tool_infos (tools may have changed availability)
                            tool_infos = self._build_tool_infos()

                            # Call BAML end-to-end with updated history
                            try:
                                parsed_response = await self.prompt.call_llm(
                                    tool_infos=tool_infos,
                                    context_data=context_data,
                                    user_input=None,  # No new user input, tool result is in history
                                    history=history,
                                    user_settings=user_settings,
                                )
                            except (AgentLogicError, ProviderError) as llm_error:
                                # These are already mapped and have proper retryable flags
                                logger.error(
                                    f"LLM call failed in ReAct loop (step {step}): {llm_error}",
                                    exc_info=True,
                                )
                                error_msg = f"LLM call failed: {llm_error.message}"
                                self.history_manager.add_error_message(history, error_msg)
                                await self.session_manager.save_session(session_id, history)
                                return {
                                    "response": error_msg,
                                    "type": "error",
                                    "session_id": session_id,
                                    "retryable": llm_error.retryable,
                                }
                            except Exception as llm_error:
                                # Fallback for unmapped errors
                                logger.error(
                                    f"Unexpected LLM error in ReAct loop (step {step}): {llm_error}",
                                    exc_info=True,
                                )
                                error_msg = f"LLM call failed: {str(llm_error)}"
                                self.history_manager.add_error_message(history, error_msg)
                                await self.session_manager.save_session(session_id, history)
                                return {
                                    "response": error_msg,
                                    "type": "error",
                                    "session_id": session_id,
                                }

                            # Continue loop to check if it's FinalResponse or another ToolCall
                            continue

                        except Exception as tool_error:
                            logger.error(
                                f"Tool execution failed for '{tool_name}': {tool_error}",
                                exc_info=True,
                            )
                            error_msg = f"Tool execution failed: {str(tool_error)}"
                            self.history_manager.add_error_message(history, error_msg)
                            await self.session_manager.save_session(session_id, history)
                            return {
                                "response": error_msg,
                                "type": "error",
                                "session_id": session_id,
                            }
                    else:
                        # Unknown response type
                        logger.warning(
                            f"Unknown response type for session {session_id}: {type(parsed_response)}"
                        )
                        error_msg = f"Unknown response type: {type(parsed_response).__name__}"
                        self.history_manager.add_error_message(history, error_msg)
                        await self.session_manager.save_session(session_id, history)
                        return {
                            "response": error_msg,
                            "type": "error",
                            "session_id": session_id,
                        }

                # Max steps reached without final response
                logger.warning(
                    f"ReAct loop reached max steps ({max_steps}) without final response for session {session_id}"
                )
                error_msg = f"Maximum ReAct steps ({max_steps}) reached without final answer"
                self.history_manager.add_error_message(history, error_msg)
                await self.session_manager.save_session(session_id, history)
                return {
                    "response": error_msg,
                    "type": "error",
                    "session_id": session_id,
                }

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
        decision: str,
    ) -> dict[str, Any]:
        """
        Resume processing after a HITL action approval/denial.

        Loads the pending action, executes the tool if approved, and continues the ReAct loop.

        Args:
            session_id: Session identifier
            approval_id: Approval identifier for the pending action
            decision: "approve" or "deny"

        Returns:
            Response dict with result from continued agent loop
        """
        was_approved = decision.lower() == "approve"

        logger.info(
            f"Resuming action for session {session_id}: approval_id={approval_id}, "
            f"decision={decision}"
        )

        # Load pending action from database
        pending_action = await self.session_manager.load_pending_action(session_id)
        if not pending_action or pending_action.get("approval_id") != approval_id:
            error_msg = f"Pending action with approval_id {approval_id} not found"
            logger.warning(error_msg)
            return {
                "response": error_msg,
                "type": "error",
                "session_id": session_id,
            }

        # Load session history
        history = await self.session_manager.load_session(session_id)
        lock = self.lock_manager.get_lock(session_id)

        async with lock:
            tool_name = pending_action["tool_name"]
            tool_args = pending_action["tool_args"]
            tool_call_id = pending_action.get("tool_call_id")  # noqa: F841 - Reserved for future use

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
                if hasattr(tool, "input_schema") and tool.input_schema:
                    validated_args = tool.input_schema.model_validate(tool_args)
                    tool_result = await tool.execute(**validated_args.model_dump())
                else:
                    tool_result = await tool.execute(**tool_args)

                # Convert tool result to string for history
                if isinstance(tool_result, dict):
                    import json

                    tool_result_str = json.dumps(tool_result, indent=2)
                else:
                    tool_result_str = str(tool_result)

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
                self.history_manager.trim_history(history)

                # Get complete tool metadata
                tool_infos = self._build_tool_infos()

                # Call BAML end-to-end (no new user input, tool result is in history)
                try:
                    parsed_response = await self.prompt.call_llm(
                        tool_infos=tool_infos,
                        context_data=context_data,
                        user_input=None,  # No new user input, tool result is in history
                        history=history,
                        user_settings=user_settings,
                    )
                except (AgentLogicError, ProviderError) as llm_error:
                    # These are already mapped and have proper retryable flags
                    logger.error(
                        f"LLM call failed after action approval: {llm_error}", exc_info=True
                    )
                    error_msg = f"LLM call failed: {llm_error.message}"
                    self.history_manager.add_error_message(history, error_msg)
                    await self.session_manager.save_session(session_id, history)
                    return {
                        "response": error_msg,
                        "type": "error",
                        "session_id": session_id,
                        "retryable": llm_error.retryable,
                    }
                except Exception as llm_error:
                    # Fallback for unmapped errors
                    logger.error(
                        f"Unexpected LLM error after action approval: {llm_error}", exc_info=True
                    )
                    error_msg = f"LLM call failed: {str(llm_error)}"
                    self.history_manager.add_error_message(history, error_msg)
                    await self.session_manager.save_session(session_id, history)
                    return {
                        "response": error_msg,
                        "type": "error",
                        "session_id": session_id,
                    }

                # Import BAML types for type checking
                from baml_client.types import FinalResponse

                # Continue the ReAct loop properly (same logic as process_message)
                max_steps = user_settings.react_max_steps
                step = 0

                while step < max_steps:
                    step += 1
                    logger.debug(
                        f"ReAct loop step {step}/{max_steps} for session {session_id} (resume_action)"
                    )

                    if isinstance(parsed_response, FinalResponse):
                        # Final answer: save full JSON object to history
                        final_response_dict = parsed_response.model_dump()
                        self.history_manager.add_final_response(history, final_response_dict)
                        await self.session_manager.save_session(session_id, history)
                        return {
                            "response": parsed_response.answer,
                            "type": "final_response",
                            "session_id": session_id,
                            "thought": parsed_response.thought,
                            "content": final_response_dict,  # Include full JSON for UI
                        }
                    elif hasattr(parsed_response, "tool_name") and hasattr(parsed_response, "args"):
                        # Another tool call - handle it (same logic as process_message)
                        tool_name = parsed_response.tool_name
                        tool_args = getattr(parsed_response, "args", {})
                        explanation = parsed_response.thought

                        logger.info(f"Agent Thought: {explanation}")
                        logger.debug(f"Received tool call for session {session_id}: {tool_name}")

                        # Save full JSON object for model feedback
                        tool_call_dict = parsed_response.model_dump()
                        self.history_manager.add_tool_call(history, tool_call_dict)

                        # Look up the tool
                        tool = self.plugin_loader.get_tool(tool_name)
                        if tool is None:
                            error_msg = f"Tool '{tool_name}' not found or not available"
                            self.history_manager.add_error_message(history, error_msg)
                            await self.session_manager.save_session(session_id, history)
                            return {
                                "response": error_msg,
                                "type": "error",
                                "session_id": session_id,
                            }

                        # Check if EXTERNAL (requires approval) or INTERNAL
                        tool_classification = getattr(tool, "classification", "EXTERNAL")
                        hitl_rationale = None

                        if not isinstance(tool_args, dict):
                            tool_args = dict(tool_args) if hasattr(tool_args, "__dict__") else {}

                        if tool_classification == "EXTERNAL":
                            hitl_rationale = tool_args.pop("rationale", None)
                            if not hitl_rationale:
                                raise ValueError(
                                    f"Protocol Violation: External tool '{tool_name}' requires a 'rationale' field"
                                )

                            logger.info(f"HITL Rationale for '{tool_name}': {hitl_rationale}")

                            from app.agent.schemas import ActionRequestResponse

                            approval_id = str(uuid.uuid4())
                            action_request = ActionRequestResponse(
                                approval_id=approval_id,
                                explanation=hitl_rationale,
                                tool_name=tool_name,
                                tool_args=tool_args,
                                tool_call_id=str(uuid.uuid4()),
                            )

                            pending_action = {
                                "approval_id": approval_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "tool_call_id": str(uuid.uuid4()),
                                "explanation": hitl_rationale,
                            }
                            await self.session_manager.save_session_with_pending_action(
                                session_id, history, pending_action
                            )

                            logger.info(
                                f"EXTERNAL tool '{tool_name}' requires approval (approval_id: {approval_id})"
                            )

                            return {
                                "response": action_request.explanation,
                                "explanation": action_request.explanation,
                                "type": "action_request",
                                "session_id": session_id,
                                "approval_id": approval_id,
                                "tool_name": tool_name,
                                "tool_args": tool_args,
                                "content": tool_call_dict,  # Include full JSON for UI
                            }

                        # INTERNAL tool: Execute immediately
                        try:
                            if hasattr(tool, "input_schema") and tool.input_schema:
                                validated_args = tool.input_schema.model_validate(tool_args)
                                tool_result = await tool.execute(**validated_args.model_dump())
                            else:
                                tool_result = await tool.execute(**tool_args)

                            if isinstance(tool_result, dict):
                                import json

                                tool_result_str = json.dumps(tool_result, indent=2)
                            else:
                                tool_result_str = str(tool_result)

                            self.history_manager.add_tool_result(
                                history, tool_name, tool_result_str
                            )
                            await self.session_manager.save_session(session_id, history)
                            self.history_manager.trim_history(history)

                            tool_infos = self._build_tool_infos()

                            # Continue loop with updated history
                            try:
                                parsed_response = await self.prompt.call_llm(
                                    tool_infos=tool_infos,
                                    context_data=context_data,
                                    user_input=None,
                                    history=history,
                                    user_settings=user_settings,
                                )
                            except (AgentLogicError, ProviderError) as llm_error:
                                error_msg = f"LLM call failed: {llm_error.message}"
                                self.history_manager.add_error_message(history, error_msg)
                                await self.session_manager.save_session(session_id, history)
                                return {
                                    "response": error_msg,
                                    "type": "error",
                                    "session_id": session_id,
                                    "retryable": llm_error.retryable,
                                }
                            except Exception as llm_error:
                                error_msg = f"LLM call failed: {str(llm_error)}"
                                self.history_manager.add_error_message(history, error_msg)
                                await self.session_manager.save_session(session_id, history)
                                return {
                                    "response": error_msg,
                                    "type": "error",
                                    "session_id": session_id,
                                }

                            continue  # Loop back to check FinalResponse vs ToolCall

                        except Exception as tool_error:
                            logger.error(f"Tool execution failed: {tool_error}", exc_info=True)
                            error_msg = f"Tool execution failed: {str(tool_error)}"
                            self.history_manager.add_error_message(history, error_msg)
                            await self.session_manager.save_session(session_id, history)
                            return {
                                "response": error_msg,
                                "type": "error",
                                "session_id": session_id,
                            }
                    else:
                        # Unknown response type
                        error_msg = f"Unknown response type: {type(parsed_response).__name__}"
                        self.history_manager.add_error_message(history, error_msg)
                        await self.session_manager.save_session(session_id, history)
                        return {
                            "response": error_msg,
                            "type": "error",
                            "session_id": session_id,
                        }

                # Max steps reached
                error_msg = f"Maximum ReAct steps ({max_steps}) reached without final answer"
                self.history_manager.add_error_message(history, error_msg)
                await self.session_manager.save_session(session_id, history)
                return {
                    "response": error_msg,
                    "type": "error",
                    "session_id": session_id,
                }

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

    async def close(self) -> None:
        """Clean up all resources."""
        # Close prompt if it has a close method (transport now handled by BAML)
        if hasattr(self.prompt, "close"):
            await self.prompt.close()

        logger.info("Agent service shut down")
