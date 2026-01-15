import { createSignal, createRoot } from "solid-js";
import { historyLoader } from "./historyLoader";

/**
 * Session Manager
 *
 * Handles session tracking, history loading, and cross-tab synchronization.
 * This is separate from ANP (which only handles lease fencing for notifications).
 */
function createSessionManager() {
  const [currentSessionId, setCurrentSessionId] = createSignal<string | null>(
    null
  );
  const [pendingSessionTitle, setPendingSessionTitle] = createSignal<
    string | null
  >(null);

  // Cross-tab synchronization channel for session changes
  const channel = new BroadcastChannel("mmcp_session_sync");

  channel.onmessage = (event) => {
    if (event.data.type === "SESSION_CHANGE") {
      // Ignore local changes (our own broadcasts)
      if (event.data.isLocal) {
        return;
      }

      // This is a cross-tab sync
      const newSessionId = event.data.sessionId;
      setCurrentSessionId(newSessionId);
    }
  };

  const _setSessionId = (sessionId: string | null, isLocal = false) => {
    setCurrentSessionId(sessionId);
    channel.postMessage({ type: "SESSION_CHANGE", sessionId, isLocal });
  };

  // Load a session (user navigation)
  const loadSession = (sessionId: string | null) => {
    _setSessionId(sessionId, true);
  };

  // Set session ID programmatically
  const setSessionId = (sessionId: string | null) => {
    _setSessionId(sessionId, true);
  };

  // Clear current session
  const clearSession = () => {
    _setSessionId(null);
    setPendingSessionTitle(null);
    // Reset history loader so that previously opened sessions
    // will reload their messages after returning from "new chat" state
    historyLoader.reset();
  };

  // Set pending title for new session
  const setPendingTitle = (title: string | null) => {
    setPendingSessionTitle(title);
  };

  return {
    currentSessionId,
    pendingSessionTitle,
    loadSession,
    setSessionId,
    clearSession,
    setPendingTitle,
  };
}

// Export as a singleton
export const sessionManager = createRoot(createSessionManager);
