import { createSignal, createRoot } from "solid-js";

/**
 * Chat State Manager
 *
 * Handles UI state for chat operations (loading/busy state).
 * This is separate from ANP (which only handles notification routing).
 */
function createChatState() {
  const [isBusy, setIsBusy] = createSignal(false);

  return {
    isBusy,
    setIsBusy,
  };
}

// Export as a singleton
export const chatState = createRoot(createChatState);
