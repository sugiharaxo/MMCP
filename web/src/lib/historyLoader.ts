import { fetchChatHistory } from "./chat";
import type { DisplayMessage } from "./messageStore";

export class HistoryLoader {
  private lastProcessedSessionId: string | null = null;

  /**
   * Load chat history for a session if it hasn't been loaded yet
   * @param sessionId The session ID to load history for
   * @returns Promise that resolves to the history messages or null if already loaded
   */
  async loadHistory(sessionId: string | null): Promise<DisplayMessage[] | null> {
    // Don't fetch history for new sessions or temporary sessions
    if (!sessionId || sessionId.startsWith("temp-")) {
      return null;
    }

    // Don't re-process the same session
    if (this.lastProcessedSessionId === sessionId) {
      return null;
    }

    this.lastProcessedSessionId = sessionId;

    try {
      const history = await fetchChatHistory(sessionId);

      const historyMessages: DisplayMessage[] = history.map((msg) => ({
        id: msg.id,
        sender: msg.sender,
        content: msg.content,
        handler: msg.handler,
        isPersistent: true, // Messages from history should be ACKed if they're notifications
        // Copy action_request fields if present
        ...(msg.type && { type: msg.type }),
        ...(msg.approval_id && { approval_id: msg.approval_id }),
        ...(msg.tool_name && { tool_name: msg.tool_name }),
        ...(msg.tool_args && { tool_args: msg.tool_args }),
        ...(msg.explanation && { explanation: msg.explanation }),
        ...(msg.action_status && { action_status: msg.action_status }),
      }));

      return historyMessages;
    } catch (error) {
      console.error("Failed to load chat history:", error);
      return null;
    }
  }

  /**
   * Reset the loader state (useful for testing or when switching contexts)
   */
  reset() {
    this.lastProcessedSessionId = null;
  }
}

// Export singleton instance
export const historyLoader = new HistoryLoader();
