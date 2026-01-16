import { chatState } from "./chatState";
import { messageStore } from "./messageStore";
import type { ChatResponse } from "./anp/types";

export async function* sendMessageStream(
  message: string,
  sessionId: string | null = null
): AsyncGenerator<ChatResponse> {
  chatState.setIsBusy(true);
  // Enter "thinking" state at the start of a new request
  messageStore.setIsThinking(true);
  try {
    const response = await fetch("/api/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: sessionId, // null for new session, existing ID to resume
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      console.error("Chat failed:", error);
      return;
    }

    if (!response.body) {
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // Split by newline for NDJSON
      const lines = buffer.split("\n");
      // Keep the last incomplete line in buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          yield JSON.parse(trimmed) as ChatResponse;
        } catch (e) {
          console.error("Failed to parse chunk:", e, trimmed);
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      try {
        yield JSON.parse(buffer.trim()) as ChatResponse;
      } catch (e) {
        console.error("Failed to parse final chunk:", e, buffer);
      }
    }
  } catch (error) {
    console.error("Chat streaming failed:", error);
  } finally {
    chatState.setIsBusy(false);
    // Fallback: ensure we don't get stuck in thinking state
    messageStore.setIsThinking(false);
  }
}

export async function sendMessage(
  message: string,
  sessionId: string | null = null
): Promise<ChatResponse | null> {
  let lastResponse: ChatResponse | null = null;
  for await (const chunk of sendMessageStream(message, sessionId)) {
    lastResponse = chunk;
  }
  return lastResponse;
}

/**
 * Stream response from agent after resolving a pending action (HITL approval/denial).
 *
 * TEMPORARY LIMITATION: Currently only supports session-scoped actions.
 * Global-scope actions (ANP Address=USER with null session_id) are not yet supported.
 *
 * TODO: Add support for global-scope actions:
 * - Accept sessionId: string | null
 * - Handle null session_id by querying EventLedger instead of ChatSession
 * - Backend must support null session_id in ActionResponse schema
 *
 * @param sessionId - Session identifier (required, but will support null for global actions)
 * @param approvalId - Approval identifier for the pending action
 * @param decision - User decision: "approve" or "deny"
 * @yields ChatResponse chunks from the agent's continued execution
 */
export async function* respondToActionStream(
  sessionId: string,
  approvalId: string,
  decision: "approve" | "deny"
): AsyncGenerator<ChatResponse> {
  chatState.setIsBusy(true);
  // Enter "thinking" state while resolving an action
  messageStore.setIsThinking(true);
  try {
    const response = await fetch("/api/v1/chat/respond", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        approval_id: approvalId,
        decision,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      console.error("Action response failed:", error);
      return;
    }

    if (!response.body) {
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // Split by newline for NDJSON
      const lines = buffer.split("\n");
      // Keep the last incomplete line in buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          yield JSON.parse(trimmed) as ChatResponse;
        } catch (e) {
          console.error("Failed to parse chunk:", e, trimmed);
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      try {
        yield JSON.parse(buffer.trim()) as ChatResponse;
      } catch (e) {
        console.error("Failed to parse final chunk:", e, buffer);
      }
    }
  } catch (error) {
    console.error("Action response streaming failed:", error);
  } finally {
    chatState.setIsBusy(false);
    messageStore.setIsThinking(false);
  }
}

export async function respondToAction(
  sessionId: string,
  approvalId: string,
  decision: "approve" | "deny"
): Promise<ChatResponse | null> {
  let lastResponse: ChatResponse | null = null;
  for await (const chunk of respondToActionStream(
    sessionId,
    approvalId,
    decision
  )) {
    lastResponse = chunk;
  }
  return lastResponse;
}

export interface HistoryMessage {
  id: string;
  sender: "user" | "agent" | "system";
  content: string;
  handler?: "system" | "agent";
  // Action request fields (when type === "action_request")
  type?: "action_request";
  approval_id?: string;
  tool_name?: string;
  tool_args?: Record<string, any>;
  explanation?: string;
  action_status?: "pending" | "approved" | "denied";
}

export async function fetchChatHistory(
  sessionId: string
): Promise<HistoryMessage[]> {
  const response = await fetch(
    `/api/v1/chat/history?session_id=${encodeURIComponent(sessionId)}`
  );
  if (!response.ok) {
    const error = await response.json();
    console.error("Failed to fetch chat history:", error);
    return [];
  }
  const data = await response.json();
  return data.messages || [];
}
