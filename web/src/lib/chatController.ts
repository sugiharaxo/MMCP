import { createEffect } from "solid-js";
import { sendMessageStream, respondToActionStream } from "./chat";
import { sessionManager } from "./sessionManager";
import { chatState } from "./chatState";
import { messageStore, type ChatResponse } from "./messageStore";
import { historyLoader } from "./historyLoader";

export class ChatController {
  private historyEffectInitialized = false;

  /**
   * Send a message and handle the streaming response
   */
  async sendMessage(text: string): Promise<void> {
    if (!text || chatState.isBusy()) return;

    // Add user message
    const userMsgId = crypto.randomUUID();
    messageStore.addMessage({
      id: userMsgId,
      sender: "user",
      content: text,
      isPersistent: false,
    });

    // Create placeholder for agent response
    const agentMsgId = crypto.randomUUID();

    try {
      // Stream the response
      let finalSessionId: string | null = null;
      for await (const chunk of sendMessageStream(
        text,
        sessionManager.currentSessionId()
      )) {
        // Capture session_id from backend
        if (chunk.session_id && !finalSessionId) {
          finalSessionId = chunk.session_id;
        }

        this.handleStreamChunk(chunk, agentMsgId);
      }

      // Update session if backend created a new one
      if (
        finalSessionId &&
        finalSessionId !== sessionManager.currentSessionId()
      ) {
        const firstLine = text.split("\n")[0].trim() || text.trim();
        sessionManager.setPendingTitle(firstLine);
        sessionManager.setSessionId(finalSessionId);
      }

      // Finalize streaming content
      this.finalizeStreamingResponse(agentMsgId);
    } catch (error) {
      console.error("Chat streaming failed:", error);
      messageStore.updateStreamingContent("Error: Failed to send message");
    }
  }

  /**
   * Resolve a pending action (approve/deny)
   */
  async resolveAction(approved: boolean, approvalId: string): Promise<void> {
    // Find the action request message by approval_id
    const messages = messageStore.messages();
    const actionMsg = messages.find(
      (msg) => msg.type === "action_request" && msg.approval_id === approvalId
    );

    if (!actionMsg || !actionMsg.approval_id) {
      console.warn(`Action request with approval_id ${approvalId} not found`);
      return;
    }

    // Mark the action as resolved immediately in the UI
    const status = approved ? "approved" : "denied";
    messageStore.updateMessage(actionMsg.id, { action_status: status });

    const decision = approved ? "approve" : "deny";
    const actionSessionId = sessionManager.currentSessionId();

    // TODO: Handle global-scope actions (session_id = null)
    if (!actionSessionId) {
      console.warn("Global-scope action resolution not yet implemented");
      return;
    }

    try {
      // Stream the response
      const agentMsgId = crypto.randomUUID();
      for await (const chunk of respondToActionStream(
        actionSessionId,
        approvalId,
        decision
      )) {
        this.handleStreamChunk(chunk, agentMsgId);
      }

      // Finalize streaming content
      this.finalizeStreamingResponse(agentMsgId);
    } catch (error) {
      console.error("Action resolution failed:", error);
      messageStore.updateStreamingContent("Error: Failed to resolve action");
    }
  }

  private handleStreamChunk(chunk: ChatResponse, agentMsgId: string): void {
    if (chunk.type === "action_request") {
      // Add action request as a message in the chat flow
      messageStore.addMessage({
        id: agentMsgId,
        sender: "agent",
        content:
          chunk.explanation || chunk.response || "Action requires approval",
        handler: "agent",
        isPersistent: false,
        type: "action_request",
        approval_id: chunk.approval_id,
        tool_name: chunk.tool_name,
        tool_args: chunk.tool_args,
        explanation: chunk.explanation,
        action_status: "pending",
      });
      messageStore.clearStreamingContent();
      // Action cards are user-visible; stop "Thinking..." once we reach this stage.
      messageStore.setIsThinking(false);
    } else if (chunk.type === "tool_use") {
      // Add tool execution message
      const toolUseId = crypto.randomUUID();
      messageStore.addMessage({
        id: toolUseId,
        sender: "system",
        content: chunk.content || "",
        handler: "system",
        isPersistent: false,
      });
      // Tool execution is also explicit feedback; don't keep "Thinking..." spinning.
      messageStore.setIsThinking(false);
    } else {
      // Regular streaming text
      messageStore.updateStreamingContent(chunk.response || "");
      // Once we have user-visible content, stop showing "Thinking..."
      if (chunk.response && chunk.response.trim().length > 0) {
        messageStore.setIsThinking(false);
      }
    }
  }

  private finalizeStreamingResponse(agentMsgId: string): void {
    const finalContent = messageStore.streamingContent();
    if (finalContent) {
      messageStore.addMessage({
        id: agentMsgId,
        sender: "agent",
        content: finalContent,
        isPersistent: false,
      });
    }
    messageStore.clearStreamingContent();
  }

  /**
   * Initialize the history loading effect exactly once.
   * Called from ChatView's lifecycle, not at module load time,
   * to avoid Solid creating reactive computations while the
   * singleton itself is being constructed.
   */
  initHistoryLoading(): void {
    if (this.historyEffectInitialized) return;
    this.historyEffectInitialized = true;

    createEffect(() => {
      const sessionId = sessionManager.currentSessionId();

      // Load history when session changes
      if (sessionId) {
        historyLoader.loadHistory(sessionId).then((history) => {
          if (history) {
            // Clear existing messages and set history
            messageStore.clearMessages();
            messageStore.setMessagesList(history);
          }
        });
      } else {
        // Clear messages for new session state
        messageStore.clearMessages();
      }
    });
  }
}

// Export singleton instance
export const chatController = new ChatController();
