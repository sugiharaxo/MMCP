import { createSignal, createRoot } from "solid-js";

export interface DisplayMessage {
  id: string;
  sender: "user" | "agent" | "system";
  content: string;
  handler?: "system" | "agent";
  isPersistent?: boolean; // True for messages from history/API that should be ACKed
  // Action request fields (when type === "action_request")
  type?: "action_request";
  approval_id?: string;
  tool_name?: string;
  tool_args?: Record<string, any>;
  explanation?: string;
  action_status?: "pending" | "approved" | "denied";
}

export interface ChatResponse {
  type: string;
  session_id?: string;
  approval_id?: string;
  tool_name?: string;
  tool_args?: Record<string, any>;
  explanation?: string;
  content?: string;
  response?: string;
}

function createMessageStore() {
  const [messages, setMessages] = createSignal<DisplayMessage[]>([]);
  const [streamingContent, setStreamingContent] = createSignal("");
  const [pendingAction, setPendingAction] = createSignal<ChatResponse | null>(
    null
  );
  const [isThinking, setIsThinking] = createSignal(false);

  // Actions for managing messages
  const addMessage = (message: DisplayMessage) => {
    setMessages((prev) => [...prev, message]);
  };

  const clearMessages = () => {
    setMessages([]);
    setStreamingContent("");
    setPendingAction(null);
    setIsThinking(false);
  };

  const setMessagesList = (newMessages: DisplayMessage[]) => {
    setMessages(newMessages);
  };

  const updateMessage = (messageId: string, updates: Partial<DisplayMessage>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === messageId ? { ...msg, ...updates } : msg))
    );
  };

  // Actions for managing streaming
  const updateStreamingContent = (content: string) => {
    setStreamingContent(content);
  };

  const clearStreamingContent = () => {
    setStreamingContent("");
    setIsThinking(false);
  };

  // Actions for managing pending actions
  const setPendingActionData = (action: ChatResponse | null) => {
    setPendingAction(action);
  };

  // Computed values
  const isEmpty = () => {
    return messages().length === 0 && !streamingContent() && !pendingAction();
  };

  return {
    // State
    messages,
    streamingContent,
    pendingAction,
    isThinking,

    // Actions
    addMessage,
    clearMessages,
    setMessagesList,
    updateMessage,
    updateStreamingContent,
    clearStreamingContent,
    setPendingActionData,
    setIsThinking,

    // Computed
    isEmpty,
  };
}

// Export as singleton
export const messageStore = createRoot(createMessageStore);
