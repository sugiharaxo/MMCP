import {
  onMount,
  onCleanup,
  For,
  Show,
  createEffect,
  createSignal,
} from "solid-js";
import { sessionManager } from "../lib/sessionManager";
import { chatState } from "../lib/chatState";
import { messageStore } from "../lib/messageStore";
import { chatController } from "../lib/chatController";
import { notificationManager } from "../lib/notificationManager";
import { ActionCard } from "./ActionCard";
import ChatInput from "./ChatInput";
import ScrollArea from "./ScrollArea";
import MarkdownRenderer from "./MarkdownRenderer";
import StealthButton from "./StealthButton";

export default function ChatView() {
  const currentSessionId = () => sessionManager.currentSessionId();
  let scrollEl: HTMLDivElement | undefined;
  let previousMessageCount = 0;
  const [copiedMessageIds, setCopiedMessageIds] = createSignal(
    new Set<string>()
  );
  const copyTimeouts = new Map<string, number>();

  onMount(() => {
    // Initialize side-effectful singletons once the view is mounted
    chatController.initHistoryLoading();
    notificationManager.initialize();
  });

  onCleanup(() => {
    notificationManager.cleanup();
    copyTimeouts.forEach((timeout) => clearTimeout(timeout));
    copyTimeouts.clear();
  });

  // Auto-scroll to bottom when messages change
  createEffect(() => {
    const sessionId = currentSessionId();
    const messages = messageStore.messages();
    const streaming = messageStore.streamingContent();
    const action = messageStore.pendingAction();

    const el = scrollEl;
    if (!el) return;

    // Reset per-session state when session changes
    const messageCount = messages.length;
    const hasMessages = messageCount > 0;
    const messagesJustLoadedForSession =
      sessionId && hasMessages && previousMessageCount === 0;

    // On initial history load for a session (0 -> N messages), jump directly to bottom (no animation)
    if (messagesJustLoadedForSession) {
      // Ensure DOM has updated before jumping
      setTimeout(() => {
        if (!scrollEl) return;
        scrollEl.scrollTop = scrollEl.scrollHeight;
      }, 0);
      return;
    }

    // Guard: if user has scrolled up away from the bottom, don't auto-scroll.
    const threshold = 48; // px tolerance from the very bottom
    const distanceFromBottom = el.scrollHeight - el.clientHeight - el.scrollTop;
    if (distanceFromBottom > threshold) return;

    // For new messages / streaming / actions while user is at bottom, smooth-scroll
    if (hasMessages || streaming || action) {
      setTimeout(() => {
        if (!scrollEl) return;
        // Use a direct jump; no smooth animation to avoid visible scroll
        scrollEl.scrollTop = scrollEl.scrollHeight;
      }, 0);
    }

    previousMessageCount = messageCount;
  });

  const handleSend = async (text: string) => {
    await chatController.sendMessage(text);
  };

  const handleActionResolve = async (approved: boolean) => {
    await chatController.resolveAction(approved);
  };

  const handleCopyMessage = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedMessageIds((prev) => new Set(prev).add(messageId));
      const existingTimeout = copyTimeouts.get(messageId);
      if (existingTimeout) clearTimeout(existingTimeout);
      const timeout = setTimeout(() => {
        setCopiedMessageIds((prev) => {
          const next = new Set(prev);
          next.delete(messageId);
          return next;
        });
        copyTimeouts.delete(messageId);
      }, 3000);
      copyTimeouts.set(messageId, timeout);
    } catch (error) {
      console.error("Failed to copy message:", error);
    }
  };

  /**
   * Determine when to show the full chat view (messages area + footer input).
   *
   * We want a "centered input" when the user hasn't picked a session
   * and hasn't started typing yet, but as soon as they send the first
   * message in a brand new chat we should:
   * - show their message immediately
   * - stream the agent response
   * - pin the input to the bottom (like an active session)
   *
   * Rely on actual conversation activity (messages / streaming / actions)
   * as well as an explicitly-selected session, instead of requiring a
   * persisted session ID from the backend.
   */
  const hasActiveConversation = () => {
    const id = currentSessionId();
    const hasMessages = messageStore.messages().length > 0;
    const hasStreaming = !!messageStore.streamingContent();
    const hasPendingAction = !!messageStore.pendingAction();

    return !!id || hasMessages || hasStreaming || hasPendingAction;
  };

  return (
    <div class="h-full flex flex-col bg-[var(--bg-primary)] font-sans">
      <div class="flex-1 relative overflow-hidden min-h-0">
        {/* Messages / actions – only visible when there's a real session */}
        <div
          class="absolute inset-0 flex flex-col transition-all duration-200 ease-out"
          style={{
            opacity: hasActiveConversation() ? "1" : "0",
            transform: hasActiveConversation()
              ? "translateY(0)"
              : "translateY(8px)",
            "pointer-events": hasActiveConversation() ? "auto" : "none",
          }}
        >
          <ScrollArea
            class="flex-1 px-4 sm:px-8 pb-24 pt-6"
            ref={(el) => (scrollEl = el)}
          >
            <div class="mx-auto w-full max-w-3xl space-y-3">
              <For each={messageStore.messages()}>
                {(msg) => {
                  const isUser = msg.sender === "user";
                  const isAgent = msg.sender === "agent";
                  const isSystem = !isUser && !isAgent;

                  return (
                    <div
                      class={`w-full flex ${
                        isUser
                          ? "justify-start md:justify-end"
                          : "justify-start"
                      }`}
                      {...(msg.isPersistent
                        ? { "data-notification-id": msg.id }
                        : {})}
                      ref={(el) => {
                        // Observe notifications for acknowledgment
                        if (msg.handler === "system" && msg.isPersistent) {
                          notificationManager.observeNotification(el, msg.id);
                        }
                      }}
                    >
                      <div
                        class={
                          isUser
                            ? "inline-block max-w-full md:max-w-[70%] rounded-tl-2xl rounded-tr-2xl rounded-bl-2xl rounded-br-lg px-4 py-2 bg-[var(--bg-tertiary)] border border-[var(--border-secondary)] text-[var(--color-primary-highlight)] shadow-sm text-[15px] text-left"
                            : isAgent
                            ? "w-full px-1 sm:px-2 text-[15px] text-[var(--color-primary)] text-left"
                            : "inline-block max-w-full rounded-md px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border-secondary)] text-[var(--color-primary)] text-xs text-left"
                        }
                      >
                        <Show when={isSystem}>
                          <div class="mb-1 text-[10px] uppercase tracking-wide text-[var(--color-primary)]/70">
                            {msg.sender.toUpperCase()}
                          </div>
                        </Show>

                        <Show
                          when={isUser}
                          fallback={
                            <>
                              <MarkdownRenderer
                                content={msg.content}
                                class={isAgent ? "mmcp-markdown" : ""}
                              />
                              <Show when={isAgent}>
                                <div class="mt-2 flex justify-start pl-2 py-1">
                                  <StealthButton
                                    onClick={() =>
                                      handleCopyMessage(msg.id, msg.content)
                                    }
                                    class="text-xs opacity-60 hover:opacity-100"
                                    title="Copy message"
                                  >
                                    <Show
                                      when={copiedMessageIds().has(msg.id)}
                                      fallback={
                                        <svg
                                          class="w-4.5 h-4.5 transform scale-x-[-1]"
                                          fill="none"
                                          stroke="currentColor"
                                          viewBox="0 0 512 512"
                                        >
                                          <rect
                                            width="336"
                                            height="336"
                                            x="128"
                                            y="128"
                                            fill="none"
                                            stroke="currentColor"
                                            stroke-linejoin="round"
                                            stroke-width="32"
                                            rx="57"
                                            ry="57"
                                          />
                                          <path
                                            fill="none"
                                            stroke="currentColor"
                                            stroke-linecap="round"
                                            stroke-linejoin="round"
                                            stroke-width="32"
                                            d="m383.5 128 .5-24a56.16 56.16 0 0 0-56-56H112a64.19 64.19 0 0 0-64 64v216a56.16 56.16 0 0 0 56 56h24"
                                          />
                                        </svg>
                                      }
                                    >
                                      <svg
                                        class="w-4.5 h-4.5"
                                        fill="none"
                                        stroke="currentColor"
                                        viewBox="0 0 24 24"
                                      >
                                        <path
                                          stroke-linecap="round"
                                          stroke-linejoin="round"
                                          stroke-width="2"
                                          d="M5 13l4 4L19 7"
                                        />
                                      </svg>
                                    </Show>
                                  </StealthButton>
                                </div>
                              </Show>
                            </>
                          }
                        >
                          <div class="whitespace-pre-wrap break-words">
                            {msg.content}
                          </div>
                        </Show>
                      </div>
                    </div>
                  );
                }}
              </For>

              {/* Thinking indicator */}
              <Show when={messageStore.isThinking()}>
                <div class="w-full flex justify-start">
                  <div class="px-1 sm:px-2 text-[13px] tracking-wide text-[var(--color-primary-highlight)]/80 animate-pulse">
                    Thinking...
                  </div>
                </div>
              </Show>

              {/* Streaming content (no pulsing) */}
              <Show when={messageStore.streamingContent()}>
                <div class="w-full flex justify-start">
                  <div class="w-full px-1 sm:px-2 text-[15px] text-[var(--color-primary-highlight)] text-left">
                    <MarkdownRenderer
                      content={messageStore.streamingContent() || ""}
                    />
                  </div>
                </div>
              </Show>
            </div>
          </ScrollArea>

          <Show when={messageStore.pendingAction()}>
            {(action) => (
              <div class="px-8 pb-4">
                <ActionCard
                  toolName={action().tool_name || "unknown"}
                  toolArgs={action().tool_args || {}}
                  approvalId={action().approval_id || ""}
                  explanation={
                    action().explanation || "No explanation provided"
                  }
                  onResolve={handleActionResolve}
                />
              </div>
            )}
          </Show>
        </div>

        {/* Single canonical ChatInput – lerped between center and footer */}
        <div
          class="absolute left-0 right-0 p-8 flex justify-center transition-all duration-200 ease-out pointer-events-none"
          style={{
            top: hasActiveConversation() ? "auto" : "50%",
            bottom: hasActiveConversation() ? "0" : "auto",
            transform: hasActiveConversation()
              ? "translateY(0)"
              : "translateY(-50%)",
          }}
        >
          <ChatInput
            onSend={handleSend}
            disabled={
              chatState.isBusy() || messageStore.pendingAction() !== null
            }
            placeholder={hasActiveConversation() ? "" : "Talk to MMCP"}
          />
        </div>
      </div>
    </div>
  );
}
