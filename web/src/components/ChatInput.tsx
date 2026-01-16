import { createSignal, onMount } from "solid-js";

interface ChatInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export default function ChatInput(props: ChatInputProps) {
  const [input, setInput] = createSignal("");
  let textareaRef: HTMLTextAreaElement | undefined;

  const isSendDisabled = () => props.disabled || !input().trim();

  const autoResize = () => {
    if (textareaRef) {
      textareaRef.style.height = "auto";
      const newHeight = Math.min(textareaRef.scrollHeight, 200); // Max 200px height
      textareaRef.style.height = `${newHeight}px`;
    }
  };

  const handleInput = (e: Event) => {
    const target = e.currentTarget as HTMLTextAreaElement;
    setInput(target.value);
    autoResize();
  };

  const handleSend = () => {
    const text = input().trim();
    if (!text || props.disabled) return;
    props.onSend(text);
    setInput("");
    if (textareaRef) {
      textareaRef.style.height = "auto";
    }
  };

  const handleKeyPress = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  onMount(() => {
    if (textareaRef) {
      autoResize();
    }
  });

  return (
    <div class="relative w-full max-w-2xl pointer-events-auto">
      <textarea
        ref={textareaRef}
        value={input()}
        onInput={handleInput}
        onKeyPress={handleKeyPress}
        placeholder={props.placeholder || ""}
        disabled={props.disabled}
        rows={1}
        class="w-full bg-[var(--bg-secondary)] border border-[var(--border-primary)] rounded-2xl px-4 py-3 pr-12 focus:outline-none disabled:cursor-not-allowed text-zinc-100 placeholder-zinc-500 resize-none overflow-hidden break-words select-none"
        style="min-height: 48px; max-height: 200px;"
      />
      <button
        onClick={handleSend}
        disabled={isSendDisabled()}
        class="absolute right-2 top-1/2 -translate-y-1/2 flex items-center justify-center w-9 h-9 text-zinc-400 disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-default"
        classList={{ "hover:text-zinc-100": !isSendDisabled() }}
      >
        <svg
          class="w-5 h-5"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        >
          <path d="M18 15l-6-6-6 6" />
        </svg>
      </button>
    </div>
  );
}
