import { onMount, onCleanup, Show } from "solid-js";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: any;
  actions?: any;
}

export default function Modal(props: ModalProps) {
  const handleBackdropClick = (e: MouseEvent) => {
    if (e.target === e.currentTarget) {
      props.onClose();
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      props.onClose();
    }
  };

  onMount(() => {
    if (props.isOpen) {
      document.addEventListener("keydown", handleKeyDown);
    }
  });

  onCleanup(() => {
    document.removeEventListener("keydown", handleKeyDown);
  });

  return (
    <div
      class={`fixed inset-0 z-50 flex items-center justify-center transition-opacity duration-300 ease-out ${
        props.isOpen
          ? "bg-black/20 opacity-100"
          : "bg-black/0 opacity-0 pointer-events-none"
      }`}
      onClick={handleBackdropClick}
    >
      <div
        class={`bg-[var(--bg-tertiary)] rounded-lg shadow-lg max-w-md w-full mx-4 transition-all duration-300 ease-out ${
          props.isOpen ? "scale-100 opacity-100" : "scale-95 opacity-0"
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <div class="px-6 py-4">
          <h3 class="text-lg font-semibold text-zinc-100">{props.title}</h3>
        </div>
        <div class="px-6 py-4">{props.children}</div>
        <Show when={props.actions}>
          <div class="px-6 py-4 flex justify-end gap-3">{props.actions}</div>
        </Show>
      </div>
    </div>
  );
}
