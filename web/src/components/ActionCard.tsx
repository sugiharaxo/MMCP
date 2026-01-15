import { onMount, type Component } from "solid-js";

interface ActionCardProps {
  toolName: string;
  toolArgs: unknown;
  approvalId: string;
  explanation: string;
  onResolve: (approved: boolean) => void;
}

export const ActionCard: Component<ActionCardProps> = (props) => {
  // Lazily load jsfe only when an action card is actually rendered.
  // This avoids running the jsfe Solid integration during normal chats
  // that never show any tool requests, which can otherwise trigger
  // nested Solid update cycles in simple conversations.
  onMount(() => {
    // Fire and forget; the custom element will upgrade when defined.
    void import("@jsfe/form");
  });

  return (
    <div class="border border-emerald-900 bg-emerald-950/20 p-4 my-4 rounded-sm font-mono">
      <div class="text-emerald-500 text-xs mb-2 uppercase tracking-tighter font-bold">
        Action Request: {props.toolName}
      </div>
      <p class="text-zinc-300 text-sm mb-4 italic">"{props.explanation}"</p>

      <div class="bg-black/40 p-3 border border-zinc-800 mb-4">
        {/* jsfe-form handles the JSON visualization/editing */}
        <jsfe-form value={JSON.stringify(props.toolArgs)} readonly />
      </div>

      <div class="flex gap-2">
        <button
          onClick={() => props.onResolve(true)}
          class="bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-1 text-sm transition-colors"
        >
          APPROVE
        </button>
        <button
          onClick={() => props.onResolve(false)}
          class="border border-zinc-700 hover:bg-[var(--bg-tertiary)] px-4 py-1 text-sm transition-colors"
        >
          DENY
        </button>
      </div>
    </div>
  );
};
