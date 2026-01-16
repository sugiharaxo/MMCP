import { type Component, createSignal } from "solid-js";
import { JsonValue } from "./JsonRenderer";
import StealthButton from "./StealthButton";

interface ActionCardProps {
  toolName: string;
  toolArgs: unknown;
  approvalId: string;
  explanation: string;
  status?: "pending" | "approved" | "denied";
  onResolve: (approved: boolean, approvalId: string) => void;
}

export const ActionCard: Component<ActionCardProps> = (props) => {
  const [showArgs, setShowArgs] = createSignal(false);
  const status = () => props.status || "pending";
  const isResolved = () => status() !== "pending";

  if (isResolved()) {
    return (
      <div
        class="w-full px-1 sm:px-2"
        style="user-select: none; -webkit-user-select: none; -moz-user-select: none;"
      >
        <div class="border border-[var(--border-secondary)] bg-[var(--bg-secondary)] rounded-md p-4">
          <div class="text-[var(--color-primary-highlight)] text-xs uppercase tracking-wide font-semibold">
            Action {status() === "approved" ? "Approved" : "Denied"}:{" "}
            {props.toolName}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      class="w-full px-1 sm:px-2"
      style="user-select: none; -webkit-user-select: none; -moz-user-select: none;"
    >
      <div class="mmcp-border-trail border border-[var(--border-secondary)] bg-[var(--bg-secondary)] rounded-md p-4 space-y-4">
        <div class="text-[var(--color-primary-highlight)] text-xs mb-2 uppercase tracking-wide font-semibold">
          Action Request: {props.toolName}
        </div>
        <p class="text-[var(--color-primary)]  text-sm">{props.explanation}</p>

        <div class="flex gap-2">
          <button
            onClick={() => props.onResolve(true, props.approvalId)}
            class="bg-[var(--color-primary-highlight)] hover:brightness-110 text-[var(--bg-primary)] px-4 py-2 font-medium transition-all rounded cursor-pointer"
          >
            APPROVE
          </button>
          <button
            onClick={() => props.onResolve(false, props.approvalId)}
            class="border border-[var(--border-secondary)] hover:bg-[var(--bg-tertiary)] text-[var(--color-primary)] px-4 py-2  transition-colors rounded cursor-pointer"
          >
            DENY
          </button>
          <StealthButton
            onClick={() => setShowArgs(!showArgs())}
            class="ml-auto"
          >
            {showArgs() ? "Hide action details" : "Show action details"}
          </StealthButton>
        </div>

        {showArgs() && (
          <div class="bg-[var(--bg-primary)] p-3 border border-[var(--border-secondary)] rounded">
            <JsonValue value={props.toolArgs} />
          </div>
        )}
      </div>
    </div>
  );
};
