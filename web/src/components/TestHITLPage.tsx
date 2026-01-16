import { createSignal } from "solid-js";
import { ActionCard } from "./ActionCard";

export default function TestHITLPage() {
  const [showAction, setShowAction] = createSignal(false);
  const [resolved, setResolved] = createSignal<{
    approved: boolean;
    approvalId: string;
  } | null>(null);

  const testAction = {
    tool_name: "test_external_action",
    tool_args: {
      message:
        "Hello from test HITL! This demonstrates complex argument rendering.",
      options: {
        timeout: 5000,
        retries: 3,
        headers: {
          "User-Agent": "MMCP/1.0",
          Accept: "application/json",
        },
      },
      targets: [
        { type: "url", value: "https://api.example.com/users" },
        { type: "file", value: "/tmp/data.json" },
      ],
      config: {
        method: "POST",
        validate: true,
        metadata: {
          created: "2024-01-16T10:00:00Z",
          tags: ["test", "hitl", "demo"],
        },
      },
    },
    approval_id: "test-approval-123",
    explanation:
      "This is a test action to demonstrate HITL functionality with complex nested arguments. The agent wants to perform an external action that requires your approval.",
  };

  const handleResolve = (approved: boolean, approvalId: string) => {
    setResolved({ approved, approvalId });
    setShowAction(false);
  };

  return (
    <div class="h-full flex flex-col bg-[var(--bg-primary)] p-8">
      <div class="max-w-3xl mx-auto w-full space-y-6">
        <div>
          <h1 class="text-2xl font-bold text-[var(--color-primary-highlight)] mb-2">
            HITL Test Page
          </h1>
          <p class="text-[var(--color-primary)] text-sm">
            Test the Human-in-the-Loop (HITL) ActionCard component
          </p>
        </div>

        <div class="flex gap-4">
          <button
            onClick={() => setShowAction(true)}
            class="bg-[var(--color-primary-highlight)] hover:opacity-80 text-[var(--bg-primary)] px-4 py-2 text-sm font-medium transition-opacity rounded"
          >
            Test HITL
          </button>
          <button
            onClick={() => {
              setShowAction(false);
              setResolved(null);
            }}
            class="border border-[var(--border-secondary)] hover:bg-[var(--bg-tertiary)] text-[var(--color-primary)] px-4 py-2 text-sm transition-colors rounded"
          >
            Reset
          </button>
        </div>

        {resolved() && (
          <div class="w-full px-1 sm:px-2">
            <div class="border border-[var(--border-secondary)] bg-[var(--bg-secondary)] rounded-md p-4">
              <div class="text-[var(--color-primary-highlight)] text-xs uppercase tracking-wide font-semibold">
                Action {resolved()?.approved ? "Approved" : "Denied"}:{" "}
                {testAction.tool_name}
              </div>
            </div>
          </div>
        )}

        {showAction() && (
          <div class="space-y-3">
            <ActionCard
              toolName={testAction.tool_name}
              toolArgs={testAction.tool_args}
              approvalId={testAction.approval_id}
              explanation={testAction.explanation}
              onResolve={handleResolve}
            />
          </div>
        )}
      </div>
    </div>
  );
}
