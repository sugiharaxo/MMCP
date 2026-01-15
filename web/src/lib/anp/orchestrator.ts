import { createSignal, createRoot } from "solid-js";
import { webSocket } from "rxjs/webSocket";
import { filter, map, tap } from "rxjs/operators";
import { NotificationSchema, type Notification } from "./types";
import { anpManager } from "./manager";

// Global session activity signal - triggers when sessions need reordering
function createSessionActivity() {
  const [sessionActivity, setSessionActivity] = createSignal<string | null>(
    null
  );

  const triggerSessionActivity = (sessionId: string) => {
    setSessionActivity(sessionId);
    // Reset immediately so it can trigger again for the same session
    setTimeout(() => setSessionActivity(null), 0);
  };

  return {
    sessionActivity,
    triggerSessionActivity,
  };
}

export const sessionActivity = createRoot(createSessionActivity);

// Construct the WS URL (handling dev proxy vs production)
const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
const host =
  window.location.port === "3000" ? "localhost:3000" : window.location.host;
const WS_URL = `${protocol}//${host}/api/v1/notifications/ws`;

export const socket$ = webSocket<any>(WS_URL);

export const notification$ = socket$.pipe(
  // 1. Basic type check
  filter((msg) => msg.type === "notification"),
  // 2. Validate against schema
  map((msg) => {
    const result = NotificationSchema.safeParse(msg);
    return result.success ? result.data : null;
  }),
  filter((n): n is Notification => n !== null),
  // 3. Lease Fencing: Only process NEW leases to prevent duplicates on reconnect (strict inequality)
  filter((n) => n.owner_lease > anpManager.activeLease()),
  // 4. Side Effect: Update the manager's state, broadcast to other tabs, and reorder session if applicable
  tap((n) => {
    anpManager.updateLease(n.owner_lease);
    // If notification is bound to a session, reorder it to the top (agent activity indicator)
    if (n.session_id) {
      sessionActivity.triggerSessionActivity(n.session_id);
    }
  })
);

export const sendAck = (id: string, lease_id: number) => {
  // Can be sent via WS or POST. The backend accepts single objects.
  fetch("/api/v1/notifications/ack", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, lease_id, type: "ack" }),
  });
};
