import { createSignal, createRoot } from "solid-js";

/**
 * ANP Manager
 *
 * Handles lease-based ownership fencing for ANP notifications (Section 6.3).
 * This is the ONLY responsibility of ANP - notification routing and lease management.
 *
 * Per ANP spec:
 * - Lease fencing prevents race conditions between Agent and Watchdog
 * - owner_lease is a monotonic integer that increments on ownership transfer
 * - Cross-tab synchronization ensures lease consistency across browser tabs
 *
 * All other concerns (session management, UI state) are handled by separate managers.
 */
function createANPManager() {
  const [activeLease, setActiveLease] = createSignal<number>(0);

  // Cross-tab synchronization for lease fencing
  // Per ANP spec Section 6.3: Lease-based ownership fencing prevents double delivery
  const channel = new BroadcastChannel("mmcp_lease_fence");

  channel.onmessage = (event) => {
    if (
      event.data.type === "LEASE_UPDATE" &&
      event.data.lease > activeLease()
    ) {
      setActiveLease(event.data.lease);
    }
  };

  const updateLease = (newLease: number) => {
    if (newLease > activeLease()) {
      setActiveLease(newLease);
      channel.postMessage({ type: "LEASE_UPDATE", lease: newLease });
    }
  };

  return {
    activeLease,
    updateLease,
  };
}

// Export as a singleton
export const anpManager = createRoot(createANPManager);
