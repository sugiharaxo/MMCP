import { notification$, sendAck } from "./anp/orchestrator";
import { anpManager } from "./anp/manager";
import { messageStore, type DisplayMessage } from "./messageStore";

export class NotificationManager {
  private subscription: any = null;
  private observer: IntersectionObserver | null = null;
  private pendingAcks = new Set<string>();
  private debounceTimer: number | undefined;

  /**
   * Initialize the notification manager
   * Should be called in component onMount
   */
  initialize() {
    this.setupObserver();
    this.setupNotifications();
  }

  /**
   * Cleanup the notification manager
   * Should be called in component onCleanup
   */
  cleanup() {
    if (this.subscription) {
      this.subscription.unsubscribe();
    }
    if (this.observer) {
      this.observer.disconnect();
    }
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
  }

  /**
   * Observe a notification element for acknowledgment
   */
  observeNotification(element: HTMLElement, _notificationId: string) {
    if (this.observer) {
      this.observer.observe(element);
    }
  }

  private setupObserver() {
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.getAttribute("data-notification-id");
            if (id) this.pendingAcks.add(id);
          }
        });

        // Debounce acknowledgment
        if (this.debounceTimer) clearTimeout(this.debounceTimer);
        this.debounceTimer = window.setTimeout(() => {
          if (this.pendingAcks.size > 0) {
            const lease = anpManager.activeLease();
            this.pendingAcks.forEach((id) => {
              sendAck(id, lease);
            });
            this.pendingAcks.clear();
          }
        }, 500);
      },
      { threshold: 0.5 }
    );
  }

  private setupNotifications() {
    // Handle ASYNC Notifications
    this.subscription = notification$.subscribe((notification) => {
      // Prevent duplicates
      const existingMessages = messageStore.messages();
      if (existingMessages.find((m) => m.id === notification.id)) {
        return;
      }

      const notificationMessage: DisplayMessage = {
        id: notification.id,
        sender: notification.routing.handler === "system" ? "system" : "agent",
        content: notification.content,
        handler: notification.routing.handler,
        isPersistent: true, // ANP notifications should be ACKed
      };

      messageStore.addMessage(notificationMessage);
    });
  }
}

// Export singleton instance
export const notificationManager = new NotificationManager();
