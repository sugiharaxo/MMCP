class AGUI {
  constructor() {
    this.ws = null;
    this.currentSessionId = null;
    this.pendingAction = null;

    this.initElements();
    this.initEventListeners();
    this.loadSessions();
  }

  initElements() {
    this.sessionSelect = document.getElementById("session-select");
    this.newSessionBtn = document.getElementById("new-session-btn");
    this.connectionStatus = document.getElementById("connection-status");
    this.messages = document.getElementById("messages");
    this.messageInput = document.getElementById("message-input");
    this.sendBtn = document.getElementById("send-btn");
    this.actionModal = document.getElementById("action-modal");
    this.actionContent = document.getElementById("action-content");
    this.approveActionBtn = document.getElementById("approve-action");
    this.denyActionBtn = document.getElementById("deny-action");
  }

  initEventListeners() {
    this.newSessionBtn.addEventListener("click", () => this.createNewSession());
    this.sessionSelect.addEventListener("change", () => this.switchSession());
    this.messageInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") this.sendMessage();
    });
    this.sendBtn.addEventListener("click", () => this.sendMessage());
    this.approveActionBtn.addEventListener("click", () => this.approveAction());
    this.denyActionBtn.addEventListener("click", () => this.denyAction());
  }

  async loadSessions() {
    try {
      const response = await fetch("/api/v1/notifications/sessions");
      const sessions = await response.json();

      this.sessionSelect.innerHTML = '<option value="">New Session</option>';
      sessions.forEach((session) => {
        const option = document.createElement("option");
        option.value = session.id;
        option.textContent = `Chat ${session.id.slice(-8)}`;
        this.sessionSelect.appendChild(option);
      });
    } catch (error) {
      console.error("Failed to load sessions:", error);
    }
  }

  async createNewSession() {
    try {
      const response = await fetch("/api/v1/notifications/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const session = await response.json();

      this.addMessage(
        "system",
        `Created new chat session: ${session.id.slice(-8)}`
      );
      await this.loadSessions();
      this.switchToSession(session.id);
    } catch (error) {
      console.error("Failed to create session:", error);
      this.addMessage("error", "Failed to create new session");
    }
  }

  switchSession() {
    const sessionId = this.sessionSelect.value;
    if (sessionId) {
      this.switchToSession(sessionId);
    } else {
      this.currentSessionId = null;
      this.addMessage("system", "Switched to new session mode");
    }
  }

  switchToSession(sessionId) {
    if (this.currentSessionId === sessionId) return;

    this.currentSessionId = sessionId;
    this.addMessage(
      "system",
      `Switched to chat session: ${sessionId.slice(-8)}`
    );

    // Disconnect and reconnect WebSocket with new session
    this.disconnect();
    this.connect();
  }

  connect() {
    if (this.ws) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/notifications/ws`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      this.connectionStatus.innerHTML = "Connected";
      this.messageInput.disabled = false;
      this.sendBtn.disabled = false;
      this.addMessage("system", "Connected to agent");
    };

    this.ws.onmessage = (event) => {
      this.handleMessage(JSON.parse(event.data));
    };

    this.ws.onclose = () => {
      this.connectionStatus.innerHTML = "Disconnected";
      this.messageInput.disabled = true;
      this.sendBtn.disabled = true;
      this.ws = null;
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.addMessage("error", "Connection error occurred");
    };
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  handleMessage(data) {
    console.log("Received message:", data);

    if (data.type === "notification") {
      // Regular notification
      this.addMessage("agent", data.content);
    } else if (data.type === "action_request") {
      // Action request - this is the HITL interruption!
      this.showActionModal(data);
    }
  }

  addMessage(type, content) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `flex ${
      type === "user" ? "justify-end" : "justify-start"
    }`;

    const bubble = document.createElement("div");
    bubble.className = `chat-bubble rounded-lg px-4 py-2 ${
      type === "user"
        ? "bg-blue-500 text-white"
        : type === "agent"
        ? "bg-gray-200 text-gray-800"
        : type === "system"
        ? "bg-yellow-100 text-yellow-800 text-sm"
        : "bg-red-100 text-red-800 text-sm"
    }`;

    if (type === "system") {
      bubble.innerHTML = `${content}`;
    } else if (type === "error") {
      bubble.innerHTML = `${content}`;
    } else {
      bubble.textContent = content;
    }

    messageDiv.appendChild(bubble);
    this.messages.appendChild(messageDiv);
    this.messages.scrollTop = this.messages.scrollHeight;
  }

  showActionModal(data) {
    this.pendingAction = data;

    this.actionContent.innerHTML = `
        <div class="space-y-3">
            <div class="bg-red-50 border border-red-200 rounded p-3">
                <h4 class="font-semibold text-red-800 mb-2">Agent Explanation:</h4>
                <p class="text-red-700">${data.explanation}</p>
            </div>

            <div class="bg-gray-50 border rounded p-3">
                <h4 class="font-semibold text-gray-800 mb-2">Technical Details:</h4>
                <p class="text-sm text-gray-600">
                    <strong>Tool:</strong> ${data.tool_name}<br>
                    <strong>Action:</strong> Execute with provided arguments
                </p>
            </div>

            <div class="bg-amber-50 border border-amber-200 rounded p-3">
                <p class="text-amber-800 text-sm">
                    This action may modify external systems or data.
                    The agent has paused and is waiting for your approval.
                </p>
            </div>
        </div>
    `;

    this.actionModal.classList.remove("hidden");
    this.addMessage("system", `⏸Agent paused for approval: ${data.tool_name}`);
  }

  async approveAction() {
    if (!this.pendingAction) return;

    try {
      this.addMessage("user", `Approved: ${this.pendingAction.tool_name}`);

      const response = await fetch(
        `/api/v1/notifications/notifications/${this.pendingAction.event_id}/approve`,
        {
          method: "POST",
        }
      );

      if (response.ok) {
        const result = await response.json();
        this.addMessage("agent", result.response);
      } else {
        const error = await response.json();
        this.addMessage(
          "error",
          `Approval failed: ${error.detail || "Unknown error"}`
        );
      }
    } catch (error) {
      console.error("Approval error:", error);
      this.addMessage("error", "Failed to approve action");
    }

    this.hideActionModal();
  }

  denyAction() {
    if (this.pendingAction) {
      this.addMessage("user", `Denied: ${this.pendingAction.tool_name}`);
      this.addMessage(
        "system",
        "Action denied. Agent will continue with alternative approach."
      );
    }
    this.hideActionModal();
  }

  hideActionModal() {
    this.actionModal.classList.add("hidden");
    this.pendingAction = null;
  }

  sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message) return;

    this.addMessage("user", message);
    this.messageInput.value = "";

    // Send via HTTP POST to /chat endpoint
    this.sendChatMessage(message);
  }

  async sendChatMessage(message) {
    try {
      const payload = {
        message: message,
        session_id: this.currentSessionId,
      };

      const response = await fetch("/api/v1/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (response.ok) {
        const result = await response.json();
        if (result.type === "action_request") {
          // This is an interruption!
          this.showActionModal(result);
        } else {
          // Regular response
          this.addMessage(
            "agent",
            result.response || result.content || "Response received"
          );
        }
      } else {
        const error = await response.json();
        this.addMessage(
          "error",
          `Chat error: ${error.detail || "Unknown error"}`
        );
      }
    } catch (error) {
      console.error("Chat error:", error);
      this.addMessage("error", "Failed to send message");
    }
  }
}

// Initialize the AG-UI when page loads
document.addEventListener("DOMContentLoaded", () => {
  window.agui = new AGUI();
});
