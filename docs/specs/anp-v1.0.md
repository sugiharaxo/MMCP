# Agentic Notification Protocol (ANP)

Notification specifications for Modular Media Control Plane (MMCP).

## 1. Abstract

The **Agentic Notification Protocol (ANP)** is a communication standard designed for the **MMCP** project to bridge asynchronous plugin events with a personal agentic runtime. An agentic runtime is an execution environment where an autonomous agent processes events, maintains context, and makes decisions about information delivery.

ANP defines a structured mechanism for routing events within a dual-observer system, where both a Human User and an Autonomous Agent act as potential consumers of information. Unlike traditional pub/sub systems which target only a user interface, or agent-to-agent protocols which ignore user awareness, ANP standardizes the **State**, **Target**, **Handler**, and **ownership** of information flow. It provides a deterministic routing layer that assigns every system event to a single responsible entity (System or Agent) for processing, ensuring that plugins can interact with the system without bypassing the Agent's context or overwhelming the User's attention.

---

## 2. Terminology & Definitions

The protocol relies on three immutable flags that define the routing logic and behavioral lifecycle of a notification. These flags form a 3-axis semantic model that distinguishes protocol-level guarantees from optional UI advisory behavior.

### 2.1. Address (The Spatial Boundary)

Defines the spatial boundary and persistence of the notification.

- **SESSION:** Bound to a specific `session_id`. The notification exists within the conversational context of an active session.
- **USER:** Bound to a `user_id`. The notification exists in the user's global session state, independent of active conversations.

**Protocol Rule:** If a notification with `Address=SESSION` is dispatched and the `session_id` no longer exists at the time of delivery, the Event Bus must automatically change the `Address` to `USER` before routing. This ensures that session-bound notifications are not lost when sessions expire, and they are delivered to the user's global scope instead.

### 2.2. Target (The Delivery Guarantee)

Defines the delivery guarantee and initial receiver. Target determines the primary routing channel and establishes the delivery contract; it does not control processing or delivery medium (see Handler).

- **USER:** The **Human** is the Target. The protocol enters a **Strict Delivery Contract**: the notification _must_ be rendered in a human-observable UI. If `Handler=AGENT` and the Agent produces no output (Silence/Veto), the Watchdog must trigger an immediate fallback to `Handler=SYSTEM` to fulfill the delivery contract.
- **AGENT:** The **Agent** is the Target. The protocol routes the event to the Agent's internal context (Agent Context). Agent channels are **autonomous:** the agent may veto, internal turn, or decide whether to notify the human. Delivery to the human is **discretionary**.

### 2.3. Handler (The Processing Responsibility)

Defines the entity responsible for processing or presenting the event. Handler determines who has responsibility for handling the message; it is independent of who receives it first (see Target). When `Handler=SYSTEM`, the Agent may still receive the event for awareness (read-only context), but is not responsible for processing.

- **SYSTEM:** The **System** handles the message directly (Toast/Pill/History). The Agent may still receive the event for awareness (read-only context), but does not process it. This is a **protocol guarantee.** Delivery is immediate and reliable.
- **AGENT:** The **Agent** handles the message autonomously (Narration/Tool Calls). The System suppresses raw delivery and renders the Agent's output instead. This is an **autonomous decision**—the agent may act, decide, or narrate to the human.

---

## 3. The Core Axiom: Channel Monopoly

ANP enforces a strict routing rule to maintain state consistency across the Runtime and UI.

> "At any discrete point in time, a notification has exactly one **Responsible Entity**. Responsibility may be transferred (Escalation), but it can never be shared. This prevents race conditions and redundant signaling."

### 3.1. Primary Delivery Channels

Channels are classified by their **system-controlled** vs **autonomous** behavior:

1.  **Channel A: The Notification Inbox (Detached) - System-Controlled**

    - _Route:_ `Address=USER` + `Handler=SYSTEM`.
    - _Definition:_ A detached UI state (e.g., Badge/Feed) for accumulating passive system information.
    - _Protocol Behavior:_ **System-controlled processing:** immediate delivery without agent intervention.
    - _UI Advisory (Optional):_ Red Dot / Bell Icon. Ephemeral toast may appear based on implementation hints.
    - _Agent Access:_ If `Target=AGENT`, the event must be made available to the Agent's reasoning loop. _How_ that is done (Full injection, Summary, or Tool-pull) is an **Implementation-Specific Strategy**, not a protocol requirement.

2.  **Channel B: The Conversation Floor (Attached) - System-Controlled**

    - _Route:_ `Address=SESSION` + `Handler=SYSTEM` + `Target=USER`.
    - _Definition:_ An active injection into the chat history (e.g., System Message/Pill).
    - _Protocol Behavior:_ **System-controlled processing:** immediate injection into chat history.
    - _UI Advisory (Optional):_ System Pill in Chat. Chat bubble or inline message.
    - _Agent Access:_ **Read-Only**. The Agent observes this as historical context but is not the originator.

3.  **Channel C: The Agent Context (Autonomous)**
    - _Route:_ (`Handler=AGENT`) **OR** (`Target=AGENT`).
    - _Definition:_ An injection into the Agent's active execution loop (System Prompt).
    - _Protocol Behavior:_ **Autonomous agent handling:** the agent may veto, create an internal turn, or decide whether to notify the human.
    - _UI Advisory (Optional):_ None initially. Agent may choose to generate chat bubbles, tool calls, or remain silent.
    - _Agent Access:_ **Actionable**. The Agent is responsible for processing, ignoring, or vocalizing the event.

---

## 4. Protocol Behavioral Matrix

The following matrices define the routing behavior enforced by the three flags (`Address`, `Target`, `Handler`). The **Protocol Outcome** column specifies the hard routing guarantee—how the flags determine delivery responsibility. The **UI Manifestation** column shows example implementations for reference; these are **not** protocol requirements. The hard implementation guarantees are defined by how the three flags are processed (see Section 9).

### 4.1. User-Target Resolution Paths (The Delivery Contract)

_Events in this group have `Target=USER`. Delivery to the human is **guaranteed**._

| Address     | Target   | Handler    | Protocol Outcome          | Human Experience (Examples)                                 | Agent Experience                                                 |
| :---------- | :------- | :--------- | :------------------------ | :---------------------------------------------------------- | :--------------------------------------------------------------- |
| **USER**    | **USER** | **SYSTEM** | **Direct Presentation**   | **Out-of-Band Toast/Inbox.** Static system-authored text.   | **Ambient Awareness.** Added to ledger; seen in next turn.       |
| **USER**    | **USER** | **AGENT**  | **Mediated Presentation** | **Agent-Authored Toast.** Narrated text. High-urgency push. | **Immediate Narration.** Forces a global proactive turn.         |
| **SESSION** | **USER** | **SYSTEM** | **Direct Presentation**   | **In-Band Pill.** Static system-authored bubble in chat.    | **Read-Only Context.** Appears in chat history for next turn.    |
| **SESSION** | **USER** | **AGENT**  | **Mediated Presentation** | **Agent-Authored Turn.** Immediate conversation bubble.     | **Immediate Narration.** Forces a session-locked proactive turn. |

_Note that UI examples in the matrices are subjective and depend on the implementation of the ANP protocol_

**Protocol Rule:** In the `Target=USER` + `Handler=AGENT` path, if the Agent produces no output, the **Watchdog** must trigger an immediate fallback to `Handler=SYSTEM` to fulfill the delivery contract.

### 4.2. Agent-Target Resolution Paths (The Reasoning Path)

_Events in this group have `Target=AGENT`. Delivery to the human is **discretionary**._

| Address     | Target    | Handler    | Protocol Outcome         | Human Experience (Examples)                           | Agent Experience                                                           |
| :---------- | :-------- | :--------- | :----------------------- | :---------------------------------------------------- | :------------------------------------------------------------------------- |
| **USER**    | **AGENT** | **SYSTEM** | **Ambient Awareness**    | **None.** Human is blind to the event.                | **Passive Context.** Fact is synchronized to global memory for next turn.  |
| **USER**    | **AGENT** | **AGENT**  | **Autonomous Execution** | **None (Initially).** Escalates only if Agent speaks. | **Immediate Processing.** Global "Internal Turn" triggered now.            |
| **SESSION** | **AGENT** | **SYSTEM** | **Ambient Awareness**    | **None.** Human is blind to the event.                | **Passive Context.** Fact is synchronized to session memory for next turn. |
| **SESSION** | **AGENT** | **AGENT**  | **Autonomous Execution** | **None (Initially).** Escalates only if Agent speaks. | **Immediate Processing.** Session "Internal Turn" triggered now.           |

**Protocol Rule:** The Agent is **only** allowed to stay silent if `Target=AGENT`. If `Target=USER` and `Handler=AGENT`, the Agent is **mandated** to narrate. If it fails, the system fallback (Escalation) ensures the fallback text or raw content is shown.

---

## 5. Lifecycle & State Transitions

ANP defines a strict state machine for compliance auditing within the `event_ledger`. State transitions are atomic and enforce ownership fencing to prevent race conditions.

### 5.1. State Machine

The protocol enforces the following ordered transitions:

```
PENDING → DISPATCHED → LOCKED → (DELIVERED | ESCALATED → (DELIVERED | FAILED) | FAILED)
```

**States:**

1.  **PENDING:** Event accepted by the Event Bus. Not yet routed to any channel.
2.  **DISPATCHED:**
    - _System Channel:_ Payload sent via WebSocket.
    - _Agent Channel:_ Payload injected into System Prompt.
3.  **LOCKED:** A transient state where either the **Agent Runtime** or the **Watchdog** has claimed ownership via a `lease_id`. This prevents concurrent delivery attempts (see Section 6.3).
4.  **DELIVERED:**
    - _System Channel:_ Client sent `ACK` (Rendered/Clicked).
    - _Agent Channel:_ Agent returned notification ID in `acknowledged_ids`.
    - _Escalation Path:_ After escalation, successful System delivery transitions to this state.
5.  **ESCALATED:** Agent execution failed or exceeded TTL; ownership forcibly reverted to System Channel (Fallback). After escalation, the event is re-routed to System channels (Channel A or B) and may transition to `DELIVERED` upon successful system delivery, or to `FAILED` if system delivery also fails.
6.  **FAILED:** A terminal "Dead Letter" state used if both Agent delivery and System escalation fail (e.g., UI disconnected, network failure). Events in this state are logged for administrative review.

**Terminal States:** `DELIVERED` and `FAILED` are immutable terminal states. Once an event reaches either state, no further state transitions are permitted. `ESCALATED` is a transitional state that represents the fallback from Agent handling to System handling; it may progress to `DELIVERED` (successful system delivery) or `FAILED` (system delivery failure).

### 5.2. Shared Awareness (User Acknowledgment)

To facilitate shared context without redundant delivery:

- **Protocol Requirement:** When a notification with `Address=SESSION` transitions to `DELIVERED` (User ACK), the Agent Runtime must be made aware that the user has seen the notification. This helps maintain shared context between the agent and user.
- **Protocol Requirement:** The Agent Runtime must interpret user-acknowledged notifications as "Informational Context" (Do not announce) rather than "New Events" (Announce). The specific mechanism for communicating this awareness (e.g., markers, flags, or metadata) is an implementation detail.

---

## 6. Safety & Escalation

### 6.1. The Internal Turn Protection

**Protocol Requirement:** Agent actions must declare their impact classification (INTERNAL or EXTERNAL) via metadata. The Event Bus uses this classification to determine promotion behavior.

When routing to **Channel C (Agent Context)** with `Target=AGENT`:

- The Agent Runtime executes an "Internal Turn" (Non-UI turn).
- **Protocol Requirement:** All actions (tools) called by an agent implementing ANP must be marked with classification metadata (`EXTERNAL` or `INTERNAL`).
- **Constraint:** If the Agent invokes any action classified as `EXTERNAL` (user-visible state mutation, e.g., File I/O, API Write, external actions), the protocol automatically promotes `Target` to **USER** and the action **must be made aware to the user** (fulfilling the delivery contract). The specific mechanism for making the user aware (chat turn, toast notification, etc.) is an implementation detail.

### 6.1.1. Classification Semantics

| Classification | Protocol Guarantee                                                                                | Developer Intent                                                                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **EXTERNAL**   | If the action is invoked, the **user must be made aware** of it. This is a hard spec requirement. | Any operation that produces **user-visible effects**, regardless of whether it is invoked in an internal turn. Implementation of delivery is flexible (toast, chat, inbox), but **visibility is mandatory**. |
| **INTERNAL**   | Invocation is **audited**, but the user **does not have to be notified**.                         | Any operation that affects only **agent-internal context**, reasoning, memory, or ephemeral processing. Plugins may take any action internally; visibility is optional.                                      |

### 6.1.2. Implementation Notes

1. **Classification Requirement:** At the time a tool is invoked by an agent during ANP processing, the tool must have a classification (`EXTERNAL` or `INTERNAL`). The protocol does not prescribe when or how this classification is assigned (e.g., at tool registration, via runtime metadata, in plugin manifests, etc.); it only requires that the classification exists at invocation time. If classification is missing or undefined at invocation, the protocol defaults to `EXTERNAL` (safety-first: ensuring user visibility for unclassified actions).
2. **Agent Ignorance:** The agent cannot determine the classification itself; this is entirely a **protocol-level constraint**. The agent simply executes the action, and the Event Bus enforces the EXTERNAL/INTERNAL rules.
3. **Separation of Concerns:** EXTERNAL vs INTERNAL describes **visibility requirements**, not technical capabilities. A plugin can perform any internal effect, but if it's EXTERNAL, the system guarantees the user will be informed.
4. **Delivery Flexibility:** EXTERNAL does **not mandate how** the notification is presented—only that it reaches the user. The UI may choose toast, pill, chat injection, or other modalities.
5. **Consistency Across Contexts:** Even in internal turns, EXTERNAL actions **must be escalated** to user-visible delivery.

### 6.2. The Watchdog (TTL)

For events routed to **Channel C** (autonomous agent handling):

1.  The Event Bus sets a `delivery_deadline`. Implementation may use urgency hints to adjust TTL, but urgency is not a protocol-level flag.
2.  If the state does not transition to `DELIVERED` before the deadline:
3.  **Escalation Protocol:**
    - Event transitions to `LOCKED` state with Watchdog as owner.
    - `owner_lease` is incremented (see Section 6.3).
    - `Handler` flag is stripped (set to `SYSTEM`).
    - `Target` is promoted to **USER** (if currently AGENT).
    - Event transitions to `ESCALATED` state.
    - Event is re-routed to **Channel A** or **Channel B** (depending on Address) for immediate system-controlled delivery.
    - Upon successful System delivery, event transitions to `DELIVERED`.
    - If System delivery fails (e.g., UI disconnected), event transitions to `FAILED`.

### 6.3. Ownership Fencing (Lease-Based)

To prevent the race condition where an Agent attempts delivery after Watchdog escalation, ANP enforces **lease-based ownership fencing**.

**Mechanism:**

- Every event has an `owner_lease` (monotonic integer) that increments on ownership transfer.
- When an event is dispatched to the Agent, the Agent Runtime receives `(event_id, lease_id)`.
- When the Watchdog escalates, it increments `owner_lease` and claims ownership.
- **Protocol Requirement:** Any Agent action (narration, tool call, ACK) must include the current `lease_id`. The Event Bus must reject actions with stale leases (lease_id mismatch).
- **Database Enforcement:** State transitions must use atomic compare-and-swap operations that verify `owner_lease` matches the expected value. If the update affects zero rows, the operation is rejected (stale lease).

This ensures that late-arriving Agent responses are silently discarded, preventing double delivery and maintaining Channel Monopoly.

### 6.4. Causal Serialization

For events where both System (system-controlled) and Agent (autonomous) channels are active:

- **Protocol Requirement:** System alerts (Channel A/B) must appear in chat history **before** Agent processing (Channel C) to ensure causal coherence.
- **Implementation Note:** The Event Bus serializes System injections and Agent turns to guarantee ordering. If an Agent turn is in progress when a System alert arrives, the System alert is queued until the Agent turn completes, then inserted first. This specifically assumes a single logical event bus and serialized agent turns.

---

## 7. Deduplication & Idempotency

The deduplication key prevents notification flooding in scenarios involving frequent status updates, such as RSS feed polling.

> **Implementation Guidance:** Avoid using ANP for things like download progress reporting. Instead, use context providers to expose status information to agents. Deduplication keys remain useful for preventing notification spam in appropriate scenarios. You would use context injection to supply the download progress, and ANP to notify the download being finished.

### 7.1. Deduplication Logic

The `deduplication_key` is an optional string field that enables upsert behavior for notifications.

- **Key Generation:** Plugins define unique keys (e.g., `rss_feed_{feed_url}`) to identify related events.
  - When a notification with a `deduplication_key` matches an existing **`PENDING`** or **`DISPATCHED`** event:
    - The existing event's `content` and `metadata` are overwritten with new data.
    - The `delivery_deadline` is reset.
  - When matching a **`DELIVERED`** or **`ESCALATED`** event:
    - A new notification is created, preserving the original event's state.

---

## 8. Data Model (Schema)

### 8.1. Canonical Event Schema

```json
{
  "id": "evt_uuid_v4",
  "content": "Human readable notification content",
  "deduplication_key": "download_progress_xyz123",
  "metadata": { "key": "value" },
  "routing": {
    "address": "session | user",
    "target": "user | agent",
    "handler": "system | agent"
  },
  "agent_instructions": {
    "directive": "Optional string for the LLM",
    "tone_hint": "concise"
  },
  "status": "pending | dispatched | locked | delivered | escalated | failed",
  "owner_lease": 1,
  "created_at": "ISO8601",
  "ack_at": null,
  "delivery_deadline": null
}
```

### 8.2. Field Constraints

- `status`: Must follow the state machine defined in Section 5.1. Terminal states (`DELIVERED` and `FAILED`) are immutable. `ESCALATED` is a transitional state that may progress to `DELIVERED` or `FAILED`.
- `owner_lease`: Monotonic integer starting at 1. Incremented on ownership transfer (escalation). Used for lease-based fencing (Section 6.3).
- `delivery_deadline`: ISO8601 timestamp. Set for events routed to Channel C (autonomous agent handling). Used by Watchdog (Section 6.2). Implementation may use urgency hints to adjust TTL, but urgency is not a protocol-level flag.

---

## 9. Known Limitations

### 9.1. Agent Context Persistence TTL

**Limitation:** The protocol guarantees that notifications with `Target=AGENT` + `Handler=SYSTEM` are delivered to the agent's system prompt, but does not specify how long these notifications remain in the agent's context window.

**Current Behavior:** Notifications are injected into the system prompt on the next agent turn. How long they persist depends on the implementation's context management strategy (e.g., character-based trimming, conversation history limits).

**Future Consideration:** A protocol-level `context_ttl` field or metadata hint could be added in a future version to allow plugins to request minimum persistence guarantees.
