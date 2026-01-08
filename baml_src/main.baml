// BAML Prompt Layer: Universal Agent Function
// Defines the contract for LLM-driven tool routing with observation feedback.

// JSON value type for flexible data structures
// Recursive types for function inputs (context, history)
// Note: These are input parameters only, not output types, so Pydantic recursion doesn't apply
type JsonValue = string | int | float | bool | JsonObject | JsonArray | null
type JsonObject = map<string, JsonValue>
type JsonArray = JsonObject[]

// Simple type for ToolCall.args to avoid Pydantic recursion
// TypeBuilder handles actual validation at runtime via @@dynamic
type ToolArgs = map<string, string>

// Final response when the agent has a complete answer
class FinalResponse {
  thought string @description("Brief internal reasoning.")
  answer string @description("The final message to the user.")
  //acknowledged_ids string[]?
  @@dynamic
}

// Tool call when the agent wants to execute a tool
class ToolCall {
  thought string @description("Brief internal reasoning.")
  tool_name string @description("Name of the tool to call.")
  // The TypeBuilder ensures the LLM knows what goes inside here.
  args ToolArgs @description("Arguments for the tool call.")
  @@dynamic
}

// Union type: Agent can either provide final answer or call a tool
type AgentResponse = FinalResponse | ToolCall

// Universal Agent function: Takes user input, tools, context, history, and client
// Returns either a final answer or a tool call
// user_input: Only provided for NEW user messages at the start of a turn.
//             During loop continuation (after tool execution), user_input is empty
//             and all messages (including the original user message) are in history.
function UniversalAgent(
  user_input: string,
  tools: string,
  context: JsonObject,
  history: JsonArray,
  merge_system: bool
) -> AgentResponse {
  // Client can be overridden via baml_options.client_registry at runtime
  // This default is only used if no client_registry is provided
  client "openai-generic/"

  prompt #"
	{% if not merge_system %}
	{{ _.role("system") }}
	{% else %}
	{{ _.role("user") }}
	{% endif %}


	### ROLE
	You are an instance of MMCP (Modular Media Control Plane), a helpful home server assistant.

	### OBJECTIVE
	Execute precise media operations or provide direct answers.
	You must output into one of your available JSON schemas.

	### CONSTRAINTS
	- Do not utilize conversational filler.
	- Be concise and deterministic.
	- If a tool fails, explain why to the user.

	{% if tools %}
	### TOOLS
	{{ tools }}
	{% endif %}

	### OUTPUT FORMAT
	{{ ctx.output_format }}

	**NEVER** output multiple schemas in the same message. **CHOOSE ONE.**

	### DECISION LOGIC
	Use the schema with `tool_name` when you want to call a tool
	Use the schema with `answer` when you want to talk directly to the user.

	EXAMPLE:
	user: "What's the weather in Tokyo?"
	assistant: {
		"thought": "The user is asking about the weather in Tokyo. I need to call the weather tool to get the information.",
		"tool_name": "weather",
		"args": {
			"city": "Tokyo"
		}
	}

	CRITICAL: **ALWAYS** output the schema with `thought` first to plan your action.

	{% if history %}
	### CONVERSATION HISTORY

	{% for msg in history %}

	{# --- Handle Tool Results (Observations) --- #}
	{% if msg.tool_result %}
	{{ _.role("user") }}
	<observation tool="{{ msg.tool_name }}">{{ msg.content }}</observation>

	{# --- Handle User Messages --- #}
	{% elif msg.role == "user" %}
	{{ _.role("user") }}
	{{ msg.content }}

	{# --- Handle Model Responses (Final or Tool) --- #}
	{% elif msg.role == "assistant" %}
	{{ _.role("assistant") }}
	{# CRITICAL: Use tojson so the model sees the exact JSON structure it outputted #}
	{{ msg.content | tojson }}
	{% endif %}

	{% endfor %}
	{% endif %}

	{% if user_input %}
	{{ _.role("user") }}
	{% if context %}
	### RUNTIME CONTEXT
	{% for key in context %}
	{{ key }}: {{ context[key] | tojson }}
	{% endfor %}

	---
	{% endif %}

	### USER INPUT
	{{ user_input }}
	{% endif %}
  "#
}
