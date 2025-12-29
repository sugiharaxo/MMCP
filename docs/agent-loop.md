# MMCP Agent Loop Specification

## Overview

The MMCP Agent Loop implements a ReAct-style reasoning pattern using structured tool calling. It processes user input through multiple deterministic and probabilistic steps to provide intelligent responses about media management.

## Core Contract

### Input

- **Type**: `str` (user message)
- **Constraints**: Non-empty, UTF-8 encoded text
- **Source**: HTTP request body or direct API call

### Output

- **Success**: `str` (final response message to user)
- **Failure**: `MMCPError` exception (fatal errors only)
- **Guarantees**:
  - Never returns partial results
  - All tool outputs are sanitized for LLM consumption
  - Response is appropriate for user display
  - Trace ID is available for debugging

## Loop Structure

### Phase 1: Context Assembly (Deterministic)

1. **History Management**: Initialize or append to conversation history
2. **System Prompt Generation**: Build dynamic prompt with available tools
3. **Response Model Construction**: Create Union type `FinalResponse | ToolSchema1 | ToolSchema2 | ...`
4. **Context Trimming**: Ensure total history ≤ `llm_max_context_chars`

### Phase 2: LLM Decision (Probabilistic)

1. **Model Call**: Query LLM with current history and response model
2. **Response Routing**: Parse Union response to determine next action
3. **Validation**: Ensure response matches expected schema

### Phase 3: Tool Execution (Deterministic)

1. **Schema Mapping**: Convert Pydantic model to tool instance
2. **Argument Validation**: Use pre-validated Pydantic model
3. **Execution Isolation**: Run tool with 30s timeout
4. **Result Sanitization**: Convert output to string format

### Phase 4: Result Processing (Deterministic)

1. **Error Handling**: Map exceptions to structured error messages
2. **History Injection**: Feed results back as "Observation" messages
3. **Failure Tracking**: Increment tool failure counters
4. **Circuit Breaker**: Check for repeated tool failures

### Phase 5: Loop Control (Deterministic)

- **Continue**: If tool result received, return to Phase 2
- **Terminate**: If FinalResponse received or termination conditions met

## Deterministic vs Probabilistic Steps

### Deterministic (Always Predictable)

- Context assembly and trimming
- Tool schema mapping and validation
- Error handling and sanitization
- Circuit breaker logic
- Timeout enforcement
- History management

### Probabilistic (LLM-Dependent)

- Tool selection decisions
- Error interpretation and recovery strategies
- Response formatting and phrasing
- Multi-step reasoning flow

## Termination Conditions

### Normal Termination

- **FinalResponse**: LLM decides to answer user directly
- **Max Steps**: Loop reaches `max_steps` (default: 5) without resolution

### Abnormal Termination (MMCPError)

- **Configuration Error**: Missing required settings/API keys
- **Authentication Error**: Invalid or expired credentials
- **Provider Fatal Error**: LLM service completely unavailable

### Circuit Breakers

- **Tool Failures**: Same tool fails 3+ times in one conversation
- **LLM Validation Errors**: 2+ consecutive schema validation failures
- **Rate Limit Exhaustion**: 3+ rate limit hits without recovery

## Context Management

### Static Context

- **System Prompt**: Tool descriptions, identity, behavior rules
- **Configuration**: LLM settings, timeouts, limits
- **Environment**: Server state, available plugins

### Dynamic Context

- **Conversation History**: User/assistant message pairs
- **Tool Results**: Previous execution outcomes
- **Failure State**: Tool failure counters, retry counts

### Context Boundaries

- **Character Limit**: `llm_max_context_chars` (configurable)
- **Trimming Strategy**: Remove oldest non-system messages first
- **Preservation**: System prompt always retained

## Tool Execution Contract

### Input Contract

- **Schema**: Pydantic BaseModel with validated fields
- **Validation**: Pre-validated by LLM response parsing
- **Isolation**: No direct access to global state

### Execution Contract

- **Async Required**: All I/O operations must be async
- **Timeout**: 30 seconds maximum execution time
- **Error Handling**: Catch all exceptions, never crash loop
- **Resource Limits**: No unbounded memory/disk usage

### Output Contract

- **Type**: `Any` (string, dict, Pydantic model)
- **Sanitization**: Always converted to string for LLM
- **Structure**: Consistent format for error vs success cases
- **Completeness**: Either full result or structured error

## Error Handling Hierarchy

### Level 1: Tool Execution Errors

- **Scope**: Individual tool failures
- **Handling**: Map to ToolError, sanitize message, track failures
- **Recovery**: Feed error back to LLM for strategy adaptation

### Level 2: LLM Processing Errors

- **Scope**: Model calls, validation, parsing
- **Handling**: Retry with error context (up to limits)
- **Recovery**: Self-correction via error feedback

### Level 3: Provider Errors

- **Scope**: API failures, rate limits, authentication
- **Handling**: Exponential backoff, retry limits
- **Recovery**: Graceful degradation or fatal error

### Level 4: System Errors

- **Scope**: Configuration, setup, fatal conditions
- **Handling**: Immediate termination with MMCPError
- **Recovery**: Not recoverable, requires admin intervention

## Observability Requirements

### Structured Logging

- **Agent Decisions**: Tool selections, reasoning steps
- **Tool Calls**: Name, arguments, execution time, success/failure
- **Context Metrics**: History size, trimming events, character counts
- **Error Details**: Full stack traces with trace_id correlation

### Status Emission

- **Real-time**: Console output for development visibility
- **Correlation**: All events tagged with trace_id
- **Extensibility**: Framework for future WebSocket/UI integration

### Performance Monitoring

- **Latency**: LLM call duration, tool execution time
- **Resource Usage**: Memory, context size trends
- **Failure Rates**: Tool success rates, retry patterns

## Implementation Notes

### Union Response Model

```python
ResponseModel = FinalResponse
for schema in tool_schemas:
    ResponseModel = Union[ResponseModel, schema]
```

- Enables type-safe LLM responses
- Instructor handles discrimination automatically
- `isinstance()` routing at runtime

### Context Injection Pattern

- **Current**: Context object passed to all tool executions via declarative plugin configuration system
- **Implementation**: Namespaced environment variables (`MMCP_PLUGIN_{SLUG}_`) with SecretStr masking for credentials
- **Loading**: Two-phase system - static validation on startup, runtime context injection per tool execution

### Circuit Breaker Logic

- **Tool Level**: 3 failures → force FinalResponse
- **LLM Level**: 2 validation errors → feed back for self-correction
- **Rate Limit**: 3 hits → exponential backoff (1s, 2s, 4s)

## Future Extensions

### Context Engine Integration

- **Injection Point**: Phase 3 (tool execution)
- **Scope**: Static + runtime context passed to tools
- **Contract**: Tools declare required context dependencies

### Enhanced Observability

- **Metrics**: Prometheus-style performance monitoring
- **Tracing**: Distributed tracing across tool calls
- **Debug Mode**: Detailed execution logs for development

### Advanced Recovery

- **Fallback Tools**: Alternative implementations for failed tools
- **Partial Success**: Handle tools that succeed partially
- **State Persistence**: Resume conversations across restarts
