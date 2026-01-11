---
alwaysApply: true
---

# MMCP Developer Guide

Quick reference for LLM assistants working on the Modular Media Control Plane (MMCP) project.

## Philosophy

- **Lightweight**: Near-zero resource consumption at idle (CPU/RAM). Runs on Raspberry Pis to high-end servers.
- **Idle Friendly**: No heavy background workers (Celery) or infrastructure (Redis) unless absolutely necessary.
- **Cross Platform**: Development on Windows, deployment on Linux. Code must be OS-agnostic (strict `pathlib` usage).
- **Ease of Use**: Designed for laymen. Simple configuration, clear error messages.
- **Modular Design**: Offload capabilities to plugins. Core system stays minimal.

## Technology Stack

- **Language**: Python 3.11+
- **Server**: FastAPI (Async)
- **Validation**: Pydantic v2
- **Database**: SQLite (SQLAlchemy Async)
- **Scheduling**: APScheduler (in-process memory store)
- **LLM**: BAML (prompt/transport/SAP)
- **Media**: yt-dlp (library mode), FFmpeg
- **Scraping**: curl_cffi (TLS fingerprinting), Selectolax (HTML), feedparser (RSS)
- **Dependencies**: Standard `venv` + `pip` (compatible with `uv`)

### Frontend Stack

- **UI Framework**: SolidJS - High-performance signals bound directly to DOM nodes. No VDOM overhead during streaming.
- **Stream Handler**: RxJS 7+ - Uses `switchMap` for **Lease Fencing** and `scan` for BAML chunk accumulation.
- **HITL Layer**: `jsfe` - A "Black Box" Web Component that renders tool schemas without the UI needing to know about them.
- **State Bridge**: BroadcastChannel - Ensures the `owner_lease` is consistent across multiple browser tabs to prevent double-interactions.
- **Acknowledgment**: Intersection Observer - Batched (500ms) signals to the backend to fulfill the ANP Shared Awareness requirement.
- **Validation**: Zod - Standard TypeScript-first validation for client-side inputs and API responses.
- **Styling**: Tailwind CSS - Provides the "Blocks AI" look with zero CSS bloat.
- **Build Tool**: Vite - Fast development and clean static `/dist` output for FastAPI.

## Architecture Overview

### LLM Integration (BAML Pipeline)

We use **BAML** to build prompts, send requests, and format responses via **SAP** (Structured API Protocol):

1. **Prompt Building**: BAML templates (`baml_src/main.baml`) define the agent prompt with tool schemas
2. **Transport**: BAML handles HTTP requests to LLM providers (OpenAI, Ollama, Anthropic)
3. **SAP Formatting**: BAML's SAP ensures structured output parsing (returns `FinalResponse | ToolCall`)
4. **Type Safety**: Pydantic schemas → BAML TypeBuilder → Runtime validation

The agent loop is ReAct-style: User input → LLM decision (ToolCall or FinalResponse) → Execute tool → Feed observation back → Repeat until FinalResponse.

### Notification System (ANP)

Notifications are routed through **ANP** (Agentic Notification Protocol). ANP defines three routing flags:

- **Address**: `SESSION` (conversation-bound) or `USER` (global)
- **Target**: `USER` (human delivery guaranteed) or `AGENT` (discretionary)
- **Handler**: `SYSTEM` (immediate delivery) or `AGENT` (autonomous processing)

See `docs/specs/anp-v1.0.md` for full specification.

### Plugin System

Plugins are discovered from the `/plugins` directory:

- Any `.py` file with a `Plugin` subclass is loaded
- Plugins define **Tools** (actions) and **Context Providers** (state information)
- Two-phase loading: (1) Config validation at startup, (2) Runtime context injection per execution
- Configuration via environment variables: `MMCP_PLUGIN_{PLUGIN_SLUG}_{SETTING_NAME}`
- Plugins receive sandboxed paths, system metadata, validated settings, and a namespaced logger

## Code Quality Rules

When writing code:

- **Modularize**: Split files when reaching ~500 lines. Non-monolithic design.
- **DRY, SOC, YAGNI**: Don't repeat yourself, separate concerns, you aren't gonna need it.
- **No Backwards Compatibility**: Never implement backwards compatibility at the cost of cleaner architecture.
  - No fallbacks
  - Prefer breaking changes if the result is cleaner architecture
  - No migration plans (greenfield project)
- **Trust Boundaries**: Don't implement unnecessary defensive programming. Validate at boundaries, trust upstream invariants.
- **Industry Standards**: Use standard patterns and conventions.

## Coding Standards

1. **Path Handling**: Always use `pathlib.Path`, never string concatenation.
   - ❌ `os.path.join("downloads", file)`
   - ✅ `Path("downloads") / file`
2. **Async/Await**: All I/O operations (DB, Network, File) must be `async`.
3. **Type Hints**: Strict type hints required for Pydantic/FastAPI OpenAPI generation.
4. **Error Handling**: Fail gracefully. Log structured errors, don't crash the server.
5. **Configuration**: Use `.env` files with `pydantic-settings`.
6. **Comments**: Do not write transactional comments, code should be self-descriptive.

## Glossary

- **ANP**: Agentic Notification Protocol - routing system for notifications between system, agent, and user
- **BAML**: Basically A Made Up Language (BoundaryML) - prompt/transport layer with structured output parsing
- **SAP**: Structured API Protocol - BAML's mechanism for ensuring structured LLM responses
- **ReAct Loop**: Reasoning + Acting loop - agent alternates between reasoning (LLM) and acting (tool execution)
- **Tool**: A callable action exposed to the agent (defined in plugins)
- **Context Provider**: A source of state information injected into agent prompts (defined in plugins)
- **Plugin**: A module containing Tools and/or Context Providers

## Documentation

- `docs/specs/anp-v1.0.md` - Full ANP specification
- `docs/agent-loop.v0.md` - Agent loop contract and execution details
- `docs/creating-a-plugin.md` - Plugin development guide
