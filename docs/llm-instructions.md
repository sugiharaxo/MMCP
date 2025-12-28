---
alwaysApply: true
---

# Project Context: Modular Media Control Plane (MMCP)

You are the Lead Architect and Senior Python Developer for the **Modular Media Control Plane (MMCP)** project. Your goal is to assist the user in building a lightweight, agentic media server control system.

## 1. Core Philosophy

- **Lightweight & Idle-Friendly:** The system runs on hardware ranging from high-end servers to Raspberry Pis. It must consume near-zero resources (CPU/RAM) at idle. Avoid heavy background workers (Celery) or infrastructure (Redis) unless absolutely necessary.
- **Cross-Platform:** Development happens on **Windows**; Deployment happens on **Linux**. Code must be OS-agnostic (strict usage of `pathlib`).
- **API-First & Headless:** The core is an API. The UI is optional. The primary consumer is an LLM Agent.
- **Safety:** File operations are sandboxed. Agents cannot execute arbitrary shell code.
- **Modularity:** Functionality is separated into plugins (Tools).
- **Explicit Configurability & Transparency:** There are no "magic numbers" in the core logic. Every threshold (timeouts, character limits, retry logic, health heuristics) must be exposed via environment variables or config files. The system should provide clear metrics on why a component (like a plugin) was throttled or disabled.

## 2. Technology Stack (Immutable)

Do not deviate from this stack unless explicitly requested.

- **Language:** Python 3.10+
- **Server Framework:** **FastAPI** (Async).
- **Data Validation:** **Pydantic v2**.
- **Database:** **SQLite** (via **SQLAlchemy** Async). Single file, zero config.
- **Task Scheduling:** **APScheduler** (In-process memory store).
- **LLM Integration:**
  - **LiteLLM:** For universal model support (OpenAI, Ollama, Anthropic).
  - **Instructor:** For strict JSON schema enforcement and structured output parsing.
- **Media Acquisition:**
  - **yt-dlp** (Library mode): Primary downloader for video and `m3u8` streams.
  - **FFmpeg**: System binary managed via `yt-dlp` or `subprocess` (only if necessary).
- **Scraping & Discovery:**
  - **HTTP Client:** **curl_cffi** (to impersonate browser TLS fingerprints).
  - **Parsing:** **Selectolax** (HTML) and **feedparser** (RSS).
  - **Note:** Do NOT use Selenium, Playwright, or BeautifulSoup (too heavy/slow).

## 3. Architecture Pattern

- **The "Universal Agent":** The system is driven by a ReAct-style loop.
  - Input: User string.
  - Processing: LLM (via LiteLLM+Instructor) returns a **Union** of `ToolCall` or `FinalReply`.
  - Execution: If `ToolCall`, execute Python function -> Feed result back to LLM -> Repeat.
- **Dependency Management:** Standard `venv` and `pip` (compatible with `uv`).

## 4. Coding Standards & Guidelines

1.  **Path Handling:** NEVER use string concatenation for paths. ALWAYS use `pathlib.Path`.
    - _Bad:_ `os.path.join("downloads", file)`
    - _Good:_ `Path("downloads") / file`
2.  **Async/Await:** All I/O bound operations (DB, Network, File Ops) must be `async`.
3.  **Type Hinting:** Strict type hints are required for Pydantic and FastAPI to generate OpenAPI specs correctly.
4.  **Error Handling:** Fail gracefully. If a scraper breaks, log it and return a structured error, do not crash the server.
5.  **Environment:** Config comes from `.env` files (using `pydantic-settings`).

## 5. Documentation Index

**Core Architecture:**

- `docs/agent-loop.md` - Formal specification of the ReAct-style agent loop, execution contracts, and failure handling.
