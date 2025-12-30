# 🛠️ Developing MMCP Plugins

Welcome to the MMCP ecosystem. This guide will help you build **Tools** (actions for the agent) and **Context Providers** (state information for the agent).

## 🌍 The Plugin Environment

When you inherit from `Tool` or `ContextProvider`, your class is automatically equipped with everything it needs to interact with the system. You don't need to import paths or configure loggers; they are available directly on `self`.

| Property        | Type     | Purpose                                                                                              |
| :-------------- | :------- | :--------------------------------------------------------------------------------------------------- |
| `self.paths`    | `Object` | Access to sandboxed `pathlib.Path` folders: `.download_dir`, `.cache_dir`, `.data_dir`, `.root_dir`. |
| `self.system`   | `dict`   | System metadata: `["version"]`, `["environment"]`, `["os"]`.                                         |
| `self.settings` | `Model`  | Your validated plugin configuration (API keys, etc.).                                                |
| `self.logger`   | `Logger` | Namespaced async logger: `mmcp.{plugin}.{tool_or_key}`.                                              |

---

## 🚀 Quick Start: The Anatomy of a Plugin

A **Plugin** is a container for **Tools** (actions the agent takes) and **Context Providers** (information the agent reads). It uses a **Nested Class Pattern** to keep functionality organized and discoverable.

### Hello World

A simple plugin with no configuration or context provider.

```python
from app.api.base import Plugin, Tool

class SimplePlugin(Plugin):
    name = "simple_plugin"
    version = "1.0.0"

    class Ping(Tool):
        name = "ping_server"
        description = "Check if the media server is responding"

        async def execute(self) -> dict:
            return {"status": "pong"}
```

---

📖 **Further links to create your first plugin:**

- [Tools](#-tools) - Implementation details for actions.
- [Context Providers](#-context-providers) - Providing state to the agent.
- [Configuration & Secrets](#-configuration--secrets) - Manage API keys and settings.

---

## 🛠️ Tools

A **Tool** is a custom action MMCP can perform. The agent reads your tool named and description and inputs to decide when and how to use it.

Most tools need information from the user (e.g., a search query or a filename). We present the options to the LLM with a map using a [Pydantic model](#what-is-a-pydantic-model) named `Input`.

### How the Agent "Speaks" to your Tool

1. **You define an `Input` class**: This tells the agent exactly what fields you expect.
2. **You add `Field` descriptions**: These are instructions for the LLM (e.g., "The title of the movie").
3. **MMCP validates the data**: If the LLM sends a string where you expected an integer, MMCP catches the error before your code even runs.

```python
from typing import Literal
from pydantic import BaseModel, Field

class MediaSearch(Plugin):
    name = "media_search"
    version = "1.0.0"

    class SearchSite(Tool):
        name = "search_site"
        description = "Searches a site for media"

        # The Inputs for the Agent
        class Input(BaseModel):
            # Required field, with a minimum length of 1
            query: str = Field(..., min_length=1, description="The title to search for")
            # Optional field
            year: int | None = Field(None, description="Filter by release year")
            # Field with a default value, and a choice between two values
            media_type: Literal["movie", "tv"] = Field(
                "movie",
                description="The category of media"
            )

        async def execute(self, query: str, media_type: str) -> dict:

            self.logger.info(f"Searching for {media_type}: {query}")
            # A real implementation would actually search a site
            # and return real results and not my stupid chungus result
            # Using curl_cffi or other async libraries is encouraged
            return {"results": ["chungus"]}
```

> #### 💡 Understanding the `Input` Syntax
>
> - **The `...` (Ellipsis):** This simply means the field is **required**. If the Agent tries to call your tool without this field, MMCP will automatically stop it and ask for the missing info.
> - **The `description`:** This text isn't for other developers; it's a direct instruction to the **LLM Agent** explaining what this piece of data should be.

### Tool Metadata

Tools require specific metadata for the agent to discover and use them correctly.

| Field         | Description                                                |
| :------------ | :--------------------------------------------------------- |
| `name`        | Unique slug within the plugin. Used for agent calls.       |
| `description` | Clear explanation of what the tool does for the LLM Agent. |
| `Input`       | Nested Pydantic model defining inputs.                     |

### Implementation Pattern

When building tools, follow the pattern of:

1. **Define Input**: Use a nested `Input` class (BaseModel) for type safety.
2. **Handle Errors**: Return a dictionary with an `"error"` key on failure.
3. **Use Services**: Leverage `self.logger`, `self.settings`, `self.paths`, and `self.system`.

---

## 🧠 Context Providers

Sometimes the agent needs to know state **before** it starts thinking (e.g., "Is the VPN connected?"). You achieve this using a **Context Provider**.
Context Providers run before standard tools. They allow you to inject dynamic data into the agent's context based on the user's query.

### Required Metadata

Context Providers require specific metadata for the agent to discover and use them correctly.

| Field         | Description                                         |
| :------------ | :-------------------------------------------------- |
| `context_key` | The key where data will live in the agent's memory. |

### When to Use Context Providers

**Use Context Providers when:**

- The agent needs information _before_ deciding what to do ("What devices are online?")
- The information changes over time but doesn't require user input
- You want to avoid repeating the same check across multiple tools

**When NOT to use:**
Use regular Tools if the agent needs to _take an action_ (like "download video").

### How Context Providers Work

1. **Define the Context Key:** A unique string (e.g., `system_health`) used to identify the data in the agent's context.
2. **Optionally Determine Eligibility:** Implement `is_eligible` to check if the Context Provider should run for a specific query. This can keep the context tight or save resources, but it's also not discouraged to always provide context.
3. **Provide Data:** Implement `provide_context` to fetch the data and return it via `ContextResponse`.

### ContextProvider Metadata

| Attribute            | Required | Purpose                                                                |
| :------------------- | :------- | :--------------------------------------------------------------------- |
| `context_key`        | Yes      | The name used to access this data within the agent's context.          |
| `is_eligible`        | No       | A gatekeeper function. Returns `True` to run, `False` to skip.         |
| `ttl` (Time-To-Live) | No       | How long (in seconds) to cache this result before fetching fresh data. |

### Example: System Status Provider

The following example creates a context provider that checks system health only when relevant keywords ("health", "status") appear in the query. It also uses a Time-To-Live (TTL) to cache the result.

```python
from app.api.base import ContextProvider
from app.api.schemas import ContextResponse

class SystemStatus(ContextProvider):
    context_key = "system_health"

    async def is_eligible(self, query: str) -> bool:
        # Do whatever you want here:
        # only run certain queries, only run on weekends, whatever
        is_status_query = any(k in query.lower() for k in ["status", "health"])
        return is_status_query

    async def provide_context(self) -> ContextResponse:
        # Logic: Fetch current state via self.system or self.paths
        # The Agent will use this info to decide its next action.
        return ContextResponse(
            data={"cpu_temp": "45C", "vpn": "Connected"},
            ttl=60
        )
```

---

## 🔐 Configuration & Secrets

MMCP automatically maps environment variables or database entries to your plugin through **Settings Models**. These are [Pydantic](#what-is-a-pydantic-model) classes that you define, and MMCP injects the validated data into your plugin and its tools via `self.settings`.

### Settings Model Structure

Create a Pydantic class where each field corresponds to a configurable value:

- **Regular fields** use standard types like `str`, `int`, `bool`. They appear in logs and UI as-is.
- **Secret fields** use `SecretStr` to prevent accidental exposure. When accessed directly, `SecretStr` returns `"**********"` instead of the actual value.

```python
from pydantic import BaseModel, SecretStr
class MySettings(BaseModel):
    # Regular setting (visible in logs/UI)
    api_url: str = "https://api.example.com"

    # Secret setting (masked as '**********' in logs/UI)
    api_key: SecretStr
```

You can provide default values for either type. Defaults let your plugin work even if the user hasn't set a particular environment variable.

```python
class MySettings(BaseModel):
    api_url: str = "https://api.example.com"  # default
    timeout: int = 30                         # default

    api_key: SecretStr                         # required; no default
```

### Connecting Settings to Your Plugin

Assign your settings model to both the plugin and its tools. MMCP will automatically populate `self.settings` with validated data at runtime.

```python
class MyPlugin(Plugin):
    name = "my_plugin"
    settings_model = MySettings  # Links model to the plugin
    class MyTool(Tool):
        name = "do_action"
        async def execute(self, query: str):
            # Guard: settings may be None if not configured
            if not self.settings:
                return {"error": "Plugin not configured"}
            # Access regular field directly
            base_url = self.settings.api_url
            # Access secret field by unwrapping
            raw_key = self.settings.api_key.get_secret_value()
```

The `get_secret_value()` method extracts the actual string from a `SecretStr`. You only call it when you're ready to use the secret—typically in a network request or API call.

### Setting Values via Environment Variables

MMCP maps your settings fields to environment variables using this pattern:

```
MMCP_PLUGIN_{PLUGIN_NAME}_{FIELD_NAME}
```

All parts are uppercase, with underscores separating components.
| Field Name | Environment Variable |
| :--- | :--- |
| `api_url` | `MMCP_PLUGIN_MY_PLUGIN_API_URL` |
| `timeout` | `MMCP_PLUGIN_MY_PLUGIN_TIMEOUT` |
| `api_key` | `MMCP_PLUGIN_MY_PLUGIN_API_KEY` |

---

## 📝 Logging

MMCP provides a pre-configured `self.logger` instance for every tool.
This logger is **namespaced**, meaning all log messages are automatically prefixed with your plugin and tool name (e.g., `mmcp.tmdb.lookup_metadata`). It is **async-safe**, so you can call it within `execute` without blocking the event loop. It also supports **structured logging**, allowing you to attach extra data fields to your messages.

### Common Logging Patterns

```python
async def execute(self, query: str, filename: str):
    # Log when something important happens
    self.logger.info(f"Downloaded {filename}")

    # Log for debugging (hidden in production)
    self.logger.debug("API response received", extra={"status_code": 200})

    try:
        results = await self.search_api(query)

        # Log warnings for recoverable issues
        if len(results) == 0:
            self.logger.warning("Using cached data because API is slow")

        # Structured: Attach 'count' as a separate field for log parsers
        self.logger.debug("Search completed", extra={"count": len(results)})
        return {"results": results}

    except Exception as e:
        # Log errors with context
        self.logger.error(f"Search failed for query: {query}", extra={"error": str(e)})
        return {"error": "Search failed"}
```

### Logger API Reference

The logger is namespaced with your plugin and tool name (e.g., `mmcp.tmdb.lookup_metadata`), so logs automatically include context about which component generated them.

| Method      | Parameters                 | Description                |
| :---------- | :------------------------- | :------------------------- |
| `info()`    | `message: str`, `**kwargs` | Log informational messages |
| `error()`   | `message: str`, `**kwargs` | Log error messages         |
| `debug()`   | `message: str`, `**kwargs` | Log debug messages         |
| `warning()` | `message: str`, `**kwargs` | Log warning messages       |

---

## 🌐 System Access

Every Tool and Context Provider has direct access to the system environment. We handle the complexity of paths and server metadata so you don't have to.

### Accessing System Directories

MMCP automatically creates and manages specific directories for your plugin. You access them through `self.paths`, which is a [Pydantic model](#what-is-a-pydantic-model) containing `pathlib.Path` objects.

| Directory     | Purpose                        | Access Path               |
| :------------ | :----------------------------- | :------------------------ |
| **Downloads** | Media files, downloads         | `self.paths.download_dir` |
| **Cache**     | Temporary files, API responses | `self.paths.cache_dir`    |
| **Data**      | Database, persistent storage   | `self.paths.data_dir`     |
| **Root**      | Project root directory         | `self.paths.root_dir`     |

### Accessing Server Information

Use `self.system` (a dictionary) to access runtime details and metadata about the MMCP instance.

| Property      | Type  | Description                                         |
| :------------ | :---- | :-------------------------------------------------- |
| `version`     | `str` | Current MMCP version (e.g., "0.1.0")                |
| `environment` | `str` | Runtime environment ("development" or "production") |

### Accessing Plugin Settings

Use `self.settings` ([Pydantic model](#what-is-a-pydantic-model)) to access your plugin's validated configuration. Available properties depend on your plugin's settings model.

```python
# Access settings as object attributes
api_key = self.settings.api_key.get_secret_value()  # For SecretStr fields
timeout = self.settings.timeout                      # Regular fields
```

```python
async def execute(self):
    # Retrieve the current environment (e.g., "production", "development")
    environment = self.system.get("environment")
    # Retrieve the server version string
    version = self.system.get("version")
    if environment == "production":
        self.logger.info("Running in production mode")
```

### Why Not Use `pathlib.Path` Directly?

Using raw paths allows plugins to access system files or other plugins' data, breaking security and portability. `self.paths` enforces a sandbox and handles cross-platform differences.

```python
# Cross-platform, sandboxed to your plugin's allowed data directory
data_path = self.paths.data_dir
```

### File I/O Operations

When performing file operations within managed directories, use the standard `pathlib.Path` methods for reading and writing files.

```python
def execute(self, filename: str):
    # 1. Build paths using the managed directories
    download_path = self.paths.download_dir / filename
    cache_path = self.paths.cache_dir / "temp.json"

    # 2. Write files using pathlib
    download_path.write_text("downloaded content", encoding="utf-8")

    # 3. Read files using pathlib
    if cache_path.exists():
        data = cache_path.read_text(encoding="utf-8")
```

---

## ✅ Plugin Checklist (Best Practices)

1. **Pathlib Only:** Never use string concatenation for paths. Use `/`.
2. **Describe your Fields:** The LLM reads the `description` inside your Pydantic `Field`. Be descriptive.
3. **Graceful Errors:** Don't let your plugin crash the server. Return a dictionary with an `"error"` key if something goes wrong.
4. **Keep Checks Fast:** Keep `is_available()` and `is_eligible()` fast. Avoid network calls inside them.

---

## 🛠️ Testing & Debugging

1. **Swagger UI:** Spin up the server and visit `/docs`. Your plugin's schema is generated automatically.
2. **Logs:** Managed loggers will prefix your logs: `[tmdb.search_metadata] API key missing`.

---

## Need Inspiration?

Check out existing implementations:

- `app/plugins/core_metadata/tmdb.py` (A Tool)
- `app/plugins/core_foundation/providers.py` (Context Provider)

---

## 📚 Further Reading: Pydantic & Typing

MMCP leverages **Pydantic v2** for all data validation. If you want to build complex tools with nested objects, sophisticated constraints, or custom types, we recommend exploring these resources:

- **[Pydantic: Field Configuration](https://docs.pydantic.dev/latest/concepts/fields/)**: Detailed info on using `Field()` for validation, constraints (like `gt`, `le`, `pattern`), and metadata.
- **[Pydantic: Common Types](https://docs.pydantic.dev/latest/concepts/types/)**: Learn how to use `EmailStr`, `FilePath`, `AnyHttpUrl`, and more to make your Tool inputs even safer.
- **[Python Typing: Literal](https://docs.python.org/3/library/typing.html#typing.Literal)**: The official documentation on using `Literal` for fixed-choice inputs.
- **[Python Typing: Union (`|`)](https://docs.python.org/3/library/typing.html#typing.Union)**: How to allow a field to accept multiple different types (e.g., `str | int`).

### What is a Pydantic Model?

A **Pydantic model** is a Python class that automatically validates and converts data. Think of it as a strongly-typed data container with built-in validation:

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Automatic validation & conversion
user = User(name="Alice", age="25")  # age string → int
print(user.name)  # "Alice"
print(user.age)   # 25 (converted to int)
```

Pydantic models give you:

- **Type safety**: Fields are validated against their declared types
- **Auto-conversion**: Strings to numbers, etc.
- **Default values**: Optional configuration
- **Validation errors**: Clear messages when data is invalid

---
