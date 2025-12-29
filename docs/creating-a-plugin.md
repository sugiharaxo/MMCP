# Creating a Plugin for MMCP

This guide explains how to create plugins for the Modular Media Control Plane (MMCP). Plugins extend MMCP's capabilities by providing **Tools** (actions the agent can take) and **Context Providers** (dynamic state information).

## Philosophy: Managed Services

MMCP provides **managed services** to plugins, eliminating boilerplate and ensuring consistency:

- **Managed Logging**: Pre-configured logger with plugin/tool namespacing (`self.logger`)
- **Managed Settings**: Automatic loading, validation, and injection of configuration (`self.settings`)
- **Managed Context**: Safe access to system directories and server info (`self.context`)
- **Auto-Discovery**: Nested classes are automatically discovered - no manual registration needed

You focus on your plugin's logic; MMCP handles the infrastructure.

## Available Services

### 1. Managed Logging

Every tool receives a pre-configured logger:

```python
self.logger.info("Tool executed successfully")
self.logger.error("API call failed", exc_info=True)
```

The logger is automatically namespaced: `mmcp.{plugin_name}.{tool_name}`

### 2. Managed Settings

Settings are automatically loaded from:

1. **Environment Variables**: `MMCP_PLUGIN_{PLUGIN_SLUG}_{FIELD_NAME}`
2. **Database**: User-configured values via API
3. **Defaults**: Pydantic model defaults

Access via `self.settings` (validated Pydantic model or `None`).

**Secret Handling**: Use `SecretStr` for sensitive values - they're automatically masked in logs and UI.

### 3. PluginContext

Every tool receives a `PluginContext` object via `self.context`:

```python
# Access system directories
download_path = self.context.config.download_dir / "file.mp4"
cache_path = self.context.config.cache_dir / "cache.json"

# Access server info
server_version = self.context.server_info.get("version")
```

**Available in `config`**:

- `root_dir: Path` - Root directory for file operations
- `download_dir: Path` - Directory for downloaded media files
- `cache_dir: Path` - Directory for cache files

**Available in `server_info`**:

- `version: str` - MMCP version
- `environment: str` - Environment (development/production)

## API Location & Documentation

- **API Base URL**: `http://localhost:8000`
- **Interactive API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

The API documentation is auto-generated from your plugin's Pydantic models. Your tool's `input_schema` automatically appears in the OpenAPI spec.

## Plugin Registration Patterns

### Pattern A: Nested Class Auto-Discovery (Recommended)

MMCP automatically discovers nested `Tool` and `Provider` classes. No manual registration needed.

#### Basic Plugin Structure

```python
from app.api.base import Plugin, Tool
from app.api.schemas import PluginContext
from pydantic import BaseModel, Field, SecretStr

# 1. Define Input Schema (what the LLM will provide)
class MyToolInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Max results")

# 2. Define Settings Model (optional)
class MyPluginSettings(BaseModel):
    """Plugin configuration."""
    api_key: SecretStr  # Required - use SecretStr for secrets
    timeout: int = 30  # Optional with default

# 3. Define Plugin with nested Tool
class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    settings_model = MyPluginSettings  # None if no settings needed

    class MyTool(Tool):  # Nesting = Auto-discovery
        name = "my_tool"
        description = "Does something useful"
        input_schema = MyToolInput
        version = "1.0.0"
        settings_model = MyPluginSettings  # Usually same as plugin

        def is_available(self, settings: MyPluginSettings | None, context: PluginContext) -> bool:
            """Check if tool is available (e.g., API key configured)."""
            return settings is not None and bool(settings.api_key.get_secret_value())

        async def execute(self, query: str, limit: int = 10) -> dict:
            """
            Execute the tool's logic.

            Args come from input_schema - already validated by Pydantic.
            Access settings via self.settings, context via self.context.
            """
            # Access managed services
            if self.settings is None:
                return {"error": "Plugin not configured"}

            api_key = self.settings.api_key.get_secret_value()  # Unwrap SecretStr
            cache_file = self.context.config.cache_dir / "my_cache.json"

            self.logger.info(f"Executing with query: {query}")

            # Your logic here
            return {"result": "success", "data": []}
```

### Environment Variable Naming

Settings are loaded from environment variables using this pattern:

```
MMCP_PLUGIN_{PLUGIN_SLUG}_{FIELD_NAME}
```

**Examples**:

- Plugin name: `my_plugin` → Slug: `MY_PLUGIN`
- Field: `api_key` → `MMCP_PLUGIN_MY_PLUGIN_API_KEY`
- Nested field: `nested.field` → `MMCP_PLUGIN_MY_PLUGIN_NESTED__FIELD` (double underscore)

**Slugification Rules**:

- Non-alphanumeric characters become underscores
- Uppercased
- Multiple underscores collapsed

### Context Providers Pattern

Context providers supply dynamic state to the LLM before the ReAct loop begins:

```python
from app.api.base import Plugin, Provider
from app.api.schemas import ContextResponse, PluginContext

class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    settings_model = None

    class MyProvider(Provider):  # Nesting = Auto-discovery
        context_key = "my_service"  # Unique key for this provider

        async def is_eligible(self, query: str) -> bool:
            """Lightweight check - should this provider run?"""
            return "my_service" in query.lower()

        async def provide_context(self, context: PluginContext) -> ContextResponse:
            """Fetch dynamic context data."""
            return ContextResponse(
                data={
                    "status": "online",
                    "items": 42,
                },
                ttl=300,  # Cache for 5 minutes
                provider_name="MyServiceProvider",
            )
```

**Context Provider Guidelines**:

- Keep `is_eligible()` lightweight (no I/O)
- Use `provide_context()` for actual data fetching
- Return `ContextResponse` with `data`, `ttl`, and `provider_name`
- Data is automatically truncated if too large (configurable)

## Complete Example: TMDb Metadata Plugin

Here's a real-world example from the codebase:

```python
from typing import Any
from curl_cffi.requests import AsyncSession, RequestsError
from pydantic import BaseModel, Field, SecretStr
from app.api.base import Plugin, Tool
from app.api.schemas import PluginContext

# 1. Input Schema
class TMDbMetadataInput(BaseModel):
    title: str = Field(..., description="The movie or TV show title")
    year: int | None = Field(None, description="Release year (optional)")
    type: str = Field("movie", description="Type: 'movie' or 'tv'")

# 2. Settings Model
class TMDbSettings(BaseModel):
    """TMDb plugin configuration."""
    api_key: SecretStr  # Required
    language: str = "en-US"  # Optional default

# 3. Plugin with Tool
class TMDb(Plugin):
    name = "tmdb"
    version = "1.0.0"
    settings_model = TMDbSettings

    class Lookup(Tool):
        name = "tmdb_lookup_metadata"
        description = "Finds movie/TV show metadata using TMDb API"
        input_schema = TMDbMetadataInput
        version = "1.0.0"
        settings_model = TMDbSettings

        def is_available(self, settings: TMDbSettings | None, _context: PluginContext) -> bool:
            """Check if API key is configured."""
            if settings is None:
                return False
            try:
                return bool(settings.api_key.get_secret_value())
            except Exception:
                return False

        async def execute(self, title: str, year: int | None = None, type: str = "movie") -> dict[str, Any]:
            """Query TMDb API for metadata."""
            if self.settings is None:
                return {"error": "TMDb API key not configured"}

            api_key = self.settings.api_key.get_secret_value()
            url = f"https://api.themoviedb.org/3/search/{type}"

            headers = {"Authorization": f"Bearer {api_key}"}

            try:
                async with AsyncSession(impersonate="chrome") as session:
                    response = await session.get(url, params={"query": title}, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                results = data.get("results", [])
                if not results:
                    return {"error": f"No results found for '{title}'"}

                release_date = results[0].get("release_date")
                year = None
                if release_date:
                    try:
                        year = int(release_date.split("-")[0])
                    except (ValueError, IndexError):
                        pass

                return {
                    "title": results[0].get("title"),
                    "year": year,
                    "overview": results[0].get("overview"),
                    "tmdb_id": results[0].get("id"),
                }
            except RequestsError as e:
                return {"error": f"Network error: {str(e)}"}
```

**Configuration**:

```bash
# Set environment variable
export MMCP_PLUGIN_TMDB_API_KEY="your_api_key_here"
```

## Best Practices

1. **Use Type Hints**: All settings and input schemas should be fully typed
2. **Handle Errors Gracefully**: Return structured error dicts, don't raise exceptions
3. **Use SecretStr**: For API keys, passwords, tokens - automatic masking
4. **Async I/O**: All network/file operations should be `async`
5. **Descriptive Fields**: Use Pydantic `Field(..., description="...")` for better LLM understanding
6. **Optional Settings**: Use `settings_model = None` if no configuration needed
7. **Availability Checks**: Implement `is_available()` to check configuration state
8. **Path Handling**: Always use `pathlib.Path`, never string concatenation

## Plugin Directory Structure

Place your plugin in `app/plugins/{plugin_name}/`:

```
app/plugins/
  my_plugin/
    __init__.py      # Can be empty or export Plugin class
    plugin.py        # Your Plugin class definition
```

The loader automatically discovers plugins in this directory.

## Testing Your Plugin

1. **Start MMCP**: `python -m uvicorn main:app --reload`
2. **Check API Docs**: Visit `http://localhost:8000/docs`
3. **Verify Discovery**: Your tool should appear in the tools list
4. **Test Configuration**: Use the settings API to configure your plugin
5. **Test Execution**: Use the agent API to trigger your tool

## Next Steps

- See `app/plugins/core_metadata/tmdb.py` for a complete Tool example
- See `app/plugins/core_foundation/providers.py` for a complete Provider example
- Check `docs/agent-loop.md` for agent interaction patterns
- Review `docs/devs.md` for project philosophy and standards
