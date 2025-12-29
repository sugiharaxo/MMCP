# Installation instructions

## Using uv (Recommended)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Clone this repository using git or [download](https://github.com/sugiharaxo/MMCP/archive/refs/heads/main.zip) and extract the zip to your desired location
3. Open a terminal inside that folder, and run `uv run main.py`

To interact with MMCP, find the IP of your server and go to `http://YOUR_IP:8000`

## Using Python

1. Download and install [Python 3.10+](https://www.python.org/downloads/). (Ensure "Add Python to PATH" is checked).
2. Clone this repository using git or download and extract the zip to your desired location

3. (Windows) Open a terminal inside that folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

3. (Linux) Open a terminal inside that folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

To interact with MMCP, find the IP of your server and go to http://<ip>:8000

## Plugin Configuration

MMCP uses a declarative plugin configuration system with namespaced environment variables. Each plugin has its own configuration requirements that must be set before the system will start.

### Environment Variable Naming Convention

All plugin settings follow the pattern: `MMCP_PLUGIN_{PLUGIN_SLUG}_{SETTING_NAME}`

- **PLUGIN_SLUG**: The plugin identifier (e.g., `TMDB`, `YOUTUBE`)
- **SETTING_NAME**: The specific setting name (e.g., `API_KEY`, `TIMEOUT_SECONDS`)
- **Format**: All uppercase with underscores

### Setting Environment Variables

### Using .env

Create a `.env` file in your MMCP root directory:

```bash
# TMDB Plugin Configuration
MMCP_PLUGIN_TMDB_API_KEY=your_tmdb_api_key_here
MMCP_PLUGIN_TMDB_REQUEST_TIMEOUT=30
MMCP_PLUGIN_TMDB_MAX_RETRIES=3

# Core System Configuration
MMCP_PLUGIN_CORE_DOWNLOADS_DIR=./downloads
MMCP_PLUGIN_CORE_CACHE_DIR=./cache
MMCP_PLUGIN_CORE_TEMP_DIR=./temp
MMCP_PLUGIN_CORE_MAX_FILE_SIZE_MB=1024

# LLM Configuration
MMCP_PLUGIN_LLM_MODEL=gpt-4
MMCP_PLUGIN_LLM_API_KEY=your_openai_api_key_here
MMCP_PLUGIN_LLM_MAX_CONTEXT_CHARS=8000
```

### Finding Plugin Requirements

Each plugin documents its required and optional configuration settings. Check the plugin's documentation or look for configuration classes in the plugin code. The system will report missing required settings at startup with clear error messages.

### Security Notes

- **Sensitive Values**: API keys and passwords use `SecretStr` and are automatically masked in logs
- **File Permissions**: Ensure your `.env` file has appropriate permissions (readable only by the MMCP process user)
- **Environment Isolation**: Plugin configurations are namespaced to prevent conflicts

### Cross-Platform Considerations

- **Path Separators**: Use forward slashes `/` in environment variables - MMCP handles OS-specific path conversion
- **Case Sensitivity**: Environment variable names are case-sensitive on Linux, case-insensitive on Windows
- **Path Encoding**: Use absolute paths or paths relative to the project root
