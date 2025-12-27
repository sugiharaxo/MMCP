# Modular Media Control Plane (MMCP)

A lightweight, agentic media server control system designed to run efficiently on hardware ranging from high-end servers to Raspberry Pis.

## Quick Start

### The Fast Way (Recommended)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv run main.py`.
   _(This automatically creates an environment and installs everything.)_

### The Standard Way

1. Download and install [Python 3.10+](https://www.python.org/downloads/). (Ensure "Add Python to PATH" is checked).
2. Download and extract MMCP to your desired location

### Windows

3. Open a Terminal in that folder and run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

### Linux

3. Open your terminal and run:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python3 main.py
   ```

## Testing the Interface

Once running, go to:

- **Status Check:** `http://localhost:8000/`
- **Interactive API Docs:** `http://localhost:8000/docs`
