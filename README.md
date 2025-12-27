# Modular Media Control Plane (MMCP)

A lightweight, agentic media server control system designed to run efficiently on hardware ranging from high-end servers to Raspberry Pis.

## Quick Start

### Windows

1. Download and install [Python 3.10+](https://www.python.org/downloads/). (Ensure "Add Python to PATH" is checked).
2. Open a folder and extract MMCP.
3. Open a Terminal (PowerShell) in that folder and run:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

### Linux

1. Open your terminal and run:
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
