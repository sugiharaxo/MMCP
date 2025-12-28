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
