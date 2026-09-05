"""Run the local EdgeWeaver API and built frontend."""

from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
