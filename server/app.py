"""
Server entry point for CodeReview OpenEnv.

This module provides the main() entry point used by:
  - pyproject.toml [project.scripts] server = "server.app:main"
  - openenv serve
  - uv run server

It imports and runs the FastAPI app defined in the root app.py.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(host: str = "0.0.0.0", port: int = 7860, workers: int = 1) -> None:
    """Start the CodeReview OpenEnv server."""
    import uvicorn

    uvicorn.run(
        "app:app",
        host=host,
        port=int(os.environ.get("PORT", port)),
        workers=workers,
    )


if __name__ == "__main__":
    main()
