#!/usr/bin/env python3
"""Deprecated compatibility shim for the Node arXiv Daily CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    plugin_dir = root / "plugin"
    cli = plugin_dir / "arxiv-daily-cli.cjs"
    if not cli.exists():
        print(
            "arxiv_daily.py is retired. Build the Node CLI first:\n"
            "  cd plugin && npm install && npm run build\n"
            "Then run:\n"
            "  cd plugin && npm run cli -- run-pending",
            file=sys.stderr,
        )
        return 1

    args = sys.argv[1:] or ["run-pending"]
    try:
        return subprocess.call(["node", str(cli), *args], cwd=plugin_dir)
    except FileNotFoundError:
        print(
            "arxiv_daily.py is retired and now delegates to the Node CLI, "
            "but `node` was not found on PATH.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
