#!/usr/bin/env python3
"""
Simple development runner that restarts the Hermes app when source files change.
Requires: pip install watchgod

Usage:
    python dev.py

It will run `python -m hermes.app` and restart automatically on file changes.
"""
import sys
import subprocess
from pathlib import Path

try:
    from watchgod import run_process
except Exception as e:
    print("watchgod is required for dev auto-reload. Install with: pip install watchgod")
    raise

ROOT = Path(__file__).resolve().parent


def _run_hermes():
    # Use the same interpreter to run the app module
    subprocess.run([sys.executable, "-m", "hermes.app"]) 


if __name__ == "__main__":
    print("Starting Hermes in dev mode (auto-reload enabled). Watching for changes in:", ROOT)
    # run_process will start _run_hermes and restart it on filesystem changes
    run_process(str(ROOT), _run_hermes)
