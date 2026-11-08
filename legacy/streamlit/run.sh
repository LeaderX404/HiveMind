#!/usr/bin/env bash
set -e
# HiveMind Phase 1 launcher (Linux/Mac).
# Creates a local virtual environment, installs the UI dependencies,
# then opens HiveMind in the browser. No terminal knowledge needed.

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found."
    echo "Install Python 3.9+ from https://www.python.org/downloads/ and try again."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "First run: preparing the environment, 1-2 minutes..."
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet streamlit psutil

echo "Opening HiveMind in your browser..."
.venv/bin/streamlit run app.py --server.headless false
