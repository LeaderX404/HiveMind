@echo off
REM HiveMind Phase 1 launcher (Windows).
REM Creates a local virtual environment, installs the UI dependencies,
REM then opens HiveMind in the browser. No terminal knowledge needed.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo python not found.
    echo Install Python 3.9+ from https://www.python.org/downloads/ and try again.
    exit /b 1
)

if not exist ".venv" (
    echo First run: preparing the environment, 1-2 minutes...
    python -m venv .venv
)

".venv\Scripts\python" -m pip install --quiet --upgrade pip
".venv\Scripts\python" -m pip install --quiet streamlit psutil

echo Opening HiveMind in your browser...
".venv\Scripts\streamlit" run app.py --server.headless false
