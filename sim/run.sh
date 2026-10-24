#!/usr/bin/env bash
# Quick launch script for Project Zomboid CCTV Tracker Workstation
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

chmod +x "$DIR/run.sh" 2>/dev/null || true

# Explicitly use the virtual environment Python (bypasses system python3 missing packages)
if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON_EXEC="$DIR/.venv/bin/python"
elif [ -f "$DIR/.venv/bin/python3" ]; then
    PYTHON_EXEC="$DIR/.venv/bin/python3"
else
    PYTHON_EXEC="python3"
fi

echo "[LAUNCHER] Launching Pelco/Traffic-Pro CCTV Workstation..."
echo "[LAUNCHER] Working directory: $DIR"
echo "[LAUNCHER] Python binary: $PYTHON_EXEC"
exec "$PYTHON_EXEC" "$DIR/main.py" "$@"
