#!/usr/bin/env bash
# ==============================================================================
# Pelco/Traffic-Pro CCTV Workstation Runner (Linux / macOS)
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

chmod +x "$DIR/run.sh" 2>/dev/null || true

# 1. Locate Python 3 binary
if command -v python3 >/dev/null 2>&1; then
    SYSTEM_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    SYSTEM_PYTHON="python"
else
    echo "[!] Error: Neither 'python3' nor 'python' was found in your PATH." >&2
    echo "[!] Please install Python 3.8+ (e.g. sudo apt install python3 python3-pip python3-venv)" >&2
    exit 1
fi

# 2. Virtual Environment Management (.venv)
VENV_DIR="$DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Setting up isolated Python virtual environment at: $VENV_DIR"
    if "$SYSTEM_PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "[✓] Virtual environment created successfully."
        if [ -f "$DIR/requirements.txt" ]; then
            echo "[*] Installing dependencies from requirements.txt..."
            "$VENV_DIR/bin/pip" install --upgrade pip 2>/dev/null || true
            "$VENV_DIR/bin/pip" install -r "$DIR/requirements.txt"
        fi
    else
        echo "[!] Warning: Could not create .venv with standard python3 -m venv."
        echo "[!] Falling back to system python environment."
    fi
fi

if [ -f "$VENV_DIR/bin/python" ]; then
    PYTHON_EXEC="$VENV_DIR/bin/python"
    echo "[*] Using virtual environment: $VENV_DIR"
else
    PYTHON_EXEC="$SYSTEM_PYTHON"
    echo "[*] Using system environment: $SYSTEM_PYTHON"
fi

# 3. Ensure required dependencies are installed
if ! "$PYTHON_EXEC" -c "import pygame, cv2, numpy" >/dev/null 2>&1; then
    if [ -f "$DIR/requirements.txt" ]; then
        echo "[*] Installing/verifying dependencies from requirements.txt..."
        "$PYTHON_EXEC" -m pip install --upgrade pip 2>/dev/null || true
        "$PYTHON_EXEC" -m pip install -r "$DIR/requirements.txt"
    fi
fi

echo "[LAUNCHER] Launching Pelco/Traffic-Pro CCTV Workstation..."
echo "[LAUNCHER] Working directory: $DIR"
echo "[LAUNCHER] Python binary: $PYTHON_EXEC"
exec "$PYTHON_EXEC" "$DIR/main.py" "$@"
