#!/usr/bin/env bash
# ==============================================================================
# Multi-Camera Vehicle Re-ID & Telemetry HUD Runner (Linux / macOS)
# ==============================================================================

set -e

# Resolve project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "Multi-Camera Vehicle Re-ID & Telemetry HUD Runner"
echo "Platform: $(uname -s) ($(uname -m))"
echo "Directory: $SCRIPT_DIR"
echo "=============================================================================="

# 1. Locate Python 3 binary
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "[!] Error: Neither 'python3' nor 'python' was found in your PATH." >&2
    echo "[!] Please install Python 3.8+ (e.g., sudo apt install python3 python3-pip python3-venv)" >&2
    exit 1
fi

PYTHON_VER=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "[*] Detected Python: $PYTHON_BIN (v$PYTHON_VER)"

# 2. Virtual environment management (.venv)
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Setting up isolated Python virtual environment at: $VENV_DIR"
    if "$PYTHON_BIN" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "[✓] Virtual environment created successfully."
    else
        echo "[!] Note: Could not create .venv with standard python3 -m venv."
        echo "[!] Falling back to system python environment."
    fi
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    PYTHON_EXEC="$VENV_DIR/bin/python"
    PIP_EXEC="$VENV_DIR/bin/pip"
    echo "[*] Using virtual environment: $VENV_DIR"
else
    PYTHON_EXEC="$PYTHON_BIN"
    PIP_EXEC="$PYTHON_BIN -m pip"
    echo "[*] Using system environment: $PYTHON_BIN"
fi

# 3. Check and install dependencies
if [ -f "requirements.txt" ]; then
    echo "[*] Checking and verifying dependencies from requirements.txt..."
    if ! $PIP_EXEC install -r requirements.txt 2>/dev/null; then
        echo "[!] Standard pip install encountered externally managed environment. Retrying with --break-system-packages..."
        $PIP_EXEC install -r requirements.txt --break-system-packages 2>/dev/null || {
            echo "[!] Warning: pip install failed. Attempting to proceed with available system packages..."
        }
    fi
fi

# 4. Check for graphical display
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "[!] No DISPLAY or WAYLAND_DISPLAY environment variable detected."
    echo "[*] Adding '--no-player' flag to run in headless / batch processing mode."
    set -- "$@" "--no-player"
fi

# 5. Launch pipeline
echo "[*] Launching Multi-Camera Re-ID pipeline..."
echo "=============================================================================="
exec "$PYTHON_EXEC" multi_cam_reid.py "$@"
