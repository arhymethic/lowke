@echo off
echo ==============================================================================
echo Multi-Camera & Single-Camera Vehicle Tracking & Telemetry HUD Runner
echo ==============================================================================

set PYTHON_BIN=C:\Users\advik\python311\python.exe

if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

echo [*] Checking dependencies...
"%PYTHON_BIN%" -m pip install -r requirements.txt

echo [*] Launching Pipeline...
"%PYTHON_BIN%" multi_cam_reid.py %*

pause
