@echo off
setlocal enabledelayedexpansion

title Multi-Camera Vehicle Tracking & Telemetry HUD (Unreal Pipeline)
echo ==============================================================================
echo Multi-Camera Vehicle Tracking & Telemetry HUD Runner (Windows)
echo ==============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: 1. Check for local virtual environment
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
    echo [*] Using local virtual environment: .venv
    goto :DEPS
)

:: 2. Locate system Python
set "PYTHON_EXEC="
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXEC=python"
) else (
    where py >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXEC=py -3"
    )
)

if "%PYTHON_EXEC%"=="" (
    echo [!] ERROR: Python 3 was not found in your system PATH.
    echo [!] Please install Python 3.8+ from https://www.python.org/
    echo [!] Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [*] Detected Python launcher: %PYTHON_EXEC%

:: 3. Setup virtual environment if not present
if not exist "%SCRIPT_DIR%.venv" (
    echo [*] Creating isolated virtual environment at .venv...
    %PYTHON_EXEC% -m venv "%SCRIPT_DIR%.venv"
    if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
        set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
        echo [✓] Virtual environment created successfully.
        goto :INSTALL
    ) else (
        echo [!] Warning: Could not create .venv, falling back to system Python.
    )
) else (
    if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
        set "PYTHON_EXEC=%SCRIPT_DIR%.venv\Scripts\python.exe"
    )
)

:DEPS
if not exist "%SCRIPT_DIR%requirements.txt" goto :RUN
"%PYTHON_EXEC%" -c "import cv2, torch, ultralytics" >nul 2>nul
if %errorlevel% equ 0 goto :RUN

:INSTALL
echo [*] Installing/verifying dependencies from requirements.txt...
"%PYTHON_EXEC%" -m pip install --upgrade pip
"%PYTHON_EXEC%" -m pip install -r "%SCRIPT_DIR%requirements.txt"

:RUN
echo [*] Launching Multi-Camera Re-ID Pipeline...
"%PYTHON_EXEC%" "%SCRIPT_DIR%multi_cam_reid.py" %*
if %errorlevel% neq 0 (
    echo.
    echo [!] Pipeline exited with error code %errorlevel%.
    pause
)
