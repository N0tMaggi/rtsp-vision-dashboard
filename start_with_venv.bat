@echo off
setlocal

set "PYTHON_BIN=python"
where py >nul 2>&1
if %errorlevel%==0 set "PYTHON_BIN=py -3"

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating virtual environment (.venv)...
  %PYTHON_BIN% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

echo [SETUP] Installing/updating dependencies in venv...
python -m pip install --upgrade pip
python -m pip install flask opencv-python ultralytics mediapipe
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  exit /b 1
)

echo [RUN] Starting SecurityCam...
python main.py

endlocal
