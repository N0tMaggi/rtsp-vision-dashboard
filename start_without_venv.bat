@echo off
setlocal

set "PYTHON_BIN=python"
where py >nul 2>&1
if %errorlevel%==0 set "PYTHON_BIN=py -3"

echo [SETUP] Installing/updating global dependencies...
%PYTHON_BIN% -m pip install --upgrade pip
%PYTHON_BIN% -m pip install flask opencv-python ultralytics mediapipe
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  exit /b 1
)

echo [RUN] Starting SecurityCam (without venv)...
%PYTHON_BIN% main.py

endlocal
