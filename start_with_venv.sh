#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] python3/python not found in PATH"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[SETUP] Creating virtual environment (.venv)..."
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

echo "[SETUP] Installing/updating dependencies in venv..."
python -m pip install --upgrade pip
python -m pip install flask opencv-python ultralytics mediapipe

echo "[RUN] Starting SecurityCam..."
python main.py
