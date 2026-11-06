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

echo "[SETUP] Installing/updating global dependencies..."
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install flask opencv-python ultralytics mediapipe onvif-zeep

# TensorRT for "Build for GPU" (NVIDIA only). Installed automatically when an
# NVIDIA GPU is found; skip with SECURITYCAM_INSTALL_TENSORRT=0.
if [ "${SECURITYCAM_INSTALL_TENSORRT:-1}" != "0" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if "$PYTHON_BIN" -c "import tensorrt, onnx, onnxslim" >/dev/null 2>&1; then
      echo "[SETUP] TensorRT already installed."
    else
      echo "[SETUP] NVIDIA GPU found - installing TensorRT export tools (large download, one time)..."
      "$PYTHON_BIN" -m pip install tensorrt onnx onnxslim onnxscript onnx-graphsurgeon polygraphy nvidia-modelopt lief || echo "[WARN] TensorRT install failed - continuing without GPU optimization."
    fi
  else
    echo "[SETUP] No NVIDIA GPU found - skipping TensorRT (GPU optimization unavailable)."
  fi
fi

echo "[RUN] Starting SecurityCam (without venv)..."
"$PYTHON_BIN" main.py
