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
python -m pip install flask opencv-python ultralytics mediapipe onvif-zeep

# TensorRT for "Build for GPU" (NVIDIA only). Installed automatically when an
# NVIDIA GPU is found; skip with SECURITYCAM_INSTALL_TENSORRT=0.
if [ "${SECURITYCAM_INSTALL_TENSORRT:-1}" != "0" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if python -c "import tensorrt, onnx, onnxslim" >/dev/null 2>&1; then
      echo "[SETUP] TensorRT already installed."
    else
      echo "[SETUP] NVIDIA GPU found - installing TensorRT export tools (large download, one time)..."
      python -m pip install tensorrt onnx onnxslim onnxscript onnx-graphsurgeon polygraphy nvidia-modelopt lief || echo "[WARN] TensorRT install failed - continuing without GPU optimization."
    fi
  else
    echo "[SETUP] No NVIDIA GPU found - skipping TensorRT (GPU optimization unavailable)."
  fi
fi

# OpenVINO for the Intel NPU (optional, Compute tab). Installed automatically only
# when an Intel NPU is found; nothing changes unless you pick a device in the
# Compute tab. Skip with SECURITYCAM_INSTALL_OPENVINO=0.
if [ "${SECURITYCAM_INSTALL_OPENVINO:-1}" != "0" ]; then
  if [ -e /dev/accel/accel0 ] || (lsmod 2>/dev/null | grep -q intel_vpu); then
    if python -c "import openvino" >/dev/null 2>&1; then
      echo "[SETUP] OpenVINO already installed."
    else
      echo "[SETUP] Intel NPU found - installing OpenVINO (optional, for the Compute tab)..."
      python -m pip install openvino || echo "[WARN] OpenVINO install failed - Intel NPU/GPU stay unavailable, everything else works."
    fi
  fi
fi

echo "[RUN] Starting SecurityCam..."
python main.py
