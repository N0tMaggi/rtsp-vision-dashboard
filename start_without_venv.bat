@echo off
setlocal

set "PYTHON_BIN=python"
where py >nul 2>&1
if %errorlevel%==0 set "PYTHON_BIN=py -3"

echo [SETUP] Installing/updating global dependencies...
%PYTHON_BIN% -m pip install --upgrade pip
%PYTHON_BIN% -m pip install flask opencv-python ultralytics mediapipe onvif-zeep
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  exit /b 1
)

rem --- CUDA build of PyTorch (PyPI only ships CPU-only torch on Windows) -----
if not defined SECURITYCAM_TORCH_CUDA set "SECURITYCAM_TORCH_CUDA=cu134"
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :skip_cuda
%PYTHON_BIN% -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if not errorlevel 1 goto :skip_cuda
echo [SETUP] NVIDIA GPU found but torch is CPU-only - installing CUDA build ^(%SECURITYCAM_TORCH_CUDA%^)...
%PYTHON_BIN% -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/%SECURITYCAM_TORCH_CUDA%
if errorlevel 1 echo [WARN] CUDA torch install failed - continuing on CPU.
:skip_cuda

rem --- TensorRT for "Build for GPU" (NVIDIA only) -------------------------
rem Installed automatically when an NVIDIA GPU is found (nvidia-smi on PATH).
rem Skip it with:  set SECURITYCAM_INSTALL_TENSORRT=0
if "%SECURITYCAM_INSTALL_TENSORRT%"=="0" goto :skip_trt
where nvidia-smi >nul 2>&1
if errorlevel 1 (
  echo [SETUP] No NVIDIA GPU found - skipping TensorRT ^(GPU optimization unavailable^).
  goto :skip_trt
)
%PYTHON_BIN% -c "import tensorrt, onnx, onnxslim" >nul 2>&1
if not errorlevel 1 (
  echo [SETUP] TensorRT already installed.
  goto :skip_trt
)
echo [SETUP] NVIDIA GPU found - installing TensorRT export tools ^(large download, one time^)...
%PYTHON_BIN% -m pip install tensorrt onnx onnxslim onnxscript onnx-graphsurgeon polygraphy nvidia-modelopt lief
if errorlevel 1 echo [WARN] TensorRT install failed - continuing without GPU optimization.
:skip_trt

rem --- OpenVINO for the Intel NPU (optional, Compute tab) --------------------
rem Installed automatically only when an Intel NPU is found. Nothing changes
rem unless you pick a device in the Compute tab. Skip with:  set SECURITYCAM_INSTALL_OPENVINO=0
if "%SECURITYCAM_INSTALL_OPENVINO%"=="0" goto :skip_ov
powershell -NoProfile -Command "if (Get-PnpDevice -Class ComputeAccelerator -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -match 'Intel' }) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 goto :skip_ov
%PYTHON_BIN% -c "import openvino" >nul 2>&1
if not errorlevel 1 (
  echo [SETUP] OpenVINO already installed.
  goto :skip_ov
)
echo [SETUP] Intel NPU found - installing OpenVINO ^(optional, for the Compute tab^)...
%PYTHON_BIN% -m pip install openvino
if errorlevel 1 echo [WARN] OpenVINO install failed - Intel NPU/GPU stay unavailable, everything else works.
:skip_ov

echo [RUN] Starting SecurityCam (without venv)...
%PYTHON_BIN% main.py

endlocal
