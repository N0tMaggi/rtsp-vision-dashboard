@echo off
setlocal

set "PYTHON_BIN=python"
where py >nul 2>&1
if %errorlevel%==0 set "PYTHON_BIN=py -3"

if not exist ".venv\Scripts\python.exe" (
  echo [SETUP] Creating virtual environment ^(.venv^)...
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

rem --- CUDA build of PyTorch ------------------------------------------------
rem PyPI only ships CPU-only torch on Windows. If an NVIDIA GPU is present and
rem the installed torch has no CUDA, swap in the CUDA build from pytorch.org.
rem Override the CUDA flavour with e.g.  set SECURITYCAM_TORCH_CUDA=cu128
if not defined SECURITYCAM_TORCH_CUDA set "SECURITYCAM_TORCH_CUDA=cu134"
where nvidia-smi >nul 2>&1
if errorlevel 1 goto :skip_cuda
python -c "import sys, torch; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if not errorlevel 1 goto :skip_cuda
echo [SETUP] NVIDIA GPU found but torch is CPU-only - installing CUDA build ^(%SECURITYCAM_TORCH_CUDA%^)...
python -m pip install --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/%SECURITYCAM_TORCH_CUDA%
if errorlevel 1 echo [WARN] CUDA torch install failed - continuing on CPU.
:skip_cuda

rem --- optional: TensorRT for "Optimize for GPU" (large download, NVIDIA only) ---
rem Enable with:  set SECURITYCAM_INSTALL_TENSORRT=1
if "%SECURITYCAM_INSTALL_TENSORRT%"=="1" (
  echo [SETUP] Installing optional TensorRT export dependencies...
  python -m pip install tensorrt onnx onnxslim
  if errorlevel 1 echo [WARN] TensorRT install failed - GPU optimization stays unavailable.
)

echo [RUN] Starting SecurityCam...
python main.py

endlocal
