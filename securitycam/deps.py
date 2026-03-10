"""Dependency helpers: torch/YOLO loading, cascades, and GPU metrics."""
import subprocess
import sys
import cv2

from . import config, state


def install_deps():
    """Auto-install torch + ultralytics when enabled and missing."""
    if not config.AUTO_INSTALL_DEPS:
        print("[DEPS] AUTO_INSTALL_DEPS = False, skipping.")
        return

    print("[DEPS] Attempting to install torch (CUDA) + ultralytics...")

    try:
        cmd_torch = [
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--pre",
            "torch", "torchvision", "torchaudio",
            "--index-url", config.TORCH_CUDA_INDEX_URL,
        ]
        print("[DEPS] ->", " ".join(cmd_torch))
        subprocess.run(cmd_torch, check=True)

        cmd_ultra = [
            sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"
        ]
        print("[DEPS] ->", " ".join(cmd_ultra))
        subprocess.run(cmd_ultra, check=True)

    except Exception as e:
        print("[DEPS] Auto-install failed:", e)
        print("[DEPS] Install dependencies manually if needed.")


def ensure_torch_and_yolo():
    """
    Import torch + YOLO.
    - If GPU is available and supported by this torch build -> DEVICE='cuda:0'
    - Otherwise -> DEVICE='cpu'
    """
    try:
        import torch as _torch
        from ultralytics import YOLO as _YOLO
        state.torch = _torch
        state.YOLO = _YOLO
    except ImportError:
        print("[DEPS] torch/ultralytics not installed. Trying auto-install ...")
        install_deps()
        try:
            import torch as _torch
            from ultralytics import YOLO as _YOLO
            state.torch = _torch
            state.YOLO = _YOLO
        except Exception as e:
            print("[DEPS] Torch/Ultralytics still unavailable:", e)
            state.YOLO_AVAILABLE = False
            state.DEVICE = "cpu"
            state.CUDA_AVAILABLE = False
            return

    state.YOLO_AVAILABLE = True

    state.DEVICE = "cpu"
    state.CUDA_AVAILABLE = False

    if state.torch.cuda.is_available():
        try:
            name = state.torch.cuda.get_device_name(0)
            major, minor = state.torch.cuda.get_device_capability(0)
            cap = major * 10 + minor

            print(f"[CUDA] GPU detected: {name}, Compute Capability = sm_{cap}")
            arch_list = []
            try:
                arch_list = state.torch.cuda.get_arch_list()
            except Exception:
                arch_list = []
            print(f"[CUDA] Torch arch list: {arch_list}")

            gpu_arch = f"sm_{cap}"
            if gpu_arch in arch_list:
                print("[CUDA] GPU is compatible -> device = cuda:0")
                state.DEVICE = "cuda:0"
                state.CUDA_AVAILABLE = True
            else:
                print(f"[CUDA] GPU capability sm_{cap} is not supported by this torch build.")
                print("[CUDA] -> Falling back to CPU (device='cpu')")
                print("[CUDA] Tip: install a newer CUDA torch build (nightly/current).")
                state.DEVICE = "cpu"
                state.CUDA_AVAILABLE = False

        except Exception as e:
            print("[CUDA] GPU detection error:", e)
            state.DEVICE = "cpu"
            state.CUDA_AVAILABLE = False
    else:
        print("[CUDA] torch.cuda.is_available() == False -> no CUDA")

    print(f"[DEVICE] final: DEVICE={state.DEVICE}, CUDA_AVAILABLE={state.CUDA_AVAILABLE}")


def ensure_face_cascade():
    """Lazy load Haarcascade for face detection."""
    if state.face_cascade is None:
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if cascade.empty():
                cascade = None
            state.face_cascade = cascade
        except Exception:
            state.face_cascade = None
    return state.face_cascade


def ensure_hand_detector(camera_state):
    """
    Lazy load MediaPipe Hands for hand skeletons.
    Returns detector or None if not available.
    """
    if camera_state.hand_detector is not None:
        return camera_state.hand_detector
    try:
        import mediapipe as mp
        state.mp_hands = mp.solutions.hands
        camera_state.hand_detector = state.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
    except Exception:
        camera_state.hand_detector = None
    return camera_state.hand_detector


def get_yolo_model(camera_state):
    """
    Lazy-load YOLO model.
    Note: device is controlled at prediction time via `device=DEVICE`,
    not via `.to(...)`.
    """
    if not state.YOLO_AVAILABLE:
        return None
    if camera_state.yolo_model is None:
        source_path = config.model_path(camera_state.yolo_model_name)
        model_source = str(source_path) if source_path.exists() else camera_state.yolo_model_name
        print(f"[YOLO] Loading model {model_source} (DEVICE={state.DEVICE})...")
        camera_state.yolo_model = state.YOLO(model_source)
        print("[YOLO] Model ready:", camera_state.yolo_model_name)
    return camera_state.yolo_model


def get_gpu_snapshot():
    """
    Return basic GPU metrics.
    Falls back to CPU values if no GPU is available.
    """
    if not state.CUDA_AVAILABLE or state.torch is None:
        return {
            "name": state.DEVICE,
            "util": 0,
            "mem_used": 0,
            "mem_total": 0,
        }

    name = state.DEVICE
    util = 0
    mem_used = 0
    mem_total = 0

    # 1) Try nvidia-smi (GPU utilization in %)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=0.5
        )
        if proc.returncode == 0 and proc.stdout.strip():
            parts = proc.stdout.strip().split(",")
            if len(parts) >= 3:
                util = int(parts[0].strip())
                mem_used = int(parts[1].strip())
                mem_total = int(parts[2].strip())
    except Exception:
        pass

    # 2) Enrich with torch memory metrics when available
    try:
        device_props = state.torch.cuda.get_device_properties(0)
        name = device_props.name
        mem_total = max(mem_total, int(device_props.total_memory / (1024 * 1024)))  # bytes -> MB
        mem_used_torch = int(state.torch.cuda.memory_allocated(0) / (1024 * 1024))
        mem_used = max(mem_used, mem_used_torch)
    except Exception:
        pass

    return {
        "name": name,
        "util": util,
        "mem_used": mem_used,
        "mem_total": mem_total,
    }
