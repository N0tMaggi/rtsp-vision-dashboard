"""Dependency helpers: torch/YOLO loading, cascades, and GPU metrics."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

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


# --- Downloadable model assets ------------------------------------------------
# Face and hand features need model files that recent wheels no longer ship
# (opencv-python 5 has no cv2/data, mediapipe >= 0.10.30 has no mp.solutions).
# They are fetched into models/ in the background the first time the feature
# is switched on; the render loop never blocks on a download.

ASSETS = {
    "face_cascade": (
        "haarcascade_frontalface_default.xml",
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
    ),
    "hand_landmarker": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    ),
}
_ASSET_RETRY_S = 60.0
_asset_lock = threading.Lock()
asset_status: dict[str, str] = {}          # name -> missing | downloading | ready | failed
_asset_retry_at: dict[str, float] = {}


def _download(name: str, filename: str, url: str) -> None:
    target = config.MODELS_DIR / filename
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[ASSET] Downloading {filename} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "SecurityCam"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        if tmp.stat().st_size < 1024:
            raise ValueError("download too small")
        os.replace(tmp, target)
        print(f"[ASSET] {filename} ready ({target.stat().st_size // 1024} KB).")
        with _asset_lock:
            asset_status[name] = "ready"
    except Exception as exc:  # network down, proxy, disk full ...
        print(f"[ASSET] Download of {filename} failed: {exc}. Retrying in {_ASSET_RETRY_S:.0f} s.")
        tmp.unlink(missing_ok=True)
        with _asset_lock:
            asset_status[name] = "failed"
            _asset_retry_at[name] = time.monotonic() + _ASSET_RETRY_S


def asset_path(name: str) -> Path | None:
    """Local path of an asset, or None while it is (being) downloaded."""
    filename, url = ASSETS[name]
    path = config.MODELS_DIR / filename
    if path.is_file():
        asset_status[name] = "ready"
        return path
    with _asset_lock:
        if asset_status.get(name) == "downloading":
            return None
        if asset_status.get(name) == "failed" and time.monotonic() < _asset_retry_at.get(name, 0.0):
            return None
        asset_status[name] = "downloading"
    threading.Thread(target=_download, args=(name, filename, url), daemon=True,
                     name=f"AssetDownload-{name}").start()
    return None


def ensure_face_cascade():
    """Haar cascade for face detection (bundled copy if present, else downloaded)."""
    if state.face_cascade is not None:
        return state.face_cascade
    candidates = []
    bundled = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if bundled:
        candidates.append(Path(bundled) / ASSETS["face_cascade"][0])
    if candidates and candidates[0].is_file():
        path = candidates[0]
    else:
        path = asset_path("face_cascade")
        if path is None:
            return None
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        print(f"[ASSET] {path.name} could not be loaded - deleting it so it is downloaded again.")
        path.unlink(missing_ok=True)
        asset_status["face_cascade"] = "missing"
        return None
    state.face_cascade = cascade
    return cascade


class _TasksHandDetector:
    """MediaPipe Tasks HandLandmarker behind the old ``Hands.process()`` interface."""

    def __init__(self, model_path: Path):
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = 0

    def process(self, rgb):
        import mediapipe as mp

        ts = max(self._last_ts + 1, int(time.perf_counter() * 1000))   # VIDEO mode needs increasing ms
        self._last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        return SimpleNamespace(
            multi_hand_landmarks=[SimpleNamespace(landmark=hand) for hand in result.hand_landmarks]
        )


def ensure_hand_detector(camera_state):
    """
    Hand landmark detector for hand skeletons (per camera).
    Uses the legacy mp.solutions API when available, otherwise the Tasks API
    with a downloaded model. Returns None while unavailable.
    """
    if camera_state.hand_detector is not None:
        return camera_state.hand_detector
    if time.monotonic() < _hand_retry_at.get(getattr(camera_state, "camera_id", ""), 0.0):
        return None
    try:
        import mediapipe as mp
    except Exception:
        _hand_retry_at[getattr(camera_state, "camera_id", "")] = time.monotonic() + _ASSET_RETRY_S
        return None
    try:
        if hasattr(mp, "solutions"):
            state.mp_hands = mp.solutions.hands
            camera_state.hand_detector = state.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.4,
            )
        else:
            path = asset_path("hand_landmarker")
            if path is None:
                return None
            camera_state.hand_detector = _TasksHandDetector(path)
    except Exception as exc:
        print(f"[ASSET] Hand detector could not be created: {exc}. Retrying in {_ASSET_RETRY_S:.0f} s.")
        camera_state.hand_detector = None
        _hand_retry_at[getattr(camera_state, "camera_id", "")] = time.monotonic() + _ASSET_RETRY_S
    return camera_state.hand_detector


_hand_retry_at: dict[str, float] = {}


def get_yolo_model(camera_state):
    """
    Lazy-load YOLO model.
    Note: device is controlled at prediction time via `device=DEVICE`,
    not via `.to(...)`.
    """
    if not state.YOLO_AVAILABLE:
        return None
    if camera_state.yolo_model is None:
        name = camera_state.yolo_model_name
        source_path = config.model_path(name)
        if not source_path.exists():
            # Download straight into models/ - a bare name would make Ultralytics
            # drop the weights into whatever directory the server was started from.
            try:
                from ultralytics.utils.downloads import attempt_download_asset

                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                attempt_download_asset(source_path)
            except Exception as exc:
                print(f"[YOLO] Download of {name} into models/ failed: {exc}")
        model_source = str(source_path) if source_path.exists() else name
        print(f"[YOLO] Loading model {model_source} (DEVICE={state.DEVICE})...")
        camera_state.yolo_model = state.YOLO(model_source)
        _adopt_stray_weights(name)
        print("[YOLO] Model ready:", name)
    return camera_state.yolo_model


def _adopt_stray_weights(name: str) -> None:
    """Move weights Ultralytics saved next to the cwd (or project root) into models/."""
    target = config.model_path(name)
    if target.exists():
        return
    for folder in {Path.cwd(), config.PROJECT_ROOT}:
        stray = folder / name
        if stray.is_file():
            try:
                config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stray), str(target))
                print(f"[YOLO] Moved {name} into models/.")
            except OSError as exc:
                print(f"[YOLO] Could not move {name} into models/: {exc}")
            return


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
