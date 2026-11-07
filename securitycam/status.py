"""Camera status payload (``/status`` and the live stream) and the cached GPU sampler.

``deps.get_gpu_snapshot()`` shells out to ``nvidia-smi`` (tens to hundreds of
ms, more under load). It used to run on every ``/status`` request; now a
background thread samples it about once per second and requests only read the
cache. The sampler runs only while someone reads the status (it stops after
``GPU_IDLE_S`` without readers), so an idle server spawns no processes and no
thread is left behind.
"""
from __future__ import annotations

import threading
import time

from . import config, deps, state, targets

GPU_INTERVAL_S = 1.0
GPU_IDLE_S = 15.0


class GpuSampler:
    """Samples ``deps.get_gpu_snapshot`` in a daemon thread while it is being read."""

    def __init__(self, interval_s: float = GPU_INTERVAL_S, idle_s: float = GPU_IDLE_S):
        self.interval_s = interval_s
        self.idle_s = idle_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_read = 0.0
        self.samples = 0

    def snapshot(self) -> dict:
        """The latest cached GPU metrics (never blocks on nvidia-smi)."""
        with self._lock:
            self._last_read = time.monotonic()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="GpuSampler", daemon=True)
                self._thread.start()
        return dict(state.gpu_stats)

    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def stop(self) -> None:
        """Stop the thread now (tests); the next ``snapshot()`` restarts it."""
        with self._lock:
            self._last_read = 0.0
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._wake.clear()

    def _run(self) -> None:
        while True:
            try:
                snap = deps.get_gpu_snapshot()
            except Exception:   # never let a metrics hiccup kill the sampler
                snap = None
            if snap:
                state.gpu_stats.update(snap)
                self.samples += 1
            self._wake.wait(self.interval_s)
            with self._lock:
                if time.monotonic() - self._last_read > self.idle_s:
                    self._thread = None
                    return


GPU = GpuSampler()


def camera_payload(camera_state: state.CameraRuntime) -> dict:
    return {
        "id": camera_state.camera_id,
        "label": camera_state.label,
        "ip": camera_state.ip,
        "stream_path": camera_state.stream_path,
        "has_stream": bool(camera_state.rtsp_url),
        "source_kind": camera_state.source_kind,   # "rtsp" | "file" | "url" (video analysis)
    }


def status_payload(camera_state: state.CameraRuntime) -> dict:
    return {
        "camera": camera_payload(camera_state),
        "fps": camera_state.stats["fps"],
        "inference_fps": camera_state.stats["inference_fps"],
        "last_infer_ms": camera_state.stats["last_infer_ms"],
        "num_detections": camera_state.stats["num_detections"],
        "faces": camera_state.stats.get("faces", 0),
        "classes": camera_state.stats["classes"],
        "resolution": camera_state.stats["resolution"],
        "total_frames": camera_state.stats["total_frames"],
        "last_update": camera_state.stats["last_update"],
        "yolo_enabled": camera_state.yolo_enabled,
        "yolo_available": state.YOLO_AVAILABLE,
        "yolo_model": camera_state.yolo_model_name if state.YOLO_AVAILABLE else None,
        "available_models": config.AVAILABLE_MODELS,
        "device": state.DEVICE,
        "cuda_available": state.CUDA_AVAILABLE,
        "esp_settings": camera_state.esp_settings,
        "gpu": GPU.snapshot(),
        "stream_error": camera_state.stream_error,
        "locked": targets.get_locked(camera_state),
        "zones": targets.zones_payload(camera_state),
        "assets": dict(getattr(deps, "asset_status", {}) or {}),
    }
