"""Mutable runtime state shared across cameras and worker threads."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from . import config

# torch / YOLO imports get assigned lazily once at startup
torch = None
YOLO = None

YOLO_AVAILABLE = False
CUDA_AVAILABLE = False
DEVICE = "cpu"

face_cascade = None
mp_hands = None


def _default_stats() -> dict[str, Any]:
    return {
        "fps": 0.0,
        "inference_fps": 0.0,
        "last_infer_ms": 0.0,
        "num_detections": 0,
        "faces": 0,
        "classes": [],
        "resolution": "0x0",
        "total_frames": 0,
        "last_update": 0.0,
    }


@dataclass
class CameraRuntime:
    camera_id: str
    label: str
    ip: str
    stream_path: str
    rtsp_url: str
    yolo_enabled: bool = False
    yolo_model: Any = None
    yolo_model_name: str = config.YOLO_MODEL_NAME
    esp_settings: dict[str, Any] = field(default_factory=lambda: config.DEFAULT_ESP_SETTINGS.copy())
    track_history: dict[Any, list[tuple[int, int, float]]] = field(default_factory=dict)
    max_track_history: int = 30
    smoothed_boxes: dict[Any, dict[str, Any]] = field(default_factory=dict)
    last_faces: list[tuple[int, int, int, int]] = field(default_factory=list)
    last_face_frame: int = -999
    last_hands: list[list[tuple[int, int]]] = field(default_factory=list)
    last_hand_frame: int = -999
    hand_detector: Any = None
    latest_frame: Any = None
    latest_results: Any = None
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    stats: dict[str, Any] = field(default_factory=_default_stats)
    stream_error: str = ""
    worker_thread: threading.Thread | None = None


def _runtime_from_camera(camera: config.CameraConfig) -> CameraRuntime:
    return CameraRuntime(
        camera_id=camera.id,
        label=camera.label,
        ip=camera.ip,
        stream_path=camera.stream_path,
        rtsp_url=camera.rtsp_url,
    )


camera_states: dict[str, CameraRuntime] = {
    camera.id: _runtime_from_camera(camera)
    for camera in config.CAMERAS
}


gpu_stats = {
    "name": "CPU",
    "util": 0,
    "mem_used": 0,
    "mem_total": 0,
}


def get_camera_state(camera_id: str) -> CameraRuntime | None:
    return camera_states.get(camera_id)


def list_camera_states() -> tuple[CameraRuntime, ...]:
    return tuple(camera_states.values())
