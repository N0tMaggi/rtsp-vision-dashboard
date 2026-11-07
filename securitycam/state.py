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
CUDA_REASON = ""          # why CUDA is not used (set by deps.ensure_torch_and_yolo), shown in the UI
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
    last_faces: list[tuple[int, int, int, int]] = field(default_factory=list)
    last_face_frame: int = -999
    last_hands: list[list[tuple[int, int]]] = field(default_factory=list)
    last_hand_frame: int = -999
    hand_detector: Any = None
    # Guards track_history / esp_motion / last_faces / last_hands / hand_detector
    # while the ESP overlay runs.
    esp_lock: threading.RLock = field(default_factory=threading.RLock)
    # Render-time motion model (esp.motion.MotionState), created lazily.
    esp_motion: Any = None
    # Locked target id (same id space as /targets) or None; user zones (normalized boxes).
    esp_locked: int | None = None
    esp_zones: list[dict[str, Any]] = field(default_factory=list)
    esp_zone_seq: int = 0
    # Last results object folded into the motion model (ingested once per inference).
    esp_last_results: Any = None
    # (results, num_detections, classes) cache so stats are computed once per result.
    results_meta: Any = None
    latest_frame: Any = None
    # time.perf_counter() capture time of latest_frame / of the frame latest_results came from
    latest_frame_ts: float = 0.0
    latest_results: Any = None
    latest_results_ts: float | None = None
    frame_lock: threading.Lock = field(default_factory=threading.Lock)
    stats: dict[str, Any] = field(default_factory=_default_stats)
    stream_error: str = ""
    worker_thread: threading.Thread | None = None
    # Video sources (securitycam/sources.py): "rtsp" for configured cameras,
    # "file" / "url" for virtual cameras opened from the video library.
    source_kind: str = "rtsp"
    source: Any = None              # sources.VideoSource for virtual cameras
    source_epoch: int = 0           # bumped on loop/seek cuts (stale inference is dropped)
    closed: bool = False            # virtual camera closed -> its YOLO worker exits


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
