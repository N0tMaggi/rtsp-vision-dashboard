"""Shared pieces of the recognition package: settings snapshot, per-camera runtime, labels."""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .. import config

DATA_DIR = config.PROJECT_ROOT / "data"

PET_CLASSES = frozenset({"cat", "dog"})
VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})
PERSON_CLASS = "person"


def is_pet(label: Any) -> bool:
    """True for COCO pet classes (cat, dog). Used by the alarm package to skip pets."""
    return isinstance(label, str) and label.strip().lower() in PET_CLASSES


def is_vehicle(label: Any) -> bool:
    return isinstance(label, str) and label.strip().lower() in VEHICLE_CLASSES


def _b(raw: Mapping[str, Any], key: str) -> bool:
    v = raw.get(key, config.DEFAULT_ESP_SETTINGS.get(key, False))
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return bool(v)


def _f(raw: Mapping[str, Any], key: str, lo: float, hi: float) -> float:
    try:
        v = float(raw.get(key, config.DEFAULT_ESP_SETTINGS.get(key, lo)))
    except (TypeError, ValueError):
        v = float(config.DEFAULT_ESP_SETTINGS.get(key, lo))
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class RecogSettings:
    tracking: bool
    reid: bool
    known_faces: bool
    face_match_threshold: float
    face_detector: str
    plates: bool
    pets_show: bool
    pets_ignore_alarms: bool
    color_pet: Tuple[int, int, int]
    color_known: Tuple[int, int, int]
    color_unknown: Tuple[int, int, int]

    @property
    def any_worker_feature(self) -> bool:
        return self.known_faces or self.plates

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RecogSettings":
        from ..esp.colors import parse_hex

        def col(key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
            v = raw.get(key, config.DEFAULT_ESP_SETTINGS.get(key))
            return (parse_hex(v) if isinstance(v, str) else None) or fallback

        det = str(raw.get("face_detector", "yunet")).strip().lower()
        return cls(
            tracking=_b(raw, "tracking"),
            reid=_b(raw, "reid"),
            known_faces=_b(raw, "known_faces"),
            face_match_threshold=_f(raw, "face_match_threshold", 0.2, 0.8),
            face_detector=det if det in ("yunet", "haar") else "yunet",
            plates=_b(raw, "plates"),
            pets_show=_b(raw, "pets_show"),
            pets_ignore_alarms=_b(raw, "pets_ignore_alarms"),
            color_pet=col("color_pet", (67, 159, 255)),
            color_known=col("color_known", (161, 224, 126)),
            color_unknown=col("color_unknown", (90, 90, 255)),
        )


# ----------------------------------------------------------------------------- per-camera runtime
@dataclass
class FaceHit:
    """One recognized face in *source frame* pixels."""
    box: Tuple[int, int, int, int]       # x, y, w, h
    name: Optional[str]                  # None = unknown
    person_id: Optional[str]
    score: float
    recognizable: bool                   # big enough for a meaningful embedding


@dataclass
class PlateRead:
    text: str
    conf: float
    box: Tuple[int, int, int, int]       # plate box x1, y1, x2, y2 (source pixels)


@dataclass
class TrackInfo:
    name: Optional[str] = None
    person_id: Optional[str] = None
    name_score: float = 0.0
    plate_votes: Dict[str, float] = field(default_factory=dict)
    plate_reads: int = 0
    plate_best: Optional[str] = None
    plate_conf: float = 0.0
    last_plate_try: float = 0.0
    last_seen: float = 0.0


@dataclass
class RecogRuntime:
    camera_id: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    track_info: Dict[int, TrackInfo] = field(default_factory=dict)
    faces: List[FaceHit] = field(default_factory=list)
    faces_ts: float = 0.0                 # perf_counter of the frame the faces came from
    faces_shape: Tuple[int, int] = (0, 0)
    plates_now: List[Tuple[Tuple[int, int, int, int], PlateRead, Optional[int]]] = field(default_factory=list)
    plates_ts: float = 0.0
    plate_log: deque = field(default_factory=lambda: deque(maxlen=60))
    reentry: Any = None                   # reid.ReentryMatcher
    worker: Optional[threading.Thread] = None
    pending: Any = None                   # (frame, detections, ts) snapshot for the worker
    wake: threading.Event = field(default_factory=threading.Event)
    tracker_cfg: Optional[str] = None
    model_ref: Any = None
    errors: Dict[str, str] = field(default_factory=dict)


_runtimes: Dict[str, RecogRuntime] = {}
_runtimes_lock = threading.Lock()


def runtime(camera_state) -> RecogRuntime:
    cam_id = getattr(camera_state, "camera_id", "default")
    rt = _runtimes.get(cam_id)
    if rt is None:
        with _runtimes_lock:
            rt = _runtimes.get(cam_id)
            if rt is None:
                rt = _runtimes[cam_id] = RecogRuntime(camera_id=cam_id)
    return rt


def peek_runtime(camera_id: str) -> Optional[RecogRuntime]:
    return _runtimes.get(camera_id)
