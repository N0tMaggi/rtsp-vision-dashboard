"""Detection extraction from Ultralytics results.

Per-track state (smoothing, extrapolation, fades, history) lives in ``motion.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from . import masks as seg_masks

KEYPOINT_CONF_THRESHOLD = 0.5
VELOCITY_WINDOW = 5              # history samples used for velocity estimation

HistoryPoint = Tuple[int, int, float]


@dataclass(frozen=True)
class Detection:
    index: int                   # row in the raw results (keypoints are indexed by this)
    obj_id: int                  # tracker id, -1 if untracked
    cls_id: int
    cls_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    conf: float
    keypoints: Optional[np.ndarray] = None    # (K, 2) int pixel coords
    kp_visible: Optional[np.ndarray] = None   # (K,) bool
    history: Tuple[HistoryPoint, ...] = field(default_factory=tuple)
    alpha: float = 1.0                        # render opacity (fade in / out)
    vx: float = 0.0                           # smoothed center velocity, px/s
    vy: float = 0.0
    track_key: Any = None
    ref_height: int = 0                       # last observed box height (stable between inferences)
    # silhouette chams (masks.py): box-relative uint8 alpha crop from a Segment model, else None
    mask: Optional[np.ndarray] = None

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _to_numpy(value: Any) -> Optional[np.ndarray]:
    """Convert a torch tensor / numpy array / sequence to numpy without per-element syncs."""
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return str(cls_id)


def extract_detections(results: Any, settings) -> List[Detection]:
    """Turn ``results[0]`` into raw (unsmoothed) detections, honoring ``person_only``.

    Tensors are moved to the CPU once per field instead of once per box; the
    pipeline calls this once per *new* result, not once per rendered frame.
    """
    if not results:
        return []
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    confs = _to_numpy(boxes.conf).reshape(-1).astype(float)
    classes = _to_numpy(boxes.cls).reshape(-1).astype(int)
    raw_ids = getattr(boxes, "id", None)
    ids = _to_numpy(raw_ids).reshape(-1).astype(int) if raw_ids is not None else None
    names = getattr(r, "names", {}) or {}

    kp_xy = kp_conf = None
    # Keypoints are always taken when the model has them (cheap: one transfer per result):
    # besides the skeleton / head overlays they drive the aim points of the targets API.
    kp = getattr(r, "keypoints", None)
    if kp is not None and getattr(kp, "xy", None) is not None:
        kp_xy = _to_numpy(kp.xy)
        if getattr(kp, "conf", None) is not None:
            kp_conf = _to_numpy(kp.conf)

    # Segment models: masks are taken like keypoints (one small crop per kept box, once per result)
    mask_src = seg_masks.prepare(r)

    detections: List[Detection] = []
    for i in range(len(xyxy)):
        coords = xyxy[i]
        if not np.all(np.isfinite(coords)):
            continue
        cls_id = int(classes[i])
        cls_name = _class_name(names, cls_id)
        if settings.person_only and cls_name.lower() != "person":
            continue
        if confs[i] < settings.confidence_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in coords)

        keypoints = visible = None
        if kp_xy is not None and i < len(kp_xy) and len(kp_xy[i]) > 0:
            pts = kp_xy[i]
            keypoints = pts.astype(np.int32)
            # Missing keypoints are reported as (0, 0); also respect per-point confidence.
            visible = (pts[:, 0] > 0) & (pts[:, 1] > 0) & np.isfinite(pts).all(axis=1)
            if kp_conf is not None and i < len(kp_conf):
                visible &= kp_conf[i] >= KEYPOINT_CONF_THRESHOLD

        detections.append(Detection(
            index=i,
            obj_id=int(ids[i]) if ids is not None and i < len(ids) else -1,
            cls_id=cls_id,
            cls_name=cls_name,
            x1=x1, y1=y1, x2=x2, y2=y2,
            cx=(x1 + x2) // 2, cy=(y1 + y2) // 2,
            conf=float(confs[i]),
            keypoints=keypoints,
            kp_visible=visible,
            mask=seg_masks.crop(mask_src, i, coords) if mask_src is not None else None,
        ))
    return detections


def estimate_velocity(history: Sequence[HistoryPoint]) -> Optional[Tuple[float, float]]:
    """Pixels/second over the last ``VELOCITY_WINDOW`` samples, or ``None``."""
    if len(history) < VELOCITY_WINDOW:
        return None
    first, last = history[-VELOCITY_WINDOW], history[-1]
    dt = last[2] - first[2]
    if dt <= 1e-3:
        return None
    return (last[0] - first[0]) / dt, (last[1] - first[1]) / dt
