"""Re-identification: BoT-SORT with ReID + an app-level re-entry memory.

* Short occlusions: Ultralytics BoT-SORT with ``with_reid: True``. With
  ``model: auto`` it reuses the detector's own features (YOLO .pt models), so
  no extra network runs; for RT-DETR / TensorRT engines Ultralytics falls back
  to a small classifier (yolo26n-cls.pt, downloaded once). A longer
  ``track_buffer`` keeps lost tracks alive for a few seconds.
* Leaving and coming back (after the tracker dropped the track): a new tracker
  id is compared with recently lost ids of the same class using a cheap
  colour descriptor (HSV histograms of the upper/lower half of the box). A
  confident match maps the new id back to the old one, so names, plate votes,
  the target lock and history stay with the person.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

TRACKER_DIR = Path(__file__).resolve().parent / "trackers"
TRACKER_REID = TRACKER_DIR / "botsort_reid.yaml"
TRACKER_DEFAULT = "botsort.yaml"

REENTRY_MEMORY_S = 120.0     # how long a lost id can be re-claimed
REENTRY_MIN_LOST_S = 0.4     # an id must be gone this long before it can be re-claimed
REENTRY_SIM = 0.82           # Bhattacharyya coefficient needed for a re-entry match
REENTRY_MARGIN = 0.04        # best match must beat the runner-up by this much
DESC_EVERY = 4               # update the descriptor of an active id every n observations
CHECK_FIRST_N = 3            # a new id is compared against the lost ids on its first n observations
MAX_LOST = 64


REID_FALLBACK_WEIGHTS = "yolo26n-cls.pt"


def native_features(model) -> bool:
    """True if BoT-SORT can reuse the detector's own features (YOLO .pt detect/pose/segment models)."""
    if type(model).__name__ == "RTDETR":
        return False
    ckpt = str(getattr(model, "ckpt_path", "") or "")
    return ckpt.endswith(".pt") and getattr(model, "task", "detect") in ("detect", "pose", "segment")


def _classifier_tracker() -> Path:
    """ReID tracker yaml that points the fallback classifier into models/ (not the cwd)."""
    from .. import config

    out = config.MODELS_DIR / "trackers" / "botsort_reid_cls.yaml"
    weights = (config.MODELS_DIR / REID_FALLBACK_WEIGHTS).as_posix()
    text = TRACKER_REID.read_text(encoding="utf-8").replace(
        "model: auto", f'model: "{weights}"', 1)
    if not out.is_file() or out.read_text(encoding="utf-8") != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return out


def tracker_for(tracking: bool, reid: bool, model=None) -> str:
    if not (tracking and reid and TRACKER_REID.is_file()):
        return TRACKER_DEFAULT
    if model is None or native_features(model):
        return str(TRACKER_REID)
    try:
        return str(_classifier_tracker())
    except OSError:
        return str(TRACKER_REID)


def descriptor(frame: np.ndarray, box) -> Optional[np.ndarray]:
    """L1-normalized HSV histogram (upper + lower half) of a box crop, or None for tiny boxes."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    # shrink a bit: box borders are mostly background
    dx, dy = (x2 - x1) // 6, (y2 - y1) // 12
    x1, y1, x2, y2 = max(0, x1 + dx), max(0, y1 + dy), min(w, x2 - dx), min(h, y2 - dy)
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None
    crop = cv2.resize(frame[y1:y2, x1:x2], (32, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    parts = []
    for half in (hsv[:32], hsv[32:]):
        hist = cv2.calcHist([half], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
        parts.append(hist)
    d = np.concatenate(parts).astype(np.float32)
    s = float(d.sum())
    return d / s if s > 0 else None


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya coefficient of two L1-normalized histograms (1 = identical)."""
    if a is None or b is None:
        return 0.0
    return float(np.sum(np.sqrt(a * b)))


@dataclass
class _IdState:
    canonical: int
    cls: int
    desc: Optional[np.ndarray] = None
    seen: int = 0
    last_seen: float = 0.0


@dataclass
class ReentryMatcher:
    ids: Dict[int, _IdState] = field(default_factory=dict)        # raw tracker id -> state
    alias: Dict[int, int] = field(default_factory=dict)           # raw tracker id -> canonical id
    lost: Dict[int, Tuple[int, np.ndarray, float]] = field(default_factory=dict)  # canonical -> (cls, desc, t)
    active_canonical: Dict[int, int] = field(default_factory=dict)  # canonical -> raw id currently using it
    reentries: int = 0

    def reset(self) -> None:
        self.ids.clear()
        self.alias.clear()
        self.lost.clear()
        self.active_canonical.clear()

    def process(self, frame: np.ndarray, rows: List[Tuple[int, int, Tuple[float, float, float, float]]],
                now: Optional[float] = None) -> Dict[int, int]:
        """Feed one frame's (raw_id, cls, box) rows; returns raw_id -> canonical id."""
        now = time.monotonic() if now is None else now
        present = {rid for rid, _, _ in rows}
        out: Dict[int, int] = {}
        used = set()

        # ids that disappeared -> remember them as "lost"
        for rid, st in list(self.ids.items()):
            if rid in present:
                continue
            if now - st.last_seen >= REENTRY_MIN_LOST_S:
                if st.desc is not None:
                    self.lost[st.canonical] = (st.cls, st.desc, st.last_seen)
                if self.active_canonical.get(st.canonical) == rid:
                    self.active_canonical.pop(st.canonical, None)
                del self.ids[rid]
        for cid in [c for c, (_, _, t) in self.lost.items() if now - t > REENTRY_MEMORY_S]:
            del self.lost[cid]
        if len(self.lost) > MAX_LOST:
            for cid, _ in sorted(self.lost.items(), key=lambda kv: kv[1][2])[: len(self.lost) - MAX_LOST]:
                del self.lost[cid]

        for rid, cls, box in rows:
            st = self.ids.get(rid)
            if st is None:
                st = self.ids[rid] = _IdState(canonical=self.alias.get(rid, rid), cls=cls)
            st.seen += 1
            st.last_seen = now
            need_desc = st.desc is None or st.seen % DESC_EVERY == 0 or st.seen <= CHECK_FIRST_N
            if need_desc:
                d = descriptor(frame, box)
                if d is not None:
                    st.desc = d if st.desc is None else 0.7 * st.desc + 0.3 * d
            # new id: try to re-claim a lost one
            if st.canonical == rid and st.seen <= CHECK_FIRST_N and st.desc is not None and self.lost:
                cands = sorted(
                    ((similarity(st.desc, desc), cid) for cid, (c, desc, _) in self.lost.items()
                     if c == cls and cid not in self.active_canonical and cid not in used),
                    reverse=True,
                )
                if cands and cands[0][0] >= REENTRY_SIM and (len(cands) == 1 or cands[0][0] - cands[1][0] >= REENTRY_MARGIN):
                    cid = cands[0][1]
                    st.canonical = cid
                    self.alias[rid] = cid
                    self.lost.pop(cid, None)
                    self.reentries += 1
            canonical = st.canonical
            holder = self.active_canonical.get(canonical)
            if canonical in used or (holder is not None and holder != rid and holder in present):
                canonical = rid            # never show two boxes with the same id
            used.add(canonical)
            self.active_canonical[canonical] = rid
            out[rid] = canonical
        return out


def remap_results(matcher: ReentryMatcher, frame: np.ndarray, results) -> int:
    """Rewrite tracker ids in ``results[0].boxes`` in place; returns the number of changed ids."""
    if not results:
        return 0
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or not getattr(boxes, "is_track", False) or len(boxes) == 0:
        return 0
    data = boxes.data
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    rows = [(int(r[-3]), int(r[-1]), (float(r[0]), float(r[1]), float(r[2]), float(r[3]))) for r in arr]
    mapping = matcher.process(frame, rows)
    updates = [(i, mapping[rid]) for i, (rid, _, _) in enumerate(rows) if mapping.get(rid, rid) != rid]
    if not updates:
        return 0
    # Ultralytics creates result tensors under torch.inference_mode(); writing into
    # them afterwards raises "Inplace update to inference tensor outside InferenceMode".
    # A clone made outside inference mode is a normal tensor we may modify.
    new = data.clone() if hasattr(data, "clone") else np.array(data, copy=True)
    for i, cid in updates:
        new[i, -3] = cid
    boxes.data = new
    return len(updates)
