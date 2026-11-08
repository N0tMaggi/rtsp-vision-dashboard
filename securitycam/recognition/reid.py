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
* Occlusions (walking behind a parked car): the tracker often hands out a new
  id within a few frames. Ids that vanished this very frame are candidates
  too, and a new id that shows up where the old one would be by now
  (constant-velocity guess) needs far less colour similarity; the upper-body
  histogram alone counts as well, since the car hides the legs.
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
CHECK_FIRST_N = 3            # appearance-only re-entry: first n observations of a new id
CLAIM_WINDOW_S = 1.5         # a new id may re-claim an old one during its first 1.5 s ...
CLAIM_MAX_OBS = 30           # ... and first n observations
MAX_LOST = 64
# Motion gating for occlusions (walking behind a parked car): a new id that appears where a
# lost id would be by now (constant velocity) needs much less appearance similarity.
SPATIAL_MAX_GAP_S = 4.0      # only for ids lost this recently
SPATIAL_PREDICT_MAX_S = 2.0  # extrapolate the motion at most this long
SPATIAL_RADIUS_H = 1.0       # match radius in box heights ...
SPATIAL_RADIUS_GROW_H = 0.8  # ... growing per second of gap
SPATIAL_SIM = 0.55           # appearance needed together with a motion match (half-hidden people)
MOTION_WEIGHT = 0.3          # how much a good position counts in the ranking
MAX_SPEED_H = 3.0            # velocity cap in box heights per second
EDGE_LOST_MARGIN = 0.03      # lost while touching a frame edge (fraction of the frame) ...
EDGE_BACK_MARGIN = 0.08      # ... and back near the same edge within SPATIAL_MAX_GAP_S
EDGE_SCORE = 0.3             # counts as a weak motion match


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


def similarity_upper(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of the upper halves only: robust when a car / wall hides the legs."""
    if a is None or b is None:
        return 0.0
    n = a.shape[0] // 2
    ua, ub = a[:n], b[:n]
    sa, sb = float(ua.sum()), float(ub.sum())
    if sa <= 0 or sb <= 0:
        return 0.0
    return float(np.sum(np.sqrt((ua / sa) * (ub / sb))))


Box = Tuple[float, float, float, float]


def _center_h(box: Box) -> Tuple[float, float, float]:
    """Top-center point + height. The top (head) stays put when a car hides the legs, the center doesn't."""
    return (box[0] + box[2]) * 0.5, box[1], max(1.0, box[3] - box[1])


@dataclass
class _IdState:
    canonical: int
    cls: int
    desc: Optional[np.ndarray] = None
    seen: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    box: Optional[Box] = None
    vel: Tuple[float, float] = (0.0, 0.0)     # center velocity in px / s (smoothed)

    def observe(self, box: Box, now: float) -> None:
        if self.box is not None and now > self.last_seen:
            dt = now - self.last_seen
            (cx0, cy0, _), (cx1, cy1, h) = _center_h(self.box), _center_h(box)
            vx, vy = (cx1 - cx0) / dt, (cy1 - cy0) / dt
            cap = MAX_SPEED_H * h                # boxes jump when the hidden part changes: cap it
            norm = (vx * vx + vy * vy) ** 0.5
            if norm > cap:
                vx, vy = vx * cap / norm, vy * cap / norm
            a = 0.3 if self.seen > 2 else 1.0
            self.vel = (self.vel[0] + a * (vx - self.vel[0]), self.vel[1] + a * (vy - self.vel[1]))
        self.box = box
        self.last_seen = now


@dataclass
class _Lost:
    cls: int
    desc: np.ndarray
    t: float
    box: Optional[Box] = None
    vel: Tuple[float, float] = (0.0, 0.0)


def _edges(box: Box, size: Tuple[int, int], margin: float) -> set:
    w, h = size
    out = set()
    if box[0] <= w * margin:
        out.add("left")
    if box[2] >= w * (1 - margin):
        out.add("right")
    if box[1] <= h * margin:
        out.add("top")
    if box[3] >= h * (1 - margin):
        out.add("bottom")
    return out


def _motion_score(lost: _Lost, box: Box, now: float, size: Optional[Tuple[int, int]] = None) -> Optional[float]:
    """0..1 when ``box`` is where the lost id would be by now (constant velocity), else None.

    Leaving through a frame edge and coming back through the same edge soon after also
    counts (weakly): the person walked out of the picture for a moment."""
    if lost.box is None:
        return None
    gap = max(0.0, now - lost.t)
    if gap > SPATIAL_MAX_GAP_S:
        return None
    if size is not None:
        out_edges = _edges(lost.box, size, EDGE_LOST_MARGIN)
        if out_edges and out_edges & _edges(box, size, EDGE_BACK_MARGIN):
            near = _motion_score(lost, box, now, None)
            return max(near or 0.0, EDGE_SCORE)
    cx0, cy0, h0 = _center_h(lost.box)
    cx1, cy1, h1 = _center_h(box)
    g = min(gap, SPATIAL_PREDICT_MAX_S)
    px, py = cx0 + lost.vel[0] * g, cy0 + lost.vel[1] * g
    h = max(h0, h1)
    dist = (((cx1 - px) ** 2 + (cy1 - py) ** 2) ** 0.5) / h
    radius = SPATIAL_RADIUS_H + SPATIAL_RADIUS_GROW_H * gap
    if dist > radius:
        return None
    # partly hidden is fine (only the head above a car: ~15 %), a much bigger box is another object
    if not 0.15 <= h1 / h0 <= 3.0:
        return None
    return 1.0 - dist / radius


@dataclass
class ReentryMatcher:
    ids: Dict[int, _IdState] = field(default_factory=dict)        # raw tracker id -> state
    alias: Dict[int, int] = field(default_factory=dict)           # raw tracker id -> canonical id
    lost: Dict[int, _Lost] = field(default_factory=dict)          # canonical -> last appearance / motion
    active_canonical: Dict[int, int] = field(default_factory=dict)  # canonical -> raw id currently using it
    reentries: int = 0

    def reset(self) -> None:
        self.ids.clear()
        self.alias.clear()
        self.lost.clear()
        self.active_canonical.clear()

    def _candidates(self, cls: int, present: set, used: set) -> Dict[int, Tuple[Optional[int], _Lost]]:
        """canonical -> (raw id or None, info): lost ids plus ids that vanished just now
        (the tracker often hands out a new id within a few frames)."""
        out: Dict[int, Tuple[Optional[int], _Lost]] = {
            cid: (None, lo) for cid, lo in self.lost.items()
            if lo.cls == cls and cid not in self.active_canonical and cid not in used}
        for rid, st in self.ids.items():
            if rid in present or st.cls != cls or st.desc is None or st.canonical in used:
                continue
            holder = self.active_canonical.get(st.canonical)
            if holder is not None and holder != rid:
                continue
            out[st.canonical] = (rid, _Lost(st.cls, st.desc, st.last_seen, st.box, st.vel))
        return out

    @staticmethod
    def _claim(st: _IdState, box: Box, cands, now: float, size: Optional[Tuple[int, int]] = None) -> Optional[int]:
        scored = []
        for cid, (_, lo) in cands.items():
            sim = max(similarity(st.desc, lo.desc), similarity_upper(st.desc, lo.desc))
            motion = _motion_score(lo, box, now, size)
            if motion is not None and sim >= SPATIAL_SIM:
                scored.append((sim + MOTION_WEIGHT * motion, cid))
            elif sim >= REENTRY_SIM and st.seen <= CHECK_FIRST_N:
                # appearance alone (came back somewhere else): only on the first observations,
                # as before, so a longer claim window doesn't merge different people
                scored.append((sim, cid))
        if not scored:
            return None
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] - scored[1][0] < REENTRY_MARGIN:
            return None            # ambiguous: two people could be this one
        return scored[0][1]

    def process(self, frame: np.ndarray, rows: List[Tuple[int, int, Box]],
                now: Optional[float] = None) -> Dict[int, int]:
        """Feed one frame's (raw_id, cls, box) rows; returns raw_id -> canonical id."""
        now = time.monotonic() if now is None else now
        present = {rid for rid, _, _ in rows}
        out: Dict[int, int] = {}
        used = set()

        # ids gone for a while -> remember them as "lost"
        for rid, st in list(self.ids.items()):
            if rid in present:
                continue
            if now - st.last_seen >= REENTRY_MIN_LOST_S:
                if st.desc is not None:
                    self.lost[st.canonical] = _Lost(st.cls, st.desc, st.last_seen, st.box, st.vel)
                if self.active_canonical.get(st.canonical) == rid:
                    self.active_canonical.pop(st.canonical, None)
                del self.ids[rid]
        for cid in [c for c, lo in self.lost.items() if now - lo.t > REENTRY_MEMORY_S]:
            del self.lost[cid]
        if len(self.lost) > MAX_LOST:
            for cid, _ in sorted(self.lost.items(), key=lambda kv: kv[1].t)[: len(self.lost) - MAX_LOST]:
                del self.lost[cid]

        for rid, cls, box in rows:
            st = self.ids.get(rid)
            if st is None:
                st = self.ids[rid] = _IdState(canonical=self.alias.get(rid, rid), cls=cls, first_seen=now)
            st.seen += 1
            claiming = st.canonical == rid and now - st.first_seen <= CLAIM_WINDOW_S and st.seen <= CLAIM_MAX_OBS
            if st.desc is None or st.seen % DESC_EVERY == 0 or claiming:
                d = descriptor(frame, box)
                if d is not None:
                    st.desc = d if st.desc is None else 0.7 * st.desc + 0.3 * d
            # new id: try to re-claim a lost / just vanished one
            if claiming and st.desc is not None:
                cands = self._candidates(cls, present, used)
                size = (frame.shape[1], frame.shape[0]) if frame is not None and frame.ndim >= 2 else None
                cid = self._claim(st, box, cands, now, size) if cands else None
                if cid is not None:
                    prev = cands[cid][0]
                    if prev is not None:          # vanished just now: take over its motion too
                        old = self.ids.pop(prev, None)
                        if old is not None:
                            st.vel = old.vel
                        if self.active_canonical.get(cid) == prev:
                            self.active_canonical.pop(cid, None)
                    st.canonical = cid
                    self.alias[rid] = cid
                    self.lost.pop(cid, None)
                    self.reentries += 1
            st.observe(box, now)
            canonical = st.canonical
            holder = self.active_canonical.get(canonical)
            if canonical in used or (holder is not None and holder != rid and holder in present):
                canonical = rid            # never show two boxes with the same id
            used.add(canonical)
            self.active_canonical[canonical] = rid
            out[rid] = canonical
        return out


def remap_results(matcher: ReentryMatcher, frame: np.ndarray, results, now: Optional[float] = None) -> int:
    """Rewrite tracker ids in ``results[0].boxes`` in place; returns the number of changed ids."""
    if not results:
        return 0
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or not getattr(boxes, "is_track", False) or len(boxes) == 0:
        return 0
    data = boxes.data
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    rows = [(int(r[-3]), int(r[-1]), (float(r[0]), float(r[1]), float(r[2]), float(r[3]))) for r in arr]
    mapping = matcher.process(frame, rows, now)
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
