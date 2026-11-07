"""Aim points (body regions) of live targets for the crosshair / PTZ auto-track.

Every target in the ``/targets`` payload gets ``aim: {key: [u, v]}`` (normalized
like ``box``) and ``aim_src``. Persons get ``head, face, neck, chest, belly,
groin, feet, center``; everything else only ``center``.

With a pose model the points come from the COCO-17 keypoints (mapped into the
same motion-predicted display space as the box); a point whose keypoints are
not visible falls back to a fixed fraction of the box height. ``aim_src`` is
"pose" when at least the torso came from keypoints.

The points are smoothed per track (time-based EMA). Caller holds the ESP lock.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

PERSON_KEYS = ("head", "face", "neck", "chest", "belly", "groin", "feet", "center")
BOX_FRACTIONS = {"head": 0.07, "face": 0.10, "neck": 0.17, "chest": 0.30, "belly": 0.45,
                 "groin": 0.53, "feet": 0.97, "center": 0.5}
SMOOTH_TAU_S = 0.12          # ~alpha 0.5 at 12 Hz
RESET_GAP_S = 1.0            # no update for this long: restart the EMA

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE = 5, 6, 11, 12, 15, 16

Point = Tuple[float, float]


def _mean(points: Sequence[Optional[Point]]) -> Optional[Point]:
    pts = [p for p in points if p is not None]
    if not pts:
        return None
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def _lerp(a: Point, b: Point, t: float) -> Point:
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def box_points(box: Sequence[float], person: bool) -> Dict[str, Point]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    if not person:
        return {"center": (cx, (y1 + y2) * 0.5)}
    h = y2 - y1
    return {k: (cx, y1 + f * h) for k, f in BOX_FRACTIONS.items()}


def pose_points(box: Sequence[float], kps: Any, vis: Any) -> Tuple[Dict[str, Point], bool]:
    """Aim points from keypoints already in box space; box fractions where they're missing.

    Returns (points, torso_from_keypoints).
    """
    out = box_points(box, True)
    if kps is None or vis is None or len(kps) <= R_ANKLE or len(vis) <= R_ANKLE:
        return out, False

    def p(i: int) -> Optional[Point]:
        return (float(kps[i][0]), float(kps[i][1])) if bool(vis[i]) else None

    nose = p(NOSE)
    eyes = _mean([p(L_EYE), p(R_EYE)])
    face = nose or eyes
    shoulder = _mean([p(L_SHOULDER), p(R_SHOULDER)])
    hip = _mean([p(L_HIP), p(R_HIP)])
    ankles = _mean([p(L_ANKLE), p(R_ANKLE)])
    head_base = _mean([p(L_EYE), p(R_EYE), p(L_EAR), p(R_EAR)]) or nose
    box_h = max(1e-6, box[3] - box[1])

    if face is not None:
        out["face"] = face
    if head_base is not None:
        ref = eyes or head_base
        dist = math.hypot(shoulder[0] - ref[0], shoulder[1] - ref[1]) if shoulder is not None else 0.12 * box_h
        out["head"] = (head_base[0], head_base[1] - 0.35 * dist)
    if shoulder is not None:
        out["neck"] = _lerp(shoulder, face, 0.2) if face is not None else shoulder
    torso = shoulder is not None and hip is not None
    if torso:
        out["chest"] = _lerp(shoulder, hip, 0.25)
        out["belly"] = _lerp(shoulder, hip, 0.65)
        length = math.hypot(hip[0] - shoulder[0], hip[1] - shoulder[1])
        out["groin"] = (hip[0], hip[1] + 0.12 * length)
    if ankles is not None:
        out["feet"] = ankles
    return out, torso


def _norm(points: Dict[str, Point], fw: float, fh: float) -> Dict[str, List[float]]:
    return {k: [min(1.0, max(0.0, x / fw)), min(1.0, max(0.0, y / fh))] for k, (x, y) in points.items()}


def compute(tr, box: Sequence[float], fw: float, fh: float) -> Tuple[Dict[str, List[float]], str]:
    """Unsmoothed normalized aim points of a track whose display box (px) is ``box``."""
    det = tr.det
    person = det.cls_name.lower() == "person"
    if not person:
        return _norm(box_points(box, False), fw, fh), "box"
    kps, vis = det.keypoints, det.kp_visible
    if kps is None or vis is None:
        return _norm(box_points(box, True), fw, fh), "box"
    # keypoints refer to the observed box; map them onto the predicted display box
    obs = tr.obs
    ow, oh = max(obs[2] - obs[0], 1.0), max(obs[3] - obs[1], 1.0)
    sx, sy = (box[2] - box[0]) / ow, (box[3] - box[1]) / oh
    mapped = [((float(k[0]) - obs[0]) * sx + box[0], (float(k[1]) - obs[1]) * sy + box[1]) for k in kps]
    points, torso = pose_points(box, mapped, vis)
    return _norm(points, fw, fh), "pose" if torso else "box"


def _store(camera_state) -> Dict[Any, Tuple[float, Dict[str, List[float]]]]:
    s = getattr(camera_state, "esp_aim", None)
    if not isinstance(s, dict):
        s = {}
        camera_state.esp_aim = s
    return s


def smoothed(camera_state, key, aim: Dict[str, List[float]], t: float) -> Dict[str, List[float]]:
    store = _store(camera_state)
    prev = store.get(key)
    if prev is not None and prev[1].keys() == aim.keys():
        pt, pa = prev
        dt = t - pt
        if dt <= 0:
            return {k: [round(v[0], 4), round(v[1], 4)] for k, v in pa.items()}
        if dt < RESET_GAP_S:
            a = 1.0 - math.exp(-dt / SMOOTH_TAU_S)
            aim = {k: [pa[k][0] + (v[0] - pa[k][0]) * a, pa[k][1] + (v[1] - pa[k][1]) * a] for k, v in aim.items()}
    store[key] = (t, aim)
    return {k: [round(v[0], 4), round(v[1], 4)] for k, v in aim.items()}


def prune(camera_state, alive_keys) -> None:
    store = _store(camera_state)
    if len(store) > len(alive_keys):
        for k in [k for k in store if k not in alive_keys]:
            del store[k]
