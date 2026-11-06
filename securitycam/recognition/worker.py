"""Per-camera recognition worker: known faces + license plates, off the render path.

The YOLO worker hands over (frame, detections) snapshots through
``submit()``; when YOLO is off the worker reads the latest camera frame
itself (faces still work, plates need vehicle detections). Results are cached
per track id in ``RecogRuntime.track_info`` and as short-lived per-frame lists
for the overlay.
"""
from __future__ import annotations

import base64
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import faces as F
from . import gallery
from . import plates as P
from .common import (PERSON_CLASS, FaceHit, RecogRuntime, RecogSettings, TrackInfo, is_vehicle,
                     runtime)

FACE_INTERVAL_S = 0.25
PLATE_INTERVAL_S = 0.35
PLATE_RETRY_S = 0.5           # per vehicle track while the vote is young
PLATE_SETTLED_RETRY_S = 3.0   # per vehicle track once it has many reads
PLATE_SETTLED_READS = 8
MAX_FACES = 8
MAX_VEHICLES = 4
TRACK_INFO_TTL_S = 300.0
SNAPSHOT_MAX_AGE_S = 1.0

Det = Tuple[int, str, Tuple[int, int, int, int], float]    # (track id or -1, class name, box, conf)


def detections_from_results(results) -> List[Det]:
    if not results:
        return []
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    data = boxes.data
    arr = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
    names = getattr(r, "names", {}) or {}
    tracked = bool(getattr(boxes, "is_track", False))
    out: List[Det] = []
    for row in arr:
        cls = int(row[-1])
        name = names.get(cls, str(cls)) if isinstance(names, dict) else str(cls)
        tid = int(row[-3]) if tracked else -1
        out.append((tid, str(name), (int(row[0]), int(row[1]), int(row[2]), int(row[3])), float(row[-2])))
    return out


def submit(camera_state, frame, results, frame_ts: float) -> None:
    rt = runtime(camera_state)
    rt.pending = (frame, detections_from_results(results), frame_ts, time.monotonic())
    ensure_worker(camera_state)
    rt.wake.set()


def ensure_worker(camera_state) -> None:
    rt = runtime(camera_state)
    t = rt.worker
    if t is not None and t.is_alive():
        return
    with rt.lock:
        if rt.worker is not None and rt.worker.is_alive():
            return
        rt.worker = threading.Thread(target=_loop, args=(camera_state, rt), daemon=True,
                                     name=f"RecognitionWorker-{rt.camera_id}")
        rt.worker.start()


def _snapshot(camera_state, rt: RecogRuntime):
    snap = rt.pending
    if snap is not None and time.monotonic() - snap[3] <= SNAPSHOT_MAX_AGE_S:
        return snap[0], snap[1], snap[2]
    with camera_state.frame_lock:
        frame, ts = camera_state.latest_frame, camera_state.latest_frame_ts
    return frame, [], ts


def _loop(camera_state, rt: RecogRuntime) -> None:
    last_face = last_plate = 0.0
    last_frame_id = None
    while True:
        rt.wake.wait(0.25)
        rt.wake.clear()
        try:
            s = RecogSettings.from_mapping(camera_state.esp_settings)
            if not s.any_worker_feature:
                with rt.lock:
                    rt.faces = []
                    rt.plates_now = []
                time.sleep(0.5)
                continue
            frame, dets, ts = _snapshot(camera_state, rt)
            if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
                time.sleep(0.2)
                continue
            now = time.monotonic()
            fresh = id(frame) != last_frame_id
            if s.known_faces and fresh and now - last_face >= FACE_INTERVAL_S:
                last_face = now
                run_faces(rt, s, frame, dets, ts)
            elif not s.known_faces and rt.faces:
                with rt.lock:
                    rt.faces = []
            if s.plates and fresh and dets and now - last_plate >= PLATE_INTERVAL_S:
                last_plate = now
                run_plates(rt, frame, dets, ts)
            elif not s.plates and rt.plates_now:
                with rt.lock:
                    rt.plates_now = []
            last_frame_id = id(frame)
            _prune(rt)
        except Exception as exc:     # never let the worker die
            rt.errors["worker"] = str(exc)
            print(f"[RECOG] Worker error for {rt.camera_id}: {exc}")
            time.sleep(1.0)


def _prune(rt: RecogRuntime) -> None:
    cutoff = time.monotonic() - TRACK_INFO_TTL_S
    with rt.lock:
        for tid in [t for t, ti in rt.track_info.items() if ti.last_seen < cutoff]:
            del rt.track_info[tid]


def _info(rt: RecogRuntime, tid: int) -> TrackInfo:
    ti = rt.track_info.get(tid)
    if ti is None:
        ti = rt.track_info[tid] = TrackInfo()
    ti.last_seen = time.monotonic()
    return ti


def face_owner(face_box, dets: List[Det]) -> Optional[int]:
    """Track id of the person whose upper body contains the face centre (smallest such box)."""
    fx, fy, fw, fh = face_box
    cx, cy = fx + fw / 2, fy + fh / 2
    best, best_area = None, None
    for tid, cls, (x1, y1, x2, y2), _ in dets:
        if cls.lower() != PERSON_CLASS or tid == -1:
            continue
        if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.5 * (y2 - y1):
            area = (x2 - x1) * (y2 - y1)
            if best_area is None or area < best_area:
                best, best_area = tid, area
    return best


def recognize_frame(frame: np.ndarray, threshold: float) -> Optional[List[Tuple[np.ndarray, FaceHit]]]:
    """YuNet + SFace + gallery on one frame; None if the face models aren't ready."""
    rows = F.detect(frame)
    if rows is None:
        return None
    out = []
    order = np.argsort(-(rows[:, 2] * rows[:, 3])) if len(rows) else []
    for i in list(order)[:MAX_FACES]:
        row = rows[i]
        x, y, w, h = (int(v) for v in row[:4])
        recognizable = w >= F.MIN_RECOGNIZABLE_PX
        name = pid = None
        score = 0.0
        if recognizable:
            emb = F.embed(frame, row)
            if emb is None:
                recognizable = False
            else:
                name, pid, score = gallery.match(emb, threshold)
        out.append((row, FaceHit(box=(x, y, w, h), name=name, person_id=pid, score=score,
                                 recognizable=recognizable)))
    return out


def run_faces(rt: RecogRuntime, s: RecogSettings, frame: np.ndarray, dets: List[Det], ts: float) -> None:
    found = recognize_frame(frame, s.face_match_threshold)
    if found is None:
        rt.errors["faces"] = "models_loading"
        return
    rt.errors.pop("faces", None)
    hits = [hit for _, hit in found]
    with rt.lock:
        rt.faces = hits
        rt.faces_ts = ts
        rt.faces_shape = frame.shape[:2]
        for hit in hits:
            if not hit.name:
                continue
            tid = face_owner(hit.box, dets)
            if tid is None:
                continue
            ti = _info(rt, tid)
            if ti.name is None or ti.name == hit.name or hit.score > ti.name_score + 0.05:
                ti.name, ti.person_id = hit.name, hit.person_id
                ti.name_score = max(hit.score, ti.name_score * 0.98) if ti.name == hit.name else hit.score


def _thumb_b64(img: np.ndarray) -> Optional[str]:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, 160 / max(w, 1))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def run_plates(rt: RecogRuntime, frame: np.ndarray, dets: List[Det], ts: float) -> None:
    reader = P.get_reader()
    if reader is None:
        rt.errors["plates"] = P.status().get("error") or "models_loading"
        return
    rt.errors.pop("plates", None)
    now = time.monotonic()
    vehicles = sorted((d for d in dets if is_vehicle(d[1])),
                      key=lambda d: -(d[2][2] - d[2][0]) * (d[2][3] - d[2][1]))[:MAX_VEHICLES]
    current = []
    for tid, cls, box, _ in vehicles:
        if tid != -1:
            with rt.lock:
                ti = _info(rt, tid)
                wait = PLATE_SETTLED_RETRY_S if ti.plate_reads >= PLATE_SETTLED_READS else PLATE_RETRY_S
                if now - ti.last_plate_try < wait:
                    continue
                ti.last_plate_try = now
        read = reader.read_vehicle(frame, box)
        if read is None:
            continue
        current.append((box, read, tid if tid != -1 else None))
        x1, y1, x2, y2 = read.box
        thumb = _thumb_b64(frame[y1:y2, x1:x2])
        with rt.lock:
            if tid != -1:
                ti = _info(rt, tid)
                ti.plate_reads += 1
                best, share = P.vote(ti.plate_votes, read.text, read.conf)
                ti.plate_best, ti.plate_conf = best, share
                _log_plate(rt, f"t{tid}", best or read.text, share, ti.plate_reads, tid, cls, thumb)
            else:
                _log_plate(rt, f"p{P.normalize_plate(read.text)}", P.normalize_plate(read.text),
                           read.conf, 1, None, cls, thumb)
    with rt.lock:
        rt.plates_now = current
        rt.plates_ts = ts


def _log_plate(rt: RecogRuntime, key: str, text: str, conf: float, reads: int, tid, cls: str, thumb) -> None:
    """Insert/update one entry of the per-camera plate log (caller holds rt.lock)."""
    wall = time.time()
    for entry in rt.plate_log:
        if entry["key"] == key:
            entry.update(plate=text, conf=round(float(conf), 3), reads=reads if tid is not None else entry["reads"] + 1,
                         last_seen=wall)
            if thumb:
                entry["thumb"] = thumb
            return
    rt.plate_log.appendleft({
        "key": key, "plate": text, "conf": round(float(conf), 3), "reads": reads,
        "track_id": tid, "vehicle": cls, "first_seen": wall, "last_seen": wall, "thumb": thumb,
    })
