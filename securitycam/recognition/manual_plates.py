"""Manual plate annotation: frozen frames, OCR on a user-drawn box, manual plate-log entries.

Flow (stage "Mark Plate"):

1. ``freeze()`` keeps a copy of the camera's current raw frame (for a paused
   video that is exactly the frame on screen) plus the vehicle targets at that
   moment, under a short-lived id. The browser shows that still image while
   the user draws, so normalized box coordinates refer to the same pixels.
2. ``read_box()`` crops the drawn box from the frozen frame and runs the plate
   OCR on several variants (padding, small rotations, a detector-refined box)
   and returns the best text plus alternatives.
3. ``confirm()`` writes a plate-log entry with ``source: "manual"`` (crop,
   vehicle and frame images) and links it to the vehicle track whose box
   contains the plate, which also feeds the plate into that track's vote.

``reread_crop()`` runs the same multi-variant OCR on an existing entry's crop.
Everything runs on the HTTP thread (CPU / OpenVINO plate models, no CUDA).
"""
from __future__ import annotations

import base64
import secrets
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import plates as P
from . import platelog
from .common import is_vehicle, runtime

FREEZE_TTL_S = 900.0
MAX_FROZEN = 6
PADS = ((0.0, 0.0), (0.06, 0.15), (0.14, 0.32))       # (x, y) margin relative to the box size
ANGLES = (0.0, -4.0, 4.0, -8.0, 8.0)
MIN_BOX_PX = 6
MAX_CANDIDATES = 5

_frozen: Dict[str, Dict[str, Any]] = {}
_frozen_lock = threading.Lock()
_reader_lock = threading.Lock()
_reader: Optional[P.PlateReader] = None
_reader_key: Optional[str] = None


class ManualPlateError(ValueError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


# ----------------------------------------------------------------------------- frozen frames
def _prune() -> None:
    cutoff = time.monotonic() - FREEZE_TTL_S
    for k in [k for k, v in _frozen.items() if v["mono"] < cutoff]:
        del _frozen[k]
    while len(_frozen) > MAX_FROZEN:
        del _frozen[min(_frozen, key=lambda k: _frozen[k]["mono"])]


def freeze(camera_state, frame: Optional[np.ndarray] = None, targets: Optional[List[dict]] = None) -> Dict[str, Any]:
    """Keep the current frame (and its vehicle targets) under an id; returns {id, w, h, ...}."""
    if frame is None:
        with camera_state.frame_lock:
            frame = camera_state.latest_frame
    if frame is None or getattr(frame, "ndim", 0) != 3:
        raise ManualPlateError("no_frame", 409)
    if targets is None:
        from .. import targets as T

        targets = T.targets_payload(camera_state).get("targets", [])
    vehicles = [{"id": int(t["id"]), "label": str(t.get("label")), "box": list(t["box"])}
                for t in targets if is_vehicle(t.get("label"))]
    fid = secrets.token_hex(8)
    h, w = frame.shape[:2]
    with _frozen_lock:
        _prune()
        _frozen[fid] = {"camera": camera_state.camera_id, "frame": frame.copy(), "mono": time.monotonic(),
                        "wall": time.time(), "vehicles": vehicles}
        _prune()
    return {"id": fid, "camera": camera_state.camera_id, "w": w, "h": h, "vehicles": vehicles,
            "url": f"/api/cameras/{camera_state.camera_id}/freeze/{fid}.jpg"}


def frozen(camera_id: str, fid: Any) -> Dict[str, Any]:
    if not isinstance(fid, str) or len(fid) != 16 or any(c not in "0123456789abcdef" for c in fid):
        raise ManualPlateError("unknown_frame", 404)
    with _frozen_lock:
        item = _frozen.get(fid)
    if item is None or item["camera"] != camera_id:
        raise ManualPlateError("unknown_frame", 404)
    return item


def frozen_jpeg(camera_id: str, fid: str) -> bytes:
    item = frozen(camera_id, fid)
    ok, buf = cv2.imencode(".jpg", item["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ManualPlateError("encode_failed", 500)
    return buf.tobytes()


# ----------------------------------------------------------------------------- OCR
def get_reader():
    """One shared reader for the HTTP threads (Flask spawns a thread per request)."""
    global _reader, _reader_key
    from .. import compute, deps

    key = compute.small_key("plates")
    if _reader is not None and _reader_key == key:
        return _reader
    paths = [deps.asset_path(k) for k in ("plate_detector", "plate_ocr", "plate_ocr_config")]
    if any(p is None for p in paths):
        raise ManualPlateError("models_loading", 503)
    try:
        _reader, _reader_key = P.PlateReader(*paths), key
    except Exception as exc:
        print(f"[PLATES] Manual OCR unavailable: {exc}")
        raise ManualPlateError("ocr_unavailable", 503) from exc
    return _reader


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if not angle:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _crop(frame: np.ndarray, box, mx: float, my: float) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * mx, (y2 - y1) * my
    a, b = int(max(0, x1 - px)), int(max(0, y1 - py))
    c, d = int(min(w, x2 + px)), int(min(h, y2 + py))
    if c - a < MIN_BOX_PX or d - b < MIN_BOX_PX // 2:
        return None
    return frame[b:d, a:c]


def _score(text: str, conf: float) -> float:
    n = len(text)
    return conf * (1.0 if P.MIN_PLATE_CHARS <= n <= 10 else 0.6 if n >= 2 else 0.0)


def read_variants(reader, frame: np.ndarray, box: Tuple[int, int, int, int], refine: bool = True) -> Dict[str, Any]:
    """OCR one plate box with padded / rotated / detector-refined variants; best text first."""
    variants: List[Tuple[str, np.ndarray]] = []
    for mx, my in PADS:
        crop = _crop(frame, box, mx, my)
        if crop is None:
            continue
        for a in ANGLES:
            variants.append((f"pad{mx:.2f}/rot{a:+.0f}", _rotate(crop, a)))
    refined = None
    if refine:
        region = _crop(frame, box, 0.6, 0.9)
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        ox = int(max(0, x1 - (x2 - x1) * 0.6))
        oy = int(max(0, y1 - (y2 - y1) * 0.9))
        if region is not None and region.shape[0] >= 16 and region.shape[1] >= 24:
            try:
                found = reader.detect(region)
            except Exception:
                found = []
            if found:
                (bx1, by1, bx2, by2), _ = found[0]
                refined = (ox + bx1, oy + by1, ox + bx2, oy + by2)
                crop = _crop(frame, refined, 0.06, 0.12)
                if crop is not None:
                    variants.append(("detector", crop))
    best: Dict[str, Dict[str, Any]] = {}
    for name, img in variants:
        try:
            text, conf = reader.read(img)
        except Exception:
            continue
        text = P.normalize_plate(text)
        if not text:
            continue
        sc = _score(text, conf)
        cur = best.get(text)
        if cur is None or sc > cur["score"]:
            best[text] = {"text": text, "conf": round(float(conf), 3), "score": sc, "variant": name}
        else:
            cur["support"] = cur.get("support", 1) + 1
    ranked = sorted(best.values(), key=lambda c: (-(c["score"] + 0.02 * c.get("support", 1)), c["text"]))
    top = ranked[0] if ranked else None
    return {
        "text": top["text"] if top else "",
        "conf": top["conf"] if top else 0.0,
        "variant": top["variant"] if top else None,
        "candidates": [{"text": c["text"], "conf": c["conf"]} for c in ranked[:MAX_CANDIDATES]],
        "refined_box": list(refined) if refined else None,
        "tried": len(variants),
    }


def thumb_data_url(img: Optional[np.ndarray], max_w: int = 320) -> Optional[str]:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    k = min(4.0, max_w / max(1, w))
    if abs(k - 1.0) > 0.05:
        img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                         interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def parse_box(raw: Any, w: int, h: int) -> Tuple[int, int, int, int]:
    """Normalized [x1, y1, x2, y2] -> clipped integer pixel box (raises on garbage / tiny boxes)."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ManualPlateError("invalid_box")
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        raise ManualPlateError("invalid_box") from None
    if not all(np.isfinite(v) for v in vals):
        raise ManualPlateError("invalid_box")
    x1, x2 = sorted(min(1.0, max(0.0, v)) for v in (vals[0], vals[2]))
    y1, y2 = sorted(min(1.0, max(0.0, v)) for v in (vals[1], vals[3]))
    box = (int(x1 * w), int(y1 * h), int(round(x2 * w)), int(round(y2 * h)))
    if box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX // 2:
        raise ManualPlateError("box_too_small")
    return box


def read_box(camera_id: str, fid: Any, box_n: Any) -> Dict[str, Any]:
    item = frozen(camera_id, fid)
    frame = item["frame"]
    h, w = frame.shape[:2]
    box = parse_box(box_n, w, h)
    reader = get_reader()
    with _reader_lock:
        out = read_variants(reader, frame, box)
    out["crop"] = thumb_data_url(_crop(frame, box, 0.08, 0.2))
    out["vehicle"] = _vehicle_for(item, box_n)
    return out


def _vehicle_for(item: Dict[str, Any], box_n) -> Optional[Dict[str, Any]]:
    """Smallest frozen vehicle box that contains the plate box centre."""
    try:
        cx = (float(box_n[0]) + float(box_n[2])) / 2
        cy = (float(box_n[1]) + float(box_n[3])) / 2
    except (TypeError, ValueError, IndexError):
        return None
    best, area = None, None
    for v in item["vehicles"]:
        x1, y1, x2, y2 = v["box"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            a = (x2 - x1) * (y2 - y1)
            if area is None or a < area:
                best, area = v, a
    return dict(best) if best else None


def confirm(camera_state, fid: Any, box_n: Any, plate: Any, ocr_conf: Any = 0.0) -> Dict[str, Any]:
    """Create the manual plate-log entry (and feed the plate into the linked track's vote)."""
    item = frozen(camera_state.camera_id, fid)
    frame = item["frame"]
    h, w = frame.shape[:2]
    box = parse_box(box_n, w, h)
    text = platelog.normalize(plate)
    if not 2 <= len(text) <= 16:
        raise ManualPlateError("invalid_plate")
    try:
        conf = max(0.0, min(1.0, float(ocr_conf or 0.0)))
    except (TypeError, ValueError):
        conf = 0.0
    veh = _vehicle_for(item, box_n)
    if veh is not None:
        vx1, vy1, vx2, vy2 = veh["box"]
        vbox = (int(vx1 * w), int(vy1 * h), int(vx2 * w), int(vy2 * h))
    else:
        bw, bh = box[2] - box[0], box[3] - box[1]
        vbox = (box[0] - 2 * bw, box[1] - 4 * bh, box[2] + 2 * bw, box[3] + 2 * bh)
    entry = platelog.get_log().add_manual(
        camera_state.camera_id, text, frame, box, vbox, track_id=veh["id"] if veh else None,
        vehicle=veh["label"] if veh else "vehicle", ocr_conf=conf, now=item["wall"])
    if veh is not None:
        rt = runtime(camera_state)
        with rt.lock:
            ti = rt.track_info.get(veh["id"])
            if ti is not None:
                ti.plate_votes[text] = ti.plate_votes.get(text, 0.0) + 10.0   # a human beats the OCR
                ti.plate_best, ti.plate_conf = P.vote(ti.plate_votes, "", 0.0)
                ti.plate_reads = max(ti.plate_reads, 8)
    return entry


def reread_crop(img: np.ndarray) -> Dict[str, Any]:
    """Multi-variant OCR on a stored plate crop (the crop already has a margin)."""
    if img is None or img.size == 0:
        raise ManualPlateError("no_image", 404)
    h, w = img.shape[:2]
    reader = get_reader()
    with _reader_lock:
        # the stored crop has ~12 % / 25 % margin: also try the inner box
        mx, my = int(w * 0.09), int(h * 0.16)
        out = read_variants(reader, img, (mx, my, w - mx, h - my), refine=False)
    return out


def read_target(camera_state, target: Dict[str, Any]) -> Dict[str, Any]:
    """"Read Plate" on a live vehicle target right now (votes into its track like the worker)."""
    with camera_state.frame_lock:
        frame = camera_state.latest_frame
    if frame is None or getattr(frame, "ndim", 0) != 3:
        raise ManualPlateError("no_frame", 409)
    frame = frame.copy()
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = target["box"]
    box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    reader = get_reader()
    with _reader_lock:
        read = reader.read_vehicle(frame, box)
    snap = freeze(camera_state, frame, [target])
    if read is None:
        return {"found": False, "freeze": snap}
    tid = int(target["id"])
    rt = runtime(camera_state)
    with rt.lock:
        ti = rt.track_info.get(tid)
        if ti is not None:
            ti.plate_reads += 1
            ti.plate_best, ti.plate_conf = P.vote(ti.plate_votes, read.text, read.conf)
    px1, py1, px2, py2 = read.box
    return {
        "found": True, "text": P.normalize_plate(read.text), "conf": round(float(read.conf), 3),
        "box": [round(px1 / w, 4), round(py1 / h, 4), round(px2 / w, 4), round(py2 / h, 4)],
        "crop": thumb_data_url(frame[py1:py2, px1:px2]), "freeze": snap,
    }
