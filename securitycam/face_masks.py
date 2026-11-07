"""Custom face-cover images ("Image" face blur mode).

Uploaded images are decoded with OpenCV, downscaled to ``MAX_SIDE`` and
re-encoded as PNG (keeps alpha, drops metadata and anything that isn't a
picture) into ``data/face_masks/<id>.png``. The renderer composites them over
faces; a missing image falls back to a black box, never to the bare face.

Uploads are JSON (``data:`` URL / base64) because the app-wide CSRF guard only
accepts JSON writes.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from flask import Blueprint, jsonify, request, send_file

from . import config

DATA_DIR = Path(os.getenv("SECURITYCAM_DATA_DIR", "").strip() or (config.PROJECT_ROOT / "data"))
MASK_DIR = DATA_DIR / "face_masks"
INDEX = "index.json"
MAX_BYTES = 6 * 1024 * 1024        # decoded upload size
MAX_PIXELS = 40_000_000            # refuse decompression bombs before decoding fully
MAX_SIDE = 512
MAX_IMAGES = 32
NAME_MAX = 40
ID_RE = re.compile(r"^[a-f0-9]{12}$")

_lock = threading.Lock()
_cache: dict[str, Optional[np.ndarray]] = {}       # id -> BGRA (None = missing)
_scaled: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


class MaskError(ValueError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------- storage
def _index_path() -> Path:
    return MASK_DIR / INDEX


def _read_index() -> list[dict[str, Any]]:
    try:
        raw = json.loads(_index_path().read_text("utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and ID_RE.match(item["id"]):
            out.append({"id": item["id"], "name": str(item.get("name") or "Image")[:NAME_MAX],
                        "w": int(item.get("w") or 0), "h": int(item.get("h") or 0),
                        "alpha": bool(item.get("alpha")), "created": float(item.get("created") or 0)})
    return out


def _write_index(items: list[dict[str, Any]]) -> None:
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _index_path().with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=1), "utf-8")
    tmp.replace(_index_path())


def _public(item: dict[str, Any]) -> dict[str, Any]:
    return {**item, "url": f"/api/face-masks/{item['id']}.png"}


def list_masks() -> list[dict[str, Any]]:
    with _lock:
        return [_public(i) for i in _read_index() if (MASK_DIR / f"{i['id']}.png").is_file()]


def _decode_payload(data: Any) -> bytes:
    if not isinstance(data, str) or not data:
        raise MaskError("invalid_image")
    if data.startswith("data:"):
        head, _, data = data.partition(",")
        if ";base64" not in head:
            raise MaskError("invalid_image")
    if len(data) > MAX_BYTES * 4 // 3 + 16:
        raise MaskError("too_large", 413)
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise MaskError("invalid_image") from None
    if len(raw) > MAX_BYTES:
        raise MaskError("too_large", 413)
    return raw


def _image_size(raw: bytes) -> Optional[tuple[int, int]]:
    """Width/height from PNG / JPEG / WebP / GIF headers (cheap bomb check)."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
        return int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little")
    return None


def _normalize(raw: bytes) -> np.ndarray:
    size = _image_size(raw)
    if size and size[0] * size[1] > MAX_PIXELS:
        raise MaskError("too_large", 413)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim not in (2, 3) or img.size == 0:
        raise MaskError("invalid_image")
    if img.dtype != np.uint8:      # 16-bit PNGs
        img = cv2.convertScaleAbs(img, alpha=255.0 / max(1.0, float(img.max())))
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] != 4:
        raise MaskError("invalid_image")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS:
        raise MaskError("too_large", 413)
    k = MAX_SIDE / max(h, w)
    if k < 1:
        img = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    return img


def add_mask(name: Any, data: Any) -> dict[str, Any]:
    raw = _decode_payload(data)
    img = _normalize(raw)
    label = " ".join(str(name or "").split())[:NAME_MAX] or "Image"
    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise MaskError("invalid_image")
    with _lock:
        items = _read_index()
        if len(items) >= MAX_IMAGES:
            raise MaskError("too_many_images", 409)
        mid = secrets.token_hex(6)
        MASK_DIR.mkdir(parents=True, exist_ok=True)
        (MASK_DIR / f"{mid}.png").write_bytes(png.tobytes())
        item = {"id": mid, "name": label, "w": int(img.shape[1]), "h": int(img.shape[0]),
                "alpha": bool((img[:, :, 3] < 250).any()), "created": time.time()}
        items.append(item)
        _write_index(items)
        _cache.pop(mid, None)
    return _public(item)


def rename_mask(mask_id: str, name: Any) -> Optional[dict[str, Any]]:
    label = " ".join(str(name or "").split())[:NAME_MAX]
    if not label:
        raise MaskError("invalid_name")
    with _lock:
        items = _read_index()
        item = next((i for i in items if i["id"] == mask_id), None)
        if item is None:
            return None
        item["name"] = label
        _write_index(items)
    return _public(item)


def delete_mask(mask_id: str) -> bool:
    with _lock:
        items = _read_index()
        keep = [i for i in items if i["id"] != mask_id]
        if len(keep) == len(items):
            return False
        _write_index(keep)
        (MASK_DIR / f"{mask_id}.png").unlink(missing_ok=True)
        _cache.pop(mask_id, None)
        for key in [k for k in _scaled if k[0] == mask_id]:
            _scaled.pop(key, None)
    return True


# ---------------------------------------------------------------------------- rendering
def load(mask_id: str) -> Optional[np.ndarray]:
    """BGRA image for ``mask_id`` (cached), or None."""
    if not mask_id or not ID_RE.match(mask_id):
        return None
    hit = _cache.get(mask_id, False)
    if hit is not False:
        return hit
    img = cv2.imread(str(MASK_DIR / f"{mask_id}.png"), cv2.IMREAD_UNCHANGED)
    if img is not None and (img.ndim != 3 or img.shape[2] != 4):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA if img.ndim == 2 else cv2.COLOR_BGR2BGRA)
    if len(_cache) > 64:
        _cache.clear()
    _cache[mask_id] = img
    return img


def scaled(mask_id: str, rw: int, rh: int, fit: str) -> Optional[tuple[np.ndarray, np.ndarray, int, int]]:
    """(BGR float32, alpha float32 HxWx1, x offset, y offset) of the image sized for an rw x rh ROI."""
    img = load(mask_id)
    if img is None:
        return None
    key = (mask_id, rw, rh, fit)
    hit = _scaled.get(key)
    if hit is None:
        h, w = img.shape[:2]
        if fit == "stretch":
            tw, th = rw, rh
        else:
            k = (max if fit == "cover" else min)(rw / w, rh / h)
            tw, th = max(1, round(w * k)), max(1, round(h * k))
        sized = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA if tw < w else cv2.INTER_LINEAR)
        # crop (cover) or pad (contain) to exactly rw x rh
        canvas = np.zeros((rh, rw, 4), np.uint8)
        ox, oy = (rw - tw) // 2, (rh - th) // 2
        sx, sy = max(0, -ox), max(0, -oy)
        dx, dy = max(0, ox), max(0, oy)
        cw, ch = min(tw - sx, rw - dx), min(th - sy, rh - dy)
        canvas[dy:dy + ch, dx:dx + cw] = sized[sy:sy + ch, sx:sx + cw]
        bgr = canvas[:, :, :3].astype(np.float32)
        alpha = (canvas[:, :, 3:4].astype(np.float32)) / 255.0
        hit = (bgr, alpha)
        if len(_scaled) > 128:
            _scaled.clear()
        _scaled[key] = hit
    return hit[0], hit[1], 0, 0


def composite(roi: np.ndarray, mask_id: str, fit: str) -> bool:
    """Draw the image over ``roi`` in place. False if the image is missing."""
    rh, rw = roi.shape[:2]
    # 4px buckets keep the cache small while faces move; resize to the exact ROI is cheap.
    bw, bh = max(4, rw // 4 * 4), max(4, rh // 4 * 4)
    got = scaled(mask_id, bw, bh, fit)
    if got is None:
        return False
    bgr, alpha, _, _ = got
    if (bw, bh) != (rw, rh):
        bgr = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (rw, rh), interpolation=cv2.INTER_LINEAR)[:, :, None]
    out = roi.astype(np.float32) * (1.0 - alpha) + bgr * alpha
    roi[:] = out.astype(np.uint8)
    return True


def reset_cache() -> None:
    _cache.clear()
    _scaled.clear()


# ---------------------------------------------------------------------------- API
blueprint = Blueprint("face_masks", __name__)


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


@blueprint.route("/api/face-masks")
def api_list():
    resp = jsonify({"masks": list_masks(), "max": MAX_IMAGES, "max_bytes": MAX_BYTES})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.route("/api/face-masks", methods=["POST"])
def api_add():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("invalid_body")
    try:
        return jsonify({"ok": True, "mask": add_mask(body.get("name"), body.get("data"))})
    except MaskError as exc:
        return _err(exc.code, exc.status)


@blueprint.route("/api/face-masks/<mask_id>", methods=["PATCH"])
def api_rename(mask_id: str):
    if not ID_RE.match(mask_id):
        return _err("not_found", 404)
    body = request.get_json(silent=True)
    try:
        item = rename_mask(mask_id, body.get("name") if isinstance(body, dict) else None)
    except MaskError as exc:
        return _err(exc.code, exc.status)
    return jsonify({"ok": True, "mask": item}) if item else _err("not_found", 404)


@blueprint.route("/api/face-masks/<mask_id>", methods=["DELETE"])
def api_delete(mask_id: str):
    if not ID_RE.match(mask_id) or not delete_mask(mask_id):
        return _err("not_found", 404)
    return jsonify({"ok": True})


@blueprint.route("/api/face-masks/<mask_id>.png")
def api_image(mask_id: str):
    path = MASK_DIR / f"{mask_id}.png"
    if not ID_RE.match(mask_id) or not path.is_file():
        return _err("not_found", 404)
    resp = send_file(path, mimetype="image/png", max_age=3600)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp
