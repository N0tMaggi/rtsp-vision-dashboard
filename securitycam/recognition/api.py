"""HTTP API of the recognition package (Flask blueprint under /api/recognition)."""
from __future__ import annotations

import base64
import threading
import time
import uuid
from typing import Any, Dict

import cv2
import numpy as np
from flask import Blueprint, abort, jsonify, request, send_file

from .. import config, deps, state
from . import engines, faces, gallery, plates, worker
from .common import RecogSettings, peek_runtime, runtime
from .gallery import GalleryError

blueprint = Blueprint("recognition", __name__, url_prefix="/api/recognition")

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
CANDIDATE_TTL_S = 300.0
_candidates: Dict[str, Dict[str, Any]] = {}      # candidate id -> {embedding, thumb, ts}
_cand_lock = threading.Lock()


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


def _camera_or_404(camera_id: str):
    cam = state.get_camera_state(camera_id)
    if cam is None:
        abort(404)
    return cam


def _asset_state(name: str) -> str:
    if (config.MODELS_DIR / deps.ASSETS[name][0]).is_file():
        return "ready"
    return deps.asset_status.get(name, "missing")


def _faces_status() -> Dict[str, Any]:
    api_ok = hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
    return {
        "available": api_ok,
        "assets": {"face_yunet": _asset_state("face_yunet"), "face_sface": _asset_state("face_sface")},
        "error": None if api_ok else "opencv_too_old",
        "install_cmd": None if api_ok else "python -m pip install --upgrade opencv-contrib-python",
        "people": len(gallery.list_people()),
    }


def _ensure_assets(*names: str) -> None:
    for n in names:
        deps.asset_path(n)        # starts the background download if missing


@blueprint.get("/status")
def status():
    cam_id = request.args.get("camera")
    rt = peek_runtime(cam_id) if cam_id else None
    return jsonify({
        "faces": _faces_status(),
        "plates": plates.status(),
        "tensorrt": engines.capabilities(),
        "reid": {"available": True, "reentries": getattr(getattr(rt, "reentry", None), "reentries", 0)},
        "errors": dict(rt.errors) if rt else {},
    })


# ----------------------------------------------------------------------------- people gallery
@blueprint.get("/people")
def people_list():
    return jsonify({"people": gallery.list_people()})


def _decode_upload() -> np.ndarray | None:
    f = request.files.get("photo")
    if f is None:
        return None
    raw = f.read(MAX_UPLOAD_BYTES + 1)
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img


@blueprint.post("/people")
def people_add():
    """Add a face sample: multipart (name + photo) or JSON {name, candidate_id}."""
    _ensure_assets("face_yunet", "face_sface")
    if request.files:
        name = request.form.get("name", "")
        img = _decode_upload()
        if img is None:
            return _err("invalid_image")
        try:
            name = gallery.clean_name(name)
        except GalleryError as exc:
            return _err(exc.code)
        rows = faces.detect(img)
        if rows is None or not faces.sface_ready():
            return _err("models_loading", 503)
        row = faces.largest(rows)
        if row is None:
            return _err("no_face", 422)
        emb = faces.embed(img, row)
        if emb is None:
            return _err("no_face", 422)
        thumb = faces.aligned_thumb(img, row)
    else:
        data = request.get_json(silent=True) or {}
        name = data.get("name", "")
        cid = data.get("candidate_id")
        with _cand_lock:
            cand = _candidates.get(cid) if isinstance(cid, str) else None
        if cand is None:
            return _err("unknown_candidate", 404)
        emb, thumb = cand["embedding"], cand["thumb"]
    try:
        person = gallery.add_sample(name, emb, thumb)
    except GalleryError as exc:
        return _err(exc.code)
    return jsonify({"ok": True, "person": person})


@blueprint.patch("/people/<person_id>")
def people_rename(person_id: str):
    data = request.get_json(silent=True) or {}
    try:
        person = gallery.rename(person_id, data.get("name"))
    except GalleryError as exc:
        return _err(exc.code, 404 if exc.code == "unknown_person" else 400)
    return jsonify({"ok": True, "person": person})


@blueprint.delete("/people/<person_id>")
def people_delete(person_id: str):
    try:
        ok = gallery.delete(person_id)
    except GalleryError:
        ok = False
    if not ok:
        return _err("unknown_person", 404)
    # forget names attributed to tracks
    for cam in state.list_camera_states():
        rt = peek_runtime(cam.camera_id)
        if rt is None:
            continue
        with rt.lock:
            for ti in rt.track_info.values():
                if ti.person_id == person_id:
                    ti.name = ti.person_id = None
                    ti.name_score = 0.0
    return jsonify({"ok": True})


@blueprint.get("/people/<person_id>/thumb")
def people_thumb(person_id: str):
    try:
        path = gallery.thumb_path(person_id)
    except GalleryError:
        path = None
    if path is None or not path.is_file():
        abort(404)
    resp = send_file(path, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


# ----------------------------------------------------------------------------- capture from the live frame
def _prune_candidates() -> None:
    cutoff = time.monotonic() - CANDIDATE_TTL_S
    with _cand_lock:
        for k in [k for k, v in _candidates.items() if v["ts"] < cutoff]:
            del _candidates[k]


@blueprint.get("/cameras/<camera_id>/face_candidates")
def face_candidates(camera_id: str):
    """Faces in the current frame (with thumbnails) that can be added to the gallery."""
    cam = _camera_or_404(camera_id)
    _ensure_assets("face_yunet", "face_sface")
    with cam.frame_lock:
        frame = cam.latest_frame
    if frame is None:
        return _err("no_frame", 409)
    frame = frame.copy()
    s = RecogSettings.from_mapping(cam.esp_settings)
    found = worker.recognize_frame(frame, s.face_match_threshold)
    if found is None or not faces.sface_ready():
        return _err("models_loading", 503)
    _prune_candidates()
    h, w = frame.shape[:2]
    out = []
    for row, hit in found:
        if not hit.recognizable:
            continue
        emb = faces.embed(frame, row)
        thumb = faces.aligned_thumb(frame, row)
        if emb is None or thumb is None:
            continue
        cid = uuid.uuid4().hex[:12]
        with _cand_lock:
            _candidates[cid] = {"embedding": emb, "thumb": thumb, "ts": time.monotonic()}
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        x, y, fw, fh = hit.box
        out.append({
            "id": cid,
            "box": [round(x / w, 4), round(y / h, 4), round((x + fw) / w, 4), round((y + fh) / h, 4)],
            "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None,
            "match": hit.name,
            "score": round(hit.score, 3),
        })
    return jsonify({"ok": True, "faces": out})


# ----------------------------------------------------------------------------- plates
@blueprint.get("/cameras/<camera_id>/plates")
def plate_list(camera_id: str):
    cam = _camera_or_404(camera_id)
    rt = runtime(cam)
    with rt.lock:
        items = [dict(e) for e in rt.plate_log]
    items.sort(key=lambda e: -e["last_seen"])
    resp = jsonify({"plates": items})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.delete("/cameras/<camera_id>/plates")
def plate_clear(camera_id: str):
    cam = _camera_or_404(camera_id)
    rt = runtime(cam)
    with rt.lock:
        rt.plate_log.clear()
        for ti in rt.track_info.values():
            ti.plate_votes.clear()
            ti.plate_best = None
            ti.plate_reads = 0
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------- models / TensorRT
@blueprint.get("/models")
def model_list():
    active = {c.yolo_model_name for c in state.list_camera_states()}
    engine_in_use = {}
    for c in state.list_camera_states():
        if c.yolo_model is not None:
            engine_in_use[c.yolo_model_name] = engines.uses_engine(c.yolo_model)
    models = engines.catalog(active)
    for m in models:
        m["engine_active"] = bool(engine_in_use.get(m["name"]))
    return jsonify({"models": models, "tensorrt": engines.capabilities()})


@blueprint.post("/models/<path:model_name>/engine")
def engine_build(model_name: str):
    try:
        job = engines.start_export(model_name)
    except engines.EngineError as exc:
        code = 409 if exc.code == "busy" else 400
        return jsonify({"ok": False, "error": exc.code, "tensorrt": engines.capabilities()}), code
    return jsonify({"ok": True, "job": job}), 202


@blueprint.delete("/models/<path:model_name>/engine")
def engine_delete(model_name: str):
    try:
        existed = engines.delete_engine(model_name)
    except engines.EngineError as exc:
        return _err(exc.code, 409 if exc.code == "busy" else 400)
    return jsonify({"ok": True, "deleted": existed})
