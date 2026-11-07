"""HTTP routes behind the stage's right-click menu: tags, solo / hidden, freeze frame,
manual plate annotation, "Read Plate" and "Add Face To People" for one target.

Every write takes JSON (the app-wide CSRF guard applies); ids are validated
before use and image routes only serve in-memory frozen frames by random id.
"""
from __future__ import annotations

import base64
import time
import uuid

import cv2
from flask import Blueprint, Response, abort, jsonify, request

from . import state, tags, targets
from .recognition import manual_plates as MP
from .tags import TagError

blueprint = Blueprint("context", __name__)


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


def _camera(camera_id: str):
    cam = state.get_camera_state(camera_id)
    if cam is None:
        abort(404)
    return cam


def _json() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _visible_target(cam, target_id: int):
    for t in targets.targets_payload(cam).get("targets", []):
        if t["id"] == target_id:
            return t
    return None


def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ----------------------------------------------------------------------------- tags
@blueprint.get("/api/tags")
def tag_catalog():
    return _no_store(jsonify({"tags": tags.STORE.catalog(), "palette": list(tags.PALETTE),
                              "max_label": tags.LABEL_MAX}))


def _key_kind(key: str) -> str:
    return {"p": "person", "f": "face", "v": "plate", "t": "track"}[key[0]]


@blueprint.post("/api/cameras/<camera_id>/targets/<int:target_id>/tags")
def tag_add(camera_id: str, target_id: int):
    cam = _camera(camera_id)
    data = _json()
    if _visible_target(cam, target_id) is None:
        return _err("unknown_target", 404)
    keys = tags.target_keys(cam, target_id)
    try:
        out = tags.STORE.add(keys, data.get("label"), data.get("color"))
    except TagError as exc:
        return _err(exc.code)
    return jsonify({"ok": True, "tags": out, "owner": _key_kind(keys[0])})


@blueprint.delete("/api/cameras/<camera_id>/targets/<int:target_id>/tags")
def tag_remove(camera_id: str, target_id: int):
    cam = _camera(camera_id)
    data = _json()
    keys = tags.target_keys(cam, target_id)
    try:
        removed = tags.STORE.remove(keys, data.get("label"))
    except TagError as exc:
        return _err(exc.code)
    if not removed:
        return _err("unknown_tag", 404)
    return jsonify({"ok": True, "tags": tags.STORE.get(keys)})


# ----------------------------------------------------------------------------- solo / hidden (visual only)
@blueprint.get("/api/cameras/<camera_id>/view")
def view_get(camera_id: str):
    return _no_store(jsonify(tags.view_state(_camera(camera_id))))


@blueprint.post("/api/cameras/<camera_id>/view")
def view_set(camera_id: str):
    cam = _camera(camera_id)
    data = _json()
    action = data.get("action")
    tid = data.get("target_id")
    if tid is not None and (isinstance(tid, bool) or not isinstance(tid, int)):
        return _err("invalid_target")
    if action == "clear":
        view = tags.clear_view(cam)
    elif action == "unsolo":
        view = tags.set_solo(cam, None)
    elif action == "unhide":
        view = tags.unhide(cam, tid)
    elif action in ("solo", "hide"):
        if tid is None:
            return _err("invalid_target")
        target = _visible_target(cam, tid)
        if target is None:
            return _err("unknown_target", 404)
        view = tags.set_solo(cam, target) if action == "solo" else tags.hide(cam, target)
    else:
        return _err("invalid_action")
    return jsonify({"ok": True, "view": view})


# ----------------------------------------------------------------------------- freeze + manual plates
@blueprint.post("/api/cameras/<camera_id>/freeze")
def freeze(camera_id: str):
    cam = _camera(camera_id)
    _json()
    try:
        return jsonify({"ok": True, **MP.freeze(cam)})
    except MP.ManualPlateError as exc:
        return _err(exc.code, exc.status)


@blueprint.get("/api/cameras/<camera_id>/freeze/<fid>.jpg")
def freeze_image(camera_id: str, fid: str):
    _camera(camera_id)
    try:
        data = MP.frozen_jpeg(camera_id, fid)
    except MP.ManualPlateError:
        abort(404)
    resp = Response(data, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=900"
    return resp


@blueprint.post("/api/cameras/<camera_id>/manual_plate/read")
def manual_read(camera_id: str):
    _camera(camera_id)
    data = _json()
    try:
        return _no_store(jsonify({"ok": True, **MP.read_box(camera_id, data.get("freeze_id"), data.get("box"))}))
    except MP.ManualPlateError as exc:
        return _err(exc.code, exc.status)


@blueprint.post("/api/cameras/<camera_id>/manual_plate")
def manual_confirm(camera_id: str):
    cam = _camera(camera_id)
    data = _json()
    try:
        entry = MP.confirm(cam, data.get("freeze_id"), data.get("box"), data.get("plate"), data.get("ocr_conf"))
    except MP.ManualPlateError as exc:
        return _err(exc.code, exc.status)
    except ValueError as exc:
        return _err(str(exc))
    from .recognition.platelog_api import _labels, _out

    return jsonify({"ok": True, "entry": _out(entry, _labels())})


@blueprint.post("/api/cameras/<camera_id>/targets/<int:target_id>/read_plate")
def read_plate(camera_id: str, target_id: int):
    cam = _camera(camera_id)
    _json()
    target = _visible_target(cam, target_id)
    if target is None:
        return _err("unknown_target", 404)
    try:
        return _no_store(jsonify({"ok": True, **MP.read_target(cam, target)}))
    except MP.ManualPlateError as exc:
        return _err(exc.code, exc.status)


# ----------------------------------------------------------------------------- face of one target -> gallery
@blueprint.post("/api/cameras/<camera_id>/targets/<int:target_id>/face")
def target_face(camera_id: str, target_id: int):
    """The face of one person target as a gallery candidate (then POST /api/recognition/people)."""
    from .recognition import api as recog_api
    from .recognition import faces, worker
    from .recognition.common import RecogSettings

    cam = _camera(camera_id)
    _json()
    target = _visible_target(cam, target_id)
    if target is None:
        return _err("unknown_target", 404)
    with cam.frame_lock:
        frame = cam.latest_frame
    if frame is None:
        return _err("no_frame", 409)
    frame = frame.copy()
    recog_api._ensure_assets("face_yunet", "face_sface")
    found = worker.recognize_frame(frame, RecogSettings.from_mapping(cam.esp_settings).face_match_threshold)
    if found is None or not faces.sface_ready():
        return _err("models_loading", 503)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = target["box"]
    det = [(target_id, "person", (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)), 1.0)]
    for row, hit in found:
        if not hit.recognizable or worker.face_owner(hit.box, det) != target_id:
            continue
        emb = hit.embedding if hit.embedding is not None else faces.embed(frame, row)
        thumb = faces.aligned_thumb(frame, row)
        if emb is None or thumb is None:
            continue
        recog_api._prune_candidates()
        cid = uuid.uuid4().hex[:12]
        with recog_api._cand_lock:
            recog_api._candidates[cid] = {"embedding": emb, "thumb": thumb, "ts": time.monotonic()}
        ok, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return jsonify({"ok": True, "candidate": {
            "id": cid, "match": hit.name, "score": round(hit.score, 3),
            "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else None,
        }})
    return _err("no_face", 422)
