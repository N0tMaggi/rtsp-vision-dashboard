"""HTTP API for alarms & events (Flask blueprint, registered in ``webapp.create_app``)."""
from __future__ import annotations

import re
import time

import cv2
from flask import Blueprint, Response, abort, jsonify, request, send_file

from .. import anchor, geo, state
from . import clips, conf, engine, notify
from .events import events, save_snapshot
from .store import AlertInputError, store

blueprint = Blueprint("alerts", __name__)
_ZONE_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_EVENT_FILES = {"thumb.jpg": "image/jpeg", "snapshot.jpg": "image/jpeg"}


def _camera_or_404(camera_id: str):
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        abort(404)
    return camera_state


def _bad(exc: AlertInputError):
    return jsonify({"ok": False, "error": exc.code}), 400


def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ----------------------------------------------------------------------------- config
def _view(camera_id: str):
    cs = state.get_camera_state(camera_id)
    return anchor.view_for(cs) if cs is not None else anchor.INACTIVE


def _wire_out(view, wire: dict) -> dict:
    """Tripwire in the current view (``a``/``b`` projected; ``drawn_*``, ``anchor``, ``world``, ``in_view``)."""
    out = dict(wire)
    out.update(anchor.wire_api_fields(view, wire))
    return out


def _alerts_payload(camera_id: str) -> dict:
    cfg = store.payload(camera_id)
    eng = engine.peek(camera_id)
    counts = eng.counts() if eng else {}
    view = _view(camera_id)
    cfg["tripwires"] = [_wire_out(view, w) for w in cfg["tripwires"]]
    for w in cfg["tripwires"]:
        c = counts.get(w["id"], {"in": 0, "out": 0})
        w["count_in"], w["count_out"] = c["in"], c["out"]
    return {
        "zone_rules": cfg["zone_rules"],
        "tripwires": cfg["tripwires"],
        "abandoned_notify": cfg["abandoned_notify"],
        "abandon_classes": sorted(conf.ABANDON_CLASSES),
        "channels": notify.channels(),
        "clip": {"pre_s": conf.CLIP_PRE_S, "post_s": conf.CLIP_POST_S},
    }


@blueprint.route("/api/cameras/<camera_id>/alerts", methods=["GET"])
def alerts_get(camera_id: str):
    _camera_or_404(camera_id)
    return _no_store(jsonify(_alerts_payload(camera_id)))


@blueprint.route("/api/cameras/<camera_id>/alerts", methods=["PATCH"])
def alerts_patch(camera_id: str):
    _camera_or_404(camera_id)
    try:
        store.set_options(camera_id, request.get_json(silent=True))
    except AlertInputError as exc:
        return _bad(exc)
    return jsonify({"ok": True, **_alerts_payload(camera_id)})


@blueprint.route("/api/cameras/<camera_id>/zones/<zone_id>/rule", methods=["PUT", "PATCH"])
def zone_rule_put(camera_id: str, zone_id: str):
    camera_state = _camera_or_404(camera_id)
    if not _ZONE_ID.match(zone_id) or not any(z.get("id") == zone_id for z in list(camera_state.esp_zones)):
        return jsonify({"ok": False, "error": "unknown_zone"}), 404
    try:
        rule = store.set_rule(camera_id, zone_id, request.get_json(silent=True))
    except AlertInputError as exc:
        return _bad(exc)
    return jsonify({"ok": True, "rule": rule})


@blueprint.route("/api/cameras/<camera_id>/zones/<zone_id>/rule", methods=["DELETE"])
def zone_rule_delete(camera_id: str, zone_id: str):
    _camera_or_404(camera_id)
    if not store.delete_rule(camera_id, zone_id):
        return jsonify({"ok": False, "error": "unknown_rule"}), 404
    return jsonify({"ok": True})


@blueprint.route("/api/cameras/<camera_id>/tripwires", methods=["GET"])
def tripwires_get(camera_id: str):
    _camera_or_404(camera_id)
    return _no_store(jsonify({"tripwires": _alerts_payload(camera_id)["tripwires"]}))


def _with_anchor(camera_state, data, current: dict | None = None):
    """Points from the client are in the current view: anchor them at the current pose.

    A patch that moves only one endpoint first maps the other one into the current view.
    """
    if not isinstance(data, dict) or ("a" not in data and "b" not in data) or "anchor" in data:
        return data
    pose = anchor.pose_now(camera_state)
    data = dict(data)
    if current is not None and ("a" not in data or "b" not in data):
        key = "b" if "a" in data else "a"
        old = anchor.anchor_of(current)
        if not old.close_to(pose, 1e-7):
            uv, front = geo.transfer((current[key],), old, pose, anchor.lens_for(camera_state.camera_id))
            if not front[0]:
                raise AlertInputError("point_not_visible")
            data[key] = [float(uv[0, 0]), float(uv[0, 1])]
    data["anchor"] = pose.as_dict()
    return data


@blueprint.route("/api/cameras/<camera_id>/tripwires", methods=["POST"])
def tripwire_create(camera_id: str):
    camera_state = _camera_or_404(camera_id)
    try:
        wire = store.add_tripwire(camera_id, _with_anchor(camera_state, request.get_json(silent=True)))
    except AlertInputError as exc:
        return _bad(exc)
    return jsonify({"ok": True, "tripwire": _wire_out(_view(camera_id), wire)})


@blueprint.route("/api/cameras/<camera_id>/tripwires/<wire_id>", methods=["PATCH"])
def tripwire_update(camera_id: str, wire_id: str):
    camera_state = _camera_or_404(camera_id)
    try:
        data = _with_anchor(camera_state, request.get_json(silent=True), store.get_tripwire(camera_id, wire_id))
        wire = store.update_tripwire(camera_id, wire_id, data)
    except AlertInputError as exc:
        return _bad(exc)
    if wire is None:
        return jsonify({"ok": False, "error": "unknown_tripwire"}), 404
    return jsonify({"ok": True, "tripwire": _wire_out(_view(camera_id), wire)})


@blueprint.route("/api/cameras/<camera_id>/tripwires/<wire_id>", methods=["DELETE"])
def tripwire_delete(camera_id: str, wire_id: str):
    _camera_or_404(camera_id)
    if not store.delete_tripwire(camera_id, wire_id):
        return jsonify({"ok": False, "error": "unknown_tripwire"}), 404
    return jsonify({"ok": True})


@blueprint.route("/api/cameras/<camera_id>/tripwires/<wire_id>/reset", methods=["POST"])
def tripwire_reset(camera_id: str, wire_id: str):
    _camera_or_404(camera_id)
    if not any(w["id"] == wire_id for w in store.camera_config(camera_id)["tripwires"]):
        return jsonify({"ok": False, "error": "unknown_tripwire"}), 404
    engine.get(camera_id).reset_counts(wire_id)
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------- events
def _event_out(meta: dict) -> dict:
    out = dict(meta)
    base = f"/api/events/{meta['camera']}/{meta['id']}"
    out["thumb_url"] = f"{base}/thumb.jpg" if meta.get("thumb") else None
    out["snapshot_url"] = f"{base}/snapshot.jpg" if meta.get("thumb") else None
    clip = meta.get("clip") or {}
    if clip.get("status") == "ready":
        if clip.get("type") == "video":
            out["clip_url"] = f"{base}/clip"
        elif clip.get("type") == "jpeg_seq":
            out["frames_url"] = f"{base}/frames/"
    return out


@blueprint.route("/api/events", methods=["GET"])
def events_list():
    camera_id = request.args.get("camera") or None
    if camera_id is not None and state.get_camera_state(camera_id) is None:
        abort(404)
    try:
        since = float(request.args["since"]) if request.args.get("since") else None
        limit = max(1, min(500, int(request.args.get("limit", 200))))
    except ValueError:
        return jsonify({"ok": False, "error": "invalid_query"}), 400
    items = [_event_out(m) for m in events.list(camera_id, since, limit)]
    active = []
    for cs in state.list_camera_states():
        if camera_id is not None and cs.camera_id != camera_id:
            continue
        eng = engine.peek(cs.camera_id)
        if eng is not None:
            active += [dict(a, camera=cs.camera_id) for a in eng.active()]
    return _no_store(jsonify({"events": items, "active": active, "now": time.time()}))


@blueprint.route("/api/events/<camera_id>/<event_id>", methods=["GET"])
def event_get(camera_id: str, event_id: str):
    meta = events.get(camera_id, event_id)
    if meta is None:
        abort(404)
    return _no_store(jsonify(_event_out(meta)))


@blueprint.route("/api/events/<camera_id>/<event_id>", methods=["DELETE"])
def event_delete(camera_id: str, event_id: str):
    if not events.delete(camera_id, event_id):
        return jsonify({"ok": False, "error": "unknown_event"}), 404
    return jsonify({"ok": True})


@blueprint.route("/api/events/<camera_id>/<event_id>/<name>", methods=["GET"])
def event_file(camera_id: str, event_id: str, name: str):
    meta = events.get(camera_id, event_id)
    path = events.event_dir(camera_id, event_id)
    if meta is None or path is None:
        abort(404)
    if name == "clip":
        clip = meta.get("clip") or {}
        if clip.get("status") != "ready" or clip.get("type") != "video" or clip.get("file") not in ("clip.webm", "clip.mp4"):
            abort(404)
        file_path, mime = path / clip["file"], clip.get("mime", "video/webm")
    elif name in _EVENT_FILES:
        file_path, mime = path / name, _EVENT_FILES[name]
    else:
        abort(404)
    if not file_path.is_file():
        abort(404)
    resp = send_file(file_path, mimetype=mime, conditional=True, max_age=3600)
    return resp


@blueprint.route("/api/events/<camera_id>/<event_id>/frames/<int:index>.jpg", methods=["GET"])
def event_frame(camera_id: str, event_id: str, index: int):
    path = events.event_dir(camera_id, event_id)
    if path is None or index < 0 or index > 99999:
        abort(404)
    file_path = path / "frames" / f"{index:05d}.jpg"
    if not file_path.is_file():
        abort(404)
    return send_file(file_path, mimetype="image/jpeg", max_age=3600)


# ----------------------------------------------------------------------------- snapshot / notify
def _current_jpeg(camera_state):
    """Newest rendered frame (with overlay); falls back to the raw camera frame."""
    latest = clips.ring(camera_state.camera_id).latest()
    if latest is not None and time.time() - latest[0] < 5.0:
        return latest[1]
    with camera_state.frame_lock:
        frame = camera_state.latest_frame
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return buf.tobytes() if ok else None


@blueprint.route("/api/cameras/<camera_id>/snapshot", methods=["POST"])
def snapshot(camera_id: str):
    camera_state = _camera_or_404(camera_id)
    jpeg = _current_jpeg(camera_state)
    if jpeg is None:
        return jsonify({"ok": False, "error": "no_frame"}), 409
    try:
        path = save_snapshot(camera_id, jpeg)
    except OSError:
        return jsonify({"ok": False, "error": "write_failed"}), 500
    resp = Response(jpeg, mimetype="image/jpeg")
    resp.headers["Content-Disposition"] = f'attachment; filename="{path.name}"'
    resp.headers["X-Snapshot-File"] = path.name
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.route("/api/notify/test", methods=["POST"])
def notify_test():
    data = request.get_json(silent=True) or {}
    camera_state = state.get_camera_state(str(data.get("camera", ""))) if data.get("camera") else None
    if not any(notify.channels().values()):
        return jsonify({"ok": False, "error": "no_channels", "channels": notify.channels()}), 400
    image = _current_jpeg(camera_state) if camera_state is not None else None
    event = {"camera": camera_state.camera_id if camera_state else "test",
             "camera_label": camera_state.label if camera_state else "SecurityCam",
             "ts": time.time(), "message": "Test notification"}
    result = notify.deliver(event, image)
    return jsonify({"ok": all(v == "ok" for v in result.values()), "result": result})
