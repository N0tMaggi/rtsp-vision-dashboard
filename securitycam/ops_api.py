"""HTTP API for operation & persistence: presets/schedules, settings
import/export, multi-camera overview and ONVIF PTZ.

Registered with one line in ``webapp.create_app()``; registering the
blueprint also restores ``data/state.json`` and starts the preset scheduler.
"""
from __future__ import annotations

import json
import time

from flask import Blueprint, Response, jsonify, request

from . import config, persistence, presets, ptz, state

blueprint = Blueprint("ops", __name__)

MAX_IMPORT_BYTES = 2 * 1024 * 1024
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
# Requests that never change configuration (no save attempt needed).
_NO_SAVE_PREFIXES = ("/api/settings/import",)


@blueprint.record_once
def _on_register(setup_state) -> None:
    presets.register()
    ptz.register()
    persistence.init()
    if setup_state.app.config.get("SECURITYCAM_SCHEDULER", True):
        presets.SCHEDULER.start()


@blueprint.after_app_request
def _persist_after_change(response):
    # Any successful mutating request may have changed user configuration
    # (ESP settings, model, zones, ...). Saving is debounced and skipped when
    # the serialized state did not change, so this is cheap.
    if request.method not in _MUTATING or response.status_code >= 400:
        return response
    path = request.path
    if path.startswith(_NO_SAVE_PREFIXES) or ("/ptz/" in path and not path.endswith("/ptz/settings")):
        return response
    persistence.mark_dirty()
    return response


def _err(code: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": code, **extra}), status


@blueprint.before_request
def _require_json_body():
    # A body on a mutating route must be JSON: browsers can send text/plain or
    # form bodies cross-site without a CORS preflight, JSON they cannot.
    if request.method in _MUTATING and not request.is_json:
        if request.content_length or request.headers.get("Transfer-Encoding"):
            return _err("unsupported_media_type", 415)
    return None


def _body():
    return request.get_json(silent=True)


# ---------------------------------------------------------------------------- overview (grid)
@blueprint.route("/api/overview")
def overview():
    resp = jsonify(overview_payload())
    resp.headers["Cache-Control"] = "no-store"
    return resp


def overview_payload() -> dict:
    """fps / detections / live flag of every camera (grid tiles; also pushed by the live stream)."""
    now = time.time()
    cams = []
    for cs in state.list_camera_states():
        stats = cs.stats
        last = float(stats.get("last_update") or 0.0)
        cams.append({
            "id": cs.camera_id,
            "label": cs.label,
            "has_stream": bool(cs.rtsp_url),
            "fps": round(float(stats.get("fps") or 0.0), 1),
            "num_detections": int(stats.get("num_detections") or 0),
            "yolo_enabled": bool(cs.yolo_enabled),
            "resolution": stats.get("resolution", "0x0"),
            "stream_error": cs.stream_error,
            "live": bool(cs.rtsp_url) and not cs.stream_error and last > 0 and now - last < 3.0,
            "preset": (presets.STORE.last_applied.get(cs.camera_id) or {}).get("preset_id"),
        })
    return {"cameras": cams, "ts": now}


# ---------------------------------------------------------------------------- presets
def _preset_error(exc: presets.PresetError):
    return _err(exc.code, exc.status)


@blueprint.route("/api/presets")
def presets_list():
    camera_id = request.args.get("camera_id") or None
    return jsonify(presets.payload(camera_id))


@blueprint.route("/api/presets", methods=["POST"])
def presets_create():
    try:
        preset = presets.STORE.create(_body())
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({"ok": True, "preset": preset})


@blueprint.route("/api/presets/<preset_id>", methods=["PATCH"])
def presets_update(preset_id: str):
    try:
        preset = presets.STORE.update(preset_id, _body())
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({"ok": True, "preset": preset})


@blueprint.route("/api/presets/<preset_id>", methods=["DELETE"])
def presets_delete(preset_id: str):
    try:
        removed = presets.STORE.delete(preset_id)
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({"ok": True, "removed_schedules": removed})


@blueprint.route("/api/presets/<preset_id>/apply", methods=["POST"])
def presets_apply(preset_id: str):
    data = _body() or {}
    camera_id = data.get("camera_id") if isinstance(data, dict) else None
    if camera_id == "all":
        camera_id = None
    try:
        cams = presets.STORE.apply(preset_id, camera_id)
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({
        "ok": True,
        "cameras": cams,
        "esp_settings": {cid: state.camera_states[cid].esp_settings for cid in cams},
    })


@blueprint.route("/api/schedules", methods=["POST"])
def schedules_create():
    try:
        sched = presets.STORE.create_schedule(_body())
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({"ok": True, "schedule": sched})


@blueprint.route("/api/schedules/<schedule_id>", methods=["PATCH"])
def schedules_update(schedule_id: str):
    try:
        sched = presets.STORE.update_schedule(schedule_id, _body())
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({"ok": True, "schedule": sched})


@blueprint.route("/api/schedules/<schedule_id>", methods=["DELETE"])
def schedules_delete(schedule_id: str):
    try:
        presets.STORE.delete_schedule(schedule_id)
    except presets.PresetError as exc:
        return _preset_error(exc)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------- import / export
@blueprint.route("/api/settings/export")
def settings_export():
    doc = persistence.STORE.document()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    resp = Response(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True), mimetype="application/json")
    resp.headers["Content-Disposition"] = f'attachment; filename="securitycam-settings-{stamp}.json"'
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.route("/api/settings/import", methods=["POST"])
def settings_import():
    if not request.is_json:
        return _err("unsupported_media_type", 415)
    if request.content_length is None:
        return _err("length_required", 411)
    if request.content_length > MAX_IMPORT_BYTES:
        return _err("too_large", 413)
    doc = request.get_json(silent=True)
    if doc is None:
        return _err("invalid_json")
    dry_run = request.args.get("dry_run", "").lower() in {"1", "true", "yes"}
    try:
        clean, warnings = persistence.STORE.validate_document(doc)
    except ValueError as exc:
        return _err("invalid_document", detail=str(exc))
    diff = persistence.STORE.diff(clean)
    if dry_run:
        return jsonify({"ok": True, "dry_run": True, "sections": sorted(clean), "diff": diff, "warnings": warnings})
    persistence.STORE.apply_document(clean)
    return jsonify({"ok": True, "dry_run": False, "sections": sorted(clean), "applied": len(diff), "warnings": warnings})


@blueprint.route("/api/settings/state")
def settings_state():
    s = persistence.STORE
    return jsonify({
        "path": str(s.path),
        "loaded": s.loaded,
        "pending": s.pending(),
        "last_saved_at": s.last_saved_at,
        "last_error": s.last_error,
        "sections": list(s.sections()),
        "schema_version": persistence.SCHEMA_VERSION,
    })


# ---------------------------------------------------------------------------- PTZ
def _ptz_ctl(camera_id: str):
    if state.get_camera_state(camera_id) is None:
        return None, _err("unknown_camera", 404)
    if not ptz.available():
        return None, _err("ptz_unavailable", 501, install_hint=ptz.INSTALL_HINT)
    if not ptz.configured(camera_id):
        return None, _err("not_configured", 409)
    ctl = ptz.get_controller(camera_id)
    if ctl is None:
        return None, _err("unknown_camera", 404)
    return ctl, None


def _ptz_call(camera_id: str, fn):
    ctl, error = _ptz_ctl(camera_id)
    if error:
        return error
    try:
        extra = fn(ctl) or {}
    except ptz.PtzNotReady as exc:
        # Commands while disconnected kick off a (backoff-guarded) connect.
        outcome = ctl.request_connect()
        return _err("not_ready", 409, state=exc.state, connect=outcome, ptz=ptz.status(camera_id))
    except ptz.PtzUnsupportedOp as exc:
        return _err("unsupported", 400, feature=str(exc))
    except ptz.PtzBusy:
        return _err("busy", 429)
    return jsonify({"ok": True, **extra, "ptz": ptz.status(camera_id)})


@blueprint.route("/api/cameras/<camera_id>/ptz")
def ptz_status(camera_id: str):
    if state.get_camera_state(camera_id) is None:
        return _err("unknown_camera", 404)
    resp = jsonify(ptz.status(camera_id))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.route("/api/cameras/<camera_id>/ptz/connect", methods=["POST"])
def ptz_connect(camera_id: str):
    ctl, error = _ptz_ctl(camera_id)
    if error:
        return error
    outcome = ctl.request_connect()
    code = 429 if outcome == "backoff" else 200
    return jsonify({"ok": outcome != "backoff", "connect": outcome, "ptz": ptz.status(camera_id)}), code


def _num(data, key, default=0.0):
    if not isinstance(data, dict):
        return default
    v = data.get(key, default)
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else default


@blueprint.route("/api/cameras/<camera_id>/ptz/move", methods=["POST"])
def ptz_move(camera_id: str):
    data = _body()
    ptz.autotrack.note_manual(camera_id)
    return _ptz_call(camera_id, lambda c: c.move(_num(data, "x"), _num(data, "y"), _num(data, "zoom")))


@blueprint.route("/api/cameras/<camera_id>/ptz/stop", methods=["POST"])
def ptz_stop(camera_id: str):
    ptz.autotrack.note_manual(camera_id)
    return _ptz_call(camera_id, lambda c: c.stop())


@blueprint.route("/api/cameras/<camera_id>/ptz/center", methods=["POST"])
def ptz_center(camera_id: str):
    data = _body()
    if not isinstance(data, dict) or not all(isinstance(data.get(k), (int, float)) and not isinstance(data.get(k), bool) for k in ("u", "v")):
        return _err("invalid_point")
    ptz.autotrack.note_manual(camera_id)

    def go(c):
        x, y = c.center_on(data["u"], data["v"])
        return {"translation": {"x": round(x, 4), "y": round(y, 4)}}

    return _ptz_call(camera_id, go)


@blueprint.route("/api/cameras/<camera_id>/ptz/home", methods=["POST"])
def ptz_home(camera_id: str):
    ptz.autotrack.note_manual(camera_id)
    return _ptz_call(camera_id, lambda c: c.home())


@blueprint.route("/api/cameras/<camera_id>/ptz/presets", methods=["POST"])
def ptz_preset_save(camera_id: str):
    data = _body()
    name = data.get("name") if isinstance(data, dict) else None
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 32:
        return _err("invalid_name")
    return _ptz_call(camera_id, lambda c: c.save_preset(" ".join(name.split())))


@blueprint.route("/api/cameras/<camera_id>/ptz/presets/<token>/goto", methods=["POST"])
def ptz_preset_goto(camera_id: str, token: str):
    ptz.autotrack.note_manual(camera_id)
    return _ptz_call(camera_id, lambda c: c.goto_preset(token))


@blueprint.route("/api/cameras/<camera_id>/ptz/presets/<token>", methods=["DELETE"])
def ptz_preset_delete(camera_id: str, token: str):
    return _ptz_call(camera_id, lambda c: c.remove_preset(token))


# ---------------------------------------------------------------------------- PTZ world anchoring
@blueprint.route("/api/cameras/<camera_id>/ptz/pose")
def ptz_pose(camera_id: str):
    if state.get_camera_state(camera_id) is None:
        return _err("unknown_camera", 404)
    resp = jsonify(ptz.pose_status(camera_id))
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _pose_fusion(camera_id: str):
    if state.get_camera_state(camera_id) is None:
        return None, _err("unknown_camera", 404)
    f = ptz.pose.peek(camera_id)
    if f is None or not f.ptz_known:
        return None, _err("not_ptz", 409)
    return f, None


@blueprint.route("/api/cameras/<camera_id>/ptz/pose/reanchor", methods=["POST"])
def ptz_pose_reanchor(camera_id: str):
    f, error = _pose_fusion(camera_id)
    if error:
        return error
    f.reanchor()
    return jsonify({"ok": True, "pose": ptz.pose_status(camera_id)})


@blueprint.route("/api/cameras/<camera_id>/ptz/pose/sync", methods=["POST"])
def ptz_pose_sync(camera_id: str):
    f, error = _pose_fusion(camera_id)
    if error:
        return error
    if not f.sync_onvif():
        return _err("no_onvif_position", 409, pose=ptz.pose_status(camera_id))
    return jsonify({"ok": True, "pose": ptz.pose_status(camera_id)})


@blueprint.route("/api/cameras/<camera_id>/ptz/pose/calibrate", methods=["POST"])
def ptz_pose_calibrate(camera_id: str):
    """Auto FOV calibration: relative pan by ``degrees`` (default 20) and back, compared with vision."""
    data = _body()
    degrees = _num(data, "degrees", 20.0)
    if not 5.0 <= abs(degrees) <= 60.0:
        return _err("invalid_degrees")
    f, error = _pose_fusion(camera_id)
    if error:
        return error
    ctl, error = _ptz_ctl(camera_id)
    if error:
        return error
    if ctl.state != "ready":
        return _err("not_ready", 409, state=ctl.state)
    if not ctl.caps.get("relative"):
        return _err("unsupported", 400, feature="relative")
    if f.calibration.get("state") == "running":
        return _err("busy", 429)
    import threading as _threading
    _threading.Thread(target=ptz.pose.calibrate_fov, args=(camera_id, ctl, float(degrees)),
                      name=f"PTZ-calibrate-{camera_id}", daemon=True).start()
    return jsonify({"ok": True, "pose": ptz.pose_status(camera_id)})


@blueprint.route("/api/cameras/<camera_id>/ptz/settings", methods=["PATCH"])
def ptz_settings(camera_id: str):
    if config.get_camera(camera_id) is None:
        return _err("unknown_camera", 404)
    try:
        settings = ptz.update_settings(camera_id, _body())
    except ValueError:
        return _err("invalid_settings")
    return jsonify({"ok": True, "settings": settings})


# ---------------------------------------------------------------------------- PTZ auto-track
@blueprint.route("/api/cameras/<camera_id>/ptz/autotrack", methods=["PATCH"])
def ptz_autotrack(camera_id: str):
    if state.get_camera_state(camera_id) is None:
        return _err("unknown_camera", 404)
    try:
        payload, outcome = ptz.autotrack.update_config(camera_id, _body())
    except ValueError:
        return _err("invalid_autotrack")
    out = {"ok": True, "autotrack": payload}
    if outcome is not None:
        out["connect"] = outcome
    return jsonify(out)


@blueprint.route("/api/cameras/<camera_id>/ptz/autotrack/target", methods=["POST"])
def ptz_autotrack_target(camera_id: str):
    cs = state.get_camera_state(camera_id)
    if cs is None:
        return _err("unknown_camera", 404)
    data = _body()
    if not isinstance(data, dict) or "target_id" not in data:
        return _err("invalid_target")
    target_id = data["target_id"]
    if target_id is not None and (isinstance(target_id, bool) or not isinstance(target_id, int) or target_id < 0):
        return _err("invalid_target")
    try:
        payload, outcome = ptz.autotrack.set_target(camera_id, target_id)
    except ptz.autotrack.TargetNotVisible:
        return _err("target_not_visible", 409)
    out = {"ok": True, "autotrack": payload, "locked": target_id if target_id is not None else cs.esp_locked}
    if outcome is not None:
        out["connect"] = outcome
    return jsonify(out)
