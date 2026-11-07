"""HTTP API of the persistent plate log (nested blueprint: /api/recognition/platelog).

All mutating routes take JSON or no body, so the app-wide CSRF guard applies.
Entry ids and image names are validated against strict patterns before any
path is built (no traversal).
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
from typing import Any, Optional

from flask import Blueprint, Response, abort, jsonify, request, send_file

from .. import state, targets
from . import platelog
from .common import is_vehicle, peek_runtime

blueprint = Blueprint("platelog", __name__, url_prefix="/platelog")

LIVE_MAX_AGE_S = 600.0      # entries older than this are never "live"
TRACK_FRESH_S = 8.0         # the recognition worker must have touched the track this recently


def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


def _float_arg(name: str) -> Optional[float]:
    raw = request.args.get(name)
    if raw in (None, ""):
        return None
    v = float(raw)               # ValueError -> 400 by the caller
    if not math.isfinite(v):
        raise ValueError(name)
    return v


def _filters() -> dict:
    camera = request.args.get("camera") or None
    if camera is not None and not platelog.CAMERA_RE.match(camera):
        raise ValueError("camera")
    q = (request.args.get("q") or "")[:32]
    return {"camera": camera, "q": q, "since": _float_arg("since"), "until": _float_arg("until"),
            "watch_only": request.args.get("watch") in ("1", "true")}


def _labels() -> dict:
    return {c.camera_id: c.label for c in state.list_camera_states()}


def _out(e: dict, labels: Optional[dict] = None) -> dict:
    out = dict(e)
    base = f"/api/recognition/platelog/{e['id']}"
    imgs = e.get("images") or {}
    v = int(float(e.get("image_ts") or 0) * 1000)     # cache buster: images are replaced by better reads
    for name in platelog.IMAGE_NAMES:
        out[f"{name}_url"] = f"{base}/{name}.jpg?v={v}" if imgs.get(name) else None
    out["camera_label"] = (labels or {}).get(e["camera"], e["camera"])
    return out


# ----------------------------------------------------------------------------- list / summary
@blueprint.get("")
def log_list():
    try:
        f = _filters()
        limit = max(1, min(500, int(request.args.get("limit", 100))))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return _err("invalid_query")
    log = platelog.get_log()
    items = log.query(**f)
    labels = _labels()
    return _no_store(jsonify({
        "items": [_out(e, labels) for e in items[offset:offset + limit]],
        "total": len(items), "offset": offset, "limit": limit, "now": time.time(),
        "version": log.version,
    }))


@blueprint.get("/summary")
def log_summary():
    try:
        seen = _float_arg("seen") or 0.0
    except ValueError:
        return _err("invalid_query")
    return _no_store(jsonify(summary_payload(seen)))


def summary_payload(seen: float = 0.0, recent: int = 0) -> dict:
    """Badge summary (``GET .../summary`` and the live stream).

    ``recent`` > 0 adds the ``first_seen`` times of the newest entries, so a
    stream client can count unread reads against its own "seen" mark.
    """
    log = platelog.get_log()
    return {**log.summary(seen, recent), "enabled": log.enabled,
            "max_count": log.max_count, "max_age_days": log.max_age_days}


@blueprint.get("/plates")
def log_groups():
    """Sightings grouped by plate (OCR-confusion tolerant)."""
    try:
        f = _filters()
        limit = max(1, min(500, int(request.args.get("limit", 200))))
    except ValueError:
        return _err("invalid_query")
    groups = platelog.get_log().groups(**f)
    return _no_store(jsonify({"plates": groups[:limit], "total": len(groups), "now": time.time()}))


@blueprint.get("/plates/<plate>")
def log_sightings(plate: str):
    """Every logged sighting of one plate (all cameras), newest first."""
    key = platelog.fold(plate)
    if not 2 <= len(key) <= 16 or len(plate) > 32:
        return _err("invalid_plate")
    labels = _labels()
    items = [_out(e, labels) for e in platelog.get_log().query(plate_key=key)]
    by_camera: dict = {}
    for e in items:
        by_camera[e["camera"]] = by_camera.get(e["camera"], 0) + 1
    return _no_store(jsonify({
        "plate_key": key, "plate": items[0]["plate"] if items else platelog.normalize(plate),
        "count": len(items), "by_camera": by_camera,
        "first_seen": min((e["first_seen"] for e in items), default=None),
        "last_seen": max((e["last_seen"] for e in items), default=None),
        "items": items,
    }))


# ----------------------------------------------------------------------------- export
_CSV_FIELDS = ["id", "camera", "camera_label", "plate", "confidence", "ocr_confidence", "reads", "first_seen",
               "last_seen", "first_seen_iso", "last_seen_iso", "track_id", "vehicle", "watchlist_note"]


def _iso(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def _cell(v: Any) -> Any:
    # spreadsheet formula injection guard
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


@blueprint.get("/export")
def log_export():
    fmt = (request.args.get("format") or "csv").lower()
    if fmt not in ("csv", "json"):
        return _err("invalid_format")
    try:
        f = _filters()
    except ValueError:
        return _err("invalid_query")
    labels = _labels()
    items = platelog.get_log().query(**f)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if fmt == "json":
        rows = []
        for e in items:
            o = _out(e, labels)
            o.pop("match", None)
            o["first_seen_iso"], o["last_seen_iso"] = _iso(e["first_seen"]), _iso(e["last_seen"])
            rows.append(o)
        body = json.dumps({"exported": time.time(), "count": len(rows), "items": rows}, indent=2)
        resp = Response(body, mimetype="application/json")
        resp.headers["Content-Disposition"] = f'attachment; filename="plate-log-{stamp}.json"'
        return _no_store(resp)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_CSV_FIELDS)
    for e in items:
        w.writerow([_cell(v) for v in (
            e["id"], e["camera"], labels.get(e["camera"], e["camera"]), e["plate"], e.get("conf"),
            e.get("ocr_conf"), e.get("reads"), round(e["first_seen"], 3), round(e["last_seen"], 3),
            _iso(e["first_seen"]), _iso(e["last_seen"]), e.get("track_id"), e.get("vehicle"),
            (e.get("watch") or {}).get("note", "") if e.get("watch") else "",
        )])
    resp = Response("﻿" + buf.getvalue(), mimetype="text/csv")
    resp.headers["Content-Disposition"] = f'attachment; filename="plate-log-{stamp}.csv"'
    return _no_store(resp)


# ----------------------------------------------------------------------------- watchlist
@blueprint.get("/watchlist")
def watch_list():
    return _no_store(jsonify({"plates": platelog.get_log().watchlist()}))


@blueprint.post("/watchlist")
def watch_add():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("invalid_input")
    try:
        item = platelog.get_log().watch_add(data.get("plate"), data.get("note", ""))
    except ValueError as exc:
        return _err(str(exc))
    return jsonify({"ok": True, "item": item})


@blueprint.delete("/watchlist/<watch_id>")
def watch_remove(watch_id: str):
    if not platelog.get_log().watch_remove(watch_id):
        return _err("unknown_watch", 404)
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------- single entry
def _entry_or_404(entry_id: str) -> dict:
    e = platelog.get_log().get(entry_id)
    if e is None:
        abort(404)
    return e


@blueprint.get("/<entry_id>")
def entry_get(entry_id: str):
    return _no_store(jsonify(_out(_entry_or_404(entry_id), _labels())))


@blueprint.delete("/<entry_id>")
def entry_delete(entry_id: str):
    if not platelog.get_log().delete(entry_id):
        return _err("unknown_entry", 404)
    return jsonify({"ok": True})


@blueprint.delete("")
def log_clear():
    camera = request.args.get("camera") or None
    if camera is not None and not platelog.CAMERA_RE.match(camera):
        return _err("invalid_camera")
    return jsonify({"ok": True, "deleted": platelog.get_log().clear(camera)})


@blueprint.patch("/<entry_id>")
def entry_set_plate(entry_id: str):
    """Correct an entry's text (after "Re-Read" or by hand)."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return _err("invalid_input")
    _entry_or_404(entry_id)
    try:
        e = platelog.get_log().set_plate(entry_id, data.get("plate"))
    except ValueError as exc:
        return _err(str(exc))
    if e is None:
        return _err("unknown_entry", 404)
    return jsonify({"ok": True, "entry": _out(e, _labels())})


@blueprint.post("/<entry_id>/reocr")
def entry_reocr(entry_id: str):
    """Run the multi-variant plate OCR again on the stored crop (does not change the entry)."""
    import cv2

    from .manual_plates import ManualPlateError, reread_crop

    _entry_or_404(entry_id)
    path = platelog.get_log().image_path(entry_id, "plate")
    img = cv2.imread(str(path)) if path is not None else None
    try:
        out = reread_crop(img)
    except ManualPlateError as exc:
        return _err(exc.code, exc.status)
    return _no_store(jsonify({"ok": True, **out}))


@blueprint.get("/<entry_id>/<name>.jpg")
def entry_image(entry_id: str, name: str):
    path = platelog.get_log().image_path(entry_id, name)
    if path is None:
        abort(404)
    resp = send_file(path, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


# ----------------------------------------------------------------------------- live target
def live_target(e: dict) -> Optional[dict]:
    """The live target for an entry's vehicle, if its track is still visible on that camera."""
    tid = e.get("track_id")
    if tid is None or time.time() - float(e.get("last_seen", 0)) > LIVE_MAX_AGE_S:
        return None
    cam = state.get_camera_state(e["camera"])
    rt = peek_runtime(e["camera"])
    if cam is None or rt is None:
        return None
    with rt.lock:
        ti = rt.track_info.get(int(tid))
        fresh = ti is not None and time.monotonic() - ti.last_seen <= TRACK_FRESH_S
        same = fresh and bool(ti.plate_best) and platelog.similar(ti.plate_best, e["plate"])
    if not same:                 # track gone, or the id now belongs to another vehicle
        return None
    for t in targets.targets_payload(cam).get("targets", []):
        if t.get("id") == int(tid) and is_vehicle(t.get("label")):
            return {"camera": e["camera"], "target_id": int(tid), "box": t.get("box"), "label": t.get("label")}
    return None


@blueprint.get("/<entry_id>/live")
def entry_live(entry_id: str):
    hit = live_target(_entry_or_404(entry_id))
    return _no_store(jsonify({"live": hit is not None, **(hit or {})}))


@blueprint.post("/<entry_id>/live")
def entry_show_live(entry_id: str):
    """Lock the vehicle's live target on its camera (the UI then switches camera + follows)."""
    hit = live_target(_entry_or_404(entry_id))
    if hit is None:
        return _err("not_live", 409)
    cam = state.get_camera_state(hit["camera"])
    if cam is None or not targets.set_lock(cam, hit["target_id"]):
        return _err("not_live", 409)
    return jsonify({"ok": True, **hit})
