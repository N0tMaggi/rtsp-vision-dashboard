"""HTTP API for the ESP visuals: motion heatmap PNG and night-vision / heatmap status.

GET  /api/cameras/<id>/heatmap.png?window=5m|1h|24h&w=<px>   RGBA PNG (204 if empty)
GET  /api/cameras/<id>/visuals                               heatmap + night-vision info
POST /api/cameras/<id>/heatmap/reset                         clear the accumulated heat
"""
from __future__ import annotations

from flask import Blueprint, Response, abort, jsonify, request

from . import state
from .esp import heatmap, nightvision
from .esp.pipeline import _esp_lock

visuals_bp = Blueprint("visuals", __name__)


def _camera_or_404(camera_id: str):
    cs = state.get_camera_state(camera_id)
    if cs is None:
        abort(404)
    return cs


def _window(raw) -> str:
    w = str(raw or "").strip().lower()
    return w if w in heatmap.WINDOWS else ""


@visuals_bp.route("/api/cameras/<camera_id>/heatmap.png")
def heatmap_png(camera_id: str):
    cs = _camera_or_404(camera_id)
    window = _window(request.args.get("window")) or \
        _window(cs.esp_settings.get("heatmap_window")) or heatmap.DEFAULT_WINDOW
    try:
        width = int(request.args.get("w", 0))
    except (TypeError, ValueError):
        width = 0
    with _esp_lock(cs):
        png = heatmap.png_bytes(cs, window, width)
    if png is None:
        resp = Response(status=204)
    else:
        resp = Response(png, mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["X-Heatmap-Window"] = window
    return resp


@visuals_bp.route("/api/cameras/<camera_id>/heatmap/reset", methods=["POST"])
def heatmap_reset(camera_id: str):
    cs = _camera_or_404(camera_id)
    with _esp_lock(cs):
        heatmap.reset(cs)
    return jsonify({"ok": True})


@visuals_bp.route("/api/cameras/<camera_id>/visuals")
def visuals_info(camera_id: str):
    cs = _camera_or_404(camera_id)
    mode = str(cs.esp_settings.get("night_vision", "off"))
    with _esp_lock(cs):
        hm = heatmap.info(cs)
        nv = nightvision.info(cs)
    nv["mode"] = mode
    if mode != "auto":
        nv["active"] = mode == "on"
    resp = jsonify({"heatmap": hm, "night_vision": nv})
    resp.headers["Cache-Control"] = "no-store"
    return resp
