"""HTTP API of the Enhance view.

POST   /api/cameras/<id>/enhance      {region?, model?, scale?, denoise?, device?, include_overlay?} -> 202 job
GET    /api/enhance                   history (newest first), models, device, notice
GET    /api/enhance/<job>             status / progress / result metadata
POST   /api/enhance/<job>/cancel      cancel a queued or running job
DELETE /api/enhance/<job>             delete a result (cancels it first when still running)
GET    /api/enhance/<job>/<file>      original.png | enhanced.png | overlay.png | meta.json
"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file

from .. import state
from . import marking, models
from .jobs import FILES, manager
from .region import EnhanceInputError, SCALES, parse_options, valid_job_id

blueprint = Blueprint("enhance", __name__)


def _no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


@blueprint.route("/api/cameras/<camera_id>/enhance", methods=["POST"])
def enhance_start(camera_id: str):
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        abort(404)
    if not request.is_json:
        return _err("json_required", 415)
    try:
        opts = parse_options(request.get_json(silent=True), models.MODEL_IDS, models.DEFAULT_MODEL)
        job = manager.submit(camera_state, opts)
    except EnhanceInputError as exc:
        return _err(exc.code, exc.status)
    return _no_store(jsonify({"ok": True, "job": job})), 202


@blueprint.route("/api/enhance", methods=["GET"])
def enhance_list():
    return _no_store(jsonify({
        "jobs": manager.list(),
        "models": models.catalog_payload(),
        "default_model": models.DEFAULT_MODEL,
        "scales": list(SCALES),
        "device": models.pick_device("auto"),
        "notice": marking.NOTICE,
    }))


@blueprint.route("/api/enhance/<job_id>", methods=["GET"])
def enhance_get(job_id: str):
    job = manager.get(job_id)
    if job is None:
        abort(404)
    return _no_store(jsonify(job))


@blueprint.route("/api/enhance/<job_id>/cancel", methods=["POST"])
def enhance_cancel(job_id: str):
    if not valid_job_id(job_id):
        abort(404)
    if not request.is_json:
        return _err("json_required", 415)
    if not manager.cancel(job_id):
        return _err("not_running", 409)
    return jsonify({"ok": True})


@blueprint.route("/api/enhance/<job_id>", methods=["DELETE"])
def enhance_delete(job_id: str):
    if not valid_job_id(job_id):
        abort(404)
    try:
        if not manager.delete(job_id):
            abort(404)
    except OSError:
        return _err("delete_failed", 409)
    return jsonify({"ok": True})


@blueprint.route("/api/enhance/<job_id>/<name>", methods=["GET"])
def enhance_file(job_id: str, name: str):
    path = manager.file_path(job_id, name)
    if path is None:
        abort(404)
    download = request.args.get("download") == "1"
    stem = "enhanced-AI" if name == "enhanced.png" else name.rsplit(".", 1)[0]
    resp = send_file(path, mimetype=FILES[name], as_attachment=download,
                     download_name=f"securitycam-{job_id[:8]}-{stem}.{name.rsplit('.', 1)[1]}", max_age=0)
    resp.headers["Cache-Control"] = "private, max-age=3600"   # results never change
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp
