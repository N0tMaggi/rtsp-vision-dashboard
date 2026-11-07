"""HTTP API of the Sound ESP (``/api/sound``). Writes are JSON only.

GET   /api/sound/config        rules, curated classes, model + privacy info
PATCH /api/sound/config        partial rule update (see ``rules.validate_patch``)
GET   /api/sound/<camera_id>   live payload (polling fallback of the SSE ``sound`` topic)
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from .. import state
from . import classes, classifier, engine
from .rules import RULES, RuleError, validate_patch

blueprint = Blueprint("sound", __name__, url_prefix="/api/sound")
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
PRIVACY_NOTE = "Audio stays on this computer: it is analysed locally and never recorded, stored or uploaded."


@blueprint.record_once
def _on_register(setup_state) -> None:
    register()


_registered = False


def register() -> None:
    """Persistence section, alarm pre-roll hook, supervisor (idempotent)."""
    global _registered
    if _registered:
        return
    _registered = True
    try:
        from .. import persistence

        persistence.register_section("sound", RULES.dump, RULES.load, RULES.validate)
        RULES.listeners.append(persistence.mark_dirty)
    except Exception as exc:
        print(f"[Sound] persistence unavailable: {exc}")
    try:
        from ..alerts import clips, engine as alert_engine  # noqa: F401  (sets clips.wants_buffer first)

        prev = clips.wants_buffer
        clips.wants_buffer = lambda cid: prev(cid) or engine.wants_buffer(cid)
    except Exception as exc:
        print(f"[Sound] alarm pre-roll hook unavailable: {exc}")
    # the supervisor thread starts lazily (first sound payload request / SSE poll with sound on)


@blueprint.before_request
def _require_json_body():
    if request.method in _MUTATING and not request.is_json:
        if request.content_length or request.headers.get("Transfer-Encoding"):
            return _err("unsupported_media_type", 415)
    return None


def _err(code: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": code, **extra}), status


def config_payload() -> dict:
    return {
        "ok": True,
        "rules": RULES.config(),
        "classes": classes.meta(),
        "model": {"name": "YAMNet (AudioSet, 521 classes)", "license": classifier.MODEL_LICENSE,
                  "status": classifier.model_status(), "url": classifier.MODEL_URL},
        "classifier": classifier.CLASSIFIER.info(),
        "privacy": PRIVACY_NOTE,
    }


@blueprint.get("/config")
def get_config():
    resp = jsonify(config_payload())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.patch("/config")
def patch_config():
    try:
        clean = validate_patch(request.get_json(silent=True))
    except RuleError as exc:
        return _err(exc.code, detail=exc.detail)
    RULES.apply(clean)
    return jsonify(config_payload())


@blueprint.get("/<camera_id>")
def get_live(camera_id: str):
    cs = state.get_camera_state(camera_id)
    if cs is None:
        return _err("unknown_camera", 404)
    engine.ensure_started()
    resp = jsonify(engine.payload(cs) or {"camera": camera_id, "on": False})
    resp.headers["Cache-Control"] = "no-store"
    return resp
