"""HTTP API of the automatic face ids (nested blueprint: /api/recognition/unknown).

All writes are JSON (app-wide CSRF guard); ids are integers, so image paths
can't be traversed.
"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request, send_file

from . import unknown_faces as U
from .gallery import GalleryError

blueprint = Blueprint("unknown_faces", __name__, url_prefix="/unknown")


def _err(code: str, status: int = 400):
    return jsonify({"ok": False, "error": code}), status


def _uid_ok(uid: int) -> bool:
    return 0 < uid < U.UID_MAX


@blueprint.get("")
def unknown_list():
    U.apply_merges()
    resp = jsonify({"faces": U.get_store().list()})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.get("/<int:uid>/thumb")
def unknown_thumb(uid: int):
    path = U.get_store().thumb_path(uid) if _uid_ok(uid) else None
    if path is None:
        abort(404)
    resp = send_file(path, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


@blueprint.post("/<int:uid>/name")
def unknown_name(uid: int):
    """Name This Person: the identity's samples move into the known gallery (new or existing name)."""
    from .. import tags

    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not _uid_ok(uid):
        return _err("invalid_input")
    try:
        person = U.promote(uid, data.get("name"))
    except U.UnknownFaceError as exc:
        return _err(exc.code, 404)
    except GalleryError as exc:
        return _err(exc.code)
    if person.get("id"):
        tags.STORE.move(f"f:{uid}", f"p:{person['id']}")
        U.relabel_tracks(uid, name=person.get("name"), person_id=person["id"])
    return jsonify({"ok": True, "person": person})


@blueprint.post("/<int:uid>/merge")
def unknown_merge(uid: int):
    data = request.get_json(silent=True)
    into = data.get("into") if isinstance(data, dict) else None
    if not _uid_ok(uid) or isinstance(into, bool) or not isinstance(into, int) or not _uid_ok(into):
        return _err("invalid_input")
    try:
        U.get_store().merge(uid, into)
    except U.UnknownFaceError as exc:
        return _err(exc.code, 404 if exc.code == "unknown_face" else 400)
    U.apply_merges()
    return jsonify({"ok": True, "into": into})


@blueprint.delete("/<int:uid>")
def unknown_delete(uid: int):
    from .. import tags

    if not _uid_ok(uid) or not U.get_store().delete(uid):
        return _err("unknown_face", 404)
    tags.STORE.drop(f"f:{uid}")
    U.relabel_tracks(uid, face_uid=None)
    return jsonify({"ok": True})
