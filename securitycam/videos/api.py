"""HTTP API for video analysis: library (upload/list/rename/delete), sources, playback.

Mutating routes take JSON bodies, except the multipart upload itself; all of
them sit behind the app-wide Origin / Sec-Fetch-Site CSRF guard.
"""
from __future__ import annotations

import math

from flask import Blueprint, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.formparser import parse_form_data

from .. import sources, state
from .library import (
    ALLOW_URLS,
    ALLOWED_EXTENSIONS,
    LIBRARY,
    MAX_UPLOAD_BYTES,
    LibraryError,
    LimitedWriter,
    SizeLimitExceeded,
    valid_id,
)

blueprint = Blueprint("videos", __name__)

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}
_FORM_OVERHEAD = 1024 * 1024        # multipart boundaries / headers on top of the file itself


def _err(code: str, status: int = 400, **extra):
    return jsonify({"ok": False, "error": code, **extra}), status


@blueprint.record_once
def _on_register(_setup_state) -> None:
    LIBRARY.cleanup_incoming()


@blueprint.before_request
def _require_json_body():
    # Only the upload is multipart; every other body must be JSON (browsers
    # cannot send JSON cross-site without a CORS preflight).
    if request.method not in _MUTATING or request.endpoint == "videos.upload":
        return None
    if not request.is_json and (request.content_length or request.headers.get("Transfer-Encoding")):
        return _err("unsupported_media_type", 415)
    return None


def _body() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _open_ids() -> dict[str, str]:
    """video id -> camera id of the currently open file sources."""
    out = {}
    for cs in state.list_camera_states():
        src = cs.source
        if src is not None and src.kind == "file":
            out[src.video_id] = cs.camera_id
    return out


def _video_out(meta: dict, open_ids: dict[str, str] | None = None) -> dict:
    open_ids = _open_ids() if open_ids is None else open_ids
    vid = meta["id"]
    return {
        "id": vid,
        "name": meta.get("name", ""),
        "original_name": meta.get("original_name", ""),
        "size": int(meta.get("size", 0)),
        "duration": float(meta.get("duration", 0.0)),
        "fps": float(meta.get("fps", 0.0)),
        "frames": int(meta.get("frames", 0)),
        "width": int(meta.get("width", 0)),
        "height": int(meta.get("height", 0)),
        "codec": meta.get("codec", ""),
        "ext": meta.get("ext", ""),
        "created": float(meta.get("created", 0.0)),
        "thumb_url": f"/api/videos/{vid}/thumb.jpg" if meta.get("has_thumb") else None,
        "camera_id": open_ids.get(vid),
    }


def _library_error(exc: LibraryError):
    return _err(exc.code, exc.status)


# ---------------------------------------------------------------------------- library
@blueprint.route("/api/videos", methods=["GET"])
def list_videos():
    open_ids = _open_ids()
    videos = [_video_out(m, open_ids) for m in LIBRARY.list()]
    resp = jsonify({
        "videos": videos,
        "total_bytes": sum(v["size"] for v in videos),
        "max_total_bytes": LIBRARY.max_total,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "extensions": list(ALLOWED_EXTENSIONS),
        "allow_urls": ALLOW_URLS,
        "sources": sources.list_sources(),
    })
    resp.headers["Cache-Control"] = "no-store"
    return resp


@blueprint.route("/api/videos", methods=["POST"])
def upload():
    if request.mimetype != "multipart/form-data":
        return _err("multipart_required", 415)
    length = request.content_length
    if length is not None and length > MAX_UPLOAD_BYTES + _FORM_OVERHEAD:
        return _err("too_large", 413, max_bytes=MAX_UPLOAD_BYTES)

    sinks: list[LimitedWriter] = []

    def stream_factory(total_content_length, content_type, filename, content_length=None):
        if sinks:
            raise LibraryError("one_file_only")          # exactly one file per request
        sink = LimitedWriter(LIBRARY.new_incoming(), MAX_UPLOAD_BYTES)
        sinks.append(sink)
        return sink

    try:
        try:
            _, form, files = parse_form_data(
                request.environ,
                stream_factory=stream_factory,
                max_form_memory_size=1024 * 1024,   # text fields only; also bounds one read chunk
                max_content_length=MAX_UPLOAD_BYTES + _FORM_OVERHEAD,
                silent=False,
                max_form_parts=16,
            )
        except SizeLimitExceeded:
            return _err("too_large", 413, max_bytes=MAX_UPLOAD_BYTES)
        except LibraryError as exc:
            return _library_error(exc)
        except HTTPException as exc:
            code = exc.code or 400
            return _err("too_large" if code == 413 else "bad_upload", code)
        except ValueError:
            return _err("bad_upload", 400)
        upload_file = files.get("file")
        if upload_file is None or not sinks:
            return _err("missing_file", 400)
        sink = sinks[0]
        sink.close()
        try:
            meta, pruned = LIBRARY.add(sink.path, upload_file.filename or "", keep=set(_open_ids()))
        except LibraryError as exc:
            return _library_error(exc)
        return jsonify({"ok": True, "video": _video_out(meta), "pruned": pruned})
    finally:
        for sink in sinks:
            sink.close()
            try:
                sink.path.unlink()
            except OSError:
                pass


@blueprint.route("/api/videos/<video_id>", methods=["GET"])
def get_video(video_id: str):
    try:
        return jsonify(_video_out(LIBRARY.get(video_id)))
    except LibraryError as exc:
        return _library_error(exc)


@blueprint.route("/api/videos/<video_id>", methods=["PATCH"])
def rename_video(video_id: str):
    try:
        meta = LIBRARY.rename(video_id, _body().get("name"))
    except LibraryError as exc:
        return _library_error(exc)
    cs = sources.find_by_video(video_id)
    if cs is not None:
        cs.label = meta["name"]
        cs.source.label = meta["name"]
    return jsonify({"ok": True, "video": _video_out(meta)})


@blueprint.route("/api/videos/<video_id>", methods=["DELETE"])
def delete_video(video_id: str):
    if not valid_id(video_id):
        return _err("unknown_video", 404)
    cs = sources.find_by_video(video_id)
    if cs is not None:
        sources.close(cs.camera_id)            # releases the file handle (needed on Windows)
    try:
        LIBRARY.delete(video_id)
    except LibraryError as exc:
        return _library_error(exc)
    except OSError:
        return _err("delete_failed", 500)
    return jsonify({"ok": True, "closed": cs.camera_id if cs is not None else None})


@blueprint.route("/api/videos/<video_id>/thumb.jpg", methods=["GET"])
def video_thumb(video_id: str):
    try:
        path = LIBRARY.thumb_path(video_id)
    except LibraryError as exc:
        return _library_error(exc)
    if path is None:
        return _err("no_thumb", 404)
    resp = send_file(path, mimetype="image/jpeg", max_age=3600)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


def _camera_out(cs) -> dict:
    return {
        "id": cs.camera_id,
        "label": cs.label,
        "ip": cs.ip,
        "stream_path": cs.stream_path,
        "has_stream": True,
        "source_kind": cs.source_kind,
    }


def _copy_from(data: dict) -> str | None:
    cid = data.get("copy_settings_from")
    return cid if isinstance(cid, str) and state.get_camera_state(cid) is not None else None


@blueprint.route("/api/videos/<video_id>/open", methods=["POST"])
def open_video(video_id: str):
    data = _body()
    try:
        meta = LIBRARY.get(video_id)
        path = LIBRARY.video_path(video_id)
    except LibraryError as exc:
        return _library_error(exc)
    loop = data.get("loop", True)
    cs = sources.open_file(video_id, str(path), meta.get("name") or "Video", float(meta.get("fps", 25.0)),
                           int(meta.get("frames", 0)), loop=loop if isinstance(loop, bool) else True,
                           copy_from=_copy_from(data))
    return jsonify({"ok": True, "camera_id": cs.camera_id, "camera": _camera_out(cs),
                    "playback": cs.source.playback_payload()})


@blueprint.route("/api/videos/<video_id>/close", methods=["POST"])
def close_video(video_id: str):
    if not valid_id(video_id):
        return _err("unknown_video", 404)
    cs = sources.find_by_video(video_id)
    if cs is None:
        return _err("not_open", 404)
    sources.close(cs.camera_id)
    return jsonify({"ok": True, "camera_id": cs.camera_id})


# ---------------------------------------------------------------------------- sources
@blueprint.route("/api/sources", methods=["GET"])
def list_sources():
    return jsonify({"sources": sources.list_sources(), "allow_urls": ALLOW_URLS})


@blueprint.route("/api/sources/url", methods=["POST"])
def open_url_source():
    if not ALLOW_URLS:
        return _err("urls_disabled", 403)
    data = _body()
    try:
        cs = sources.open_url(data.get("url"), data.get("label"), copy_from=_copy_from(data))
    except sources.SourceError as exc:
        return _err(exc.code, exc.status)
    return jsonify({"ok": True, "camera_id": cs.camera_id, "camera": _camera_out(cs),
                    "playback": cs.source.playback_payload()})


@blueprint.route("/api/sources/<camera_id>/close", methods=["POST"])
def close_source(camera_id: str):
    if not sources.close(camera_id):
        return _err("unknown_source", 404)
    return jsonify({"ok": True, "camera_id": camera_id})


def _source_or_none(camera_id: str):
    src = sources.get_source(camera_id)
    return src


@blueprint.route("/api/sources/<camera_id>/playback", methods=["GET"])
def playback_get(camera_id: str):
    src = _source_or_none(camera_id)
    if src is None:
        return _err("unknown_source", 404)
    resp = jsonify(src.playback_payload())
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _number(value, lo: float | None = None, hi: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if not math.isfinite(v):
        return None
    if lo is not None and v < lo:
        return None
    if hi is not None and v > hi:
        return None
    return v


@blueprint.route("/api/sources/<camera_id>/playback", methods=["POST"])
def playback_post(camera_id: str):
    """Playback control. Body: {"action": "play"|"pause"|"toggle"|"seek"|"step"|"restart",
    "seconds"|"fraction"|"delta" (seek), "frames" (step), "speed", "loop"} - fields combine."""
    src = _source_or_none(camera_id)
    if src is None:
        return _err("unknown_source", 404)
    pb = src.playback
    if pb is None:
        return _err("not_seekable", 409)
    data = _body()
    action = data.get("action")
    if action is not None and action not in ("play", "pause", "toggle", "seek", "step", "restart"):
        return _err("invalid_action")

    if "speed" in data:
        speed = _number(data["speed"], sources.SPEED_MIN, sources.SPEED_MAX)
        if speed is None:
            return _err("invalid_speed")
        pb.set_speed(speed)
    if "loop" in data:
        if not isinstance(data["loop"], bool):
            return _err("invalid_loop")
        pb.set_loop(data["loop"])

    if action == "play":
        pb.play()
    elif action == "pause":
        pb.pause()
    elif action == "toggle":
        pb.toggle()
    elif action == "restart":
        pb.restart()
    elif action == "seek":
        if "fraction" in data:
            v = _number(data["fraction"], 0.0, 1.0)
            if v is None:
                return _err("invalid_seek")
            pb.seek_fraction(v)
        elif "seconds" in data:
            v = _number(data["seconds"], 0.0)
            if v is None:
                return _err("invalid_seek")
            pb.seek_seconds(v)
        elif "delta" in data:
            v = _number(data["delta"], -86400.0, 86400.0)
            if v is None:
                return _err("invalid_seek")
            pb.seek_seconds(max(0.0, pb.frame / pb.fps + v))
        else:
            return _err("invalid_seek")
    elif action == "step":
        v = _number(data.get("frames", 1), -100, 100)
        if v is None or int(v) == 0:
            return _err("invalid_step")
        pb.step(int(v))
    return jsonify({"ok": True, "playback": pb.payload()})
