"""Flask application wiring and camera-specific HTTP handlers."""
from __future__ import annotations

from pathlib import Path

from urllib.parse import urlsplit

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import config, deps, state, targets
from .live import blueprint as live_blueprint
from .recognition.api import blueprint as recognition_api
from .status import camera_payload as _camera_payload, status_payload as _status_payload
from .streaming import gen_frames
from .validation import ZoneError, parse_lock_id
from .visuals_api import visuals_bp


def _get_camera_state_or_404(camera_id: str) -> state.CameraRuntime:
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        abort(404)
    return camera_state


def create_app() -> Flask:
    deps.ensure_torch_and_yolo()
    package_dir = Path(__file__).resolve().parent
    app = Flask(__name__, static_folder=str(package_dir / "static"))

    @app.before_request
    def _reject_cross_site_writes():
        # CSRF guard for every mutating route: a page on another site must not be
        # able to change settings, zones, presets, PTZ etc. through the user's
        # browser. Browsers always send Origin on cross-site POST/PUT/PATCH/DELETE;
        # non-browser clients (curl, scripts) send neither header and are allowed.
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None
        # Modern browsers label the request's relation to the page directly.
        if request.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return jsonify({"ok": False, "error": "cross_origin_request"}), 403
        host = (request.host or "").lower()
        origin = request.headers.get("Origin")
        if origin is not None:
            # "null" = sandboxed iframe / file:// page -> never trusted
            if origin == "null" or urlsplit(origin).netloc.lower() != host:
                return jsonify({"ok": False, "error": "cross_origin_request"}), 403
            return None
        referer = request.headers.get("Referer")
        if referer and urlsplit(referer).netloc.lower() != host:
            return jsonify({"ok": False, "error": "cross_origin_request"}), 403
        return None

    frontend_dir = package_dir / "static" / "app"
    from .ops_api import blueprint as ops_blueprint  # persistence, presets, grid overview, PTZ
    app.register_blueprint(ops_blueprint)
    app.register_blueprint(recognition_api)
    app.register_blueprint(live_blueprint)   # SSE push channel: /api/stream

    @app.route("/")
    def index():
        # The dashboard is a prebuilt React/TypeScript app (see ../frontend).
        if not (frontend_dir / "index.html").is_file():
            return Response(
                "Frontend build missing. Run: cd frontend && npm install && npm run build",
                status=503,
                mimetype="text/plain",
            )
        response = send_from_directory(frontend_dir, "index.html")
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.route("/api/cameras")
    def cameras():
        return jsonify({
            "cameras": [_camera_payload(camera_state) for camera_state in state.list_camera_states()],
            "default_camera_id": config.DEFAULT_CAMERA_ID,
            "available_models": list(config.AVAILABLE_MODELS),
        })

    @app.route("/video_feed")
    def video_feed_default():
        return video_feed(config.DEFAULT_CAMERA_ID)

    @app.route("/video_feed/<camera_id>")
    def video_feed(camera_id: str):
        _get_camera_state_or_404(camera_id)
        width = request.args.get("w", type=int)
        if width:  # downscaled grid tile, rendered once per size and shared
            from .scaled_feed import gen_frames_scaled
            return Response(gen_frames_scaled(camera_id, width), mimetype="multipart/x-mixed-replace; boundary=frame")
        return Response(
            gen_frames(camera_id),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/status")
    def status_default():
        return status(config.DEFAULT_CAMERA_ID)

    @app.route("/api/cameras/<camera_id>/status")
    def status(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        return jsonify(_status_payload(camera_state))

    @app.route("/toggle_yolo", methods=["POST"])
    def toggle_yolo_default():
        return toggle_yolo(config.DEFAULT_CAMERA_ID)

    @app.route("/api/cameras/<camera_id>/toggle_yolo", methods=["POST"])
    def toggle_yolo(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        camera_state.yolo_enabled = not camera_state.yolo_enabled
        return jsonify({"yolo_enabled": camera_state.yolo_enabled})

    @app.route("/update_esp", methods=["POST"])
    def update_esp_default():
        return update_esp(config.DEFAULT_CAMERA_ID)

    @app.route("/api/cameras/<camera_id>/update_esp", methods=["POST"])
    def update_esp(camera_id: str):
        from .validation import sanitize_esp_patch  # local import keeps this edit route-scoped

        camera_state = _get_camera_state_or_404(camera_id)
        data = request.get_json(silent=True)
        # Accepts every DEFAULT_ESP_SETTINGS key (plus legacy "gay_mode" -> "rgb_mode");
        # unknown keys and invalid values are dropped, numbers are clamped.
        camera_state.esp_settings.update(sanitize_esp_patch(data))
        return jsonify({"ok": True, "esp_settings": camera_state.esp_settings})

    @app.route("/set_model", methods=["POST"])
    def set_model_default():
        return set_model(config.DEFAULT_CAMERA_ID)

    @app.route("/api/cameras/<camera_id>/set_model", methods=["POST"])
    def set_model(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        data = request.get_json(silent=True) or {}
        model_name = data.get("model_name")

        if model_name not in config.AVAILABLE_MODELS:
            return jsonify({"ok": False, "error": "invalid_model"}), 400

        if model_name == camera_state.yolo_model_name:
            return jsonify({"ok": True, "yolo_model": camera_state.yolo_model_name})

        print(f"[YOLO] Switching model for {camera_id}: {camera_state.yolo_model_name} -> {model_name}")
        camera_state.yolo_model_name = model_name
        camera_state.yolo_model = None

        return jsonify({"ok": True, "yolo_model": camera_state.yolo_model_name})

    # ------------------------------------------------------------------ targets / lock / zones
    @app.route("/api/cameras/<camera_id>/targets")
    def camera_targets(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        response = jsonify(targets.targets_payload(camera_state))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/cameras/<camera_id>/lock", methods=["POST"])
    def camera_lock(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        try:
            target_id = parse_lock_id(request.get_json(silent=True))
        except ValueError:
            return jsonify({"ok": False, "error": "invalid_lock"}), 400
        if not targets.set_lock(camera_state, target_id):
            return jsonify({"ok": False, "error": "unknown_target"}), 404
        return jsonify({"ok": True, "locked": targets.get_locked(camera_state)})

    @app.route("/api/cameras/<camera_id>/zones", methods=["GET"])
    def camera_zones(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        return jsonify({"zones": targets.zones_payload(camera_state)})

    @app.route("/api/cameras/<camera_id>/zones", methods=["POST"])
    def camera_zone_create(camera_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        try:
            zone = targets.create_zone(camera_state, request.get_json(silent=True))
        except ZoneError as exc:
            return jsonify({"ok": False, "error": exc.code}), 400
        return jsonify({"ok": True, "zone": zone})

    @app.route("/api/cameras/<camera_id>/zones/<zone_id>", methods=["PATCH"])
    def camera_zone_update(camera_id: str, zone_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        try:
            zone = targets.update_zone(camera_state, zone_id, request.get_json(silent=True))
        except ZoneError as exc:
            return jsonify({"ok": False, "error": exc.code}), 400
        if zone is None:
            return jsonify({"ok": False, "error": "unknown_zone"}), 404
        return jsonify({"ok": True, "zone": zone})

    @app.route("/api/cameras/<camera_id>/zones/<zone_id>", methods=["DELETE"])
    def camera_zone_delete(camera_id: str, zone_id: str):
        camera_state = _get_camera_state_or_404(camera_id)
        if not targets.delete_zone(camera_state, zone_id):
            return jsonify({"ok": False, "error": "unknown_zone"}), 404
        return jsonify({"ok": True})

    app.register_blueprint(visuals_bp)
    from .alerts.routes import blueprint as alerts_blueprint  # alarms, tripwires, events, snapshots
    app.register_blueprint(alerts_blueprint)
    from .videos.api import blueprint as videos_blueprint  # video analysis: library, file/URL sources, playback
    app.register_blueprint(videos_blueprint)
    from .compute.api import blueprint as compute_blueprint  # compute devices: which hardware runs which workload
    app.register_blueprint(compute_blueprint)
    from .enhance.api import blueprint as enhance_blueprint  # Enhance view: AI upscaling of a frame region
    app.register_blueprint(enhance_blueprint)
    from .context_api import blueprint as context_blueprint  # right-click menu: tags, solo/hide, manual plates
    app.register_blueprint(context_blueprint)
    from .face_masks import blueprint as face_masks_blueprint  # custom face-cover images
    app.register_blueprint(face_masks_blueprint)
    return app


app = create_app()
