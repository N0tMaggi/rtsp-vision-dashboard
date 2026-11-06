"""Flask application wiring and camera-specific HTTP handlers."""
from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, abort, jsonify, request, send_from_directory

from . import config, deps, state, targets
from .streaming import gen_frames
from .validation import ZoneError, parse_lock_id
from .visuals_api import visuals_bp


def _camera_payload(camera_state: state.CameraRuntime) -> dict:
    return {
        "id": camera_state.camera_id,
        "label": camera_state.label,
        "ip": camera_state.ip,
        "stream_path": camera_state.stream_path,
        "has_stream": bool(camera_state.rtsp_url),
    }


def _get_camera_state_or_404(camera_id: str) -> state.CameraRuntime:
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        abort(404)
    return camera_state


def _status_payload(camera_state: state.CameraRuntime) -> dict:
    state.gpu_stats.update(deps.get_gpu_snapshot())
    return {
        "camera": _camera_payload(camera_state),
        "fps": camera_state.stats["fps"],
        "inference_fps": camera_state.stats["inference_fps"],
        "last_infer_ms": camera_state.stats["last_infer_ms"],
        "num_detections": camera_state.stats["num_detections"],
        "faces": camera_state.stats.get("faces", 0),
        "classes": camera_state.stats["classes"],
        "resolution": camera_state.stats["resolution"],
        "total_frames": camera_state.stats["total_frames"],
        "last_update": camera_state.stats["last_update"],
        "yolo_enabled": camera_state.yolo_enabled,
        "yolo_available": state.YOLO_AVAILABLE,
        "yolo_model": camera_state.yolo_model_name if state.YOLO_AVAILABLE else None,
        "available_models": config.AVAILABLE_MODELS,
        "device": state.DEVICE,
        "cuda_available": state.CUDA_AVAILABLE,
        "esp_settings": camera_state.esp_settings,
        "gpu": state.gpu_stats,
        "stream_error": camera_state.stream_error,
        "locked": targets.get_locked(camera_state),
        "zones": targets.zones_payload(camera_state),
        "assets": dict(getattr(deps, "asset_status", {}) or {}),
    }


def create_app() -> Flask:
    deps.ensure_torch_and_yolo()
    package_dir = Path(__file__).resolve().parent
    app = Flask(__name__, static_folder=str(package_dir / "static"))

    frontend_dir = package_dir / "static" / "app"
    from .ops_api import blueprint as ops_blueprint  # persistence, presets, grid overview, PTZ
    app.register_blueprint(ops_blueprint)

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
    return app


app = create_app()
