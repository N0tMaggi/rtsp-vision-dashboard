"""Flask application wiring and camera-specific HTTP handlers."""
from __future__ import annotations

from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request

from . import config, deps, state
from .streaming import gen_frames


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
    }


def create_app() -> Flask:
    deps.ensure_torch_and_yolo()
    package_dir = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        template_folder=str(package_dir / "templates"),
        static_folder=str(package_dir / "static"),
    )

    @app.route("/")
    def index():
        bootstrap = {
            "cameras": [_camera_payload(camera_state) for camera_state in state.list_camera_states()],
            "default_camera_id": config.DEFAULT_CAMERA_ID,
            "available_models": list(config.AVAILABLE_MODELS),
        }
        return render_template("dashboard.html", bootstrap=bootstrap)

    @app.route("/api/cameras")
    def cameras():
        return jsonify({
            "cameras": [_camera_payload(camera_state) for camera_state in state.list_camera_states()],
            "default_camera_id": config.DEFAULT_CAMERA_ID,
        })

    @app.route("/video_feed")
    def video_feed_default():
        return video_feed(config.DEFAULT_CAMERA_ID)

    @app.route("/video_feed/<camera_id>")
    def video_feed(camera_id: str):
        _get_camera_state_or_404(camera_id)
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
        camera_state = _get_camera_state_or_404(camera_id)
        data = request.get_json(silent=True) or {}
        str_keys = {"box_style"}
        float_keys = {"confidence_threshold", "line_thickness", "face_zoom_scale", "face_blur_strength", "chams_opacity"}

        for key in camera_state.esp_settings.keys():
            if key not in data:
                continue
            if key.startswith("color_") or key == "chams_color":
                camera_state.esp_settings[key] = str(data[key])
            elif key in str_keys:
                camera_state.esp_settings[key] = str(data[key])
            elif key in float_keys:
                try:
                    camera_state.esp_settings[key] = float(data[key])
                except Exception:
                    pass
            else:
                camera_state.esp_settings[key] = bool(data[key])

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

    return app


app = create_app()
