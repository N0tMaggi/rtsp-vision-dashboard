"""Flask application wiring and HTTP handlers."""
from pathlib import Path

from flask import Flask, Response, render_template, jsonify, request

from . import config, state, deps
from .streaming import gen_frames


def create_app() -> Flask:
    deps.ensure_torch_and_yolo()
    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))

    @app.route("/")
    def index():
        return render_template(
            "dashboard.html",
            camera_ip=config.CAMERA_IP,
            rtsp_path=config.STREAM_PATH,
        )

    @app.route("/video_feed")
    def video_feed():
        return Response(
            gen_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/status")
    def status():
        state.gpu_stats.update(deps.get_gpu_snapshot())

        data = {
            "fps": state.stats["fps"],
            "inference_fps": state.stats["inference_fps"],
            "last_infer_ms": state.stats["last_infer_ms"],
            "num_detections": state.stats["num_detections"],
            "faces": state.stats.get("faces", 0),
            "classes": state.stats["classes"],
            "resolution": state.stats["resolution"],
            "total_frames": state.stats["total_frames"],
            "last_update": state.stats["last_update"],
            "yolo_enabled": state.yolo_enabled,
            "yolo_available": state.YOLO_AVAILABLE,
            "yolo_model": state.yolo_model_name if state.YOLO_AVAILABLE else None,
            "available_models": config.AVAILABLE_MODELS,
            "device": state.DEVICE,
            "cuda_available": state.CUDA_AVAILABLE,
            "esp_settings": state.esp_settings,
            "gpu": state.gpu_stats,
        }
        return jsonify(data)

    @app.route("/toggle_yolo", methods=["POST"])
    def toggle_yolo():
        state.yolo_enabled = not state.yolo_enabled
        return jsonify({"yolo_enabled": state.yolo_enabled})

    @app.route("/update_esp", methods=["POST"])
    def update_esp():
        data = request.get_json(silent=True) or {}
        str_keys = {"box_style"}
        float_keys = {"confidence_threshold", "line_thickness", "face_zoom_scale", "face_blur_strength", "chams_opacity"}

        for k in state.esp_settings.keys():
            if k not in data:
                continue
            if k.startswith("color_"):
                state.esp_settings[k] = str(data[k])
            elif k in str_keys:
                state.esp_settings[k] = str(data[k])
            elif k in float_keys:
                try:
                    state.esp_settings[k] = float(data[k])
                except Exception:
                    pass
            else:
                state.esp_settings[k] = bool(data[k])

        return jsonify({"ok": True, "esp_settings": state.esp_settings})

    @app.route("/set_model", methods=["POST"])
    def set_model():
        data = request.get_json(silent=True) or {}
        model_name = data.get("model_name")

        if model_name not in config.AVAILABLE_MODELS:
            return jsonify({"ok": False, "error": "invalid_model"}), 400

        if model_name == state.yolo_model_name:
            return jsonify({"ok": True, "yolo_model": state.yolo_model_name})

        print(f"[YOLO] Switching model: {state.yolo_model_name} -> {model_name}")
        state.yolo_model_name = model_name
        state.yolo_model = None  # reload on next usage

        return jsonify({"ok": True, "yolo_model": state.yolo_model_name})

    return app


app = create_app()
