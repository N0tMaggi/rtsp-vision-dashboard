from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or "=" not in value:
            continue
        key, raw = value.split("=", 1)
        key = key.strip()
        raw = raw.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = raw


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    value = os.getenv(name)
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    value = os.getenv(name)
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


_load_dotenv(PROJECT_ROOT / ".env")

CAMERA_IP = os.getenv("SECURITYCAM_CAMERA_IP", "").strip()
USERNAME = os.getenv("SECURITYCAM_CAMERA_USERNAME", "").strip()
PASSWORD = os.getenv("SECURITYCAM_CAMERA_PASSWORD", "").strip()
STREAM_PATH = os.getenv("SECURITYCAM_CAMERA_STREAM_PATH", "stream1").strip() or "stream1"

APP_HOST = os.getenv("SECURITYCAM_APP_HOST", "0.0.0.0").strip() or "0.0.0.0"
APP_PORT = _env_int("SECURITYCAM_APP_PORT", 5000, minimum=1, maximum=65535)
APP_DEBUG = _env_bool("SECURITYCAM_APP_DEBUG", False)
SECRET_KEY = os.getenv("SECURITYCAM_SECRET_KEY", "").strip()
SESSION_COOKIE_SECURE = _env_bool("SECURITYCAM_SESSION_COOKIE_SECURE", False)
SESSION_COOKIE_SAMESITE = os.getenv("SECURITYCAM_SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
TRUSTED_HOSTS = tuple(
    host.strip()
    for host in os.getenv("SECURITYCAM_TRUSTED_HOSTS", "").split(",")
    if host.strip()
)

DISPLAY_SCALE = _env_float("SECURITYCAM_DISPLAY_SCALE", 1.0, minimum=0.2, maximum=1.0)
JPEG_QUALITY = _env_int("SECURITYCAM_JPEG_QUALITY", 85, minimum=30, maximum=95)
FACE_DETECT_EVERY = _env_int("SECURITYCAM_FACE_DETECT_EVERY", 5, minimum=1)
FACE_DETECT_SCALE = _env_float("SECURITYCAM_FACE_DETECT_SCALE", 0.5, minimum=0.2, maximum=1.0)
# Default for the per-camera "smoothing" ESP setting (0 = raw boxes, 1 = smoothest).
SMOOTH_ALPHA = _env_float("SECURITYCAM_SMOOTH_ALPHA", 0.6, minimum=0.0, maximum=1.0)
# Optional TrueType font for ESP labels (monospace recommended); system monospace fonts otherwise.
ESP_FONT = os.getenv("SECURITYCAM_ESP_FONT", "").strip()
HAND_DETECT_EVERY = _env_int("SECURITYCAM_HAND_DETECT_EVERY", 5, minimum=1)
HAND_DETECT_SCALE = _env_float("SECURITYCAM_HAND_DETECT_SCALE", 0.5, minimum=0.2, maximum=1.0)

AUTO_INSTALL_DEPS = _env_bool("SECURITYCAM_AUTO_INSTALL_DEPS", False)
TORCH_CUDA_INDEX_URL = os.getenv(
    "SECURITYCAM_TORCH_CUDA_INDEX_URL",
    "https://download.pytorch.org/whl/nightly/cu128",
).strip() or "https://download.pytorch.org/whl/nightly/cu128"

YOLO_IMG_SIZE = _env_int("SECURITYCAM_YOLO_IMG_SIZE", 640, minimum=320, maximum=1280)

AVAILABLE_MODELS = (
    # YOLO26: newest Ultralytics generation, NMS-free end-to-end, better on small/far objects.
    "yolo26n.pt",
    "yolo26n-pose.pt",
    "yolo26s.pt",
    "yolo26s-pose.pt",
    "yolo26m.pt",
    "yolo26m-pose.pt",
    "yolo26l.pt",
    "yolo26l-pose.pt",
    "yolo26x.pt",
    "yolo26x-pose.pt",
    "yolov8n.pt",
    "yolov8n-pose.pt",
    "yolov8s.pt",
    "yolov8s-pose.pt",
    "yolov8m.pt",
    "yolov8m-pose.pt",
    "yolov8l.pt",
    "yolov8l-pose.pt",
    "yolov8x.pt",
    "yolov8x-pose.pt",
    # --- recognition package: newer families (grouped in the UI by name prefix) ---
    "yolo11n.pt",
    "yolo11n-pose.pt",
    "yolo11s.pt",
    "yolo11s-pose.pt",
    "yolo11m.pt",
    "yolo11m-pose.pt",
    "rtdetr-l.pt",
    # --- instance segmentation: body outlines for Silhouette chams (securitycam/esp/silhouette.py) ---
    "yolo26n-seg.pt",
    "yolo26s-seg.pt",
    "yolo26m-seg.pt",
    "yolo26l-seg.pt",
    "yolo26x-seg.pt",
    "yolo11n-seg.pt",
    "yolo11s-seg.pt",
    "yolo11m-seg.pt",
)

YOLO_MODEL_NAME = os.getenv("SECURITYCAM_YOLO_MODEL", "yolo26s.pt").strip() or "yolo26s.pt"
if YOLO_MODEL_NAME not in AVAILABLE_MODELS:
    YOLO_MODEL_NAME = "yolo26s.pt"


DEFAULT_ESP_SETTINGS = {
    "boxes": True,
    "box_style": "corner",
    "line_thickness": 2,
    "fill_box": False,
    "chams": False,
    "chams_opacity": 0.35,
    "chams_color": "#ffd21f",
    # --- silhouette chams (securitycam/esp/silhouette.py): body masks from a Segment model ---
    "chams_mode": "box",               # box | silhouette | outline | silhouette_outline | xray
    "outline_glow": 8,                 # glow width around silhouette outlines, px at 720p (0 = off)
    "class_colors": False,
    "person_only": False,
    "highlight_center": False,
    "thermal": False,
    "names": True,
    "conf": True,
    "skeleton": True,
    "head": False,
    "snaplines": False,
    "center_dot": False,
    "tracking": False,
    "distance": False,
    "tracers": False,
    "prediction": False,
    "velocity": False,
    "rgb_mode": False,
    "face_boxes": False,
    "face_zoom": False,
    "face_blur": False,
    "face_blur_strength": 70,
    "face_blur_mode": "blur",
    "face_blur_image": "",          # id of a custom cover image (face_masks.py), "image" mode
    "face_blur_image_fit": "cover",
    "face_blur_image_underlay": True,   # blur under the image so transparent parts never show a face
    "face_zoom_scale": 2.0,
    "hand_skeleton": False,
    "confidence_threshold": 0.35,
    # render-time motion smoothing: 0 = boxes jump at inference rate, 1 = smoothest
    "smoothing": round(SMOOTH_ALPHA, 2),
    "label_background": True,
    "zones": True,
    "glow": True,
    "jpeg_quality": JPEG_QUALITY,
    # alarms & events (securitycam/alerts): rule evaluation, tripwire overlay, abandoned objects
    "alarms": True,
    "tripwires": True,
    "abandoned": False,
    "abandoned_seconds": 30,
    # PTZ cameras: zones, tripwires, heatmap and markers stay fixed in the room while the
    # camera pans / tilts / zooms (securitycam/anchor.py, securitycam/ptz/pose.py)
    "world_anchor": True,
    # theme: anthracite #222427 / signal yellow #ffd21f / white #f4f4f2
    "color_primary": "#ffd21f",
    "color_secondary": "#f4f4f2",
    "color_boxes": "#ffd21f",
    "color_skeleton": "#f4f4f2",
    "color_head": "#ff6b57",
    "color_tracers": "#ffe58a",
    "color_velocity": "#5ad1ff",
    "color_prediction": "#9be7ff",
    "color_face": "#5ad1ff",
    "color_hand": "#7ee0a1",
    "color_text": "#f4f4f2",
    # visuals: motion heatmap, loitering ramp, night vision
    "heatmap": False,
    "heatmap_window": "1h",
    "heatmap_opacity": 0.55,
    "loiter": False,
    "loiter_seconds": 120,
    "night_vision": "off",
    "night_vision_scanlines": False,
    # --- depth (securitycam/depth): monocular depth + room map, all off by default ---
    "depth": False,                    # run Depth Anything V2 Small while the camera is viewed
    "depth_view": False,               # blend the colorized depth map into the video
    "depth_view_opacity": 0.5,
    "depth_view_palette": "turbo",     # turbo | inferno
    "depth_rate": 3.0,                 # depth maps per second (0.5 .. 10)
    "depth_size": 364,                 # model input long side in px (multiple of 14, 196 .. 518)
    "depth_fov": 90,                   # horizontal FOV of a fixed camera in degrees (PTZ: lens calibration)
    "depth_scale": 0.0,                # user calibration (z = scale / disparity); 0 = auto from person heights
    # --- end depth ---
    # --- recognition (securitycam/recognition) ---
    "reid": True,                      # BoT-SORT ReID + re-entry id memory (needs tracking)
    "known_faces": False,              # label faces from the local gallery (data/faces)
    "face_match_threshold": 0.36,      # SFace cosine similarity needed for a name (OpenCV recommends 0.363)
    "face_detector": "yunet",          # yunet (falls back to haar while missing) | haar
    "face_auto_ids": True,             # with known_faces: number unknown faces as persistent "Face #n" identities
    "face_cluster_threshold": 0.45,    # SFace cosine needed to count as the same unknown face (stricter than a name match)
    "plates": False,                   # read license plates on vehicles
    "pets_show": True,
    "pets_ignore_alarms": True,        # read by the alarm package via recognition.is_pet()
    "color_pet": "#ff9f43",
    "color_known": "#7ee0a1",
    "color_unknown": "#ff5a5a",
    # --- pose ESP pack (securitycam/esp/posepack.py); every visual is off by default ---
    "iff_colors": False,               # friend (known) / foe (unknown face) / unidentified person colors
    "iff_color_friend": "#7ee0a1",
    "iff_color_foe": "#ff5a5a",
    "iff_color_unknown": "#ffd21f",
    "gaze": False,                     # head direction cone + "looking at camera"
    "posture": False,                  # standing / sitting / lying / ... + action labels
    "fall_alarm": True,                # with posture: upright -> lying fast raises a "fall" alarm event
    "status_bars": False,              # dwell / confidence / speed bars beside each box
    "bar_dwell": True,
    "bar_conf": True,
    "bar_speed": True,
    "gestures": False,                 # wave = PTZ auto-track, open palm = stop, thumbs up = snapshot
    "gesture_wave": True,
    "gesture_palm": True,
    "gesture_thumbs": True,
}


def model_path(model_name: str) -> Path:
    return MODELS_DIR / model_name


def build_rtsp_url(ip: str, username: str, password: str, stream_path: str) -> str:
    if not ip or not username or not password:
        return ""
    encoded_user = quote(username, safe="")
    encoded_pass = quote(password, safe="")
    return f"rtsp://{encoded_user}:{encoded_pass}@{ip}:554/{stream_path}"


@dataclass(frozen=True)
class CameraConfig:
    id: str
    label: str
    ip: str
    username: str
    password: str
    stream_path: str
    rtsp_url: str


def _slugify_camera_id(raw: str, fallback: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", raw.strip().lower()).strip("-")
    return value or fallback


def _camera_config(camera_id: str, label: str, ip: str, username: str, password: str, stream_path: str) -> CameraConfig:
    return CameraConfig(
        id=camera_id,
        label=label,
        ip=ip,
        username=username,
        password=password,
        stream_path=stream_path,
        rtsp_url=build_rtsp_url(ip, username, password, stream_path),
    )


def _parse_multi_camera_env(raw: str) -> tuple[CameraConfig, ...]:
    cameras: list[CameraConfig] = []
    seen_ids: set[str] = set()

    for index, chunk in enumerate(raw.split(";"), start=1):
        entry = chunk.strip()
        if not entry:
            continue

        parts = [part.strip() for part in entry.split("|")]
        if len(parts) < 5:
            continue

        camera_id = _slugify_camera_id(parts[0], f"camera-{index}")
        label = parts[1] or f"Camera {index}"
        ip = parts[2]
        username = parts[3]
        password = parts[4]
        stream_path = parts[5] if len(parts) > 5 and parts[5] else STREAM_PATH

        if camera_id in seen_ids:
            continue

        cameras.append(_camera_config(camera_id, label, ip, username, password, stream_path))
        seen_ids.add(camera_id)

    return tuple(cameras)


def _legacy_camera_config() -> CameraConfig:
    camera_id = "camera-1"
    label = CAMERA_IP or "Camera 1"
    return _camera_config(camera_id, label, CAMERA_IP, USERNAME, PASSWORD, STREAM_PATH)


CAMERAS = _parse_multi_camera_env(os.getenv("SECURITYCAM_CAMERAS", "")) or (_legacy_camera_config(),)
DEFAULT_CAMERA_ID = CAMERAS[0].id


def get_camera(camera_id: str) -> CameraConfig | None:
    for camera in CAMERAS:
        if camera.id == camera_id:
            return camera
    return None


RTSP_URL = CAMERAS[0].rtsp_url
