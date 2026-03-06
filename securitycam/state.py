"""Mutable runtime state shared across modules."""
import threading
from . import config
from .config import DEFAULT_ESP_SETTINGS

# torch / YOLO objects get assigned lazily
torch = None
YOLO = None

YOLO_AVAILABLE = False
CUDA_AVAILABLE = False
DEVICE = "cpu"
yolo_model = None
yolo_model_name = config.YOLO_MODEL_NAME
face_cascade = None
mp_hands = None
hand_detector = None

yolo_enabled = False

# ESP configuration (mutable copy)
esp_settings = DEFAULT_ESP_SETTINGS.copy()

# Tracking helpers
track_history = {}
MAX_TRACK_HISTORY = 30
smoothed_boxes = {}  # id -> dict with coords and last_seen frame idx
last_faces = []
last_face_frame = -999
last_hands = []
last_hand_frame = -999

# Threading vars
latest_frame = None
latest_results = None
frame_lock = threading.Lock()

stats = {
    "fps": 0.0,
    "inference_fps": 0.0,
    "last_infer_ms": 0.0,
    "num_detections": 0,
    "faces": 0,
    "classes": [],
    "resolution": "0x0",
    "total_frames": 0,
    "last_update": 0.0,
}

gpu_stats = {
    "name": "CPU",
    "util": 0,
    "mem_used": 0,
    "mem_total": 0,
}
