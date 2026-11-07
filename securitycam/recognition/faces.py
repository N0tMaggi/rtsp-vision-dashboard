"""YuNet face detection + SFace embeddings (OpenCV, ONNX models from opencv_zoo, all local).

Detector/recognizer instances are thread-local: OpenCV DNN nets are not safe to
share between the render threads and the recognition worker.
"""
from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import compute, deps

YUNET_SCORE = 0.7          # 0.6 lets hands through; real faces score 0.85+ (profiles ~0.7)
YUNET_NMS = 0.3
YUNET_TOP_K = 50
MIN_RECOGNIZABLE_PX = 36          # faces narrower than this give unreliable embeddings
MAX_DETECT_SIDE = 1280            # YuNet input is downscaled to at most this

_local = threading.local()
_retry_lock = threading.Lock()


def _has_api() -> bool:
    return hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")


def yunet_ready() -> bool:
    """True when YuNet can be used right now (triggers the background download otherwise)."""
    return _has_api() and deps.asset_path("face_yunet") is not None


def sface_ready() -> bool:
    return _has_api() and deps.asset_path("face_sface") is not None


def _detector() -> Optional["cv2.FaceDetectorYN"]:
    det = getattr(_local, "yunet", None)
    if det is not None:
        return det
    if not _has_api():
        return None
    path = deps.asset_path("face_yunet")
    if path is None:
        return None
    try:
        det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), YUNET_SCORE, YUNET_NMS, YUNET_TOP_K)
    except cv2.error as exc:
        print(f"[RECOG] YuNet could not be loaded: {exc}. Deleting it so it is downloaded again.")
        path.unlink(missing_ok=True)
        deps.asset_status["face_yunet"] = "missing"
        return None
    _local.yunet = det
    return det


def _recognizer() -> Optional["cv2.FaceRecognizerSF"]:
    rec = getattr(_local, "sface", None)
    if rec is not None:
        return rec
    if not _has_api():
        return None
    path = deps.asset_path("face_sface")
    if path is None:
        return None
    try:
        rec = cv2.FaceRecognizerSF.create(str(path), "")
    except cv2.error as exc:
        print(f"[RECOG] SFace could not be loaded: {exc}. Deleting it so it is downloaded again.")
        path.unlink(missing_ok=True)
        deps.asset_status["face_sface"] = "missing"
        return None
    _local.sface = rec
    return rec


def detect(frame: np.ndarray, scale: float = 1.0) -> Optional[np.ndarray]:
    """YuNet faces in ``frame`` pixels as an (N, 15) float32 array, or None if YuNet is unavailable.

    ``scale`` < 1 runs the detector on a downscaled copy (coordinates are mapped back).
    The image is additionally capped at MAX_DETECT_SIDE.
    """
    det = _detector()
    if det is None or frame is None or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    s = min(scale, MAX_DETECT_SIDE / max(h, w)) if max(h, w) * scale > MAX_DETECT_SIDE else scale
    img = frame if abs(s - 1.0) < 1e-3 else cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))),
                                                        interpolation=cv2.INTER_AREA)
    ih, iw = img.shape[:2]
    faces = compute.faces_detect(img, YUNET_SCORE, YUNET_NMS, YUNET_TOP_K)   # None unless on an OpenVINO device
    if faces is None:
        try:
            det.setInputSize((iw, ih))
            _, faces = det.detect(img)
        except cv2.error:
            return np.zeros((0, 15), np.float32)
    if faces is None or len(faces) == 0:
        return np.zeros((0, 15), np.float32)
    faces = np.asarray(faces, np.float32).reshape(-1, 15)
    if abs(s - 1.0) >= 1e-3:
        faces[:, :14] /= s
    return faces


def to_rects(faces: np.ndarray, frame_w: int, frame_h: int) -> List[Tuple[int, int, int, int]]:
    """YuNet rows -> clipped (x, y, w, h) integer rects."""
    out = []
    for row in faces:
        x, y, fw, fh = (float(v) for v in row[:4])
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(frame_w, int(x + fw)), min(frame_h, int(y + fh))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def embed(frame: np.ndarray, face_row: np.ndarray) -> Optional[np.ndarray]:
    """L2-normalized 128-d SFace feature for one YuNet face row, or None."""
    rec = _recognizer()
    if rec is None:
        return None
    try:
        aligned = rec.alignCrop(frame, face_row.reshape(1, -1) if face_row.ndim == 1 else face_row)
        feat = compute.face_feature(aligned)       # None unless on an OpenVINO device
        if feat is None:
            feat = rec.feature(aligned)
    except cv2.error:
        return None
    feat = np.asarray(feat, np.float32).reshape(-1)
    n = float(np.linalg.norm(feat))
    if n < 1e-9:
        return None
    return feat / n


def aligned_thumb(frame: np.ndarray, face_row: np.ndarray, size: int = 112) -> Optional[np.ndarray]:
    """Square crop around the face (slightly padded), for gallery thumbnails."""
    x, y, fw, fh = (float(v) for v in face_row[:4])
    h, w = frame.shape[:2]
    side = max(fw, fh) * 1.35
    cx, cy = x + fw / 2, y + fh / 2
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def largest(faces: np.ndarray) -> Optional[np.ndarray]:
    if faces is None or len(faces) == 0:
        return None
    return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
