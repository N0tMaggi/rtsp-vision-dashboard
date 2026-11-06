"""Throttled auxiliary detectors (Haar faces, MediaPipe hands).

Both functions must be called with the camera's ESP lock held: they mutate
``last_faces`` / ``last_hands`` and the MediaPipe detector is not thread-safe.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .. import config, deps, recognition

FaceBox = Tuple[int, int, int, int]
HandPoints = List[Tuple[int, int]]


def clip_rect(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> Optional[FaceBox]:
    """Clamp an (x, y, w, h) rect to the frame; ``None`` if nothing is left."""
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(frame_w, int(x + w)), min(frame_h, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def _due(last_frame: int, frame_idx: int, every: int) -> bool:
    return frame_idx - last_frame >= every or frame_idx < last_frame


def update_faces(camera_state, frame: np.ndarray, settings, frame_idx: int,
                 gray: Optional[np.ndarray] = None) -> List[FaceBox]:
    """Return current face boxes, re-running the cascade every FACE_DETECT_EVERY frames."""
    if not settings.needs_faces:
        camera_state.last_faces.clear()
        camera_state.stats["faces"] = 0
        return []

    if _due(getattr(camera_state, "last_face_frame", -999), frame_idx, config.FACE_DETECT_EVERY):
        # recognition: YuNet (better recall, side faces) when available, Haar otherwise
        yunet = recognition.detect_face_rects(camera_state, frame, config.FACE_DETECT_SCALE)
        cascade = deps.ensure_face_cascade() if yunet is None else None
        if yunet is not None:
            camera_state.last_faces[:] = yunet
        elif cascade is not None:
            h, w = frame.shape[:2]
            scale = config.FACE_DETECT_SCALE
            try:
                if gray is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                raw = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                faces: List[FaceBox] = []
                for (fx, fy, fw, fh) in raw:
                    rect = clip_rect(fx / scale, fy / scale, fw / scale, fh / scale, w, h)
                    if rect is not None:
                        faces.append(rect)
                camera_state.last_faces[:] = faces
            except cv2.error:
                pass  # keep the previous faces
        camera_state.last_face_frame = frame_idx

    faces = list(camera_state.last_faces)
    camera_state.stats["faces"] = len(faces)
    return faces


def update_hands(camera_state, frame: np.ndarray, settings, frame_idx: int) -> List[HandPoints]:
    """Return current 21-point hand landmarks, re-running MediaPipe every HAND_DETECT_EVERY frames."""
    if not settings.hand_skeleton:
        camera_state.last_hands.clear()
        return []

    if _due(getattr(camera_state, "last_hand_frame", -999), frame_idx, config.HAND_DETECT_EVERY):
        detector = deps.ensure_hand_detector(camera_state)
        if detector is not None:
            h, w = frame.shape[:2]
            scale = config.HAND_DETECT_SCALE
            try:
                small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                hand_results = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                hands: List[HandPoints] = []
                for hlm in getattr(hand_results, "multi_hand_landmarks", None) or []:
                    hands.append([(int(lm.x * w), int(lm.y * h)) for lm in hlm.landmark])
                camera_state.last_hands[:] = hands
            except Exception:
                pass  # MediaPipe raises assorted errors on odd frames; keep previous hands
        camera_state.last_hand_frame = frame_idx

    return [list(p) for p in camera_state.last_hands]
