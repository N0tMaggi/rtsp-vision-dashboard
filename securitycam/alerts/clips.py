"""Per-camera ring buffer of rendered (overlay) JPEG frames and the clip encoder.

``record_frame`` is called once per rendered frame from ``streaming.render_jpeg``.
It only stores a reference to the already encoded JPEG bytes (no copy, no
re-encode), sampled at ``CLIP_FPS`` and bounded by age and total bytes, so the
cost on the render path is a few dict/deque operations. Frames are only kept
while the camera has something that can raise an alarm (``wants_buffer``).

Clip format is probed once: WebM/VP8, then MP4/H.264 (both play in Chrome);
if this OpenCV build has neither encoder, the clip is stored as a JPEG sequence.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import conf

# Set by the engine: camera_id -> bool (does this camera need a pre-roll buffer?)
wants_buffer: Callable[[str], bool] = lambda camera_id: False


class FrameRing:
    def __init__(self, max_bytes: int, keep_s: float, fps: float):
        self.max_bytes = max_bytes
        self.keep_s = keep_s
        self.min_dt = 0.85 / max(0.1, fps)
        self.frames: deque[tuple[float, bytes]] = deque()
        self.bytes = 0
        self.last: Optional[tuple[float, bytes]] = None
        self.lock = threading.Lock()

    def add(self, t: float, jpeg: bytes, keep: bool) -> None:
        with self.lock:
            self.last = (t, jpeg)
            frames = self.frames
            if not keep:
                if frames:
                    frames.clear()
                    self.bytes = 0
                return
            if frames and t - frames[-1][0] < self.min_dt:
                return
            frames.append((t, jpeg))
            self.bytes += len(jpeg)
            cutoff = t - self.keep_s
            while frames and (frames[0][0] < cutoff or self.bytes > self.max_bytes):
                self.bytes -= len(frames.popleft()[1])

    def between(self, t0: float, t1: float) -> list[tuple[float, bytes]]:
        with self.lock:
            return [f for f in self.frames if t0 <= f[0] <= t1]

    def first_after(self, t: float) -> Optional[tuple[float, bytes]]:
        with self.lock:
            for f in self.frames:
                if f[0] >= t:
                    return f
            if self.last is not None and self.last[0] >= t:
                return self.last
        return None

    def latest(self) -> Optional[tuple[float, bytes]]:
        with self.lock:
            return self.last


_rings: dict[str, FrameRing] = {}
_rings_lock = threading.Lock()


def ring(camera_id: str) -> FrameRing:
    r = _rings.get(camera_id)
    if r is None:
        with _rings_lock:
            r = _rings.get(camera_id)
            if r is None:
                r = FrameRing(conf.CLIP_BUFFER_MB * 1024 * 1024, conf.CLIP_PRE_S + conf.CLIP_POST_S + 3.0,
                              conf.CLIP_FPS)
                _rings[camera_id] = r
    return r


def record_frame(camera_state, jpeg: bytes) -> None:
    """Render-path hook: remember the rendered JPEG (never raises)."""
    try:
        cid = camera_state.camera_id
        ring(cid).add(time.time(), jpeg, wants_buffer(cid))
    except Exception:  # pragma: no cover - must never break streaming
        pass


# ----------------------------------------------------------------------------- encoding
_CANDIDATES = {
    "webm": ("VP80", ".webm", "video/webm"),
    "mp4": ("avc1", ".mp4", "video/mp4"),
}
_probe_lock = threading.Lock()
_probed: Optional[list] = None


def _probe_one(fourcc: str, ext: str) -> bool:
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 5.0, (64, 48))
        if not writer.isOpened():
            return False
        for i in range(3):
            writer.write(np.full((48, 64, 3), 40 * i, np.uint8))
        writer.release()
        return os.path.getsize(path) > 0
    except Exception:
        return False
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def video_formats() -> list:
    """Usable (fourcc, ext, mime) in preference order (probed once, may be empty)."""
    global _probed
    with _probe_lock:
        if _probed is None:
            order = {"auto": ("webm", "mp4"), "webm": ("webm",), "mp4": ("mp4",), "jpeg": ()}.get(
                conf.CLIP_FORMAT, ("webm", "mp4"))
            _probed = [_CANDIDATES[k] for k in order if _probe_one(*_CANDIDATES[k][:2])]
            names = ", ".join(f[0] for f in _probed) or "none (JPEG sequence)"
            print(f"[Alerts] Clip encoders: {names}")
        return list(_probed)


def _fit(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    k = max_w / float(w)
    return cv2.resize(img, (max_w, max(2, int(round(h * k))) // 2 * 2), interpolation=cv2.INTER_AREA)


def write_clip(frames: list[tuple[float, bytes]], out_dir: Path, formats: Optional[list] = None) -> dict:
    """Write ``frames`` (time, jpeg) into ``out_dir``; returns clip metadata for meta.json."""
    if not frames:
        return {"status": "failed", "error": "no_frames"}
    duration = max(0.0, frames[-1][0] - frames[0][0])
    fps = (len(frames) - 1) / duration if duration > 0.2 and len(frames) > 1 else conf.CLIP_FPS
    fps = float(max(1.0, min(30.0, fps)))
    formats = video_formats() if formats is None else formats

    for fourcc, ext, mime in formats:
        path = out_dir / f"clip{ext}"
        writer = None
        try:
            size = None
            count = 0
            for _, jpeg in frames:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = _fit(img, conf.CLIP_MAX_WIDTH)
                if size is None:
                    size = (img.shape[1] // 2 * 2, img.shape[0] // 2 * 2)
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
                    if not writer.isOpened():
                        raise RuntimeError("writer_not_opened")
                if (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                writer.write(img)
                count += 1
            if writer is not None:
                writer.release()
                writer = None
            if count and path.is_file() and path.stat().st_size > 0:
                return {"status": "ready", "type": "video", "file": path.name, "mime": mime,
                        "frames": count, "fps": round(fps, 2), "duration": round(duration, 2)}
        except Exception as exc:
            print(f"[Alerts] {fourcc} clip failed ({exc.__class__.__name__}), trying next format.")
        finally:
            if writer is not None:
                writer.release()
        try:
            path.unlink()
        except OSError:
            pass

    # Fallback: JPEG sequence (frames/00000.jpg ...), played as an image sequence in the UI.
    seq_dir = out_dir / "frames"
    seq_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for _, jpeg in frames:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        fitted = _fit(img, conf.CLIP_MAX_WIDTH)
        if fitted is img:
            data = jpeg
        else:
            ok, buf = cv2.imencode(".jpg", fitted, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            data = buf.tobytes()
        (seq_dir / f"{count:05d}.jpg").write_bytes(data)
        count += 1
    if not count:
        return {"status": "failed", "error": "decode_failed"}
    return {"status": "ready", "type": "jpeg_seq", "frames": count, "fps": round(fps, 2),
            "duration": round(duration, 2)}
