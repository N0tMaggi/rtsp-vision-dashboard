"""Downscaled MJPEG streams for the multi-camera grid (``/video_feed/<id>?w=``).

Grid tiles do not need N full-resolution streams. A scaled client shares the
camera's ``CameraFeed`` (so the overlay is still rendered once per source
frame for everybody), and the downscaled JPEG is produced once per
(camera, width, frame) and shared by all clients asking for that width.
Widths snap to a few buckets so different tile sizes still share renders.

The scaled frame is made from the shared full-size JPEG with OpenCV's
reduced decoders (IMREAD_REDUCED_COLOR_2/4/8 decode at 1/2..1/8 size almost
for free), then resized and re-encoded at a slightly lower quality.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

from . import state, streaming

WIDTH_BUCKETS = (320, 480, 640, 960, 1280)
SCALED_JPEG_QUALITY_MAX = 80

_REDUCED = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2))


def snap_width(width: int | None) -> int | None:
    """Requested width -> nearest bucket (None / <= 0 means full size)."""
    if width is None or width <= 0:
        return None
    for bucket in WIDTH_BUCKETS:
        if width <= bucket:
            return bucket
    return WIDTH_BUCKETS[-1]


class _Entry:
    __slots__ = ("lock", "seq", "jpeg")

    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.jpeg: bytes | None = None


_cache: dict[tuple[str, int], _Entry] = {}
_cache_lock = threading.Lock()
encode_count = 0      # number of scaled encodes (diagnostics / tests)


def _entry(camera_id: str, width: int) -> _Entry:
    with _cache_lock:
        entry = _cache.get((camera_id, width))
        if entry is None:
            entry = _cache[(camera_id, width)] = _Entry()
        return entry


def _source_width(camera_state) -> int:
    try:
        return int(str(camera_state.stats.get("resolution", "0x0")).split("x")[0])
    except (TypeError, ValueError):
        return 0


def downscale_jpeg(jpeg: bytes, width: int, src_width: int = 0, quality: int = SCALED_JPEG_QUALITY_MAX) -> bytes | None:
    buf = np.frombuffer(jpeg, dtype=np.uint8)
    flag = cv2.IMREAD_COLOR
    if src_width > 0:
        for factor, reduced in _REDUCED:
            if src_width // factor >= width:
                flag = reduced
                break
    img = cv2.imdecode(buf, flag)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    ok, out = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return out.tobytes() if ok else None


def scaled_jpeg(camera_state, seq: int, jpeg: bytes, width: int) -> bytes | None:
    """Downscaled version of the shared frame ``seq`` (encoded once per width)."""
    global encode_count
    entry = _entry(camera_state.camera_id, width)
    with entry.lock:
        if entry.seq == seq and entry.jpeg is not None:
            return entry.jpeg
        src_w = _source_width(camera_state)
        if 0 < src_w <= width:
            out = jpeg                              # source already small enough
        else:
            quality = min(SCALED_JPEG_QUALITY_MAX, streaming._jpeg_quality(camera_state))
            out = downscale_jpeg(jpeg, width, src_w, quality)
            encode_count += 1
        if out is None:
            return None
        entry.seq, entry.jpeg = seq, out
        return out


def gen_frames_scaled(camera_id: str, width: int):
    """MJPEG generator like ``streaming.gen_frames`` but at (bucketed) ``width``."""
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        return
    width = snap_width(width)
    if width is None:
        yield from streaming.gen_frames(camera_id)
        return
    if not camera_state.rtsp_url:
        camera_state.stream_error = "No RTSP configuration found for this camera."
        return

    streaming.ensure_worker(camera_state)
    feed = streaming._acquire_feed(camera_state)
    if feed is None:
        if not camera_state.stream_error:
            camera_state.stream_error = (
                "Could not open RTSP stream - check IP, stream path and the camera account in .env."
            )
        return

    camera_state.stream_error = ""
    last_seq = 0
    try:
        while True:
            got = feed.next_jpeg(last_seq)
            if got is None:
                if feed.stream.stopped:
                    break
                if not feed.stream.grabbed:
                    camera_state.stream_error = "Camera stream is unavailable."
                continue
            last_seq, full = got
            small = scaled_jpeg(camera_state, last_seq, full, width)
            if small is None:
                continue
            camera_state.stream_error = ""
            yield streaming._mjpeg_part(small)
    finally:
        streaming._release_feed(camera_state, feed)
