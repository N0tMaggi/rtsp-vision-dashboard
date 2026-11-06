"""Scoped FFmpeg capture options for OpenCV.

OpenCV reads ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` from the process environment
on every ``cv2.VideoCapture`` open; there is no per-capture way to pass
FFmpeg options such as ``protocol_whitelist``. URL sources therefore set the
variable for the duration of their own open under an *exclusive* lock and
restore it afterwards; RTSP camera opens take the lock *shared*, so they
always see the operator's original value (e.g. ``rtsp_transport;tcp``).

On Windows the FFmpeg plugin DLL reads the msvcrt environment, which
``os.environ`` does not update, so the value is mirrored there as well.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager

VAR = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


class _SharedExclusiveLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._readers = 0
        self._writer = False

    @contextmanager
    def shared(self):
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    @contextmanager
    def exclusive(self):
        with self._cond:
            while self._writer or self._readers:
                self._cond.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


LOCK = _SharedExclusiveLock()


def _set(value: str | None) -> None:
    if value is None:
        os.environ.pop(VAR, None)
    else:
        os.environ[VAR] = value
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.cdll.msvcrt._putenv(f"{VAR}={value or ''}".encode("utf-8"))   # "VAR=" removes it
        except Exception:
            pass


@contextmanager
def options(extra: str):
    """Add ``extra`` FFmpeg capture options for opens inside the block (exclusive).

    The operator's own options stay in front; later keys win in FFmpeg's parser.
    """
    with LOCK.exclusive():
        previous = os.environ.get(VAR)
        base = (previous or "").strip()
        _set(f"{base}|{extra}" if base else extra)
        try:
            yield
        finally:
            _set(previous)


def shared():
    """Hold while opening a capture that must see the operator's own options."""
    return LOCK.shared()
