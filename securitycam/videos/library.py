"""Video library on disk: ``data/videos/<id>/{video.<ext>, meta.json, thumb.jpg}``.

Uploads are streamed to ``data/videos/.incoming/`` in chunks, then validated
by actually opening them with OpenCV (container + first frame + metadata) and
moved into place. Nothing from an upload is ever executed; the original file
name is only kept (sanitized) as display text, the path uses a random id.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import cv2

from .. import config

MB = 1024 * 1024


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, "").strip() or default))
    except ValueError:
        return default


DATA_DIR = Path(os.getenv("SECURITYCAM_DATA_DIR", "").strip() or (config.PROJECT_ROOT / "data"))
VIDEOS_DIR = DATA_DIR / "videos"
INCOMING_DIR = VIDEOS_DIR / ".incoming"
MAX_UPLOAD_BYTES = _env_int("SECURITYCAM_VIDEO_MAX_MB", 1024, 1) * MB
MAX_TOTAL_BYTES = _env_int("SECURITYCAM_VIDEO_MAX_TOTAL_MB", 10240, 1) * MB
ALLOW_URLS = os.getenv("SECURITYCAM_VIDEO_ALLOW_URLS", "true").strip().lower() in {"1", "true", "yes", "on"}
ALLOWED_EXTENSIONS = ("mp4", "mkv", "avi", "mov", "webm")
THUMB_WIDTH = 320
NAME_MAX = 120

_ID_RE = re.compile(r"^[a-f0-9]{12}$")


class LibraryError(Exception):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def valid_id(video_id: Any) -> bool:
    return isinstance(video_id, str) and bool(_ID_RE.match(video_id))


def clean_name(raw: Any, fallback: str = "Video") -> str:
    """Display name: last path component, no control chars, trimmed, bounded length."""
    if not isinstance(raw, str):
        return fallback
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    name = unicodedata.normalize("NFC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"\s+", " ", name).strip().strip(".")
    return name[:NAME_MAX] or fallback


def extension_of(filename: str) -> str:
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def probe(path: Path) -> dict[str, Any]:
    """Open ``path`` with OpenCV and read its metadata; raises LibraryError('invalid_video')."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise LibraryError("invalid_video", 415)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise LibraryError("invalid_video", 415)
        height, width = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0.5 <= fps <= 240.0):
            fps = 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frames <= 0 or frames > 10_000_000:
            # Some containers (webm, fragmented mp4) don't report a length: count it (bounded).
            frames, deadline = 1, time.monotonic() + 20.0
            while cap.grab():
                frames += 1
                if time.monotonic() > deadline:
                    break
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")
        codec = codec if codec.isprintable() else ""
        return {
            "width": int(width),
            "height": int(height),
            "fps": round(fps, 3),
            "frames": frames,
            "duration": round(frames / fps, 3),
            "codec": codec,
        }
    finally:
        cap.release()


def write_thumb(video: Path, out: Path, meta: dict[str, Any]) -> bool:
    cap = cv2.VideoCapture(str(video))
    try:
        # A frame a little into the clip is usually more telling than frame 0.
        at = int(min(meta.get("frames", 0) * 0.1, meta.get("fps", 25) * 2))
        if at > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, at)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            return False
        h, w = frame.shape[:2]
        if w > THUMB_WIDTH:
            frame = cv2.resize(frame, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return False
        out.write_bytes(buf.tobytes())
        return True
    finally:
        cap.release()


class SizeLimitExceeded(Exception):
    pass


class LimitedWriter:
    """Binary file that refuses to grow beyond ``limit`` bytes (upload sink)."""

    def __init__(self, path: Path, limit: int):
        self.path = path
        self.limit = limit
        self.size = 0
        self._fh = open(path, "wb")

    def write(self, data: bytes) -> int:
        self.size += len(data)
        if self.size > self.limit:
            raise SizeLimitExceeded()
        return self._fh.write(data)

    def seek(self, *args):
        return self._fh.seek(*args)

    def tell(self):
        return self._fh.tell()

    def read(self, *args):
        return b""

    def flush(self):
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    @property
    def closed(self) -> bool:
        return self._fh.closed


class Library:
    def __init__(self, root: Path = VIDEOS_DIR, max_total: int = MAX_TOTAL_BYTES):
        self.root = root
        self.incoming = root / ".incoming"
        self.max_total = max_total
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- paths
    def _dir(self, video_id: str) -> Path:
        if not valid_id(video_id):
            raise LibraryError("unknown_video", 404)
        return self.root / video_id

    def new_incoming(self) -> Path:
        self.incoming.mkdir(parents=True, exist_ok=True)
        return self.incoming / f"{secrets.token_hex(8)}.part"

    def cleanup_incoming(self, max_age_s: float = 6 * 3600) -> None:
        """Remove partial uploads left behind by a crash."""
        if not self.incoming.is_dir():
            return
        now = time.time()
        for p in self.incoming.glob("*.part"):
            try:
                if now - p.stat().st_mtime > max_age_s:
                    p.unlink()
            except OSError:
                pass

    # ---------------------------------------------------------------- read
    def _read_meta(self, d: Path) -> dict[str, Any] | None:
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict) or meta.get("id") != d.name:
            return None
        if not (d / meta.get("file", "")).is_file():
            return None
        return meta

    def get(self, video_id: str) -> dict[str, Any]:
        meta = self._read_meta(self._dir(video_id))
        if meta is None:
            raise LibraryError("unknown_video", 404)
        return meta

    def video_path(self, video_id: str) -> Path:
        meta = self.get(video_id)
        return self._dir(video_id) / meta["file"]

    def thumb_path(self, video_id: str) -> Path | None:
        p = self._dir(video_id) / "thumb.jpg"
        return p if p.is_file() else None

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        out = []
        for d in self.root.iterdir():
            if d.is_dir() and valid_id(d.name):
                meta = self._read_meta(d)
                if meta is not None:
                    out.append(meta)
        out.sort(key=lambda m: m.get("created", 0), reverse=True)
        return out

    def total_bytes(self) -> int:
        return sum(int(m.get("size", 0)) for m in self.list())

    # ---------------------------------------------------------------- write
    def _write_meta(self, d: Path, meta: dict[str, Any]) -> None:
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, d / "meta.json")

    def prune_for(self, incoming: int, keep: set[str]) -> list[str]:
        """Delete the oldest videos (not in ``keep``) until ``incoming`` more bytes fit."""
        removed: list[str] = []
        with self.lock:
            videos = sorted(self.list(), key=lambda m: m.get("created", 0))
            total = sum(int(m.get("size", 0)) for m in videos)
            for meta in videos:
                if total + incoming <= self.max_total:
                    break
                if meta["id"] in keep:
                    continue
                self.delete(meta["id"])
                total -= int(meta.get("size", 0))
                removed.append(meta["id"])
            if total + incoming > self.max_total:
                raise LibraryError("library_full", 413)
        return removed

    def add(self, upload: Path, original_name: str, keep: set[str] | None = None) -> tuple[dict[str, Any], list[str]]:
        """Validate a finished upload and move it into the library. Consumes ``upload``."""
        try:
            ext = extension_of(original_name)
            if ext not in ALLOWED_EXTENSIONS:
                raise LibraryError("unsupported_type", 415)
            size = upload.stat().st_size
            if size <= 0:
                raise LibraryError("empty_file", 400)
            info = probe(upload)
            pruned = self.prune_for(size, keep or set())
            with self.lock:
                video_id = secrets.token_hex(6)
                while (self.root / video_id).exists():
                    video_id = secrets.token_hex(6)
                d = self.root / video_id
                d.mkdir(parents=True)
                fname = f"video.{ext}"
                os.replace(upload, d / fname)
                name = clean_name(original_name, f"Video {video_id[:4]}")
                meta = {
                    "id": video_id,
                    "name": name,
                    "original_name": name,
                    "file": fname,
                    "ext": ext,
                    "size": size,
                    "created": time.time(),
                    **info,
                }
                meta["has_thumb"] = write_thumb(d / fname, d / "thumb.jpg", meta)
                self._write_meta(d, meta)
            return meta, pruned
        finally:
            try:
                upload.unlink()
            except OSError:
                pass

    def rename(self, video_id: str, name: Any) -> dict[str, Any]:
        if not isinstance(name, str) or not clean_name(name, ""):
            raise LibraryError("invalid_name")
        with self.lock:
            meta = self.get(video_id)
            meta["name"] = clean_name(name)
            self._write_meta(self._dir(video_id), meta)
        return meta

    def delete(self, video_id: str) -> None:
        with self.lock:
            d = self._dir(video_id)
            if not d.is_dir():
                raise LibraryError("unknown_video", 404)
            shutil.rmtree(d)


LIBRARY = Library()
