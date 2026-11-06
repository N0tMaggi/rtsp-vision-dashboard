"""Alarm events on disk: ``data/events/<camera>/<stamp>/`` with meta.json, thumb.jpg,
snapshot.jpg and a clip (``clip.webm`` / ``clip.mp4`` or ``frames/*.jpg``).

``trigger`` is called from the render path and only enqueues. A worker thread
creates the event (waits briefly for a rendered frame that shows the alarm,
writes thumbnail + metadata, queues the notification); a second thread encodes
the clip once the post-roll has been recorded. Retention (max count / max age)
is applied after every new event and on first load.
"""
from __future__ import annotations

import heapq
import json
import os
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from . import clips, conf, notify

THUMB_WIDTH = 480
FRAME_WAIT_S = 1.5          # how long event creation waits for a frame showing the alarm
CAMERA_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
EVENT_RE = re.compile(r"^\d{8}-\d{6}-\d{3}(-\d{1,3})?$")


def _atomic_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _thumb(jpeg: bytes) -> Optional[bytes]:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > THUMB_WIDTH:
        img = cv2.resize(img, (THUMB_WIDTH, max(1, int(h * THUMB_WIDTH / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


class EventStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or conf.EVENTS_DIR
        self._lock = threading.RLock()
        self._index: dict[tuple[str, str], dict] = {}
        self._loaded = False
        self._triggers: "queue.Queue[dict]" = queue.Queue(maxsize=64)
        self._jobs: list = []                     # heap of (due_monotonic, seq, cam, eid, t_wall)
        self._jobs_cond = threading.Condition()
        self._seq = 0
        self._threads_started = False

    # ------------------------------------------------------------------ paths
    def event_dir(self, camera_id: str, event_id: str) -> Optional[Path]:
        if not CAMERA_RE.match(camera_id or "") or not EVENT_RE.match(event_id or ""):
            return None
        return self.root / camera_id / event_id

    # ------------------------------------------------------------------ index
    def _load_locked(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.root.is_dir():
            return
        for cam_dir in self.root.iterdir():
            if not cam_dir.is_dir() or not CAMERA_RE.match(cam_dir.name):
                continue
            for ev_dir in cam_dir.iterdir():
                meta_path = ev_dir / "meta.json"
                if not EVENT_RE.match(ev_dir.name) or not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                clip = meta.get("clip") or {}
                if clip.get("status") == "recording":   # app stopped while recording
                    meta["clip"] = {"status": "failed", "error": "interrupted"}
                meta["id"], meta["camera"] = ev_dir.name, cam_dir.name
                self._index[(cam_dir.name, ev_dir.name)] = meta
        self._retention_locked()

    def list(self, camera_id: Optional[str] = None, since: Optional[float] = None, limit: int = 200) -> list[dict]:
        with self._lock:
            self._load_locked()
            items = [m for (cam, _), m in self._index.items()
                     if (camera_id is None or cam == camera_id) and (since is None or m.get("ts", 0) > since)]
        items.sort(key=lambda m: m.get("ts", 0), reverse=True)
        return [dict(m) for m in items[:max(1, limit)]]

    def get(self, camera_id: str, event_id: str) -> Optional[dict]:
        with self._lock:
            self._load_locked()
            m = self._index.get((camera_id, event_id))
            return dict(m) if m else None

    def delete(self, camera_id: str, event_id: str) -> bool:
        path = self.event_dir(camera_id, event_id)
        if path is None:
            return False
        with self._lock:
            self._load_locked()
            if self._index.pop((camera_id, event_id), None) is None:
                return False
        shutil.rmtree(path, ignore_errors=True)
        return True

    def _retention_locked(self) -> None:
        doomed = []
        if conf.EVENTS_MAX_AGE_DAYS > 0:
            cutoff = time.time() - conf.EVENTS_MAX_AGE_DAYS * 86400.0
            doomed += [k for k, m in self._index.items() if m.get("ts", 0) < cutoff]
        remaining = sorted((k for k in self._index if k not in set(doomed)),
                           key=lambda k: self._index[k].get("ts", 0))
        excess = len(remaining) - conf.EVENTS_MAX_COUNT
        if excess > 0:
            doomed += remaining[:excess]
        for key in doomed:
            self._index.pop(key, None)
            path = self.event_dir(*key)
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

    # ------------------------------------------------------------------ creation
    def trigger(self, event: dict) -> bool:
        """Render-path entry point: queue an event (never blocks)."""
        self._start_threads()
        try:
            self._triggers.put_nowait(event)
            return True
        except queue.Full:
            return False

    def _start_threads(self) -> None:
        if self._threads_started:
            return
        with self._lock:
            if self._threads_started:
                return
            self._threads_started = True
            threading.Thread(target=self._event_worker, name="AlertEvents", daemon=True).start()
            threading.Thread(target=self._clip_worker, name="AlertClips", daemon=True).start()

    def _new_dir(self, camera_id: str, ts: float) -> tuple[str, Path]:
        base = time.strftime("%Y%m%d-%H%M%S", time.localtime(ts)) + f"-{int((ts % 1) * 1000):03d}"
        eid = base
        n = 1
        while (self.root / camera_id / eid).exists():
            n += 1
            eid = f"{base}-{n}"
        path = self.root / camera_id / eid
        path.mkdir(parents=True, exist_ok=False)
        return eid, path

    def create(self, trig: dict, wait_s: float = FRAME_WAIT_S) -> Optional[dict]:
        cam = trig["camera"]
        if not CAMERA_RE.match(cam):
            return None
        ts = float(trig.get("ts", time.time()))
        ring = clips.ring(cam)
        # Prefer the first rendered frame after the trigger: it shows the alarm overlay.
        frame = None
        deadline = time.monotonic() + wait_s
        while frame is None:
            frame = ring.first_after(ts + 0.15)
            if frame is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if frame is None:
            frame = ring.latest()

        eid, path = self._new_dir(cam, ts)
        snapshot = frame[1] if frame else None
        has_thumb = False
        if snapshot:
            (path / "snapshot.jpg").write_bytes(snapshot)
            thumb = _thumb(snapshot)
            if thumb:
                (path / "thumb.jpg").write_bytes(thumb)
                has_thumb = True
        meta = {k: v for k, v in trig.items() if k not in ("notify",)}
        meta.update({"id": eid, "camera": cam, "ts": ts, "thumb": has_thumb,
                     "clip": {"status": "recording" if snapshot else "failed"}})
        _atomic_json(path / "meta.json", meta)
        with self._lock:
            self._load_locked()
            self._index[(cam, eid)] = meta
            self._retention_locked()

        if trig.get("notify", True):
            notify.submit(meta, snapshot, (cam, trig.get("kind"), trig.get("source_id")))
        if snapshot:
            self.schedule_clip(cam, eid, ts)
        return dict(meta)

    def _event_worker(self) -> None:
        while True:
            trig = self._triggers.get()
            try:
                self.create(trig)
            except Exception as exc:
                print(f"[Alerts] Event creation failed: {exc.__class__.__name__}: {exc}")

    # ------------------------------------------------------------------ clips
    def schedule_clip(self, camera_id: str, event_id: str, ts: float) -> None:
        due = time.monotonic() + max(0.0, ts + conf.CLIP_POST_S + 0.3 - time.time())
        with self._jobs_cond:
            self._seq += 1
            heapq.heappush(self._jobs, (due, self._seq, camera_id, event_id, ts))
            self._jobs_cond.notify()

    def finish_clip(self, camera_id: str, event_id: str, ts: float) -> Optional[dict]:
        path = self.event_dir(camera_id, event_id)
        if path is None or not path.is_dir():
            return None                      # deleted meanwhile
        frames = clips.ring(camera_id).between(ts - conf.CLIP_PRE_S, ts + conf.CLIP_POST_S)
        try:
            clip = clips.write_clip(frames, path)
        except Exception as exc:
            clip = {"status": "failed", "error": exc.__class__.__name__}
        with self._lock:
            meta = self._index.get((camera_id, event_id))
            if meta is None:                 # deleted while encoding
                shutil.rmtree(path, ignore_errors=True)
                return None
            if not path.is_dir():
                return None
            meta = dict(meta)
            meta["clip"] = clip
            self._index[(camera_id, event_id)] = meta
            try:
                _atomic_json(path / "meta.json", meta)
            except OSError:
                pass
        return meta

    def _clip_worker(self) -> None:
        while True:
            with self._jobs_cond:
                while not self._jobs or self._jobs[0][0] > time.monotonic():
                    timeout = self._jobs[0][0] - time.monotonic() if self._jobs else None
                    self._jobs_cond.wait(timeout)
                _, _, cam, eid, ts = heapq.heappop(self._jobs)
            try:
                self.finish_clip(cam, eid, ts)
            except Exception as exc:
                print(f"[Alerts] Clip failed: {exc.__class__.__name__}: {exc}")


events = EventStore()


def save_snapshot(camera_id: str, jpeg: bytes) -> Path:
    conf.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    name = f"{camera_id}_{time.strftime('%Y%m%d-%H%M%S', time.localtime(ts))}-{int((ts % 1) * 1000):03d}.jpg"
    path = conf.SNAPSHOTS_DIR / name
    path.write_bytes(jpeg)
    return path
