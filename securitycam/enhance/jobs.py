"""Enhancement jobs: capture, queue (one at a time), cancel, results on disk, retention.

Layout under ``data/enhanced/`` (``SECURITYCAM_DATA_DIR`` aware)::

    <job id>/original.png    raw camera pixels of the region (no overlay), PNG text: "not enhanced"
    <job id>/enhanced.png    upscaled + corner tag + "AI ENHANCED" band + PNG text chunks
    <job id>/overlay.png     optional: the same region of the rendered (ESP overlay) frame
    <job id>/meta.json       sidecar with everything (model, device, timings, source, region)

Request threads only copy the crop out of the shared frame and enqueue; PNG
encoding, model download/load and inference happen in one worker thread.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import conf, marking, models
from .region import EnhanceInputError, crop_box, valid_job_id

FILES = {
    "original.png": "image/png",
    "enhanced.png": "image/png",
    "overlay.png": "image/png",
    "meta.json": "application/json",
}
ACTIVE = ("queued", "downloading", "loading", "running")


@dataclass
class Capture:
    """Everything a job needs from the camera, taken at request time."""
    crop: np.ndarray
    frame_size: tuple[int, int]
    region_px: tuple[int, int, int, int]
    source: dict[str, Any]
    overlay_jpeg: bytes | None = None


@dataclass
class Job:
    id: str
    meta: dict[str, Any]
    capture: Capture | None = None
    cancel: threading.Event = field(default_factory=threading.Event)


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"


def capture_frame(camera_state, region, include_overlay: bool = False) -> Capture:
    """Crop the raw frame currently shown (paused video: the paused frame; live: the latest)."""
    with camera_state.frame_lock:
        frame = camera_state.latest_frame
        frame_ts = camera_state.latest_frame_ts
    if frame is None:
        raise EnhanceInputError("no_frame", 409)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box(region, w, h)
    crop = np.ascontiguousarray(frame[y1:y2, x1:x2]).copy()   # the reader frame is shared - never keep a view
    now = time.time()
    age = max(0.0, time.perf_counter() - frame_ts) if frame_ts else 0.0
    taken = now - age
    source: dict[str, Any] = {
        "kind": getattr(camera_state, "source_kind", "rtsp"),
        "timestamp": round(taken, 3),
        "timestamp_iso": _iso(taken),
    }
    src = getattr(camera_state, "source", None)
    if src is not None:
        try:
            pb = src.playback_payload()
            source.update({"label": src.label, "playing": bool(pb.get("playing")), "position_s": pb.get("position"),
                           "frame": pb.get("frame")})
        except Exception:
            pass
    overlay = None
    if include_overlay:
        try:
            from ..alerts import clips

            latest = clips.ring(camera_state.camera_id).latest()
            if latest is not None and now - latest[0] < 5.0:
                overlay = latest[1]
        except Exception:
            overlay = None
    return Capture(crop, (w, h), (x1, y1, x2, y2), source, overlay)


class JobManager:
    def __init__(self, root: Path | None = None, enhance_fn: Callable[..., models.Result] | None = None):
        self.root = Path(root) if root is not None else conf.ENHANCED_DIR
        self._enhance = enhance_fn or models.enhance
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._cond = threading.Condition()
        self._worker: threading.Thread | None = None
        self._running: str | None = None

    # ------------------------------------------------------------------ submit / status
    def submit(self, camera_state, opts: dict[str, Any]) -> dict[str, Any]:
        cap = capture_frame(camera_state, opts["region"], opts.get("include_overlay", False))
        spec = models.CATALOG[opts["model"]]
        out_w, out_h = cap.crop.shape[1] * opts["scale"], cap.crop.shape[0] * opts["scale"]
        if out_w * out_h > conf.MAX_OUTPUT_MP * 1_000_000:
            raise EnhanceInputError("region_too_large")
        job_id = secrets.token_hex(10)
        meta = {
            "id": job_id,
            "camera_id": camera_state.camera_id,
            "camera_label": getattr(camera_state, "label", camera_state.camera_id),
            "created_at": round(time.time(), 3),
            "status": "queued",
            "progress": 0.0,
            "options": {k: opts[k] for k in ("model", "scale", "denoise", "device")},
            "region": [round(v, 6) for v in opts["region"]],
            "region_px": list(cap.region_px),
            "frame_size": list(cap.frame_size),
            "crop_size": [cap.crop.shape[1], cap.crop.shape[0]],
            "output_size": [out_w, out_h],
            "source": cap.source,
            "model": _model_meta(spec),
            "ai_enhanced": spec.ai,
            "notice": marking.NOTICE,
            "description": marking.DESCRIPTION if spec.ai else "Upscaled with classic resampling (no AI).",
            "files": {},
            "timings": {},
            "device_used": None,
            "fallback": None,
            "error": None,
        }
        job = Job(job_id, meta, cap)
        with self._cond:
            if len(self._queue) >= conf.MAX_QUEUE:
                raise EnhanceInputError("queue_full", 429)
            self._jobs[job_id] = job
            self._queue.append(job_id)
            self._ensure_worker()
            self._cond.notify_all()
            return self._payload(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        if not valid_job_id(job_id):
            return None
        with self._cond:
            job = self._jobs.get(job_id)
            if job is not None:
                return self._payload(job)
        meta = self._read_meta(job_id)
        return self._decorate(meta) if meta else None

    def list(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        if self.root.is_dir():
            for d in self.root.iterdir():
                if d.is_dir() and valid_job_id(d.name):
                    meta = self._read_meta(d.name)
                    if meta:
                        seen[d.name] = self._decorate(meta)
        with self._cond:
            for job in self._jobs.values():
                if job.meta["status"] == "canceled":
                    seen.pop(job.id, None)
                else:
                    seen[job.id] = self._payload(job)
        return sorted(seen.values(), key=lambda m: m.get("created_at", 0), reverse=True)

    def cancel(self, job_id: str) -> bool:
        if not valid_job_id(job_id):
            return False
        with self._cond:
            job = self._jobs.get(job_id)
            if job is None or job.meta["status"] not in ACTIVE:
                return False
            job.cancel.set()
            if job_id in self._queue:
                self._queue.remove(job_id)
                job.meta.update(status="canceled", progress=0.0)
                job.capture = None
            return True

    def delete(self, job_id: str) -> bool:
        """True = deleted, False = unknown job; raises OSError when files could not be removed."""
        if not valid_job_id(job_id):
            return False
        with self._cond:
            job = self._jobs.get(job_id)
            if job is not None and job.meta["status"] in ACTIVE:
                job.cancel.set()
                if job_id in self._queue:
                    self._queue.remove(job_id)
                    self._jobs.pop(job_id, None)
            elif job is not None:
                self._jobs.pop(job_id, None)
        d = self._dir(job_id)
        if d is None or not d.is_dir():
            return job is not None
        shutil.rmtree(d, ignore_errors=True)
        if d.exists():          # Windows: a file is still open (download in progress)
            raise OSError("delete_failed")
        return True

    def file_path(self, job_id: str, name: str) -> Path | None:
        """Path of a result file, or None (unknown job / file name; no traversal possible)."""
        if not valid_job_id(job_id) or name not in FILES:
            return None
        d = self._dir(job_id)
        if d is None:
            return None
        p = d / name
        return p if p.is_file() else None

    def busy(self) -> bool:
        with self._cond:
            return self._running is not None or bool(self._queue)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Tests: block until the queue is empty and nothing runs."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.busy():
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------ internals
    def _dir(self, job_id: str) -> Path | None:
        if not valid_job_id(job_id):
            return None
        root = self.root.resolve() if self.root.exists() else self.root
        d = root / job_id
        return d

    def _read_meta(self, job_id: str) -> dict[str, Any] | None:
        d = self._dir(job_id)
        if d is None:
            return None
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(meta, dict) or meta.get("id") != job_id:
            return None
        if meta.get("status") in ACTIVE:      # left over from a previous server run
            meta["status"] = "failed"
            meta["error"] = "interrupted"
        return meta

    def _write_meta(self, job: Job) -> None:
        d = self._dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(job.meta, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, d / "meta.json")

    def _decorate(self, meta: dict[str, Any]) -> dict[str, Any]:
        out = dict(meta)
        base = f"/api/enhance/{meta['id']}"
        out["urls"] = {k: f"{base}/{v}" for k, v in (meta.get("files") or {}).items()}
        return out

    def _payload(self, job: Job) -> dict[str, Any]:
        out = self._decorate(job.meta)
        if job.meta["status"] == "queued":
            try:
                out["queue_position"] = list(self._queue).index(job.id) + 1
            except ValueError:
                pass
        return out

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._loop, daemon=True, name="EnhanceWorker")
            self._worker.start()

    def _loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                job_id = self._queue.popleft()
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                self._running = job_id
            try:
                self._run(job)
            except Exception as exc:   # never kill the worker
                print(f"[ENHANCE] Job {job_id} crashed: {type(exc).__name__}: {exc}")
                job.meta.update(status="failed", error="internal_error")
                try:
                    self._write_meta(job)
                except OSError:
                    pass
            finally:
                job.capture = None
                with self._cond:
                    self._running = None
                    # finished jobs are served from disk from now on
                    if job.meta["status"] not in ACTIVE and job.meta["status"] != "canceled":
                        self._jobs.pop(job_id, None)
                self._prune()

    def _run(self, job: Job) -> None:
        meta, cap = job.meta, job.capture
        t_start = time.perf_counter()
        d = self._dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        model_id, scale, denoise = meta["options"]["model"], meta["options"]["scale"], meta["options"]["denoise"]
        common = self._text_common(meta)

        orig_png = marking.encode_png(cap.crop, {**common, "Title": "Original crop", "Description": marking.ORIGINAL_DESCRIPTION,
                                                 "AI-Enhanced": "false"})
        (d / "original.png").write_bytes(orig_png)
        meta["files"]["original"] = "original.png"
        if cap.overlay_jpeg:
            ov = self._overlay_crop(cap.overlay_jpeg, meta["region"])
            if ov is not None:
                (d / "overlay.png").write_bytes(marking.encode_png(ov, {**common, "Title": "Original crop with overlay",
                                                                       "Description": "Rendered dashboard frame (overlay), not enhanced"}))
                meta["files"]["overlay"] = "overlay.png"
        self._write_meta(job)

        def progress(f: float) -> None:
            meta["progress"] = round(min(1.0, max(0.0, f)), 4)

        def stage(s: str) -> None:
            meta["status"] = s

        try:
            res = self._enhance(cap.crop, model_id, scale, denoise, meta["options"]["device"],
                                progress, job.cancel.is_set, stage)
        except models.Cancelled:
            self._discard(job)
            return
        except Exception as exc:
            print(f"[ENHANCE] Job {job.id} failed: {type(exc).__name__}: {exc}")
            meta.update(status="failed", error="enhance_failed")
            self._write_meta(job)
            return
        if job.cancel.is_set():
            self._discard(job)
            return

        spec = res.model
        meta["model"] = _model_meta(spec)
        meta["ai_enhanced"] = spec.ai
        if not spec.ai:
            meta["description"] = "Upscaled with classic resampling (no AI)."
        meta["device_used"] = res.device
        meta["fallback"] = res.fallback
        text = {
            **common,
            "Title": "AI ENHANCED" if spec.ai else "UPSCALED (NO AI)",
            "Description": marking.DESCRIPTION if spec.ai else "Upscaled with classic resampling (no AI) - not evidence",
            "AI-Enhanced": "true" if spec.ai else "false",
            "Model": spec.label,
            "Model-License": spec.license,
            "Scale": f"{scale}x",
            "Denoise": f"{denoise:.2f}" if spec.denoise else "-",
            "Device": res.device,
        }
        png, band = marking.mark_export(res.image, spec.short, scale, text, ai=spec.ai)
        (d / "enhanced.png").write_bytes(png)
        meta["files"]["enhanced"] = "enhanced.png"
        meta["band_px"] = band
        meta["output_size"] = [res.image.shape[1], res.image.shape[0]]
        meta["timings"] = {k: round(v, 3) for k, v in res.timings.items()}
        meta["timings"]["total_s"] = round(time.perf_counter() - t_start, 3)
        meta.update(status="done", progress=1.0, finished_at=round(time.time(), 3))
        self._write_meta(job)

    def _discard(self, job: Job) -> None:
        """Canceled: nothing stays on disk; the status remains queryable until restart."""
        job.meta.update(status="canceled", progress=0.0, files={})
        d = self._dir(job.id)
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _text_common(meta: dict[str, Any]) -> dict[str, str]:
        r = meta["region"]
        px = meta["region_px"]
        src = meta["source"]
        stamp = src.get("timestamp_iso", "")
        if src.get("position_s") is not None:
            stamp += f" (video position {src['position_s']} s, frame {src.get('frame')})"
        return {
            "Software": marking.SOFTWARE,
            "Source-Camera": f"{meta['camera_label']} ({meta['camera_id']})",
            "Source-Timestamp": stamp,
            "Region": f"normalized [{r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}]; "
                      f"pixels [{px[0]}, {px[1]}, {px[2]}, {px[3]}] of {meta['frame_size'][0]}x{meta['frame_size'][1]}",
            "Creation Time": _iso(meta["created_at"]),
            "Job": meta["id"],
        }

    @staticmethod
    def _overlay_crop(jpeg: bytes, region) -> np.ndarray | None:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = crop_box(tuple(region), w, h)
        return img[y1:y2, x1:x2].copy()

    def _prune(self) -> None:
        """Keep at most MAX_COUNT finished results and none older than MAX_AGE_DAYS."""
        if not self.root.is_dir():
            return
        with self._cond:
            protect = {j.id for j in self._jobs.values() if j.meta["status"] in ACTIVE}
            if self._running:
                protect.add(self._running)
        entries = []
        for d in self.root.iterdir():
            if not d.is_dir() or not valid_job_id(d.name) or d.name in protect:
                continue
            meta = self._read_meta(d.name)
            created = meta.get("created_at", 0) if meta else d.stat().st_mtime
            entries.append((created, d))
        entries.sort(key=lambda e: e[0], reverse=True)
        cutoff = time.time() - conf.MAX_AGE_DAYS * 86400 if conf.MAX_AGE_DAYS > 0 else None
        for i, (created, d) in enumerate(entries):
            if i >= conf.MAX_COUNT or (cutoff is not None and created < cutoff):
                shutil.rmtree(d, ignore_errors=True)


def _model_meta(spec: models.ModelSpec) -> dict[str, Any]:
    return {"id": spec.id, "label": spec.label, "ai": spec.ai, "license": spec.license, "source": spec.source}


manager = JobManager()
