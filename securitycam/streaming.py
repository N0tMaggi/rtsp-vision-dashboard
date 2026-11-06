"""Camera streaming, worker threads, and MJPEG generators.

Per camera there is one RTSP reader (``VideoStream``) shared by all browser
clients through a ``CameraFeed``. Every new source frame is rendered and
JPEG-encoded at most once; all clients of that camera get the same bytes.
Clients block on a condition variable until a newer frame exists (no busy
spinning, no re-sending identical frames, always the newest frame).
"""
from __future__ import annotations

import re
import threading
import time

import cv2

from . import config, deps, recognition, state
from .alerts.clips import record_frame
from .esp import draw_custom_esp

FRAME_WAIT_S = 2.0          # how long a client waits for a new frame before flagging the stream


def yolo_worker(camera_state):
    """
    Dedicated worker per camera.
    Inference runs independently from MJPEG encoding.
    """
    print(f"[YOLO Worker] Started for {camera_state.camera_id}.")
    last_frame_ref = None

    while True:
        if not camera_state.yolo_enabled or not state.YOLO_AVAILABLE:
            time.sleep(0.5)
            with camera_state.frame_lock:
                camera_state.latest_results = None
                camera_state.latest_results_ts = None
            last_frame_ref = None
            continue

        with camera_state.frame_lock:
            current_frame_ref = camera_state.latest_frame
            frame_ts = camera_state.latest_frame_ts

        if current_frame_ref is None or current_frame_ref is last_frame_ref:
            # No (new) frame: don't re-run inference on a frame we already processed.
            time.sleep(0.004)
            continue

        model = deps.get_yolo_model(camera_state)
        if model is None:
            time.sleep(1)
            continue

        do_track = camera_state.esp_settings["tracking"]

        try:
            started_at = time.time()
            current_conf = camera_state.esp_settings.get("confidence_threshold", 0.35)

            if do_track:
                results = model.track(current_frame_ref, persist=True, verbose=False, device=state.DEVICE, conf=current_conf,
                                      **recognition.track_kwargs(camera_state, model))
            else:
                results = model(current_frame_ref, verbose=False, device=state.DEVICE, conf=current_conf)
            last_frame_ref = current_frame_ref
            # recognition: stable re-entry ids + hand-off to the face/plate worker
            results = recognition.after_inference(camera_state, current_frame_ref, results, frame_ts)

            infer_dt = time.time() - started_at
            camera_state.stats["last_infer_ms"] = infer_dt * 1000.0
            if infer_dt > 0:
                camera_state.stats["inference_fps"] = 1.0 / infer_dt

            with camera_state.frame_lock:
                camera_state.latest_results = results
                camera_state.latest_results_ts = frame_ts

        except RuntimeError as exc:
            msg = str(exc)
            if ("no kernel image is available" in msg or "CUDA error" in msg) and state.DEVICE.startswith("cuda"):
                print(f"[YOLO Worker] CUDA crash for {camera_state.camera_id} -> fallback CPU")
                state.DEVICE = "cpu"
                state.CUDA_AVAILABLE = False
                camera_state.yolo_model = None
            else:
                print(f"[YOLO Worker] Error for {camera_state.camera_id}: {exc}")
            time.sleep(1)

        except Exception:
            # Ignore transient inference errors and continue (without spinning).
            time.sleep(0.05)


def _results_meta(camera_state, results):
    """(num_detections, classes) for ``results``, computed once per results object."""
    cached = camera_state.results_meta
    if cached is not None and cached[0] is results:
        return cached[1], cached[2]
    num_det = 0
    classes = []
    if results:
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            num_det = len(boxes)
            names = result.names
            for cls_value in boxes.cls.tolist():
                cls_id = int(cls_value)
                classes.append(names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id))
    classes = sorted(set(classes))
    camera_state.results_meta = (results, num_det, classes)
    return num_det, classes


def yolo_process_render(camera_state, frame, frame_ts=None, coord_scale=1.0):
    """
    Render-only function for the active camera frame.
    """
    with camera_state.frame_lock:
        results = camera_state.latest_results
        results_ts = camera_state.latest_results_ts

    num_det, classes = _results_meta(camera_state, results)

    # Even without results, frame-level ESP effects (thermal, face blur, crosshair) apply.
    try:
        annotated_frame = draw_custom_esp(camera_state, frame, results, frame_ts=frame_ts,
                                          results_ts=results_ts, coord_scale=coord_scale)
    except Exception as exc:
        print(f"[Render] Error for {camera_state.camera_id}: {exc}")
        return frame, num_det, classes

    return annotated_frame, num_det, classes


def _jpeg_quality(camera_state) -> int:
    try:
        q = int(camera_state.esp_settings.get("jpeg_quality", config.JPEG_QUALITY))
    except (TypeError, ValueError):
        q = config.JPEG_QUALITY
    return max(30, min(95, q))


def render_jpeg(camera_state, frame, frame_ts=None):
    """Render the overlay for one source frame and JPEG-encode it.

    ``frame`` is the shared reader frame and is never modified. Returns the
    JPEG bytes or ``None`` if encoding failed. Updates the camera stats.
    """
    coord_scale = 1.0
    if config.DISPLAY_SCALE < 1.0:
        # resize allocates a new array, so the overlay can draw on it directly
        frame = cv2.resize(frame, (0, 0), fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE,
                           interpolation=cv2.INTER_AREA)
        coord_scale = config.DISPLAY_SCALE
    elif camera_state.yolo_enabled:
        # The ESP draws in place; the reader frame is shared with the YOLO worker
        # and other clients, so draw on a private copy.
        frame = frame.copy()

    height, width = frame.shape[:2]
    stats = camera_state.stats
    stats["resolution"] = f"{width}x{height}"
    stats["total_frames"] += 1

    num_det = 0
    classes = []
    if camera_state.yolo_enabled:
        frame, num_det, classes = yolo_process_render(camera_state, frame, frame_ts, coord_scale)

    stats["num_detections"] = int(num_det)
    stats["classes"] = classes

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_quality(camera_state)])
    if not ok:
        return None
    jpeg = buffer.tobytes()
    record_frame(camera_state, jpeg)   # alarm pre-roll ring buffer + snapshot source
    return jpeg


class VideoStream:
    """
    Read frames in a dedicated thread to avoid RTSP buffer lag.
    Only the most recent frame is kept for processing; ``seq`` increases per frame
    and ``cond`` is notified whenever a new frame arrives.
    """

    def __init__(self, src, camera_state):
        self.camera_state = camera_state
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.grabbed, self.frame = self.stream.read()
        self.frame_ts = time.perf_counter()
        self.seq = 1 if self.grabbed else 0
        self.stopped = False
        self.thread = None

        if not self.grabbed:
            print(f"[VideoStream] Could not connect to {camera_state.camera_id}.")
        else:
            print(f"[VideoStream] Stream started for {camera_state.camera_id}.")
            self._publish(self.frame, self.frame_ts)

    def _publish(self, frame, ts):
        with self.camera_state.frame_lock:
            self.camera_state.latest_frame = frame
            self.camera_state.latest_frame_ts = ts

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True,
                                       name=f"VideoStream-{self.camera_state.camera_id}")
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            ts = time.perf_counter()
            with self.cond:
                self.grabbed = grabbed
                if grabbed:
                    self.frame = frame
                    self.frame_ts = ts
                    self.seq += 1
                self.cond.notify_all()

            if grabbed:
                self._publish(frame, ts)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def wait_newer(self, seq, timeout):
        """Block until a frame newer than ``seq`` exists; return (seq, frame, ts) or None."""
        deadline = time.perf_counter() + timeout
        with self.cond:
            while self.seq <= seq and not self.stopped:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)
            if self.stopped or self.frame is None:
                return None
            return self.seq, self.frame, self.frame_ts

    def release(self):
        self.stopped = True
        with self.cond:
            self.cond.notify_all()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        if self.stream:
            self.stream.release()


class CameraFeed:
    """Shared reader + per-frame JPEG cache for all clients of one camera."""

    def __init__(self, camera_state, stream):
        self.camera_state = camera_state
        self.stream = stream
        self.clients = 0
        self.render_lock = threading.Lock()
        self.jpeg_seq = 0
        self.jpeg = None
        self.last_render = None

    def next_jpeg(self, after_seq):
        """Newest encoded frame with seq > ``after_seq`` (rendered once, shared)."""
        got = self.stream.wait_newer(after_seq, FRAME_WAIT_S)
        if got is None:
            return None
        with self.render_lock:
            if self.jpeg_seq > after_seq and self.jpeg is not None:
                return self.jpeg_seq, self.jpeg          # another client already rendered it
            # Take the newest frame at this point (we may have waited for the lock).
            with self.stream.lock:
                seq, frame, ts = self.stream.seq, self.stream.frame, self.stream.frame_ts
            jpeg = render_jpeg(self.camera_state, frame, ts)
            now = time.time()
            if self.last_render is not None:
                dt = now - self.last_render
                if dt > 0:
                    self.camera_state.stats["fps"] = 1.0 / dt
            self.last_render = now
            self.camera_state.stats["last_update"] = now
            if jpeg is None:
                return None
            self.jpeg_seq, self.jpeg = seq, jpeg
            return seq, jpeg


_feeds: dict[str, CameraFeed] = {}
_feeds_lock = threading.Lock()

# Failed connects back off (5 s, doubling up to 60 s). Many cameras (e.g. Tapo)
# lock RTSP for a while after repeated failed logins, so hammering them with
# reconnects on every browser request makes a credential problem much worse.
_CONNECT_BACKOFF_MIN_S = 5.0
_CONNECT_BACKOFF_MAX_S = 60.0
_connect_failures: dict[str, tuple[int, float]] = {}   # camera_id -> (count, retry_after)


def _redact_url(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://user:***@host/... for logs."""
    return re.sub(r"(//[^:/@]*:)[^@/]*@", r"\1***@", url)


def _acquire_feed(camera_state):
    with _feeds_lock:
        feed = _feeds.get(camera_state.camera_id)
        if feed is not None and not feed.stream.stopped and feed.stream.grabbed:
            feed.clients += 1
            return feed
        if feed is not None:
            feed.stream.release()
            _feeds.pop(camera_state.camera_id, None)

        failures, retry_after = _connect_failures.get(camera_state.camera_id, (0, 0.0))
        wait = retry_after - time.monotonic()
        if wait > 0:
            camera_state.stream_error = (
                f"Camera refused the connection. Next attempt in {wait:.0f} s - "
                "check user/password of the camera account in .env."
            )
            return None

        print(f"[Camera] Connecting to {camera_state.camera_id}: {_redact_url(camera_state.rtsp_url)}")
        with camera_state.frame_lock:
            camera_state.latest_frame = None
        stream = VideoStream(camera_state.rtsp_url, camera_state)
        if not stream.grabbed:
            stream.release()
            failures += 1
            delay = min(_CONNECT_BACKOFF_MAX_S, _CONNECT_BACKOFF_MIN_S * 2 ** (failures - 1))
            _connect_failures[camera_state.camera_id] = (failures, time.monotonic() + delay)
            print(f"[Camera] Connect to {camera_state.camera_id} failed ({failures}x), retrying in {delay:.0f} s.")
            return None
        _connect_failures.pop(camera_state.camera_id, None)
        stream.start()
        feed = CameraFeed(camera_state, stream)
        feed.clients = 1
        _feeds[camera_state.camera_id] = feed
        return feed


def _release_feed(camera_state, feed):
    with _feeds_lock:
        feed.clients -= 1
        if feed.clients > 0:
            return
        if _feeds.get(camera_state.camera_id) is feed:
            _feeds.pop(camera_state.camera_id, None)
    feed.stream.release()
    with camera_state.frame_lock:
        camera_state.latest_frame = None
    print(f"[Camera] Stream closed for {camera_state.camera_id}.")


def ensure_worker(camera_state):
    thread = camera_state.worker_thread
    if thread is not None and thread.is_alive():
        return

    worker = threading.Thread(
        target=yolo_worker,
        args=(camera_state,),
        name=f"YoloWorkerThread-{camera_state.camera_id}",
        daemon=True,
    )
    worker.start()
    camera_state.worker_thread = worker


def _mjpeg_part(jpg_bytes: bytes) -> bytes:
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpg_bytes)).encode("ascii") + b"\r\n\r\n"
        + jpg_bytes + b"\r\n"
    )


def gen_frames(camera_id: str):
    """MJPEG generator for a specific camera."""
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        return

    if not camera_state.rtsp_url:
        camera_state.stream_error = "No RTSP configuration found for this camera."
        print(f"[Camera] Missing RTSP configuration for {camera_id}.")
        return

    ensure_worker(camera_state)

    feed = _acquire_feed(camera_state)
    if feed is None:
        if not camera_state.stream_error:
            camera_state.stream_error = (
                "Could not open RTSP stream - check IP, stream path and the camera account in .env."
            )
        return

    print(f"[Camera] Camera connection established for {camera_id}.")
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
            last_seq, jpg_bytes = got
            camera_state.stream_error = ""
            yield _mjpeg_part(jpg_bytes)
    finally:
        _release_feed(camera_state, feed)
