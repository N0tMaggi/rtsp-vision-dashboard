"""Camera streaming, worker threads, and MJPEG generators."""
from __future__ import annotations

import threading
import time

import cv2

from . import config, deps, state
from .esp import draw_custom_esp


def yolo_worker(camera_state):
    """
    Dedicated worker per camera.
    Inference runs independently from MJPEG encoding.
    """
    print(f"[YOLO Worker] Started for {camera_state.camera_id}.")

    while True:
        if not camera_state.yolo_enabled or not state.YOLO_AVAILABLE:
            time.sleep(0.5)
            with camera_state.frame_lock:
                camera_state.latest_results = None
            continue

        with camera_state.frame_lock:
            current_frame_ref = camera_state.latest_frame

        if current_frame_ref is None:
            time.sleep(0.01)
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
                results = model.track(current_frame_ref, persist=True, verbose=False, device=state.DEVICE, conf=current_conf)
            else:
                results = model(current_frame_ref, verbose=False, device=state.DEVICE, conf=current_conf)

            infer_dt = time.time() - started_at
            camera_state.stats["last_infer_ms"] = infer_dt * 1000.0
            if infer_dt > 0:
                camera_state.stats["inference_fps"] = 1.0 / infer_dt

            with camera_state.frame_lock:
                camera_state.latest_results = results

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
            # Ignore transient inference errors and continue.
            pass


def yolo_process_render(camera_state, frame):
    """
    Render-only function for the active camera frame.
    """
    with camera_state.frame_lock:
        results = camera_state.latest_results

    if not results:
        return frame, 0, []

    result = results[0]
    num_det = len(result.boxes)
    classes = []
    if result.boxes:
        for cls_value in result.boxes.cls:
            cls_id = int(cls_value)
            classes.append(result.names.get(cls_id, str(cls_id)))

    try:
        annotated_frame = draw_custom_esp(camera_state, frame, results)
    except Exception as exc:
        print(f"[Render] Error for {camera_state.camera_id}: {exc}")
        return frame, num_det, classes

    return annotated_frame, num_det, classes


class VideoStream:
    """
    Read frames in a dedicated thread to avoid RTSP buffer lag.
    Only the most recent frame is kept for processing.
    """

    def __init__(self, src, camera_state):
        self.camera_state = camera_state
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

        if not self.grabbed:
            print(f"[VideoStream] Could not connect to {camera_state.camera_id}.")
        else:
            print(f"[VideoStream] Stream started for {camera_state.camera_id}.")

    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

            if grabbed:
                with self.camera_state.frame_lock:
                    self.camera_state.latest_frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        if self.stream:
            self.stream.release()


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


def gen_frames(camera_id: str):
    """MJPEG generator for a specific camera."""
    camera_state = state.get_camera_state(camera_id)
    if camera_state is None:
        return

    if not camera_state.rtsp_url:
        camera_state.stream_error = "No RTSP configuration found for this camera."
        print(f"[Camera] Missing RTSP configuration for {camera_id}.")
        return

    print(f"[Camera] Connecting to {camera_id}: {camera_state.rtsp_url}")

    with camera_state.frame_lock:
        camera_state.latest_frame = None

    ensure_worker(camera_state)

    vs = VideoStream(camera_state.rtsp_url, camera_state).start()
    time.sleep(1.0)

    if not vs.grabbed:
        camera_state.stream_error = "Could not open RTSP stream."
        print(f"[Camera] Could not open RTSP stream for {camera_id}.")
        return

    print(f"[Camera] Camera connection established for {camera_id}.")
    camera_state.stream_error = ""
    prev_time = time.time()

    try:
        while True:
            ret, frame = vs.read()

            if not ret or frame is None:
                camera_state.stream_error = "Camera stream is unavailable."
                time.sleep(0.01)
                continue

            if config.DISPLAY_SCALE < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=config.DISPLAY_SCALE, fy=config.DISPLAY_SCALE)

            height, width = frame.shape[:2]
            camera_state.stats["resolution"] = f"{width}x{height}"
            camera_state.stats["total_frames"] += 1

            num_det = 0
            classes = []

            if camera_state.yolo_enabled:
                frame, num_det, classes = yolo_process_render(camera_state, frame)

            now = time.time()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            camera_state.stream_error = ""
            camera_state.stats["fps"] = fps
            camera_state.stats["num_detections"] = int(num_det)
            camera_state.stats["classes"] = sorted(set(classes))
            camera_state.stats["last_update"] = now

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY],
            )
            if not ok:
                continue

            jpg_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
    finally:
        vs.release()
        print(f"[Camera] Stream closed for {camera_id}.")
