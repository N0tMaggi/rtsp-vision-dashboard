"""Camera streaming, worker thread, and MJPEG generator."""
import threading
import time
import cv2
from . import config, state, deps
from .esp import draw_custom_esp


def yolo_worker():
    """
    Dedicated Worker Thread for Inference.
    Runs as fast as possible, updating 'latest_results'.
    Doesn't block the video feed generation.
    """
    print("[YOLO Worker] Started.")

    while True:
        if not state.yolo_enabled or not state.YOLO_AVAILABLE:
            time.sleep(0.5)
            with state.frame_lock:
                state.latest_results = None
            continue

        with state.frame_lock:
            current_frame_ref = state.latest_frame

        if current_frame_ref is None:
            time.sleep(0.01)
            continue

        model = deps.get_yolo_model()
        if model is None:
            time.sleep(1)
            continue

        do_track = state.esp_settings["tracking"]

        try:
            t0 = time.time()
            current_conf = state.esp_settings.get("confidence_threshold", 0.35)

            if do_track:
                results = model.track(current_frame_ref, persist=True, verbose=False, device=state.DEVICE, conf=current_conf)
            else:
                results = model(current_frame_ref, verbose=False, device=state.DEVICE, conf=current_conf)

            infer_dt = time.time() - t0
            state.stats["last_infer_ms"] = infer_dt * 1000.0
            if infer_dt > 0:
                state.stats["inference_fps"] = 1.0 / infer_dt

            with state.frame_lock:
                state.latest_results = results

        except RuntimeError as e:
            msg = str(e)
            if ("no kernel image is available" in msg or "CUDA error" in msg) and state.DEVICE.startswith("cuda"):
                print("[YOLO Worker] CUDA Crash -> Fallback CPU")
                state.DEVICE = "cpu"
                state.CUDA_AVAILABLE = False
                state.yolo_model = None
            else:
                print(f"[YOLO Worker] Error: {e}")
            time.sleep(1)

        except Exception:
            # Ignore transient errors
            pass


def yolo_process_render(frame):
    """
    Render-Only function.
    Reads 'latest_results' from Worker Thread and draws them on 'frame'.
    Zero Inference Delay on this thread!
    """
    with state.frame_lock:
        results = state.latest_results

    if not results:
        return frame, 0, []

    r = results[0]
    num_det = len(r.boxes)
    classes = []
    if r.boxes:
        for c in r.boxes.cls:
            cls_id = int(c)
            classes.append(r.names.get(cls_id, str(cls_id)))

    try:
        annotated_frame = draw_custom_esp(frame, results)
    except Exception as e:
        print("[Render] Error:", e)
        return frame, num_det, classes

    return annotated_frame, num_det, classes


class VideoStream:
    """
    Read frames in a dedicated thread to avoid buffer lag.
    Only the most recent frame is kept for processing.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

        if not self.grabbed:
            print("[VideoStream] Could not connect to stream.")
        else:
            print("[VideoStream] Stream started.")

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

            if grabbed:
                with state.frame_lock:
                    state.latest_frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        if self.stream:
            self.stream.release()


def gen_frames():
    """MJPEG generator for /video_feed."""
    print("[Camera] Connecting to camera:", config.RTSP_URL)

    state.latest_frame = None

    if not any(t.name == "YoloWorkerThread" for t in threading.enumerate()):
        yt = threading.Thread(target=yolo_worker, name="YoloWorkerThread", daemon=True)
        yt.start()

    vs = VideoStream(config.RTSP_URL).start()

    time.sleep(1.0)

    if not vs.grabbed:
        print("[Camera] Could not open RTSP stream.")
        return

    print("[Camera] Camera connection established.")

    prev_time = time.time()

    try:
        while True:
            ret, frame = vs.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            state.stats["resolution"] = f"{w}x{h}"
            state.stats["total_frames"] += 1

            num_det = 0
            classes = []

            if state.yolo_enabled:
                frame, num_det, classes = yolo_process_render(frame)

            now = time.time()
            dt = now - prev_time
            fps = 1.0 / dt if dt > 0 else 0.0
            prev_time = now

            state.stats["fps"] = fps
            state.stats["num_detections"] = int(num_det)
            state.stats["classes"] = sorted(list(set(classes)))
            state.stats["last_update"] = now

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
    finally:
        vs.release()
        print("[Camera] Stream closed.")
