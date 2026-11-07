""""Test speed": a short background benchmark of one workload on every available device.

YOLO is timed end to end through Ultralytics (pre-processing, inference,
post-processing) on a synthetic 1280x720 frame, exactly like the camera worker
calls it. Face and plate models are timed as raw inference per call. One job at
a time; each device gets a short warm-up and at most ~2.5 s of timed runs.
"""
from __future__ import annotations

import statistics
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .. import config, deps, gpu, state
from . import assignments, devices, ov_runtime

TIME_BUDGET_S = 2.5
MAX_RUNS = 30
WARMUP = 3

_lock = threading.Lock()
_job: Optional[Dict[str, Any]] = None


class BenchmarkError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def status() -> Optional[Dict[str, Any]]:
    with _lock:
        if _job is None:
            return None
        out = {k: v for k, v in _job.items() if not k.startswith("_")}
        out["rows"] = [dict(r) for r in _job["rows"]]
    if out["status"] == "running":
        out["elapsed_s"] = round(time.time() - out["started"], 1)
    return out


def _time(fn: Callable[[], Any]) -> float:
    for _ in range(WARMUP):
        fn()
    ts: List[float] = []
    t_end = time.perf_counter() + TIME_BUDGET_S
    while len(ts) < MAX_RUNS and (len(ts) < 5 or time.perf_counter() < t_end):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return statistics.median(ts)


def _devices_for(workload: str) -> List[str]:
    devs = []
    for d in assignments.WORKLOADS[workload]["devices"]:
        ok, _ = assignments.supports(workload, d)
        if ok and (d != "cpu" or devices.openvino_installed() or workload == "yolo"):
            devs.append(d)
    return devs


def start(workload: str, model: Optional[str] = None, runner: Optional[Callable] = None) -> Dict[str, Any]:
    """Start a benchmark job. ``runner(job, set_row)`` is injectable for tests."""
    global _job
    if workload not in assignments.WORKLOADS:
        raise BenchmarkError("invalid_workload")
    if workload == "yolo":
        if model not in config.AVAILABLE_MODELS:
            raise BenchmarkError("invalid_model")
        if runner is None and not state.YOLO_AVAILABLE:
            raise BenchmarkError("yolo_unavailable")
    with _lock:
        if _job is not None and _job["status"] == "running":
            raise BenchmarkError("busy")
        devs = ["auto"] + _devices_for(workload) if workload != "yolo" else _devices_for(workload)
        _job = {"workload": workload, "model": model if workload == "yolo" else None, "status": "running",
                "stage": "starting", "started": time.time(), "finished": None, "error": None,
                "rows": [{"device": d, "label": _row_label(workload, d), "ms": None, "error": None,
                          "status": "pending", "auto": _is_auto(workload, d)} for d in devs]}
        job = _job
    threading.Thread(target=_run, args=(job, runner or _default_runner), daemon=True,
                     name=f"ComputeBenchmark-{workload}").start()
    return status() or {}


def _is_auto(workload: str, d: str) -> bool:
    if workload == "yolo":
        return d == ("cuda" if state.CUDA_AVAILABLE else "cpu") and (d == "cuda" or not devices.openvino_installed())
    return d == "auto"


def _row_label(workload: str, d: str) -> str:
    if workload == "depth":     # depth package
        from ..depth import bench as depth_bench
        return depth_bench.row_label(d)
    if d == "auto":
        return "Auto (onnxruntime / OpenCV)" if workload == "plates" else "Auto (OpenCV)"
    if d == "cuda":
        return f"{devices.cuda_name() or 'NVIDIA GPU'}"
    if d == "cpu":
        return "CPU (OpenVINO)" if devices.openvino_installed() else "CPU (PyTorch)"
    return devices.LABELS.get(d, d)


def _set_row(job, dev: str, **kw) -> None:
    with _lock:
        for r in job["rows"]:
            if r["device"] == dev:
                r.update(kw)


def _run(job, runner) -> None:
    try:
        runner(job, lambda dev, **kw: _set_row(job, dev, **kw))
        with _lock:
            job.update(status="done", stage="done", finished=time.time())
    except Exception as exc:
        traceback.print_exc(limit=3)
        with _lock:
            job.update(status="failed", stage="failed", error=str(exc)[:300], finished=time.time())
    finally:
        with _lock:
            for r in job["rows"]:
                if r["status"] in ("pending", "running"):
                    r.update(status="skipped")
        if job["status"] == "done":
            result = {"workload": job["workload"], "model": job["model"], "finished": job["finished"],
                      "rows": [{k: r[k] for k in ("device", "label", "ms", "error", "auto")} for r in job["rows"]]}
            assignments.store_benchmark(job["workload"], job["model"] or "_", result)
            try:
                from .. import persistence

                persistence.mark_dirty()
            except Exception:
                pass


# ----------------------------------------------------------------------------- real runners
def _default_runner(job, set_row) -> None:
    {"yolo": _bench_yolo, "faces": _bench_faces, "plates": _bench_plates,
     "depth": _bench_depth}[job["workload"]](job, set_row)


def _bench_depth(job, set_row) -> None:    # depth package
    from ..depth import bench as depth_bench
    depth_bench.run(job, set_row, _time)


def _frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.full((720, 1280, 3), 90, np.uint8)
    img += rng.integers(0, 40, img.shape, dtype=np.uint8)
    return img


def _bench_yolo(job, set_row) -> None:
    from . import yolo as cy
    from ..recognition import engines

    name = job["model"]
    frame = _frame()
    for row in list(job["rows"]):
        dev = row["device"]
        set_row(dev, status="running")
        with _lock:
            job["stage"] = f"{dev}"
        model = None
        try:
            if dev == "cuda":
                src = config.model_path(name)
                model = engines.load_model(name, str(src) if src.is_file() else name)
                if engines.uses_engine(model):
                    set_row(dev, label=f"{devices.cuda_name() or 'NVIDIA GPU'} (TensorRT)")
                else:
                    set_row(dev, label=f"{devices.cuda_name() or 'NVIDIA GPU'} (PyTorch)")
                ul = state.DEVICE
            else:
                model = cy.prepare_model(name, dev, on_stage=lambda st, d=dev: set_row(d, stage=st))
                ul = model._securitycam_device
            with gpu.guard(ul):     # never overlap CUDA work of camera workers (TensorRT stream capture)
                ms = _time(lambda: model(frame, verbose=False, device=ul, conf=0.35))
            set_row(dev, ms=round(ms, 2), status="done")
        except Exception as exc:
            set_row(dev, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}",
                    status="failed")
        finally:
            del model


def _small_auto_faces():
    import cv2

    from ..recognition import faces as F

    ypath, spath = deps.asset_path("face_yunet"), deps.asset_path("face_sface")
    if ypath is None or spath is None:
        raise RuntimeError("face models are still downloading")
    det = cv2.FaceDetectorYN.create(str(ypath), "", (640, 640), F.YUNET_SCORE, F.YUNET_NMS, F.YUNET_TOP_K)
    rec = cv2.FaceRecognizerSF.create(str(spath), "")
    rng = np.random.default_rng(5)
    img = rng.integers(60, 200, (640, 640, 3), dtype=np.uint8)
    crop = rng.integers(60, 200, (112, 112, 3), dtype=np.uint8)
    return lambda: (det.detect(img), rec.feature(crop))


def _bench_faces(job, set_row) -> None:
    ypath, spath = deps.asset_path("face_yunet"), deps.asset_path("face_sface")
    if ypath is None or spath is None:
        raise RuntimeError("Face models are still downloading - try again in a moment.")
    for row in list(job["rows"]):
        dev = row["device"]
        set_row(dev, status="running")
        try:
            if dev == "auto":
                fn = _small_auto_faces()
            else:
                y, _ = ov_runtime.compile_raw(ypath, dev, (1, 3, 640, 640))
                s, _ = ov_runtime.compile_raw(spath, dev, (1, 3, 112, 112))
                ry, rs = y.create_infer_request(), s.create_infer_request()
                rng = np.random.default_rng(5)
                xy = rng.integers(60, 200, (1, 3, 640, 640)).astype(np.float32)
                xs = rng.integers(60, 200, (1, 3, 112, 112)).astype(np.float32)
                fn = lambda: (ry.infer({0: xy}), rs.infer({0: xs}))  # noqa: E731
            set_row(dev, ms=round(_time(fn), 2), status="done")
        except Exception as exc:
            set_row(dev, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}",
                    status="failed")


def _bench_plates(job, set_row) -> None:
    from ..recognition import plates as P

    paths = [deps.asset_path(k) for k in ("plate_detector", "plate_ocr", "plate_ocr_config")]
    if any(p is None for p in paths):
        raise RuntimeError("Plate models are still downloading - try again in a moment.")
    cfg = P._load_yaml(paths[2])
    h, w = int(cfg["img_height"]), int(cfg["img_width"])
    ch = 3 if str(cfg.get("image_color_mode", "grayscale")).lower() == "rgb" else 1
    rng = np.random.default_rng(3)
    xd = (rng.random((1, 3, 384, 384)) * 0.2 + 0.4).astype(np.float32)
    xo = rng.integers(60, 200, (1, h, w, ch), dtype=np.uint8)
    for row in list(job["rows"]):
        dev = row["device"]
        set_row(dev, status="running")
        try:
            if dev == "auto":
                det = P._Net(str(paths[0]), xd)
                ocr = P._Net(str(paths[1]), xo, check_finite=True)
                set_row(dev, label=f"Auto ({det.backend})")
                fn = lambda: (det(xd), ocr(xo))  # noqa: E731
            else:
                d, _ = ov_runtime.compile_raw(paths[0], dev, (1, 3, 384, 384))
                o, _ = ov_runtime.compile_raw(paths[1], dev, (1, h, w, ch))
                rd, ro = d.create_infer_request(), o.create_infer_request()
                fn = lambda: (rd.infer({0: xd}), ro.infer({0: xo}))  # noqa: E731
            set_row(dev, ms=round(_time(fn), 2), status="done")
        except Exception as exc:
            set_row(dev, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}",
                    status="failed")
