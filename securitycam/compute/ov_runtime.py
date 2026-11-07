"""OpenVINO plumbing: YOLO export, compile cache, and small-model networks.

* ``export_yolo(name)`` exports a model once into ``models/<stem>_openvino_model/``
  (FP16, static ``config.YOLO_IMG_SIZE`` square input) and reuses it afterwards.
* ``enable_compile_cache()`` makes every ``openvino.Core`` in this process use
  ``models/ov_cache`` (Ultralytics creates its own Core per model), so the NPU
  and the Intel GPU compile a model only once.
* ``OVNet`` wraps a compiled model with one infer request per thread;
  ``NetPool`` compiles small models (face / plate) in the background so no
  render or request thread ever waits for a compile.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .. import config
from . import devices

_export_locks: Dict[str, threading.Lock] = {}
_export_guard = threading.Lock()
_cache_patched = False


def enable_compile_cache() -> None:
    """Route every ``openvino.Core()`` through a subclass that sets CACHE_DIR (idempotent)."""
    global _cache_patched
    if _cache_patched:
        return
    try:
        import openvino as ov

        base = ov.Core
        cache = str(devices.CACHE_DIR)

        class _CachedCore(base):  # type: ignore[misc, valid-type]
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                try:
                    self.set_property({"CACHE_DIR": cache})
                except Exception:
                    pass

        devices.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ov.Core = _CachedCore
        _cache_patched = True
    except Exception as exc:
        print(f"[COMPUTE] OpenVINO compile cache unavailable: {exc}")
        _cache_patched = True


# ----------------------------------------------------------------------------- YOLO export
def export_dir(name: str) -> Path:
    return config.MODELS_DIR / f"{Path(name).stem}_openvino_model"


def _export_ok(name: str) -> bool:
    d = export_dir(name)
    xml = d / f"{Path(name).stem}.xml"
    if not xml.is_file() or not xml.with_suffix(".bin").is_file():
        return False
    meta = d / "metadata.yaml"
    if meta.is_file():
        try:
            text = meta.read_text(encoding="utf-8")
            # "imgsz:\n- 640\n- 640" -> exported for another input size -> export again
            if "imgsz:" in text:
                tail = text.split("imgsz:", 1)[1].split("\n")[1:3]
                sizes = [int(t.strip(" -")) for t in tail if t.strip(" -").isdigit()]
                if sizes and any(s != config.YOLO_IMG_SIZE for s in sizes):
                    return False
            if "dynamic: true" in text:
                return False
        except (OSError, ValueError):
            pass
    return True


def exported(name: str) -> bool:
    return _export_ok(name)


def export_yolo(name: str, on_stage: Optional[Callable[[str], None]] = None) -> Path:
    """Export ``name`` for OpenVINO once (blocking; call from a background thread)."""
    with _export_guard:
        lock = _export_locks.setdefault(name, threading.Lock())
    with lock:
        if _export_ok(name):
            return export_dir(name)
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        src = config.model_path(name)
        if not src.is_file():
            if on_stage:
                on_stage("downloading")
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            attempt_download_asset(src)
        if on_stage:
            on_stage("exporting")
        print(f"[COMPUTE] Exporting {name} for OpenVINO (one time)...")
        out = YOLO(str(src)).export(format="openvino", half=True, imgsz=config.YOLO_IMG_SIZE,
                                    dynamic=False, verbose=False)
        out = Path(out)
        target = export_dir(name)
        if out.resolve() != target.resolve() and out.is_dir():
            import shutil

            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(out), str(target))
        if not _export_ok(name):
            raise RuntimeError("OpenVINO export finished without a model")
        print(f"[COMPUTE] {target.name} ready.")
        return target


def ultralytics_device(dev: str) -> str:
    """Ultralytics device string for an OpenVINO-backed device id ("intel:npu", "intel:gpu.0", ...)."""
    ovd = devices.ov_device(dev) or {"npu": "NPU", "igpu": "GPU", "cpu": "CPU"}[dev]
    return f"intel:{ovd.lower()}"


# ----------------------------------------------------------------------------- small models
class OVNet:
    """Compiled OpenVINO model; thread-safe through one infer request per thread.

    ``__call__`` mirrors ``plates._Net``: first output as a numpy array.
    ``outputs(x)`` returns all outputs by name. On an inference error
    ``on_error`` is called once (the workload falls back to auto) and the
    exception is re-raised for the caller's own handling.
    """

    def __init__(self, compiled, device: str, input_dtype=np.float32, on_error: Optional[Callable[[Exception], None]] = None):
        self._cm = compiled
        self.device = device
        self.backend = f"openvino:{device}"
        self.input_dtype = input_dtype
        self._local = threading.local()
        self._on_error = on_error
        self._names = []          # all tensor names per output (get_any_name() differs per device)
        for i, o in enumerate(compiled.outputs):
            try:
                names = set(o.get_names()) or {o.get_any_name()}
            except Exception:
                names = set()
            self._names.append(names | {str(i)})

    def _req(self):
        r = getattr(self._local, "req", None)
        if r is None:
            r = self._local.req = self._cm.create_infer_request()
        return r

    def _run(self, x: np.ndarray):
        try:
            req = self._req()
            req.infer({0: np.ascontiguousarray(x, dtype=self.input_dtype)})
            return [np.array(req.get_output_tensor(i).data) for i in range(len(self._names))]
        except Exception as exc:
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    pass
            raise

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._run(x)[0]

    def outputs(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for names, arr in zip(self._names, self._run(x)):
            for n in names:
                out[n] = arr
        return out


Key = Tuple[str, str, Tuple[int, ...]]


class NetPool:
    """Background compile of (model path, device, static input shape) -> OVNet."""

    MAX_ENTRIES = 24

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[Key, Dict[str, Any]] = {}

    def get(self, workload: str, path: Path, dev: str, shape: Sequence[int],
            on_fail: Callable[[str, str], None], on_ready: Optional[Callable[[], None]] = None) -> Optional[OVNet]:
        """Ready net, or None while it compiles / after it failed (caller then uses auto)."""
        key: Key = (str(path), dev, tuple(int(s) for s in shape))
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                return e.get("net") if e["status"] == "ready" else None
            if len(self._entries) >= self.MAX_ENTRIES:
                return None
            self._entries[key] = {"status": "compiling", "started": time.time(), "workload": workload}
        threading.Thread(target=self._compile, args=(key, workload, on_fail, on_ready), daemon=True,
                         name=f"OVCompile-{workload}-{dev}").start()
        return None

    def _compile(self, key: Key, workload: str, on_fail, on_ready) -> None:
        path, dev, shape = key
        try:
            ovname = devices.ov_device(dev)
            if not ovname:
                raise RuntimeError(f"{devices.LABELS.get(dev, dev)} is not available")
            c = devices.core()
            model = c.read_model(path)
            inp = model.inputs[0]
            if list(inp.get_partial_shape()) != list(shape) or inp.get_partial_shape().is_dynamic:
                model.reshape({inp.get_any_name(): list(shape)})
            t0 = time.perf_counter()
            compiled = c.compile_model(model, ovname, {"PERFORMANCE_HINT": "LATENCY"})
            import openvino as ov

            et = compiled.inputs[0].get_element_type()
            dtype = np.uint8 if et == ov.Type.u8 else (np.float16 if et == ov.Type.f16 else np.float32)
            net = OVNet(compiled, dev, dtype, on_error=lambda exc: on_fail(workload, f"{type(exc).__name__}: {exc}"))
            # one dummy run: catches devices that compile but cannot execute the graph
            # (or return garbage, e.g. inf from an op the device gets wrong)
            outs = net._run(np.zeros(shape, dtype))
            if not all(np.all(np.isfinite(o)) for o in outs if np.issubdtype(o.dtype, np.floating)):
                raise RuntimeError(f"{devices.LABELS.get(dev, dev)} returns invalid (non-finite) values for this model")
            with self._lock:
                self._entries[key] = {"status": "ready", "net": net, "workload": workload,
                                      "compile_s": round(time.perf_counter() - t0, 2)}
            if on_ready is not None:
                on_ready()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240] if str(exc) else ''}"
            with self._lock:
                self._entries[key] = {"status": "failed", "error": msg, "workload": workload}
            if not isinstance(exc, RuntimeError) or "not available" not in str(exc):
                traceback.print_exc(limit=2)
            on_fail(workload, msg)

    def status(self, workload: str) -> Dict[str, int]:
        out = {"compiling": 0, "ready": 0, "failed": 0}
        with self._lock:
            for e in self._entries.values():
                if e.get("workload") == workload:
                    out[e["status"]] = out.get(e["status"], 0) + 1
        return out

    def clear(self, workload: Optional[str] = None) -> None:
        with self._lock:
            for k in [k for k, e in self._entries.items()
                      if (workload is None or e.get("workload") == workload) and e["status"] != "compiling"]:
                del self._entries[k]


POOL = NetPool()


def compile_raw(path: Path, dev: str, shape: Optional[Sequence[int]] = None) -> Tuple[Any, float]:
    """Synchronous compile for benchmarks: (compiled model, compile seconds)."""
    c = devices.core()
    model = c.read_model(str(path))
    if shape is not None:
        inp = model.inputs[0]
        model.reshape({inp.get_any_name(): list(shape)})
    t0 = time.perf_counter()
    compiled = c.compile_model(model, devices.ov_device(dev), {"PERFORMANCE_HINT": "LATENCY"})
    return compiled, time.perf_counter() - t0
