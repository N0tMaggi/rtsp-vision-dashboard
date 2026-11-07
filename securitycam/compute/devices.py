"""Accelerator discovery: NVIDIA CUDA (torch), Intel NPU / Intel GPU / CPU (OpenVINO).

Discovery runs once in a background thread (creating an OpenVINO ``Core``
loads the NPU/GPU drivers, ~1 s) and is cached; request handlers only read the
cache and never wait for it. OpenVINO is optional: without it the Intel
devices are listed as unavailable together with the install command.
"""
from __future__ import annotations

import importlib.util
import platform
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .. import config, state

OPENVINO_INSTALL_CMD = "python -m pip install openvino"
CACHE_DIR = config.MODELS_DIR / "ov_cache"

# Selectable device ids (besides "auto"), in display order.
DEVICE_IDS = ("cuda", "npu", "igpu", "cpu")
LABELS = {
    "auto": "Auto",
    "cuda": "NVIDIA GPU",
    "npu": "Intel NPU",
    "igpu": "Intel GPU",
    "cpu": "CPU",
}
SHORT = {"cuda": "GPU", "npu": "NPU", "igpu": "iGPU", "cpu": "CPU"}

_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None
_thread: Optional[threading.Thread] = None
_core = None
_core_lock = threading.Lock()


def openvino_installed() -> bool:
    return importlib.util.find_spec("openvino") is not None


def _make_core():
    import openvino as ov

    core = ov.Core()
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(CACHE_DIR)})
    except Exception:
        pass
    return core


# Injectable for tests (returns an object with available_devices / get_property).
CORE_FACTORY: Callable[[], Any] = _make_core


def core():
    """Shared OpenVINO Core (compile cache in models/ov_cache). Raises if OpenVINO is missing."""
    global _core
    with _core_lock:
        if _core is None:
            _core = CORE_FACTORY()
        return _core


def _prop(c, dev: str, name: str, default=None):
    try:
        return c.get_property(dev, name)
    except Exception:
        return default


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "CPU"


def _probe_openvino() -> Dict[str, Any]:
    out: Dict[str, Any] = {"installed": False, "version": None, "error": None, "devices": {}}
    if not openvino_installed():
        return out
    out["installed"] = True
    try:
        import openvino as ov

        out["version"] = str(getattr(ov, "__version__", "") or "").split("-")[0] or None
    except Exception:
        pass
    try:
        c = core()
        for dev in list(c.available_devices):
            name = str(_prop(c, dev, "FULL_DEVICE_NAME", dev))
            dtype = str(_prop(c, dev, "DEVICE_TYPE", ""))
            out["devices"][dev] = {"name": name, "type": dtype}
    except Exception as exc:
        out["error"] = str(exc).splitlines()[0][:200] if str(exc) else type(exc).__name__
    return out


def _pick(ov_devices: Dict[str, Dict[str, str]], prefix: str, want_intel: bool = False) -> Optional[str]:
    cands = [d for d in ov_devices if d == prefix or d.startswith(prefix + ".")]
    if want_intel:
        cands = [d for d in cands if "intel" in ov_devices[d]["name"].lower()]
        # an integrated Intel GPU first, then any Intel GPU (Arc)
        cands.sort(key=lambda d: (0 if "INTEGRATED" in ov_devices[d]["type"].upper() else 1, d))
    return cands[0] if cands else None


def _discover() -> Dict[str, Any]:
    t0 = time.perf_counter()
    ov_info = _probe_openvino()
    ovd = ov_info["devices"]
    npu = _pick(ovd, "NPU")
    igpu = _pick(ovd, "GPU", want_intel=True)
    cpu = _pick(ovd, "CPU")
    return {
        "openvino": ov_info,
        "ov_map": {"npu": npu, "igpu": igpu, "cpu": cpu},
        "seconds": round(time.perf_counter() - t0, 2),
        "at": time.time(),
    }


def _run_discovery() -> None:
    global _cache
    try:
        result = _discover()
    except Exception as exc:  # never let discovery take anything down
        result = {"openvino": {"installed": openvino_installed(), "version": None, "devices": {},
                               "error": str(exc)[:200]},
                  "ov_map": {"npu": None, "igpu": None, "cpu": None}, "seconds": 0.0, "at": time.time()}
    with _lock:
        _cache = result
    for fn in list(_ready_listeners):
        try:
            fn()
        except Exception as exc:
            print(f"[COMPUTE] discovery listener failed: {exc}")


_ready_listeners: list = []


def on_ready(fn) -> None:
    """Call ``fn`` after every finished discovery (assignments may become usable)."""
    _ready_listeners.append(fn)


def start() -> None:
    """Start discovery in the background (idempotent)."""
    global _thread
    with _lock:
        if _cache is not None or (_thread is not None and _thread.is_alive()):
            return
        _thread = threading.Thread(target=_run_discovery, daemon=True, name="ComputeDiscovery")
        _thread.start()


def refresh(sync: bool = False) -> None:
    """Forget the cache and discover again (``sync`` for tests)."""
    global _cache, _core
    with _lock:
        _cache = None
    with _core_lock:
        _core = None
    if sync:
        _run_discovery()
    else:
        start()


def ready() -> bool:
    with _lock:
        return _cache is not None


def wait(timeout: float) -> bool:
    start()
    t = _thread
    if t is not None and t.is_alive():
        t.join(timeout)
    return ready()


def _cached() -> Dict[str, Any]:
    with _lock:
        return dict(_cache) if _cache is not None else {}


def ov_device(dev_id: str) -> Optional[str]:
    """OpenVINO device name ("NPU", "GPU.0", "CPU") for a device id, or None."""
    return (_cached().get("ov_map") or {}).get(dev_id)


def cuda_name() -> Optional[str]:
    if not state.CUDA_AVAILABLE or state.torch is None:
        return None
    try:
        return str(state.torch.cuda.get_device_name(0))
    except Exception:
        return "NVIDIA GPU"


def availability(dev_id: str) -> tuple[bool, Optional[str]]:
    """(available, reason) of a selectable device id (OpenVINO-backed ids need discovery)."""
    if dev_id == "auto":
        return True, None
    if dev_id == "cuda":
        if state.CUDA_AVAILABLE:
            return True, None
        return False, getattr(state, "CUDA_REASON", "") or "No NVIDIA GPU with a working CUDA build of PyTorch."
    if dev_id not in ("npu", "igpu", "cpu"):
        return False, "Unknown device."
    if dev_id == "cpu" and not openvino_installed():
        # CPU without OpenVINO still works through PyTorch (YOLO) / OpenCV (faces, plates).
        return True, None
    if not openvino_installed():
        return False, f"Needs OpenVINO ({OPENVINO_INSTALL_CMD})."
    cache = _cached()
    if not cache:
        return False, "Detecting hardware…"
    ov_info = cache.get("openvino") or {}
    if ov_info.get("error") and not ov_info.get("devices"):
        return False, f"OpenVINO failed to start: {ov_info['error']}"
    if not (cache.get("ov_map") or {}).get(dev_id):
        return False, {"npu": "No Intel NPU found (or its driver is missing).",
                       "igpu": "No Intel GPU found by OpenVINO.",
                       "cpu": "OpenVINO found no CPU device."}[dev_id]
    return True, None


def snapshot() -> Dict[str, Any]:
    """Hardware overview for the API (never blocks)."""
    start()
    cache = _cached()
    ov_info = cache.get("openvino") or {"installed": openvino_installed(), "version": None, "devices": {}, "error": None}
    ov_map = cache.get("ov_map") or {}
    ovd = ov_info.get("devices") or {}
    devices: List[Dict[str, Any]] = []

    ok, reason = availability("cuda")
    devices.append({"id": "cuda", "label": LABELS["cuda"], "name": cuda_name() or "NVIDIA GPU",
                    "available": ok, "reason": reason, "backend": "PyTorch / TensorRT", "selectable": True})
    for dev_id in ("npu", "igpu"):
        ok, reason = availability(dev_id)
        ovname = ov_map.get(dev_id)
        devices.append({"id": dev_id, "label": LABELS[dev_id],
                        "name": (ovd.get(ovname) or {}).get("name") if ovname else None,
                        "ov_device": ovname, "available": ok, "reason": reason, "backend": "OpenVINO",
                        "selectable": True})
    ovcpu = ov_map.get("cpu")
    ok_ov = bool(ovcpu) and openvino_installed()
    devices.append({"id": "cpu", "label": "CPU (OpenVINO)",
                    "name": (ovd.get(ovcpu) or {}).get("name") if ovcpu else _cpu_name(),
                    "ov_device": ovcpu, "available": ok_ov,
                    "reason": None if ok_ov else (availability("npu")[1] if not openvino_installed()
                                                  else "OpenVINO found no CPU device."),
                    "backend": "OpenVINO", "selectable": True})
    devices.append({"id": "cpu_torch", "label": "CPU (PyTorch)",
                    "name": (ovd.get(ovcpu) or {}).get("name") if ovcpu else _cpu_name(),
                    "available": bool(state.YOLO_AVAILABLE),
                    "reason": None if state.YOLO_AVAILABLE else "PyTorch / Ultralytics not installed.",
                    "backend": "PyTorch", "selectable": False})
    return {
        "ready": bool(cache),
        "devices": devices,
        "openvino": {
            "installed": bool(ov_info.get("installed")),
            "version": ov_info.get("version"),
            "error": ov_info.get("error"),
            "install_cmd": None if ov_info.get("installed") else OPENVINO_INSTALL_CMD,
        },
    }
