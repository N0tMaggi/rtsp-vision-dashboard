""""Test speed" for the Depth workload (compute tab): raw inference per device.

Times Depth Anything V2 Small at the default input (364 x 210, a 16:9 frame)
on Auto (= whatever Auto resolves to) and on every available device.
"""
from __future__ import annotations

import numpy as np

from . import model

BENCH_SHAPE = (1, 3, 210, 364)


def row_label(dev: str) -> str:
    from ..compute import devices
    if dev == "auto":
        return f"Auto ({devices.LABELS.get(model.auto_device(), 'CPU')})"
    if dev == "cuda":
        return f"{devices.cuda_name() or 'NVIDIA GPU'} (onnxruntime CUDA)"
    if dev == "cpu":
        return "CPU (OpenVINO)" if devices.openvino_installed() else "CPU (onnxruntime)"
    return devices.LABELS.get(dev, dev)


def run(job, set_row, time_fn, factory=None) -> None:
    """Benchmark runner (``compute.benchmark`` calls it with its own timing helper)."""
    from .. import deps
    path = deps.asset_path(model.ASSET)
    if path is None:
        raise RuntimeError("The depth model is still downloading - try again in a moment.")
    build = factory or model.build_backend
    x = (np.random.default_rng(11).random(BENCH_SHAPE) * 2 - 1).astype(np.float32)
    for row in list(job["rows"]):
        dev = row["device"]
        set_row(dev, status="running")
        try:
            real = model.auto_device() if dev == "auto" else dev
            be = build(path, real, BENCH_SHAPE)
            set_row(dev, ms=round(time_fn(lambda: be(x)), 2), status="done")
            del be
        except Exception as exc:
            set_row(dev, error=f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}",
                    status="failed")
