"""Process-wide serialization of CUDA work.

TensorRT records CUDA graphs ("stream capture") while it builds or runs an
engine. If any other thread in the process touches the GPU during a capture,
CUDA aborts with ``cudaErrorStreamCaptureUnsupported`` /
``cudaErrorStreamCaptureInvalidated`` and the failed call poisons the work of
every thread involved. Camera workers, video sources, "Test speed" benchmarks
and "Build for GPU" exports all ran CUDA work in their own threads; overlapping
them crashed detection.

Every CUDA inference, benchmark run and engine export takes ``LOCK``. The GPU
executes kernels one after another anyway, so serializing costs very little.
It's re-entrant so nested helpers (e.g. a benchmark that loads a model) can't
deadlock themselves.
"""
from __future__ import annotations

import contextlib
import threading

LOCK = threading.RLock()


def is_cuda(device) -> bool:
    d = str(device).lower()
    return d.startswith("cuda") or d.isdigit()


def guard(device):
    """``with gpu.guard(device):`` - takes the lock only for CUDA devices."""
    return LOCK if is_cuda(device) else contextlib.nullcontext()
