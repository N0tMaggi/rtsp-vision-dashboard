"""Settings for the Enhance view (read from the environment / .env)."""
from __future__ import annotations

import os
from pathlib import Path

from .. import config
from ..config import _env_float, _env_int

DATA_DIR = Path(os.getenv("SECURITYCAM_DATA_DIR", "").strip() or (config.PROJECT_ROOT / "data"))
ENHANCED_DIR = DATA_DIR / "enhanced"

# Retention: the oldest results are removed first. 0 = unlimited age.
MAX_COUNT = _env_int("SECURITYCAM_ENHANCE_MAX_COUNT", 40, minimum=1, maximum=10000)
MAX_AGE_DAYS = _env_float("SECURITYCAM_ENHANCE_MAX_AGE_DAYS", 30.0, minimum=0.0, maximum=3650.0)

# auto (CUDA -> OpenVINO CPU -> PyTorch CPU), cuda, openvino or cpu
DEVICE = (os.getenv("SECURITYCAM_ENHANCE_DEVICE", "auto").strip().lower() or "auto")

# Tile edge (input px) for the neural models: bounds VRAM / RAM per step. Each tile
# takes the process-wide GPU lock on its own, so camera inference keeps running
# between tiles.
TILE = _env_int("SECURITYCAM_ENHANCE_TILE", 384, minimum=64, maximum=2048)
TILE_PAD = 16

# Upper bound for the output image (megapixels); larger requests are refused.
MAX_OUTPUT_MP = _env_float("SECURITYCAM_ENHANCE_MAX_OUTPUT_MP", 40.0, minimum=1.0, maximum=200.0)

# How long a job waits for its model download before falling back to Lanczos.
DOWNLOAD_WAIT_S = _env_float("SECURITYCAM_ENHANCE_DOWNLOAD_WAIT_S", 180.0, minimum=1.0, maximum=3600.0)

# Queued (not yet running) jobs at most; more are refused with 429.
MAX_QUEUE = _env_int("SECURITYCAM_ENHANCE_MAX_QUEUE", 8, minimum=1, maximum=100)
