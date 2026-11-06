"""Settings for alarms, events, clips and notifications (read from the environment / .env)."""
from __future__ import annotations

import os
from pathlib import Path

from .. import config
from ..config import _env_float, _env_int

DATA_DIR = Path(os.getenv("SECURITYCAM_DATA_DIR", "").strip() or (config.PROJECT_ROOT / "data"))
ALERTS_FILE = DATA_DIR / "alerts.json"
EVENTS_DIR = DATA_DIR / "events"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

# Event clips: seconds before / after the trigger, recording rate and size limits.
CLIP_PRE_S = _env_float("SECURITYCAM_CLIP_PRE_S", 10.0, minimum=0.0, maximum=30.0)
CLIP_POST_S = _env_float("SECURITYCAM_CLIP_POST_S", 10.0, minimum=1.0, maximum=30.0)
CLIP_FPS = _env_float("SECURITYCAM_CLIP_FPS", 10.0, minimum=2.0, maximum=30.0)
# Upper bound for the per-camera ring buffer of rendered JPEG frames (MB).
CLIP_BUFFER_MB = _env_int("SECURITYCAM_CLIP_BUFFER_MB", 96, minimum=8, maximum=2048)
CLIP_MAX_WIDTH = _env_int("SECURITYCAM_CLIP_MAX_WIDTH", 1280, minimum=320, maximum=3840)
# auto (webm/VP8 -> mp4/H.264 -> JPEG sequence), webm, mp4 or jpeg
CLIP_FORMAT = (os.getenv("SECURITYCAM_CLIP_FORMAT", "auto").strip().lower() or "auto")

# Retention of stored events (oldest are removed first). 0 = unlimited age.
EVENTS_MAX_COUNT = _env_int("SECURITYCAM_EVENTS_MAX_COUNT", 300, minimum=1, maximum=100000)
EVENTS_MAX_AGE_DAYS = _env_float("SECURITYCAM_EVENTS_MAX_AGE_DAYS", 14.0, minimum=0.0, maximum=3650.0)

# A rule/tripwire creates at most one event per cooldown; an alarm stays visible this long.
ALARM_COOLDOWN_S = _env_float("SECURITYCAM_ALARM_COOLDOWN_S", 30.0, minimum=0.0, maximum=86400.0)
ALARM_HOLD_S = _env_float("SECURITYCAM_ALARM_HOLD_S", 4.0, minimum=0.5, maximum=120.0)

# Notification channels (all optional). Never logged.
NOTIFY_COOLDOWN_S = _env_float("SECURITYCAM_NOTIFY_COOLDOWN_S", 60.0, minimum=0.0, maximum=86400.0)
NTFY_URL = os.getenv("SECURITYCAM_NTFY_URL", "").strip()
NTFY_TOKEN = os.getenv("SECURITYCAM_NTFY_TOKEN", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("SECURITYCAM_TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("SECURITYCAM_TELEGRAM_CHAT_ID", "").strip()
DISCORD_WEBHOOK_URL = os.getenv("SECURITYCAM_DISCORD_WEBHOOK_URL", "").strip()
# Telegram API base (overridable for tests / self-hosted bot API servers).
TELEGRAM_API = os.getenv("SECURITYCAM_TELEGRAM_API", "https://api.telegram.org").strip().rstrip("/")

# COCO classes that count as "objects that can be left behind".
ABANDON_CLASSES = frozenset(
    c.strip().lower()
    for c in os.getenv("SECURITYCAM_ABANDON_CLASSES", "backpack,handbag,suitcase,umbrella,laptop").split(",")
    if c.strip()
)
