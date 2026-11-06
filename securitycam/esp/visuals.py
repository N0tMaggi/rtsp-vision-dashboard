"""ESP "visuals" package: heatmap, loitering ramp, night vision (pipeline glue).

Kept out of ``EspSettings`` on purpose: ``VisualSettings`` reads its own keys
from the same settings mapping, so the core snapshot stays untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from .. import config
from . import heatmap, nightvision, trackstats
from .colors import BGR, mix

LOITER_RED: BGR = (78, 90, 255)          # #ff5a4e (theme destructive), BGR
LOITER_LABEL_MIN_S = 3.0                 # dwell shown in the label pill from this age on


def _bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _num(v: Any, default: float, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:
        return default
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class VisualSettings:
    heatmap: bool
    heatmap_window: str
    heatmap_opacity: float
    loiter: bool
    loiter_seconds: float
    night_vision: str
    night_vision_scanlines: bool

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "VisualSettings":
        D = config.DEFAULT_ESP_SETTINGS
        g = lambda k: raw.get(k, D.get(k))  # noqa: E731
        window = str(g("heatmap_window") or heatmap.DEFAULT_WINDOW).strip().lower()
        nv = str(g("night_vision") or "off").strip().lower()
        return cls(
            heatmap=_bool(g("heatmap"), False),
            heatmap_window=window if window in heatmap.WINDOWS else heatmap.DEFAULT_WINDOW,
            heatmap_opacity=_num(g("heatmap_opacity"), 0.55, 0.0, 1.0),
            loiter=_bool(g("loiter"), False),
            loiter_seconds=_num(g("loiter_seconds"), 120.0, 5.0, 3600.0),
            night_vision=nv if nv in nightvision.MODES else "off",
            night_vision_scanlines=_bool(g("night_vision_scanlines"), False),
        )


def update_state(camera_state, mst, t_now: float) -> dict:
    """Per-frame state update (ESP lock held): track stats + heatmap accumulation."""
    stats = trackstats.update(camera_state, mst, t_now)
    heatmap.accumulate(camera_state, mst, t_now)
    return stats


def frame_effects(frame, camera_state, vs: VisualSettings, t_now: float, frame_idx: int,
                  skip_night: bool = False) -> None:
    """Night vision, then heat overlay (both under every ESP stroke)."""
    if not skip_night and vs.night_vision != "off" and \
            nightvision.is_active(camera_state, vs.night_vision, frame, t_now):
        nightvision.apply(frame, frame_idx, vs.night_vision_scanlines)
    if vs.heatmap:
        heatmap.draw(frame, camera_state, vs.heatmap_window, vs.heatmap_opacity, t_now)


def loiter_level(vs: VisualSettings, stats: dict, key, t_now: float) -> float:
    if not vs.loiter:
        return 0.0
    dwell = trackstats.dwell_seconds(stats.get(key), t_now)
    return max(0.0, min(1.0, dwell / vs.loiter_seconds))


def loiter_palettes(palettes: list, detections: list, vs: VisualSettings, stats: dict, t_now: float) -> list:
    """Ramp each box color toward red with its dwell time (no-op when loiter is off)."""
    if not vs.loiter:
        return palettes
    out = []
    for det, pal in zip(detections, palettes):
        f = loiter_level(vs, stats, det.track_key, t_now)
        if f > 0.0:
            # quantize so the color (and any cached glow) changes in visible steps only
            f = round(f * 32) / 32
            pal = type(pal)(box=mix(pal.box, LOITER_RED, f), text=pal.text,
                            fill=mix(pal.fill, LOITER_RED, f), skeleton=pal.skeleton)
        out.append(pal)
    return out


def loiter_runs(runs: List, det, vs: VisualSettings, stats: dict, t_now: float, text_color: BGR) -> List:
    """Append the dwell time ("2:14") to a label's runs."""
    if not vs.loiter:
        return runs
    dwell = trackstats.dwell_seconds(stats.get(det.track_key), t_now)
    if dwell < LOITER_LABEL_MIN_S:
        return runs
    f = min(1.0, dwell / vs.loiter_seconds)
    color = mix(text_color, LOITER_RED, f)
    sep = [("  ", False, text_color)] if runs else []
    return runs + sep + [(trackstats.format_dwell(dwell), f >= 1.0, color)]

