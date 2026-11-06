"""ESP drawing for alarms & events: alarm zones, tripwires with counters, abandoned objects.

Called from ``pipeline.draw_custom_esp``:

* ``update``      (inside the ESP lock) runs the alarm engine and returns the zones
                  with alarm styling applied (red, "ALARM" in the pill)
* ``draw_under``  (after the zones) pulsing alarm outlines and tripwire lines
* ``draw_over``   (after the detection labels) tripwire counters and "NEW SINCE" pills

Nothing in here may raise into the render path.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np

from . import renderers as R
from .colors import ANTHRACITE, BGR, mix, parse_hex
from .text import has_truetype, separator

ALARM_HEX = "#ff5a4e"
ALARM_BGR: BGR = (78, 90, 255)
PULSE_HZ = 1.6


def update(camera_state, detections, w: int, h: int, t: float, zones: list) -> Tuple[list, object]:
    """Run the alarm engine; returns (zones to draw, frame info)."""
    try:
        from ..alerts import engine
        info = engine.update(camera_state, detections, w, h, t, zones,
                             getattr(camera_state, "esp_settings", None) or {})
    except Exception as exc:  # pragma: no cover - never break rendering
        print(f"[Alerts] engine error: {exc.__class__.__name__}: {exc}")
        return zones, None
    if not info.alarm_zones or not zones:
        return zones, info
    out = []
    for z in zones:
        if z.get("id") in info.alarm_zones:
            z = dict(z, color=ALARM_HEX, label=f"ALARM {separator()} {z.get('label', '')}", _alarm=True)
        out.append(z)
    return out, info


def _pulse(t: float) -> float:
    return 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * PULSE_HZ)


def _arrow_geometry(a, b, st: R.Style):
    """Midpoint, unit normal toward the positive ("in") side, arrow length."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
    return (mx, my), (nx, ny), max(14.0, 26.0 * st.ui)


def draw_under(frame: np.ndarray, info, zones: list, st: R.Style, settings) -> None:
    if info is None:
        return
    try:
        h, w = frame.shape[:2]
        p = _pulse(info.t)
        # pulsing outer outline around alarmed zones
        for z in zones:
            if not z.get("_alarm"):
                continue
            x1, y1, x2, y2 = z["box"]
            off = st.t + int(round((2 + 6 * p) * st.ui))
            rect = np.array(((x1 * w - off, y1 * h - off), (x2 * w + off, y1 * h - off),
                             (x2 * w + off, y2 * h + off), (x1 * w - off, y2 * h + off)), np.int32)
            color = mix(ALARM_BGR, (255, 255, 255), 0.35 * p)
            cv2.polylines(frame, [rect], True, ANTHRACITE, st.t + 2, R.AA)
            cv2.polylines(frame, [rect], True, color, st.t, R.AA)

        if not settings.get("tripwires", True):
            return
        for wd, _n_in, _n_out, alarm in info.wires:
            base = ALARM_BGR if alarm else (parse_hex(wd.get("color", "")) or settings_primary(settings))
            color = mix(base, (255, 255, 255), 0.35 * p) if alarm else base
            a = (wd["a"][0] * w, wd["a"][1] * h)
            b = (wd["b"][0] * w, wd["b"][1] * h)
            pa, pb = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
            cv2.line(frame, pa, pb, ANTHRACITE, st.t + 3, R.AA)
            cv2.line(frame, pa, pb, color, st.t + 1, R.AA)
            r = max(3, int(round(4 * st.ui)))
            for q in (pa, pb):
                cv2.circle(frame, q, r + 1, ANTHRACITE, -1, R.AA)
                cv2.circle(frame, q, r, color, -1, R.AA)
            (mx, my), (nx, ny), alen = _arrow_geometry(a, b, st)
            tail = (int(mx), int(my))
            tip = (int(mx + nx * alen), int(my + ny * alen))
            cv2.arrowedLine(frame, tail, tip, ANTHRACITE, st.t + 3, R.AA, tipLength=0.4)
            cv2.arrowedLine(frame, tail, tip, color, st.t + 1, R.AA, tipLength=0.4)
    except Exception as exc:  # pragma: no cover
        print(f"[Alerts] draw error: {exc.__class__.__name__}")


def settings_primary(settings) -> BGR:
    return parse_hex(str(settings.get("color_primary", ""))) or (31, 210, 255)


def _arrows() -> Tuple[str, str]:
    return ("↑", "↓") if has_truetype() else ("IN", "OUT")


def draw_over(frame: np.ndarray, info, st: R.Style, settings, text_color: BGR, background: bool,
              placed: List[tuple]) -> None:
    if info is None:
        return
    try:
        h, w = frame.shape[:2]
        dim = mix(text_color, ANTHRACITE, 0.45)
        if settings.get("tripwires", True):
            up, down = _arrows()
            for wd, n_in, n_out, alarm in info.wires:
                accent = ALARM_BGR if alarm else (parse_hex(wd.get("color", "")) or settings_primary(settings))
                runs = []
                if alarm:
                    runs += [("ALARM", True, ALARM_BGR), ("  ", False, dim)]
                runs += [(str(wd.get("label", "")), True, text_color), ("  ", False, dim),
                         (f"{up} {n_in}", False, text_color), ("  ", False, dim),
                         (f"{down} {n_out}", False, text_color)]
                tag = R.make_tag(runs, st, accent, 1.0, background)
                if tag is None:
                    continue
                a = (wd["a"][0] * w, wd["a"][1] * h)
                b = (wd["b"][0] * w, wd["b"][1] * h)
                (mx, my), (nx, ny), alen = _arrow_geometry(a, b, st)
                # on the side opposite the arrow, pushed out by the pill's extent along the
                # normal, so it covers neither the arrow nor the line
                reach = abs(nx) * tag.w * 0.5 + abs(ny) * tag.h * 0.5 + st.gap * 2
                cx, cy = mx - nx * reach, my - ny * reach
                box = (int(cx - tag.w / 2), int(cy - tag.h / 2), int(cx + tag.w / 2), int(cy + tag.h / 2))
                tag.x = min(max(0, box[0]), max(0, w - tag.w))
                tag.y = min(max(0, box[1]), max(0, h - tag.h))
                placed.append((tag.x, tag.y, tag.x + tag.w, tag.y + tag.h))
                R.draw_tag(frame, tag)

        for box, label in info.abandoned:
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
            pad = st.gap
            segs = R._dashed_rect(x1 - pad, y1 - pad, x2 + pad, y2 + pad,
                                  max(4, int(8 * st.ui)), max(3, int(5 * st.ui)))
            if segs:
                cv2.polylines(frame, segs, False, ANTHRACITE, st.t + 2, R.AA)
                cv2.polylines(frame, segs, False, ALARM_BGR, st.t, R.AA)
            tag = R.make_tag([(label, True, text_color)], st, ALARM_BGR, 1.0, background)
            if tag is not None:
                R.place_tag(tag, (x1 - pad, y1 - pad, x2 + pad, y2 + pad), w, h, st, placed)
                R.draw_tag(frame, tag)
    except Exception as exc:  # pragma: no cover
        print(f"[Alerts] draw error: {exc.__class__.__name__}")
