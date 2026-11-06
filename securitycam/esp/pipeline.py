"""ESP overlay orchestration: snapshot settings, update state under a lock, then draw."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import List, Optional

import cv2
import numpy as np

from . import alerts_render as AR
from . import motion
from . import renderers as R
from .colors import ANTHRACITE, DetectionColors, center_dot_color, mix, resolve_detection_colors
from .detections import Detection, extract_detections
from .sensors import update_faces, update_hands
from .colors import parse_hex
from .settings import EspSettings

_LOCK_INIT_GUARD = threading.Lock()
_FADE_OPAQUE = 0.985


def _esp_lock(camera_state) -> threading.RLock:
    """Per-camera lock guarding track/face/hand state (created lazily for duck-typed states)."""
    lock = getattr(camera_state, "esp_lock", None)
    if lock is None:
        with _LOCK_INIT_GUARD:
            lock = getattr(camera_state, "esp_lock", None)
            if lock is None:
                lock = threading.RLock()
                camera_state.esp_lock = lock
    return lock


def _track_history(camera_state) -> dict:
    hist = getattr(camera_state, "track_history", None)
    if hist is None:
        hist = camera_state.track_history = {}
    return hist


def _closest_to_center(detections: List[Detection], w: int, h: int) -> Optional[int]:
    if not detections:
        return None
    cx, cy = w // 2, h // 2
    return min(range(len(detections)),
               key=lambda i: (detections[i].cx - cx) ** 2 + (detections[i].cy - cy) ** 2)


def _scale_detections(dets: List[Detection], k: float) -> List[Detection]:
    if k == 1.0:
        return dets
    out = []
    for d in dets:
        x1, y1, x2, y2 = int(d.x1 * k), int(d.y1 * k), int(d.x2 * k), int(d.y2 * k)
        kp = (d.keypoints * k).astype(np.int32) if d.keypoints is not None else None
        out.append(replace(d, x1=x1, y1=y1, x2=x2, y2=y2, cx=(x1 + x2) // 2, cy=(y1 + y2) // 2, keypoints=kp))
    return out


def _update_tracks(camera_state, s: EspSettings, results, frame_ts: float,
                   results_ts: Optional[float], coord_scale: float) -> List[Detection]:
    """Fold new inference results into the motion model and step it to ``frame_ts``.

    Must be called with the ESP lock held. Extraction (tensor -> numpy) only
    happens when ``results`` is a new object (or the filter settings changed),
    not on every rendered frame.
    """
    st = motion.get_state(camera_state)
    history = _track_history(camera_state)
    signature = (s.person_only, s.needs_keypoints, s.confidence_threshold, coord_scale)
    if results is not None:
        if results is not st.last_results or signature != st.signature:
            is_new = results is not st.last_results
            dets = _scale_detections(extract_detections(results, s), coord_scale)
            t_obs = results_ts if results_ts is not None else frame_ts
            if not is_new:
                # Same result re-read with other filters: don't fake a new observation time.
                t_obs = max((tr.obs_t for tr in st.tracks.values()), default=t_obs)
            max_hist = max(2, int(getattr(camera_state, "max_track_history", 30)))
            motion.ingest(st, dets, t_obs, frame_ts, history, max_hist)
            st.last_results = results
            st.signature = signature
            camera_state.esp_last_results = results
    elif st.last_results is not None or st.tracks:
        motion.mark_all_missing(st, frame_ts)
        st.last_results = None
    return motion.step(st, frame_ts, s.smoothing, history, s.tracers)


def draw_custom_esp(camera_state, frame, results, frame_ts: Optional[float] = None,
                    results_ts: Optional[float] = None, coord_scale: float = 1.0):
    """Draw the game-style ESP overlay onto ``frame`` (in place) and return it.

    ``results`` may be ``None`` (no inference yet / YOLO off); existing boxes then
    fade out and frame-level effects (thermal, face blur, crosshair) still apply.
    ``frame_ts`` / ``results_ts`` are ``time.perf_counter()`` capture times of the
    displayed frame and of the frame the results were computed on; they drive
    box extrapolation. Both default to "now". ``coord_scale`` maps result
    coordinates onto ``frame`` (e.g. when the displayed frame was downscaled).
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        return frame

    s = EspSettings.from_mapping(camera_state.esp_settings)
    h, w = frame.shape[:2]
    t_frame = time.perf_counter() if frame_ts is None else float(frame_ts)
    now = time.time()  # wall clock, only for the rgb-mode color cycle
    frame_idx = int(camera_state.stats.get("total_frames", 0))
    st = R.style_for(w, h, s.line_thickness)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if s.thermal else None

    # ---- stateful part (short critical section) -----------------------------------
    with _esp_lock(camera_state):
        faces = update_faces(camera_state, frame, s, frame_idx, gray)
        hands = update_hands(camera_state, frame, s, frame_idx)
        detections = _update_tracks(camera_state, s, results, t_frame, results_ts, float(coord_scale))
        mst = motion.get_state(camera_state)
        mst.frame_w, mst.frame_h = w, h
        locked = getattr(camera_state, "esp_locked", None)
        if locked is not None and locked not in mst.tracks:
            camera_state.esp_locked = locked = None   # locked track was pruned
        zones, alert_info = AR.update(camera_state, detections, w, h, t_frame,
                                      list(getattr(camera_state, "esp_zones", None) or ()))
        zones = zones if s.zones else []

    # ---- frame-level effects (on the raw image, before any overlay) -----------------
    if s.face_blur and faces:
        R.anonymize_faces(frame, faces, s.face_blur_mode, s.face_blur_strength)
    if s.thermal:
        R.apply_thermal(frame, gray)
    inset = R.make_face_inset(frame, faces, s.face_zoom_scale, st.margin) if s.face_zoom else None
    if s.center_dot:
        R.draw_center_dot(frame, center_dot_color(s, now), st)

    placed: List[tuple] = []
    tags = []

    # ---- zones (under the detections; their pills are drawn on top later) ----------
    zone_tags = []
    for zone in zones:
        try:
            zx1, zy1, zx2, zy2 = zone["box"]
            zbox = (int(zx1 * w), int(zy1 * h), int(zx2 * w), int(zy2 * h))
            count = sum(1 for d in detections
                        if zbox[0] <= d.cx <= zbox[2] and zbox[1] <= d.cy <= zbox[3])
            ztag = R.draw_zone(frame, zbox, parse_hex(zone.get("color", "")) or s.color_primary,
                        str(zone.get("label", "")), count, s.color_text, st, s.label_background, placed)
            if ztag is not None:
                zone_tags.append(ztag)
        except (KeyError, TypeError, ValueError):
            continue
    AR.draw_under(frame, alert_info, zones, st, camera_state.esp_settings)

    # ---- detections ----------------------------------------------------------------
    if detections:
        closest = _closest_to_center(detections, w, h) if s.highlight_center else None
        palettes = [
            resolve_detection_colors(s, i, d.obj_id, d.cls_id, i == closest, now)
            for i, d in enumerate(detections)
        ]
        lock_idx = None
        if locked is not None:
            lock_idx = next((i for i, d in enumerate(detections) if d.track_key == locked), None)
            # dim everything that is not locked (color only: no extra blending cost)
            palettes = [
                pal if i == lock_idx else DetectionColors(
                    box=mix(pal.box, ANTHRACITE, 0.45), text=mix(pal.text, ANTHRACITE, 0.3),
                    fill=pal.fill, skeleton=mix(pal.skeleton, ANTHRACITE, 0.45))
                for i, pal in enumerate(palettes)
            ]

        # Pass 1: translucent fills, so later outlines/labels stay crisp.
        if s.boxes and (s.fill_box or s.chams):
            for det, pal in zip(detections, palettes):
                if s.fill_box:
                    R.draw_fill(frame, det, pal.fill)
                if s.chams:
                    R.draw_chams(frame, det, s.chams_color, s.chams_opacity)

        draws_body = s.boxes or s.skeleton or s.head or closest is not None

        # Pass 2: strokes. Fading tracks are drawn, then blended back toward the
        # pre-draw ROI; long lines (snaplines/tracers/motion) wait until opaque.
        for i, (det, pal) in enumerate(zip(detections, palettes)):
            opaque = det.alpha >= _FADE_OPAQUE
            backup = rect = None
            if opaque:
                if s.snaplines:
                    R.draw_snapline(frame, det, pal.box, st)
                if s.tracers and det.history:
                    R.draw_tracer(frame, det.history, (det.cx, det.cy), s.color_tracers, st)
                if s.velocity or s.prediction:
                    R.draw_motion(frame, det, s, st)
            elif draws_body:
                x0, y0, x1, y1 = R.box_extent(det, s.box_style, st)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                rect = (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
                backup = frame[y0:y1, x0:x1].copy() if rect else None

            if i == lock_idx:
                R.draw_lock(frame, det, s.box_style, s.color_primary, st, t_frame, s.center_dot)
            elif s.boxes:
                R.draw_box(frame, det, s.box_style, pal.box, st, s.glow and lock_idx is None)
            if s.skeleton:
                R.draw_skeleton(frame, det, pal.skeleton, st)
            if s.head:
                R.draw_head(frame, det, s.color_head, st)
            if i == closest:
                R.draw_highlight_marker(frame, det, pal.box, st)
            if rect is not None:
                R.apply_fade(frame, backup, rect, det.alpha)

            if s.needs_labels:
                # rgb mode cycles colors every frame: keep the label accent constant so the
                # cached label image is reused
                accent = pal.text if s.rgb_mode else pal.box
                runs = R.build_label(det, s, pal.text)
                if i == lock_idx:
                    runs = [("LOCK", True, s.color_primary)] + ([("  ", False, pal.text)] if runs else []) + runs
                    accent = s.color_primary
                tag = R.make_tag(runs, st, accent, det.alpha, s.label_background)
                if tag is not None:
                    tags.append((det, tag, i == lock_idx))
            elif i == lock_idx:
                tag = R.make_tag([("LOCK", True, s.color_primary)], st, s.color_primary, det.alpha,
                                 s.label_background)
                tags.append((det, tag, True))

    for ztag in zone_tags:
        R.draw_tag(frame, ztag)

    if detections:
        # Labels last (on top of every box); locked first, then top-most boxes first.
        tags.sort(key=lambda dt: (not dt[2], dt[0].y1))
        for det, tag, _ in tags:
            R.place_tag(tag, (det.x1, det.y1, det.x2, det.y2), w, h, st, placed)
            R.draw_tag(frame, tag)
    AR.draw_over(frame, alert_info, st, camera_state.esp_settings, s.color_text, s.label_background, placed)

    # ---- faces / hands / inset -----------------------------------------------------
    if s.face_boxes and faces:
        R.draw_face_boxes(frame, faces, s.color_face, s.color_text, st, s.label_background, placed)
    if s.hand_skeleton and hands:
        R.draw_hands(frame, hands, s.color_hand, st)
    if inset is not None:
        R.paste_inset(frame, inset, s.color_primary, st, f"ZOOM {s.face_zoom_scale:.1f}x",
                      s.color_text, s.label_background)

    return frame
