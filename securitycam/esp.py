"""ESP drawing helpers and color utilities."""
import colorsys
import time
from typing import List, Tuple

import cv2
import numpy as np

from . import config, state, deps

# COCO keypoint limb connections
SKELETON_LIMBS = [
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
]


def get_rainbow_color(offset: float = 0.0, speed: float = 1.0) -> Tuple[int, int, int]:
    """Generates BGR tuple from time for Rainbow/Gaming mode."""
    t = time.time() * speed + offset
    r, g, b = colorsys.hsv_to_rgb(t % 1.0, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return (0, 255, 0)
    r, g, b = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)

def feature_color(key: str, fallback_hex: str) -> Tuple[int, int, int]:
    """Resolve per-feature color with fallback to provided hex."""
    return hex_to_bgr(state.esp_settings.get(key, fallback_hex))


def class_id_to_bgr(cls_id: int) -> Tuple[int, int, int]:
    """Deterministic class color palette derived from class ID."""
    hue = (cls_id * 0.11) % 1.0  # spread colors around the HSV wheel
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_custom_esp(frame, results):
    """
    Game-style ESP overlay with extensive runtime customization.
    """
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    r = results[0]
    boxes = r.boxes
    keypoints = r.keypoints if hasattr(r, "keypoints") and r.keypoints is not None else None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if state.esp_settings["thermal"]:
        frame[:] = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    if state.esp_settings["gay_mode"]:
        col_primary = (0, 255, 0)
        col_secondary = (0, 255, 255)
    else:
        col_primary = hex_to_bgr(state.esp_settings.get("color_primary", "#00ff00"))
        col_secondary = hex_to_bgr(state.esp_settings.get("color_secondary", "#ff00ff"))

    if state.esp_settings["center_dot"]:
        c_col = get_rainbow_color(0.5) if state.esp_settings["gay_mode"] else (0, 0, 255)
        l = 20
        cv2.line(frame, (center_x - l, center_y), (center_x + l, center_y), c_col, 1)
        cv2.line(frame, (center_x, center_y - l), (center_x, center_y + l), c_col, 1)
        cv2.circle(frame, (center_x, center_y), 2, c_col, -1)

    if not boxes:
        return frame

    now = time.time()
    line_thickness = max(1, int(round(state.esp_settings.get("line_thickness", 2))))

    detections: List[dict] = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = r.names.get(cls_id, str(cls_id))
        bx, by = (x1 + x2) // 2, (y1 + y2) // 2

        if state.esp_settings["person_only"] and cls_name.lower() != "person":
            continue

        try:
            obj_id = int(box.id[0]) if box.id is not None else -1
        except Exception:
            obj_id = -1

        detections.append({
            "idx": i,
            "obj_id": obj_id,
            "cls_id": cls_id,
            "cls_name": cls_name,
            "coords": (x1, y1, x2, y2),
            "center": (bx, by),
            "conf": conf,
            "dist_center": (bx - center_x) ** 2 + (by - center_y) ** 2
        })

    if not detections:
        return frame

    closest_idx = None
    if state.esp_settings["highlight_center"]:
        closest_idx = min(range(len(detections)), key=lambda k: detections[k]["dist_center"])

    # Face detection throttled and scaled-down for performance
    faces = state.last_faces
    face_det_scale = state.esp_settings.get("face_zoom_scale", 2.0)
    if state.esp_settings.get("face_boxes") or state.esp_settings.get("face_zoom") or state.esp_settings.get("face_blur"):
        if state.stats["total_frames"] % config.FACE_DETECT_EVERY == 0:
            cascade = deps.ensure_face_cascade()
            if cascade is not None:
                try:
                    small_gray = cv2.resize(gray, (0, 0), fx=config.FACE_DETECT_SCALE, fy=config.FACE_DETECT_SCALE)
                    raw_faces = cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                    faces = []
                    for (fx, fy, fw, fh) in raw_faces:
                        fx = int(fx / config.FACE_DETECT_SCALE)
                        fy = int(fy / config.FACE_DETECT_SCALE)
                        fw = int(fw / config.FACE_DETECT_SCALE)
                        fh = int(fh / config.FACE_DETECT_SCALE)
                        faces.append((fx, fy, fw, fh))
                except Exception:
                    faces = state.last_faces
            state.last_faces[:] = faces
            state.last_face_frame = state.stats["total_frames"]
    state.stats["faces"] = len(faces)

    # Hand detection (MediaPipe) throttled
    hands_landmarks = state.last_hands
    if state.esp_settings.get("hand_skeleton"):
        if state.stats["total_frames"] % config.HAND_DETECT_EVERY == 0:
            detector = deps.ensure_hand_detector()
            if detector:
                try:
                    rgb_small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=config.HAND_DETECT_SCALE, fy=config.HAND_DETECT_SCALE), cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb_small)
                    hands_landmarks = []
                    if results.multi_hand_landmarks:
                        for hlm in results.multi_hand_landmarks:
                            pts = []
                            for lm in hlm.landmark:
                                x = int(lm.x * frame.shape[1])
                                y = int(lm.y * frame.shape[0])
                                pts.append((x, y))
                            hands_landmarks.append(pts)
                except Exception:
                    pass
            state.last_hands[:] = hands_landmarks
            state.last_hand_frame = state.stats["total_frames"]

    current_frame_idx = state.stats.get("total_frames", 0)

    for det_i, det in enumerate(detections):
        x1, y1, x2, y2 = det["coords"]
        conf = det["conf"]
        cls_id = det["cls_id"]
        cls_name = det["cls_name"]
        bx, by = det["center"]
        obj_id = det["obj_id"]

        # --- 1b. Smooth boxes/centers to reduce flicker ---
        key = obj_id if obj_id != -1 else f"{cls_id}_{det_i}"
        if key in state.smoothed_boxes:
            prev = state.smoothed_boxes[key]
            alpha = config.SMOOTH_ALPHA
            x1 = int(prev["x1"] * alpha + x1 * (1 - alpha))
            y1 = int(prev["y1"] * alpha + y1 * (1 - alpha))
            x2 = int(prev["x2"] * alpha + x2 * (1 - alpha))
            y2 = int(prev["y2"] * alpha + y2 * (1 - alpha))
            bx = int(prev["bx"] * alpha + bx * (1 - alpha))
            by = int(prev["by"] * alpha + by * (1 - alpha))
            conf = prev["conf"] * alpha + conf * (1 - alpha)

        state.smoothed_boxes[key] = {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "bx": bx, "by": by, "conf": conf, "last": current_frame_idx
        }

        if obj_id != -1:
            if obj_id not in state.track_history:
                state.track_history[obj_id] = []
            state.track_history[obj_id].append((bx, by, now))
            if len(state.track_history[obj_id]) > state.MAX_TRACK_HISTORY:
                state.track_history[obj_id].pop(0)

        if state.esp_settings["gay_mode"]:
            color = get_rainbow_color(offset=det_i * 0.1 + obj_id * 0.2, speed=0.5)
            text_color = (255, 255, 255)
        elif state.esp_settings["class_colors"]:
            color = class_id_to_bgr(cls_id)
            text_color = (255, 255, 255)
        else:
            color = feature_color("color_boxes", state.esp_settings.get("color_primary", "#00ff00"))
            text_color = feature_color("color_text", state.esp_settings.get("color_primary", "#00ff00"))

        if closest_idx is not None and det_i == closest_idx:
            color = (0, 215, 255)
            text_color = (0, 215, 255)
            cv2.drawMarker(frame, (bx, by), color, cv2.MARKER_CROSS, 20, line_thickness)

        if state.esp_settings["snaplines"]:
            cv2.line(frame, (center_x, h), (bx, y2), color, line_thickness, cv2.LINE_AA)

        if obj_id != -1 and obj_id in state.track_history:
            hist = state.track_history[obj_id]
            if state.esp_settings["tracers"] and len(hist) > 1:
                tracer_col = feature_color("color_tracers", state.esp_settings.get("color_secondary", "#ff00ff"))
                pts = np.array([[pt[0], pt[1]] for pt in hist], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, tracer_col, line_thickness)

            if (state.esp_settings["velocity"] or state.esp_settings["prediction"]) and len(hist) >= 5:
                subset = hist[-5:]
                dx = (subset[-1][0] - subset[0][0])
                dy = (subset[-1][1] - subset[0][1])
                dt = (subset[-1][2] - subset[0][2])
                if dt > 0:
                    vx = dx / dt
                    vy = dy / dt
                    if state.esp_settings["velocity"]:
                        vel_col = feature_color("color_velocity", state.esp_settings.get("color_secondary", "#ff00ff"))
                        end_x = int(bx + vx * 0.5)
                        end_y = int(by + vy * 0.5)
                        cv2.arrowedLine(frame, (bx, by), (end_x, end_y), vel_col, line_thickness, tipLength=0.3)
                    if state.esp_settings["prediction"]:
                        pred_col = feature_color("color_prediction", state.esp_settings.get("color_secondary", "#ff00ff"))
                        pred_x = int(bx + vx * 1.0)
                        pred_y = int(by + vy * 1.0)
                        cv2.drawMarker(frame, (pred_x, pred_y), pred_col, cv2.MARKER_TILTED_CROSS, 10, line_thickness)
                        cv2.line(frame, (bx, by), (pred_x, pred_y), (50, 50, 50), line_thickness)

        if state.esp_settings["boxes"]:
            if state.esp_settings["box_style"] == "corner":
                l = int((x2 - x1) * 0.2)
                t = line_thickness
                cv2.line(frame, (x1, y1), (x1 + l, y1), color, t)
                cv2.line(frame, (x1, y1), (x1, y1 + l), color, t)
                cv2.line(frame, (x2, y1), (x2 - l, y1), color, t)
                cv2.line(frame, (x2, y1), (x2, y1 + l), color, t)
                cv2.line(frame, (x1, y2), (x1 + l, y2), color, t)
                cv2.line(frame, (x1, y2), (x1, y2 - l), color, t)
                cv2.line(frame, (x2, y2), (x2 - l, y2), color, t)
                cv2.line(frame, (x2, y2), (x2, y2 - l), color, t)
            elif state.esp_settings["box_style"] == "3d":
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
                shift = 10
                cv2.rectangle(frame, (x1 + shift, y1 - shift), (x2 + shift, y2 - shift), color, line_thickness)
                cv2.line(frame, (x1, y1), (x1 + shift, y1 - shift), color, line_thickness)
                cv2.line(frame, (x2, y1), (x2 + shift, y1 - shift), color, line_thickness)
                cv2.line(frame, (x1, y2), (x1 + shift, y2 - shift), color, line_thickness)
                cv2.line(frame, (x2, y2), (x2 + shift, y2 - shift), color, line_thickness)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

            if state.esp_settings.get("fill_box", False):
                overlay = frame.copy()
                fill_col = color if (state.esp_settings["class_colors"] or state.esp_settings["gay_mode"]) else col_secondary
                cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_col, -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
            if state.esp_settings.get("chams", False):
                overlay = frame.copy()
                cham_col = feature_color("chams_color", "#00ffff")
                cv2.rectangle(overlay, (x1, y1), (x2, y2), cham_col, -1)
                alpha = float(state.esp_settings.get("chams_opacity", 0.35))
                alpha = max(0.0, min(alpha, 1.0))
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        label_parts = []
        if state.esp_settings["tracking"] and obj_id != -1:
            label_parts.append(f"ID:{obj_id}")
        if state.esp_settings["names"]:
            label_parts.append(cls_name.upper())
        if state.esp_settings["distance"]:
            h_px = max(y2 - y1, 1)
            dist_m = 1000.0 / h_px
            label_parts.append(f"{dist_m:.1f}m")
        if state.esp_settings["conf"]:
            label_parts.append(f"{int(conf * 100)}%")

        if label_parts:
            label_text = " // ".join(label_parts)
            cv2.putText(frame, label_text, (x1, y1 - 8), cv2.FONT_HERSHEY_PLAIN, 1.0, text_color, line_thickness, cv2.LINE_AA)

        if (state.esp_settings["skeleton"] or state.esp_settings["head"]) and keypoints is not None and len(keypoints.xy) > det["idx"]:
            kpts = keypoints.xy[det["idx"]].cpu().numpy()
            if state.esp_settings["skeleton"]:
                skeleton_col = get_rainbow_color(det_i * 0.5, 2.0) if state.esp_settings["gay_mode"] else feature_color("color_skeleton", state.esp_settings.get("color_primary", "#00ff00"))
                for p1, p2 in SKELETON_LIMBS:
                    if p1 < len(kpts) and p2 < len(kpts):
                        pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
                        pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
                        if pt1[0] > 0 and pt2[0] > 0:
                            cv2.line(frame, pt1, pt2, skeleton_col, line_thickness)

                def get_pt(idx):
                    if idx < len(kpts):
                        x, y = kpts[idx]
                        if x > 0 and y > 0:
                            return int(x), int(y)
                    return None

                nose = get_pt(0)
                l_sh, r_sh = get_pt(5), get_pt(6)
                l_hip, r_hip = get_pt(11), get_pt(12)
                mid_sh = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2) if l_sh and r_sh else None
                mid_hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2) if l_hip and r_hip else None
                if mid_sh:
                    cv2.line(frame, mid_sh, l_sh, skeleton_col, line_thickness)
                    cv2.line(frame, mid_sh, r_sh, skeleton_col, line_thickness)
                if mid_hip:
                    cv2.line(frame, mid_hip, l_hip, skeleton_col, line_thickness)
                    cv2.line(frame, mid_hip, r_hip, skeleton_col, line_thickness)
                if mid_sh and mid_hip:
                    cv2.line(frame, mid_sh, mid_hip, skeleton_col, line_thickness)
                if nose and mid_sh:
                    cv2.line(frame, nose, mid_sh, skeleton_col, line_thickness)

            if state.esp_settings["head"]:
                nk = kpts[0]
                nx, ny = int(nk[0]), int(nk[1])
                if nx > 0 and ny > 0:
                    head_rad = int((x2 - x1) / 8)
                    head_col = feature_color("color_head", "#ff0000")
                    cv2.circle(frame, (nx, ny), max(head_rad, 5), head_col, line_thickness)

    # Draw face boxes
    if state.esp_settings.get("face_boxes") and len(faces) > 0:
        face_col = feature_color("color_face", state.esp_settings.get("color_secondary", "#ff00ff"))
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), face_col, max(1, line_thickness - 1))
            cv2.putText(frame, "FACE", (fx, fy - 6), cv2.FONT_HERSHEY_PLAIN, 1.0, face_col, 1, cv2.LINE_AA)

    # Face blur alternative
    if state.esp_settings.get("face_blur") and len(faces) > 0:
        k = int(state.esp_settings.get("face_blur_strength", 15))
        if k % 2 == 0:
            k += 1
        k = max(3, k)
        for (fx, fy, fw, fh) in faces:
            roi = frame[fy:fy+fh, fx:fx+fw]
            if roi.size > 0:
                frame[fy:fy+fh, fx:fx+fw] = cv2.GaussianBlur(roi, (k, k), 0)

    # Face zoom inset (largest face)
    if state.esp_settings.get("face_zoom") and len(faces) > 0:
        f = max(faces, key=lambda b: b[2] * b[3])
        fx, fy, fw, fh = f
        pad = int(0.2 * max(fw, fh))
        x1 = max(fx - pad, 0)
        y1 = max(fy - pad, 0)
        x2 = min(fx + fw + pad, w)
        y2 = min(fy + fh + pad, h)
        face_crop = frame[y1:y2, x1:x2].copy()
        if face_crop.size > 0:
            zoom = max(1.0, float(face_det_scale))
            target_w = int(face_crop.shape[1] * zoom)
            target_h = int(face_crop.shape[0] * zoom)
            face_zoom = cv2.resize(face_crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            inset_w = min(target_w, w // 4)
            inset_h = min(target_h, h // 4)
            face_zoom = cv2.resize(face_zoom, (inset_w, inset_h))
            frame[10:10 + inset_h, 10:10 + inset_w] = face_zoom
            cv2.rectangle(frame, (8, 8), (12 + inset_w, 12 + inset_h), (255, 255, 255), 1)

    # Hand skeletons (21-point MediaPipe)
    if state.esp_settings.get("hand_skeleton") and hands_landmarks:
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # index
            (0, 9), (9, 10), (10, 11), (11, 12), # middle
            (0, 13), (13, 14), (14, 15), (15, 16),# ring
            (0, 17), (17, 18), (18, 19), (19, 20),# pinky
        ]
        for pts in hands_landmarks:
            for (a, b) in hand_connections:
                if a < len(pts) and b < len(pts):
                    hand_col = feature_color("color_hand", state.esp_settings.get("color_primary", "#00ff00"))
                    cv2.line(frame, pts[a], pts[b], hand_col, max(1, line_thickness - 1))
            for p in pts:
                hand_col = feature_color("color_hand", state.esp_settings.get("color_primary", "#00ff00"))
                cv2.circle(frame, p, 2, hand_col, -1)

    # Cleanup old smoothed boxes
    stale_keys = [k for k, v in state.smoothed_boxes.items() if current_frame_idx - v["last"] > 30]
    for k in stale_keys:
        state.smoothed_boxes.pop(k, None)

    return frame
