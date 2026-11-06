# SecurityCam

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-111111)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-0078D4)
![Status](https://img.shields.io/badge/Status-Active-2EA043)
![Config](https://img.shields.io/badge/Config-.env%20supported-2EA043)
![License](https://img.shields.io/badge/License-Private-informational)

SecurityCam is a modular RTSP camera dashboard built with Flask, OpenCV, and YOLOv8. It provides live browser streaming, runtime AI toggling, model switching, GPU/CPU telemetry, and a full advanced ESP-style overlay control panel.

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Architecture](#architecture)
4. [Repository Layout](#repository-layout)
5. [How Runtime Processing Works](#how-runtime-processing-works)
6. [Requirements](#requirements)
7. [Quick Start](#quick-start)
8. [Configuration](#configuration)
9. [API Reference](#api-reference)
10. [ESP Features and Controls](#esp-features-and-controls)
11. [Model Management](#model-management)
12. [Performance Tuning](#performance-tuning)
13. [Operational Notes](#operational-notes)
14. [Troubleshooting](#troubleshooting)
15. [Development Workflow](#development-workflow)
16. [Security Notes](#security-notes)
17. [Roadmap](#roadmap)

## Overview

SecurityCam focuses on practical, real-time camera monitoring with optional AI inference:
- RTSP stream served directly to browser as MJPEG
- multi-camera workspace with closable camera tabs
- YOLOv8 detection and pose support
- full overlay customization (visual, prediction, face, hand, style)
- runtime model switching and runtime settings updates via HTTP API

The app is designed for local/self-hosted setups and can run in pure stream mode or AI-enhanced mode.

## Core Features

- Live stream endpoint: `GET /video_feed`
- Camera-specific stream endpoint: `GET /video_feed/<camera_id>`
- Optional YOLO inference with runtime toggle: `POST /toggle_yolo`
- Runtime model switch: `POST /set_model`
- Live status and telemetry endpoint: `GET /status`
- Camera-specific API endpoints under `/api/cameras/<camera_id>/...`
- Settings, zones, presets and schedules survive restarts (`data/state.json`), with JSON export/import
- Presets (Day / Night / Away / Minimal and your own) with weekday schedules
- Multi-camera grid view with downscaled streams
- Optional real PTZ over ONVIF (pad, arrow keys, right-click to center, camera positions)
- Rich ESP overlay system with many feature groups:
  - box styles and line tuning
  - skeleton/head/snaplines/crosshair
  - tracking/tracers/velocity/prediction
  - face boxes/face zoom/face blur
  - hand skeleton overlay
  - thermal mode, class colors, center highlight
  - RGB mode and per-feature color controls

## Architecture

### High-Level Data Flow

```mermaid
flowchart LR
    CAM["RTSP Camera"] --> VS["VideoStream Thread"]
    VS --> LF["state.latest_frame"]
    LF --> YW["YOLO Worker Thread"]
    YW --> LR["state.latest_results"]
    LR --> RENDER["draw_custom_esp"]
    RENDER --> MJPEG["GET /video_feed (MJPEG Response)"]

    UI["Browser Dashboard"] --> API["/status, /toggle_yolo, /update_esp, /set_model"]
    API --> STATE["state module"]
    STATE --> YW
    STATE --> RENDER
```

### Request/Response Layer

```mermaid
flowchart TB
    BR["Browser"] --> FL["Flask App"]
    FL --> IDX["GET /"]
    FL --> VFEED["GET /video_feed"]
    FL --> ST["GET /status"]
    FL --> TGL["POST /toggle_yolo"]
    FL --> ESP["POST /update_esp"]
    FL --> MOD["POST /set_model"]
```

## Repository Layout

```text
securitycam/
├─ .env.example
├─ .gitignore
├─ README.md
├─ main.py
├─ start_with_venv.bat
├─ start_without_venv.bat
├─ start_with_venv.sh
├─ start_without_venv.sh
├─ models/
├─ frontend/              # React + TypeScript + shadcn/ui dashboard (Vite)
└─ securitycam/
   ├─ __init__.py
   ├─ config.py
   ├─ deps.py
   ├─ esp/                # ESP overlay package (settings, colors, detections, renderers, pipeline)
   ├─ state.py
   ├─ static/app/         # built frontend (output of `npm run build`)
   ├─ streaming.py
   ├─ validation.py
   └─ webapp.py
```

### Module Responsibilities

- `main.py`: entrypoint, starts Flask app
- `securitycam/config.py`: env parsing, defaults, model list, RTSP URL build
- `securitycam/state.py`: shared mutable runtime state
- `securitycam/deps.py`: dependency checks, optional auto-install, YOLO model load, GPU snapshot
- `securitycam/esp/`: ESP overlay — `settings.py` (per-frame settings snapshot), `colors.py`, `detections.py` (extraction, smoothing, track history), `sensors.py` (face/hand detection), `renderers.py` (one function per feature), `pipeline.py` (`draw_custom_esp`)
- `securitycam/streaming.py`: RTSP reader, YOLO worker, render pipeline, MJPEG generator
- `securitycam/validation.py`: validates every ESP setting sent to `update_esp`
- `securitycam/webapp.py`: serves the built frontend, route wiring and JSON endpoints
- `frontend/`: React + TypeScript + shadcn/ui source of the dashboard

## How Runtime Processing Works

### Sequence

```mermaid
sequenceDiagram
    participant C as Camera
    participant VS as VideoStream
    participant S as state.py
    participant W as YOLO Worker
    participant E as ESP Renderer
    participant B as Browser

    C->>VS: RTSP frames
    VS->>S: update latest_frame
    W->>S: read latest_frame
    W->>W: infer (detect/track)
    W->>S: update latest_results
    B->>E: request frame via /video_feed
    E->>S: read latest_results
    E->>B: encoded JPEG frame
```

### YOLO Runtime States

```mermaid
stateDiagram-v2
    [*] --> Disabled
    Disabled --> Enabled: POST /toggle_yolo
    Enabled --> Disabled: POST /toggle_yolo
    Enabled --> Enabled: POST /set_model
```

## Requirements

### Runtime

- Python 3.10+
- RTSP-compatible camera
- Windows or Linux

### Python packages

Install at least:
- `flask`
- `opencv-python`
- `ultralytics`

Optional:
- `torch`, `torchvision`, `torchaudio` (especially for CUDA)
- `mediapipe` (for hand skeleton feature)

## Quick Start

### 1) Clone and enter project

```bash
git clone https://github.com/N0tMaggi/rtsp-vision-dashboard.git
cd securitycam
```

### 2) Create virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install flask opencv-python ultralytics mediapipe
```

### 4) Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your camera credentials and preferences.

### 5) Run app

```bash
python main.py
```

### 6) Open dashboard

`http://localhost:5000`

## One-command start scripts

Use the scripts below if you want install + run in one step.

### Windows

With venv:

```bat
start_with_venv.bat
```

Without venv:

```bat
start_without_venv.bat
```

### Linux/macOS

With venv:

```bash
chmod +x start_with_venv.sh
./start_with_venv.sh
```

Without venv:

```bash
chmod +x start_without_venv.sh
./start_without_venv.sh
```

What each script does:
- verifies Python availability
- installs/upgrades required packages (`flask`, `opencv-python`, `ultralytics`, `mediapipe`)
- starts `main.py`
- venv variant creates/uses `.venv`
- non-venv variant installs into the active/global interpreter environment

## Configuration

Configuration is loaded from environment variables, optionally from `.env` via `config.py`.

### Essential camera settings

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_CAMERAS` | no | empty | Semicolon-separated camera list: `id|Label|ip|username|password|stream_path` |
| `SECURITYCAM_CAMERA_IP` | yes | empty | Camera IP / host |
| `SECURITYCAM_CAMERA_USERNAME` | yes | empty | RTSP username |
| `SECURITYCAM_CAMERA_PASSWORD` | yes | empty | RTSP password |
| `SECURITYCAM_CAMERA_STREAM_PATH` | no | `stream1` | Stream path (`stream1`, `stream2`, etc.) |

If `SECURITYCAM_CAMERAS` is set, the legacy single-camera variables are used only as fallback defaults.

### App runtime settings

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_APP_HOST` | no | `0.0.0.0` | Bind interface |
| `SECURITYCAM_APP_PORT` | no | `5000` | HTTP port |
| `SECURITYCAM_APP_DEBUG` | no | `false` | Flask debug mode |

### Security/session-related settings

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_SECRET_KEY` | recommended | `change-me` or env value | Signing key |
| `SECURITYCAM_SESSION_COOKIE_SECURE` | no | `false` | Enable only with HTTPS |
| `SECURITYCAM_SESSION_COOKIE_SAMESITE` | no | `Lax` | SameSite policy |
| `SECURITYCAM_TRUSTED_HOSTS` | no | empty | Optional comma-separated host allow list |

### Model and performance settings

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_YOLO_MODEL` | no | `yolov8n.pt` | Startup model |
| `SECURITYCAM_YOLO_IMG_SIZE` | no | `640` | Inference image size |
| `SECURITYCAM_DISPLAY_SCALE` | no | `1.0` | Display downscale factor |
| `SECURITYCAM_JPEG_QUALITY` | no | `85` | Default MJPEG quality (per camera adjustable in the ESP menu) |
| `SECURITYCAM_FACE_DETECT_EVERY` | no | `5` | Face detection cadence |
| `SECURITYCAM_FACE_DETECT_SCALE` | no | `0.5` | Face detection scaling |
| `SECURITYCAM_HAND_DETECT_EVERY` | no | `5` | Hand detection cadence |
| `SECURITYCAM_HAND_DETECT_SCALE` | no | `0.5` | Hand detection scaling |
| `SECURITYCAM_SMOOTH_ALPHA` | no | `0.6` | Default box smoothing, 0 = raw boxes at inference rate (per camera adjustable in the ESP menu) |
| `SECURITYCAM_ESP_FONT` | no | – | Path to a TrueType font for ESP labels (falls back to Consolas / DejaVu Sans Mono) |
| `SECURITYCAM_TORCH_CUDA` | no | `cu134` | CUDA flavour the Windows start scripts install when an NVIDIA GPU is present but torch is CPU-only |

### Dependency auto-install settings

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_AUTO_INSTALL_DEPS` | no | `false` | Try auto-install if imports fail |
| `SECURITYCAM_TORCH_CUDA_INDEX_URL` | no | PyTorch nightly index | Index URL used for auto-install |

### Saved state, presets and schedules

ESP settings, detection on/off, YOLO model and zones of every camera, presets, schedules and PTZ calibration are saved to `data/state.json` (git-ignored) about a second after each change and restored on startup. Writes are atomic (temp file + rename). If the file is unreadable it is moved to `data/state.json.bak` and the app starts with defaults. Export/import (with a preview of the changes) is in the ESP menu under **Backup**.

Presets live in the **Presets** menu category: the built-ins *Day*, *Night*, *Away* and *Minimal* change only the overlay toggles that matter for them; *Save current settings* stores a full snapshot for one camera or all cameras. Schedules apply a preset at a local time on chosen weekdays (two disabled examples ship: Night at 22:00, Day at 07:00).

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_DATA_DIR` | no | `data/` | Folder for `state.json` |
| `SECURITYCAM_SAVE_DEBOUNCE` | no | `1.0` | Seconds between the last change and the write (0..60) |

New features can persist their own configuration with `persistence.register_section(name, dump_fn, load_fn, validate_fn)` (see `securitycam/persistence.py`).

### Grid view

With two or more cameras the header shows a single/grid toggle (`G`). The grid uses 2 columns up to 4 cameras and 3 above; tiles stream a downscaled feed (`?w=`) instead of full-resolution video. `1`–`9` open a camera, a click or `Enter` on a tile animates it into the single view.

### PTZ (optional, ONVIF)

Real pan/tilt/zoom needs the optional package:

```bash
pip install onvif-zeep
```

Without it the **PTZ** menu category only shows this hint. ONVIF logs in with the camera account from the camera config (Tapo: enable the camera account in the Tapo app; ONVIF runs on port 2020). Turn on **PTZ Mode** to get an on-screen pad, arrow keys (Page Up/Down zoom if supported) and right-click "center this point". The login only happens when you open the PTZ panel or turn PTZ Mode on; failed logins back off from 60 s up to 30 min (network errors 5 s up to 2 min), and PTZ waits while the RTSP login of the same camera is failing, so a wrong password cannot lock the account. A held move stops automatically if the browser stops refreshing it.

| Variable | Required | Default | Description |
|---|---|---:|---|
| `SECURITYCAM_PTZ` | no | `true` | Master switch |
| `SECURITYCAM_ONVIF_PORT` | no | `2020` | ONVIF port for all cameras |
| `SECURITYCAM_ONVIF_PORTS` | no | – | Per-camera ports, `camera-1:2020,garage:8000` |
| `SECURITYCAM_ONVIF_TIMEOUT` | no | `5` | Seconds per ONVIF call |
| `SECURITYCAM_PTZ_FOV_H` / `_FOV_V` | no | `100` / `56` | Field of view (degrees) for right-click centering; tune in the UI |
| `SECURITYCAM_PTZ_PAN_RANGE` / `_TILT_RANGE` | no | `360` / `114` | Mechanical range that ONVIF's generic -1..1 relative move spans |

## API Reference

### `GET /`
Returns dashboard HTML.

### `GET /video_feed`
Returns MJPEG stream.

### `GET /video_feed/<camera_id>`
Returns MJPEG stream for the selected camera.

### `GET /status`
Returns live runtime telemetry and settings.

### `GET /api/cameras/<camera_id>/status`
Returns runtime telemetry and settings for the selected camera.

Typical fields:
- `fps`
- `inference_fps`
- `last_infer_ms`
- `num_detections`
- `faces`
- `classes`
- `resolution`
- `yolo_enabled`
- `yolo_available`
- `yolo_model`
- `available_models`
- `device`
- `cuda_available`
- `esp_settings`
- `gpu`

### `POST /toggle_yolo`
Toggles YOLO state.

Response example:

```json
{ "yolo_enabled": true }
```

### `POST /api/cameras/<camera_id>/toggle_yolo`
Toggles YOLO state for the selected camera.

### `POST /set_model`
Request:

```json
{ "model_name": "yolov8s.pt" }
```

Possible responses:
- `200`: model switched or already selected
- `400`: invalid model

### `POST /api/cameras/<camera_id>/set_model`
Switches the model for the selected camera.

### `POST /update_esp`
Accepts partial ESP setting payload and updates runtime settings.

### `POST /api/cameras/<camera_id>/update_esp`
Accepts partial ESP setting payload and updates runtime settings for the selected camera.

### Operation & persistence endpoints

All bodies are JSON (`Content-Type: application/json`); non-JSON bodies are rejected with `415`.

| Method & path | Purpose |
|---|---|
| `GET /video_feed/<camera_id>?w=640` | Downscaled MJPEG for grid tiles. Width snaps to 320/480/640/960/1280; each size is encoded once per frame and shared by all clients. |
| `GET /api/overview` | fps, detections, live flag and last preset of every camera (grid HUD). |
| `GET /api/presets?camera_id=<id>` | Presets visible for a camera (global + bound), schedules with `next_run`, `last_applied`. |
| `POST /api/presets` | `{name, camera_id: null\|id, from_camera}` saves the current settings of `from_camera`. |
| `PATCH /api/presets/<id>` | `{name?, camera_id?, capture_from?}` rename / rescope / overwrite with current settings. |
| `DELETE /api/presets/<id>` | Deletes the preset and its schedules. |
| `POST /api/presets/<id>/apply` | `{camera_id: id\|null}`; `null` applies a global preset to all cameras. |
| `POST /api/schedules`, `PATCH`/`DELETE /api/schedules/<id>` | `{preset_id, time: "HH:MM", days: [0..6] (0 = Monday), camera_id, enabled}` |
| `GET /api/settings/export` | Download of all saved configuration (`securitycam-settings-*.json`). |
| `POST /api/settings/import?dry_run=1` | Validates an export and returns the diff without applying; without `dry_run` it applies. |
| `GET /api/settings/state` | Where state is saved, last save time, last error. |
| `GET /api/cameras/<id>/ptz` | PTZ availability, connection state, capabilities, camera presets, calibration. Never logs in. |
| `POST /api/cameras/<id>/ptz/connect` | Connects over ONVIF (refused with `429` while backing off). |
| `POST /api/cameras/<id>/ptz/move` / `stop` | `{x, y, zoom}` in -1..1 continuous move; re-send within 1.5 s or the server stops the camera. |
| `POST /api/cameras/<id>/ptz/center` | `{u, v}` normalized frame point; relative move that centers it. |
| `POST /api/cameras/<id>/ptz/home` | Home position (if supported). |
| `POST /api/cameras/<id>/ptz/presets`, `POST .../presets/<token>/goto`, `DELETE .../presets/<token>` | Camera-side PTZ positions. |
| `PATCH /api/cameras/<id>/ptz/settings` | `{fov_h, fov_v, pan_range, tilt_range, speed}` calibration (persisted). |

## ESP Features and Controls

The dashboard includes the complete advanced ESP feature set.

### Visual features

- Box rendering (`full`, `corner`, `3d`)
- Skeleton overlay
- Head marker
- Snaplines
- Crosshair (`center_dot`)
- Class names
- Confidence
- Distance estimate
- Face boxes
- Face zoom inset
- Hand skeleton
- Chams
- Face blur
- Box fill
- Class colors
- Person-only filtering
- Highlight center target
- Thermal mode

### Prediction/tracking features

- Tracking IDs
- Tracers
- Velocity vectors
- Prediction markers

### Style features

- RGB rainbow mode (`rgb_mode`; the legacy key `gay_mode` is still accepted by the update endpoint)
- Primary/secondary color control
- Per-feature color overrides:
  - boxes
  - skeleton
  - head
  - tracers
  - velocity
  - prediction
  - face
  - hand
  - text
  - chams

### Key tunables

- confidence threshold
- line thickness
- face zoom scale
- chams opacity
- face blur strength

## Alarms and Events

Everything lives in `securitycam/alerts/` (Flask blueprint) and `securitycam/esp/alerts_render.py`.
Alarms are evaluated in the render path, so they run while the camera is being viewed and detection is on.

- **Zone rules**: arm any zone in the ESP menu under *Alarms*. Optional class filter (e.g. only `person`),
  minimum dwell time (seconds a target's center must stay inside) and a schedule (always, or a time window
  such as 22:00-06:00). A triggered zone pulses red with `ALARM` in its pill.
- **Tripwires**: press `T` (or *Draw Tripwire*) and drag a line. The ESP draws the line, an arrow for the
  "in" direction (a→b rotated 90° clockwise) and `↑ in  ↓ out` counters. Crossings are detected from the
  motion model's per-track center history (stable track ids; turn on Tracking for best results). A tripwire can
  raise an alarm for both directions or one, optionally only for some classes. *Flip* swaps the direction.
- **Abandoned objects**: with *Abandoned Objects* on, bags/suitcases/umbrellas/laptops
  (`SECURITYCAM_ABANDON_CLASSES`) that appear and stay put for *Abandoned After* seconds are marked
  `NEW SINCE 14:32` and create an event. Objects already visible in the first seconds after detection starts
  are treated as background.
- **Events**: each alarm stores `data/events/<camera>/<YYYYMMDD-HHMMSS-mmm>/` with `meta.json`,
  `snapshot.jpg`, `thumb.jpg` and a clip from ~10 s before to ~10 s after the trigger. Clips come from a
  bounded per-camera ring buffer of the rendered (overlay) JPEG frames. The encoder is probed once:
  WebM/VP8, then MP4/H.264 (both play in Chrome); if neither works, the clip is a JPEG sequence
  (`frames/00000.jpg ...`) that the UI plays as an animation. Old events are removed by count/age.
- **Notifications**: ntfy, Telegram and Discord (all optional, with snapshot). Sent from a background
  queue with a cooldown per rule; errors are logged without URLs or tokens.
- **Snapshots**: `S` saves the current rendered frame to `data/snapshots/` and downloads it in the browser.
- **UI**: the siren button in the header opens the event timeline (`E`), shows an unread badge and pulses
  red while an alarm is active; new alarms pop up as toasts at the top.

Rules and tripwires persist in `data/alerts.json` (atomic writes). `data/` is git-ignored.

### Alarm settings (.env)

| Variable | Default | Description |
|---|---:|---|
| `SECURITYCAM_DATA_DIR` | `./data` | Root for `alerts.json`, `events/`, `snapshots/` |
| `SECURITYCAM_CLIP_PRE_S` / `SECURITYCAM_CLIP_POST_S` | `10` / `10` | Clip seconds before / after the trigger |
| `SECURITYCAM_CLIP_FPS` | `10` | Frame rate of the pre-roll buffer and clips |
| `SECURITYCAM_CLIP_BUFFER_MB` | `96` | Max memory of the per-camera frame buffer |
| `SECURITYCAM_CLIP_MAX_WIDTH` | `1280` | Clips are downscaled to this width |
| `SECURITYCAM_CLIP_FORMAT` | `auto` | `auto`, `webm`, `mp4` or `jpeg` |
| `SECURITYCAM_EVENTS_MAX_COUNT` | `300` | Keep at most this many events (all cameras) |
| `SECURITYCAM_EVENTS_MAX_AGE_DAYS` | `14` | Delete older events (`0` = keep) |
| `SECURITYCAM_ALARM_COOLDOWN_S` | `30` | Min. seconds between events of the same rule/tripwire |
| `SECURITYCAM_ALARM_HOLD_S` | `4` | How long the red alarm style stays after the condition ends |
| `SECURITYCAM_NOTIFY_COOLDOWN_S` | `60` | Min. seconds between notifications of the same rule/tripwire |
| `SECURITYCAM_NTFY_URL` / `SECURITYCAM_NTFY_TOKEN` | empty | ntfy topic URL (e.g. `https://ntfy.sh/my-secret-topic`), optional access token |
| `SECURITYCAM_TELEGRAM_BOT_TOKEN` / `SECURITYCAM_TELEGRAM_CHAT_ID` | empty | Telegram bot (photo + caption) |
| `SECURITYCAM_DISCORD_WEBHOOK_URL` | empty | Discord webhook (message + image) |
| `SECURITYCAM_ABANDON_CLASSES` | `backpack,handbag,suitcase,umbrella,laptop` | COCO classes watched as abandoned objects |

New ESP setting keys: `alarms` (evaluate rules), `tripwires` (draw tripwires), `abandoned`, `abandoned_seconds` (5-600).

### Alarm API

| Method | Path | Purpose |
|---|---|---|
| `GET` / `PATCH` | `/api/cameras/<id>/alerts` | Rules, tripwires (with live counters), channel status / set `abandoned_notify` |
| `PUT` / `DELETE` | `/api/cameras/<id>/zones/<zone_id>/rule` | Create/update or remove a zone rule (`enabled`, `classes`, `min_dwell_s`, `schedule`, `notify`) |
| `GET` / `POST` | `/api/cameras/<id>/tripwires` | List / create (`{"a": [x, y], "b": [x, y]}`, normalized) |
| `PATCH` / `DELETE` | `/api/cameras/<id>/tripwires/<tid>` | Update (`label`, `trigger`, `alarm`, `classes`, `color`, `notify`, `flip`) / delete |
| `POST` | `/api/cameras/<id>/tripwires/<tid>/reset` | Reset counters |
| `GET` | `/api/events?camera=&since=&limit=` | Events (newest first) and currently active alarms |
| `GET` / `DELETE` | `/api/events/<camera>/<event_id>` | Event metadata / delete event |
| `GET` | `/api/events/<camera>/<event_id>/{thumb.jpg,snapshot.jpg,clip}` | Files (clip supports HTTP ranges) |
| `GET` | `/api/events/<camera>/<event_id>/frames/<n>.jpg` | JPEG-sequence clip frames |
| `POST` | `/api/cameras/<id>/snapshot` | Save + return the current rendered frame |
| `POST` | `/api/notify/test` | Send a test notification (`{"camera": "<id>"}`) |

## Model Management

Available startup/runtime model names include:
- `yolov8n.pt`, `yolov8n-pose.pt`
- `yolov8s.pt`, `yolov8s-pose.pt`
- `yolov8m.pt`, `yolov8m-pose.pt`
- `yolov8l.pt`, `yolov8l-pose.pt`
- `yolov8x.pt`, `yolov8x-pose.pt`

Model loading behavior:
- if local model file exists in `models/`, it is used
- otherwise, Ultralytics may resolve/download model by name

## Performance Tuning

Main levers:
- reduce `SECURITYCAM_YOLO_IMG_SIZE` for faster inference
- reduce `SECURITYCAM_DISPLAY_SCALE` for lower CPU/network usage
- reduce `SECURITYCAM_JPEG_QUALITY` for lower bandwidth
- increase `SECURITYCAM_FACE_DETECT_EVERY` and `SECURITYCAM_HAND_DETECT_EVERY` to reduce overhead
- adjust `SECURITYCAM_SMOOTH_ALPHA` for stability vs latency

Suggested profiles:

| Profile | YOLO_IMG_SIZE | DISPLAY_SCALE | JPEG_QUALITY | FACE_EVERY | HAND_EVERY |
|---|---:|---:|---:|---:|---:|
| Quality | 960 | 1.0 | 90 | 3 | 3 |
| Balanced | 640 | 1.0 | 85 | 5 | 5 |
| Performance | 512 | 0.8 | 75 | 8 | 8 |
| CPU-focused | 416 | 0.7 | 70 | 10 | 10 |

## Operational Notes

- YOLO can be fully disabled while streaming continues
- worker thread decouples inference from stream encoding
- if CUDA runtime errors occur, code falls back to CPU in worker logic
- metrics are refreshed and exposed through `/status`

## Troubleshooting

### No stream image

Check:
- camera IP, username, password, stream path
- RTSP URL manually in VLC
- firewall and network routing
- camera RTSP capability and concurrent session limits

### YOLO not available

Check:
- `torch` and `ultralytics` installation
- active Python environment
- startup logs from dependency initialization

### Overlay not updating

Check:
- browser console for failed API requests
- `/update_esp` response payload
- that the setting key exists in `state.esp_settings`

### High latency or low FPS

Try:
- lower `SECURITYCAM_YOLO_IMG_SIZE`
- lower `SECURITYCAM_DISPLAY_SCALE`
- disable costly overlays (face zoom, hand skeleton, tracers)
- increase detection cadence values

## Development Workflow

### Run locally

```bash
python -m venv .venv
# activate virtual environment
pip install flask opencv-python ultralytics mediapipe
python main.py
```

### Frontend (React + TypeScript + shadcn/ui)

The built dashboard lives in `securitycam/static/app/` and is committed, so the start scripts work without Node.js. To change the UI (Node 20+):

```bash
cd frontend
npm install
npm run dev     # http://localhost:5173, proxies /api and /video_feed to Flask on :5000
npm run build   # type-checks and writes securitycam/static/app/
```

### Syntax check

```bash
python -m compileall main.py securitycam
```

### Typical implementation flow

1. add/adjust the runtime setting in `config.py` (`DEFAULT_ESP_SETTINGS`) and its rule in `validation.py`
2. expose it in `frontend/src/lib/esp-schema.ts`
3. consume it in `securitycam/esp/` or `streaming.py`
4. rebuild the frontend and verify via `/status`

## Security Notes

This project now uses environment-based credentials and avoids hardcoded camera secrets in source code.

Recommended deployment hardening:
- keep `SECURITYCAM_APP_DEBUG=false`
- set a strong `SECURITYCAM_SECRET_KEY`
- use HTTPS via reverse proxy
- restrict network access to trusted clients
- avoid exposing dashboard publicly without authentication at network edge

## Recognition

Package `securitycam/recognition/`, UI category **Recognition** in the ESP menu. Everything runs locally; no image or embedding leaves the machine.

| Feature | How | Extra install |
|---|---|---|
| Re-identification (`reid`) | Ultralytics BoT-SORT with ReID (`with_reid: True`, detector features for YOLO `.pt` models, a small classifier in `models/` for RT-DETR / engines) plus a re-entry memory that gives a returning person their old id (colour descriptor, 2 min) | none |
| Known faces (`known_faces`, `face_match_threshold`) | OpenCV YuNet + SFace (opencv_zoo, downloaded into `models/`), gallery in `data/faces/<id>/` | none (OpenCV >= 4.8) |
| Face detector (`face_detector`) | YuNet by default for face boxes / blur / zoom, Haar as fallback | none |
| License plates (`plates`) | YOLOv9-t plate detector (open-image-models, MIT) + fast-plate-ocr global model (MIT), both run in OpenCV DNN; per-track majority vote | none (`onnxruntime` is used as fallback if installed) |
| Pets (`pets_show`, `pets_ignore_alarms`) | cats / dogs get their own colour and a `PET` tag; `recognition.is_pet(label)` for the alarm code | none |
| Models | YOLOv8, YOLO11 (n/s/m, detect + pose), RT-DETR-L | none |
| Optimize for GPU | Ultralytics TensorRT export (FP16) into `models/<model>.engine`, used automatically afterwards | `python -m pip install tensorrt onnx onnxslim` (or `SECURITYCAM_INSTALL_TENSORRT=1` with the start scripts) |

API (prefix `/api/recognition`): `GET /status`, `GET|POST /people` (multipart `name` + `photo`, or JSON `{name, candidate_id}`), `PATCH|DELETE /people/<id>`, `GET /people/<id>/thumb`, `GET /cameras/<id>/face_candidates`, `GET|DELETE /cameras/<id>/plates`, `GET /models`, `POST|DELETE /models/<model>/engine`.

## Roadmap

- optional dashboard authentication
- richer multi-camera support
- deeper API contract validation for `/update_esp`
- automated tests for endpoints and runtime config
