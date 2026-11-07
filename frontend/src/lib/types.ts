export interface Camera {
  id: string
  label: string
  ip: string
  stream_path: string
  has_stream: boolean
  /** "rtsp" for configured cameras; "file" / "url" for video-analysis sources. */
  source_kind?: "rtsp" | "file" | "url"
}

export interface CamerasResponse {
  cameras: Camera[]
  default_camera_id: string
  available_models?: string[]
}

export type BoxStyle = "full" | "corner" | "3d"

/** Server-side ESP settings. Keys mirror config.DEFAULT_ESP_SETTINGS. */
export type EspSettings = Record<string, boolean | number | string>

export interface GpuStats {
  name: string
  util: number
  mem_used: number
  mem_total: number
}

export interface CameraStatus {
  camera: Camera
  fps: number
  inference_fps: number
  last_infer_ms: number
  num_detections: number
  faces: number
  classes: string[]
  resolution: string
  total_frames: number
  last_update: number
  yolo_enabled: boolean
  yolo_available: boolean
  yolo_model: string | null
  available_models: string[]
  device: string
  cuda_available: boolean
  esp_settings: EspSettings
  gpu: GpuStats
  stream_error: string
  /** Locked target id (newer backends). */
  locked?: number | null
  zones?: Zone[]
  assets?: Record<string, "missing" | "downloading" | "ready" | "failed">
}

/** Normalized box [x1, y1, x2, y2], 0..1 relative to the source frame. */
export type NBox = [number, number, number, number]

/** Body points a crosshair / PTZ auto-track can aim at. Persons have all, other objects only "center". */
export type AimKey = "head" | "face" | "neck" | "chest" | "belly" | "groin" | "feet" | "center"

export interface Target {
  id: number
  label: string
  conf: number
  box: NBox
  /** Estimated distance in meters (rough, from the box height). */
  distance?: number
  /** Per-track stats (newer backends): wall-clock seconds since epoch, seconds, normalized / meters. */
  first_seen?: number
  age?: number
  dwell?: number
  path?: number
  path_m?: number
  /** Current speed in m/s (and normalized frame units / s). */
  speed?: number
  speed_n?: number
  /** Tags of the target's identity (person / face id / plate / track), newer backends. */
  tags?: TargetTag[]
  /** What recognition knows about the target. */
  identity?: TargetIdentity
  /** Hidden by the camera's solo / hide view filter (visual only). */
  hidden?: boolean
  /** Newer backends: body points, normalized like `box` (pose keypoints or box fractions). */
  aim?: Partial<Record<AimKey, [number, number]>>
  aim_src?: "pose" | "box"
  // --- depth package (only while the camera's Depth setting is on) ---
  /** Metric-ish depth from the depth map (meters); `distance` uses it when calibrated. */
  depth_m?: number
  distance_src?: "depth" | "box"
  distance_box?: number
  /** Floor-plan position in meters: [x right, z ahead] of the camera (pan ignored). */
  ground?: [number, number]
  /** Same in world bearings (PTZ, anchored): stays put while the camera pans. */
  ground_world?: [number, number]
  ground_src?: "depth" | "box"
  // --- end depth ---
}

/** Depth workload status of a camera (targets payload `depth`, `/api/cameras/<id>/depth`). */
export interface DepthStatus {
  enabled: boolean
  state: "off" | "idle" | "downloading" | "loading" | "running" | "failed" | "stopped"
  model: { state: "missing" | "downloading" | "ready" | "failed"; name: string; license: string }
  device: string | null
  device_label: string | null
  source: "auto" | "user" | null
  backend: string | null
  input: string | null
  latency_ms: number | null
  fps: number | null
  rate: number
  calibration: { mode: "user" | "auto" | "none"; scale: number | null; samples: number; confidence: number; ready: boolean }
  fov_h: number | null
  error: string | null
  note: string
  /** Coarse floor occupancy [x, z, weight 0..1] in meters (world frame when `world`). */
  scan?: { cell: number; world: boolean; pts: [number, number, number][] }
}

export interface TargetTag {
  label: string
  color: string
}

export interface TargetIdentity {
  name?: string
  person_id?: string
  /** Automatic unknown face id ("Face #n"). */
  face_id?: number
  plate?: string
  /** A face was attributed to this target in the last few seconds. */
  face?: boolean
}

export interface ViewFilterEntry {
  id: number
  epoch: number
  key: string | null
  label: string
  name: string | null
}

/** Per-camera solo / hidden targets (visual only: alarms keep counting everything). */
export interface ViewFilter {
  solo: ViewFilterEntry | null
  hidden: ViewFilterEntry[]
}

/** Camera pose of a world-anchored PTZ camera (degrees; FOV is the effective one at this zoom). */
export interface CameraPose {
  pan: number
  tilt: number
  zoom: number
  fov_h: number
  fov_v: number
  moving: boolean
}

export interface TargetsResponse {
  ts: number
  frame: { w: number; h: number }
  locked: number | null
  targets: Target[]
  /** Present while overlays are world-anchored (PTZ). */
  pose?: CameraPose
  /** Solo / hidden targets of this camera (newer backends). */
  view?: ViewFilter
  /** Depth workload status (depth package; only while Depth is on). */
  depth?: DepthStatus
}

export interface Zone {
  id: string
  label: string
  /** Bounding box in the current view (PTZ: projected from where it was drawn). */
  box: NBox
  color: string
  count: number
  /** Newer backends: outline in the current view, normalized points (clipped to the frame). */
  polygon?: [number, number][]
  /** False while a PTZ camera looks elsewhere. */
  in_view?: boolean
  anchored?: boolean
  /** Box as drawn, and the camera pose it was drawn at. */
  drawn_box?: NBox
  anchor?: { pan: number; tilt: number; zoom: number }
  /** Corner directions (azimuth, elevation in degrees). */
  world?: [number, number][]
}
