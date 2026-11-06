export interface Camera {
  id: string
  label: string
  ip: string
  stream_path: string
  has_stream: boolean
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
