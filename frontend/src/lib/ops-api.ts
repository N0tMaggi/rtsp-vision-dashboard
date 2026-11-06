/** API + types for operation & persistence: presets, schedules, backup, grid overview, PTZ. */
import type { EspSettings } from "./types"

export interface Preset {
  id: string
  name: string
  camera_id: string | null
  builtin: boolean
  settings: EspSettings
  yolo_enabled: boolean | null
  created_at: number
  updated_at: number
}

export interface Schedule {
  id: string
  preset_id: string
  time: string
  /** 0 = Monday … 6 = Sunday */
  days: number[]
  camera_id: string | null
  enabled: boolean
  next_run?: string | null
}

export interface PresetsResponse {
  presets: Preset[]
  schedules: Schedule[]
  last_applied: Record<string, { preset_id: string; at: number; source: "manual" | "schedule" }>
  now: string
}

export interface CameraOverview {
  id: string
  label: string
  has_stream: boolean
  fps: number
  num_detections: number
  yolo_enabled: boolean
  resolution: string
  stream_error: string
  live: boolean
  preset: string | null
}

export interface DiffEntry {
  path: string
  before: unknown
  after: unknown
  kind: "added" | "changed" | "removed"
}

export interface ImportResult {
  ok: boolean
  dry_run: boolean
  sections: string[]
  diff?: DiffEntry[]
  applied?: number
  warnings: string[]
  error?: string
  detail?: string
}

export interface PersistState {
  path: string
  loaded: boolean
  pending: boolean
  last_saved_at: number
  last_error: string
  sections: string[]
  schema_version: number
}

export type PtzState = "unavailable" | "idle" | "connecting" | "ready" | "unsupported" | "backoff"

export interface PtzSettings {
  fov_h: number
  fov_v: number
  pan_range: number
  tilt_range: number
  speed: number
}

export interface PtzStatus {
  available: boolean
  enabled: boolean
  install_hint: string
  configured: boolean
  port: number
  settings: PtzSettings
  state: PtzState
  error: string
  error_kind: "" | "auth" | "network" | "rtsp" | "unsupported" | "command"
  retry_in: number
  caps: { pan_tilt?: boolean; zoom?: boolean; relative?: boolean; presets?: boolean; home?: boolean }
  presets: { token: string; name: string }[]
  busy: boolean
  moving: boolean
}

const cam = (id: string) => `/api/cameras/${encodeURIComponent(id)}`

async function json<T>(res: Response): Promise<T> {
  const body = (await res.json().catch(() => ({}))) as T & { error?: string }
  if (!res.ok) throw Object.assign(new Error(body.error ?? `HTTP ${res.status}`), { status: res.status, body })
  return body
}

function send<T>(url: string, body?: unknown, method = "POST"): Promise<T> {
  return fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => json<T>(r))
}

export const opsApi = {
  overview: (signal?: AbortSignal) => fetch("/api/overview", { signal }).then((r) => json<{ cameras: CameraOverview[] }>(r)),
  /** Downscaled MJPEG for grid tiles (server snaps the width to a shared bucket). */
  tileStreamUrl: (id: string, width: number, nonce: number) => `/video_feed/${encodeURIComponent(id)}?w=${width}&v=${nonce}`,

  presets: (cameraId: string | null) =>
    fetch(`/api/presets${cameraId ? `?camera_id=${encodeURIComponent(cameraId)}` : ""}`).then((r) => json<PresetsResponse>(r)),
  createPreset: (body: { name: string; camera_id: string | null; from_camera: string; include_yolo?: boolean }) =>
    send<{ ok: boolean; preset: Preset }>("/api/presets", body),
  updatePreset: (id: string, patch: { name?: string; camera_id?: string | null; capture_from?: string }) =>
    send<{ ok: boolean; preset: Preset }>(`/api/presets/${encodeURIComponent(id)}`, patch, "PATCH"),
  deletePreset: (id: string) => send<{ ok: boolean }>(`/api/presets/${encodeURIComponent(id)}`, undefined, "DELETE"),
  applyPreset: (id: string, cameraId: string | null) =>
    send<{ ok: boolean; cameras: string[] }>(`/api/presets/${encodeURIComponent(id)}/apply`, { camera_id: cameraId }),
  createSchedule: (body: Omit<Schedule, "id" | "next_run">) => send<{ ok: boolean; schedule: Schedule }>("/api/schedules", body),
  updateSchedule: (id: string, patch: Partial<Omit<Schedule, "id" | "next_run">>) =>
    send<{ ok: boolean; schedule: Schedule }>(`/api/schedules/${encodeURIComponent(id)}`, patch, "PATCH"),
  deleteSchedule: (id: string) => send<{ ok: boolean }>(`/api/schedules/${encodeURIComponent(id)}`, undefined, "DELETE"),

  exportUrl: "/api/settings/export",
  persistState: () => fetch("/api/settings/state").then((r) => json<PersistState>(r)),
  importSettings: (text: string, dryRun: boolean) =>
    fetch(`/api/settings/import${dryRun ? "?dry_run=1" : ""}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: text,
    }).then((r) => r.json() as Promise<ImportResult>),

  ptz: (id: string, signal?: AbortSignal) => fetch(`${cam(id)}/ptz`, { signal }).then((r) => json<PtzStatus>(r)),
  ptzConnect: (id: string) => send<{ ok: boolean; ptz: PtzStatus }>(`${cam(id)}/ptz/connect`),
  ptzMove: (id: string, x: number, y: number, zoom = 0) => send<{ ok: boolean }>(`${cam(id)}/ptz/move`, { x, y, zoom }),
  ptzStop: (id: string) => send<{ ok: boolean }>(`${cam(id)}/ptz/stop`),
  ptzCenter: (id: string, u: number, v: number) => send<{ ok: boolean }>(`${cam(id)}/ptz/center`, { u, v }),
  ptzHome: (id: string) => send<{ ok: boolean }>(`${cam(id)}/ptz/home`),
  ptzSavePreset: (id: string, name: string) => send<{ ok: boolean; ptz: PtzStatus }>(`${cam(id)}/ptz/presets`, { name }),
  ptzGotoPreset: (id: string, token: string) => send<{ ok: boolean }>(`${cam(id)}/ptz/presets/${encodeURIComponent(token)}/goto`),
  ptzDeletePreset: (id: string, token: string) =>
    send<{ ok: boolean; ptz: PtzStatus }>(`${cam(id)}/ptz/presets/${encodeURIComponent(token)}`, undefined, "DELETE"),
  ptzSettings: (id: string, patch: Partial<PtzSettings>) =>
    send<{ ok: boolean; settings: PtzSettings }>(`${cam(id)}/ptz/settings`, patch, "PATCH"),
}

export const TILE_WIDTHS = [320, 480, 640, 960, 1280] as const

export function snapTileWidth(px: number): number {
  return TILE_WIDTHS.find((w) => px <= w) ?? TILE_WIDTHS[TILE_WIDTHS.length - 1]
}

export const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] as const
