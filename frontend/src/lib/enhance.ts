import { useSyncExternalStore } from "react"
import type { NBox } from "./types"

/**
 * Enhance view: API client + a tiny module-level store so anything (toolbar,
 * playback bar, keyboard, a future context menu) can open the view without
 * prop drilling.
 *
 *   openEnhance({ cameraId, region })   // region normalized [x1,y1,x2,y2]; both optional
 *
 * Without `cameraId` the camera on the stage is used; without `region` the
 * stage's current zoom viewport (full frame when not zoomed).
 * `window.dispatchEvent(new CustomEvent(OPEN_ENHANCE_EVENT, { detail: { cameraId, region } }))`
 * does the same for code that should not import this module.
 */

export type JobStatus = "queued" | "downloading" | "loading" | "running" | "saving" | "done" | "failed" | "canceled"

export interface EnhanceJob {
  id: string
  source_job: string | null
  camera_id: string
  camera_label: string
  created_at: number
  finished_at?: number
  status: JobStatus
  progress: number
  queue_position?: number
  options: { model: string; scale: number; denoise: number; device: string }
  region: NBox
  region_px: NBox
  frame_size: [number, number]
  crop_size: [number, number]
  output_size: [number, number]
  source: { kind: string; timestamp: number; timestamp_iso: string; label?: string; playing?: boolean; position_s?: number; frame?: number }
  model: { id: string; label: string; ai: boolean; license: string; source: string }
  ai_enhanced: boolean
  notice: string
  description: string
  device_used: string | null
  fallback: string | null
  timings: { download_s?: number; load_s?: number; infer_s?: number; total_s?: number }
  band_px?: number
  error: string | null
  urls: { original?: string; enhanced?: string; overlay?: string }
}

export interface EnhanceModel {
  id: string
  label: string
  ai: boolean
  denoise: boolean
  license: string
  source: string
  size_mb: number
  note: string
  ready: boolean
  downloading: boolean
}

export interface EnhanceIndex {
  jobs: EnhanceJob[]
  models: EnhanceModel[]
  default_model: string
  scales: number[]
  device: string
  notice: string
}

export interface EnhanceOptions {
  model: string
  scale: number
  denoise: number
}

export const NOTICE = "AI upscaling invents plausible detail. Use for presentation and exports, not as evidence."
export const ACTIVE: JobStatus[] = ["queued", "downloading", "loading", "running", "saving"]
export const OPEN_ENHANCE_EVENT = "securitycam:open-enhance"

export class EnhanceError extends Error {
  code: string
  constructor(code: string) {
    super(code)
    this.code = code
  }
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => null)
  if (!res.ok) throw new EnhanceError((body && (body as { error?: string }).error) || `http_${res.status}`)
  return body as T
}

const send = (url: string, method: string, body?: unknown) =>
  fetch(url, { method, headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) })

export const enhanceApi = {
  index: (signal?: AbortSignal) => fetch("/api/enhance", { signal }).then((r) => json<EnhanceIndex>(r)),
  job: (id: string, signal?: AbortSignal) => fetch(`/api/enhance/${encodeURIComponent(id)}`, { signal }).then((r) => json<EnhanceJob>(r)),
  start: (cameraId: string, body: { region?: NBox | null; source_job?: string } & EnhanceOptions) =>
    send(`/api/cameras/${encodeURIComponent(cameraId)}/enhance`, "POST", {
      ...body,
      region: body.region ?? undefined,
      include_overlay: !body.source_job,
    }).then((r) => json<{ ok: boolean; job: EnhanceJob }>(r)),
  cancel: (id: string) => send(`/api/enhance/${encodeURIComponent(id)}/cancel`, "POST", {}).then((r) => json<{ ok: boolean }>(r)),
  remove: (id: string) => send(`/api/enhance/${encodeURIComponent(id)}`, "DELETE").then((r) => json<{ ok: boolean }>(r)),
}

export function errorText(code: string): string {
  switch (code) {
    case "no_frame":
      return "No picture yet. Wait for the stream, then try again."
    case "region_too_large":
      return "That region is too large for this scale. Zoom in further or pick 2×."
    case "queue_full":
      return "Too many enhancements are queued. Wait for one to finish."
    case "unknown_source_job":
      return "The source frame of this result was deleted."
    case "enhance_failed":
    case "internal_error":
      return "Enhancement failed. Details are in the server log."
    case "interrupted":
      return "The server restarted while this was running."
    default:
      return `Request failed (${code}).`
  }
}

// ---------------------------------------------------------------------------- store
interface StageSource {
  cameraId: string
  label?: string
  /** Visible part of the frame, normalized. */
  region: () => NBox
}

export interface EnhanceState {
  open: boolean
  cameraId: string | null
  region: NBox | null
  /** Bumped on every open so the view starts a fresh capture. */
  nonce: number
  /** The video on the stage is paused (set by the playback bar). */
  videoPaused: boolean
  stageCameraId: string | null
}

let state: EnhanceState = { open: false, cameraId: null, region: null, nonce: 0, videoPaused: false, stageCameraId: null }
let stage: StageSource | null = null
const listeners = new Set<() => void>()

function set(patch: Partial<EnhanceState>) {
  state = { ...state, ...patch }
  listeners.forEach((l) => l())
}

const FULL: NBox = [0, 0, 1, 1]

/** Normalize, clamp and order a region; null for anything unusable. */
export function cleanRegion(r: unknown): NBox | null {
  if (!Array.isArray(r) || r.length !== 4 || !r.every((v) => typeof v === "number" && Number.isFinite(v))) return null
  const c = (v: number) => Math.min(1, Math.max(0, v))
  const [x1, x2] = [c(r[0]), c(r[2])].sort((a, b) => a - b)
  const [y1, y2] = [c(r[1]), c(r[3])].sort((a, b) => a - b)
  if (x2 - x1 <= 0 || y2 - y1 <= 0) return null
  return [x1, y1, x2, y2]
}

/** Open the Enhance view. Entry point for toolbar, playback bar, shortcut and context menu. */
export function openEnhance(req: { cameraId?: string; region?: NBox | null } = {}) {
  const cameraId = req.cameraId ?? stage?.cameraId ?? null
  if (!cameraId) return
  let region = cleanRegion(req.region)
  if (!region && stage && stage.cameraId === cameraId) region = cleanRegion(stage.region())
  set({ open: true, cameraId, region: region ?? FULL, nonce: state.nonce + 1 })
}

export function closeEnhance() {
  set({ open: false })
}

/** The stage registers how to read its current viewport. Returns the unregister function. */
export function registerEnhanceStage(src: StageSource): () => void {
  stage = src
  if (state.stageCameraId !== src.cameraId) set({ stageCameraId: src.cameraId })
  return () => {
    if (stage === src) {
      stage = null
      set({ stageCameraId: null })
    }
  }
}

export function setEnhanceVideoPaused(paused: boolean) {
  if (state.videoPaused !== paused) set({ videoPaused: paused })
}

export function useEnhanceState(): EnhanceState {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l)
      return () => listeners.delete(l)
    },
    () => state,
  )
}

if (typeof window !== "undefined") {
  window.addEventListener(OPEN_ENHANCE_EVENT, (e) => {
    const d = (e as CustomEvent<{ cameraId?: string; region?: NBox }>).detail ?? {}
    openEnhance({ cameraId: typeof d.cameraId === "string" ? d.cameraId : undefined, region: d.region })
  })
}

/** Visible region of a zoomed view (see use-view-transform: content point u lands at x + u·s). */
export function viewportRegion(v: { s: number; x: number; y: number }, w: number, h: number): NBox {
  if (!w || !h || v.s <= 1.001) return FULL
  return cleanRegion([-v.x / v.s / w, -v.y / v.s / h, (w - v.x) / v.s / w, (h - v.y) / v.s / h]) ?? FULL
}

// ---------------------------------------------------------------------------- prefs
const PREFS_KEY = "securitycam.enhance.prefs"

export function loadPrefs(): EnhanceOptions {
  const d: EnhanceOptions = { model: "realesr-general-x4v3", scale: 4, denoise: 0.5 }
  try {
    const raw = JSON.parse(window.localStorage.getItem(PREFS_KEY) ?? "null")
    if (raw && typeof raw === "object") {
      if (typeof raw.model === "string") d.model = raw.model
      if (raw.scale === 2 || raw.scale === 4) d.scale = raw.scale
      if (typeof raw.denoise === "number" && raw.denoise >= 0 && raw.denoise <= 1) d.denoise = raw.denoise
    }
  } catch {
    // storage unavailable: defaults
  }
  return d
}

export function savePrefs(p: EnhanceOptions) {
  try {
    window.localStorage.setItem(PREFS_KEY, JSON.stringify(p))
  } catch {
    // ignore
  }
}
