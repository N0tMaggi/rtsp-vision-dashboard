/** Persistent license-plate log (backend: securitycam/recognition/platelog*.py). */

import { ApiError } from "./recognition"
import type { NBox } from "./types"

export interface PlateWatch {
  id: string
  plate: string
  note: string
}

export interface PlateLogEntry {
  id: string
  camera: string
  camera_label: string
  plate: string
  plate_key: string
  /** Share of the per-track vote for this text, 0..1. */
  conf: number
  /** Best raw OCR confidence, 0..1. */
  ocr_conf: number
  reads: number
  first_seen: number
  last_seen: number
  track_id: number | null
  vehicle: string
  images?: { plate: boolean; vehicle: boolean; frame: boolean }
  plate_box?: NBox
  vehicle_box?: NBox
  /** The vehicle crop's own box in the frame (vehicle box + margin). */
  crop_box?: NBox
  frame_w?: number
  frame_h?: number
  quality?: number
  watch: PlateWatch | null
  /** "manual" for plates marked by hand on the stage. */
  source?: "manual"
  /** Text corrected after the fact (re-read or typed). */
  edited?: boolean
  /** Search score: 3 exact, 2 OCR-confusion tolerant, 1 near miss. */
  match?: number
  plate_url: string | null
  vehicle_url: string | null
  frame_url: string | null
}

export interface PlateGroup {
  plate_key: string
  plate: string
  count: number
  cameras: string[]
  first_seen: number
  last_seen: number
  best_id: string
  latest_id: string
  watch: PlateWatch | null
  match: number
  variants: string[]
}

export interface PlateLogFilters {
  q?: string
  camera?: string
  since?: number
  until?: number
}

export interface WatchItem {
  id: string
  plate: string
  note: string
  created: number
}

export interface LiveTarget {
  live: boolean
  camera?: string
  target_id?: number
  box?: NBox
  label?: string
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON */
  }
  if (!res.ok) throw new ApiError((body as { error?: string } | null)?.error ?? `http_${res.status}`, res.status)
  return body as T
}

const base = "/api/recognition/platelog"
const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
})

export function filterQuery(f: PlateLogFilters, extra: Record<string, string | number> = {}) {
  const p = new URLSearchParams()
  if (f.q?.trim()) p.set("q", f.q.trim())
  if (f.camera) p.set("camera", f.camera)
  if (f.since !== undefined) p.set("since", String(f.since))
  if (f.until !== undefined) p.set("until", String(f.until))
  for (const [k, v] of Object.entries(extra)) p.set(k, String(v))
  const s = p.toString()
  return s ? `?${s}` : ""
}

export const plateLogApi = {
  list: (f: PlateLogFilters, limit = 200, signal?: AbortSignal) =>
    call<{ items: PlateLogEntry[]; total: number; now: number; version: number }>(
      `${base}${filterQuery(f, { limit })}`,
      { signal },
    ),
  groups: (f: PlateLogFilters, signal?: AbortSignal) =>
    call<{ plates: PlateGroup[]; total: number }>(`${base}/plates${filterQuery(f)}`, { signal }),
  sightings: (plate: string) =>
    call<{ plate: string; count: number; by_camera: Record<string, number>; items: PlateLogEntry[] }>(
      `${base}/plates/${encodeURIComponent(plate)}`,
    ),
  summary: (seen: number, signal?: AbortSignal) =>
    call<{ total: number; new: number; latest: number; version: number; enabled: boolean }>(
      `${base}/summary?seen=${seen}`,
      { signal },
    ),
  get: (id: string) => call<PlateLogEntry>(`${base}/${encodeURIComponent(id)}`),
  remove: (id: string) => call<{ ok: true }>(`${base}/${encodeURIComponent(id)}`, json("DELETE")),
  clear: (camera?: string) =>
    call<{ ok: true; deleted: number }>(`${base}${camera ? `?camera=${encodeURIComponent(camera)}` : ""}`, json("DELETE")),
  exportUrl: (f: PlateLogFilters, format: "csv" | "json") => `${base}/export${filterQuery(f, { format })}`,
  live: (id: string, signal?: AbortSignal) => call<LiveTarget>(`${base}/${encodeURIComponent(id)}/live`, { signal }),
  showLive: (id: string) =>
    call<{ ok: true; camera: string; target_id: number; box: NBox }>(`${base}/${encodeURIComponent(id)}/live`, json("POST")),
  watchlist: () => call<{ plates: WatchItem[] }>(`${base}/watchlist`),
  watch: (plate: string, note = "") => call<{ ok: true; item: WatchItem }>(`${base}/watchlist`, json("POST", { plate, note })),
  unwatch: (id: string) => call<{ ok: true }>(`${base}/watchlist/${encodeURIComponent(id)}`, json("DELETE")),
}

/** Fired by the Recognition panel ("Open Plate Log"); App listens and opens the drawer. */
export const OPEN_PLATE_LOG_EVENT = "securitycam:open-plate-log"

export function openPlateLog() {
  window.dispatchEvent(new CustomEvent(OPEN_PLATE_LOG_EVENT))
}

export function formatPlate(plate: string) {
  // Visual grouping only: letters / digits runs separated by a thin gap (AB123CD -> AB 123 CD).
  return plate.replace(/([A-Z]+)(?=\d)|(\d+)(?=[A-Z])/g, "$1$2 ")
}

export function relativeTime(ts: number, now = Date.now() / 1000) {
  const s = Math.max(0, now - ts)
  if (s < 10) return "just now"
  if (s < 60) return `${Math.round(s)} s ago`
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}
