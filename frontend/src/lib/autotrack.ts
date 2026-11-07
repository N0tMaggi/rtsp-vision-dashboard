import { useSyncExternalStore } from "react"
import { opsApi, type AutoTrack, type AutoTrackConfig, type AutoTrackState, type PtzStatus } from "./ops-api"
import type { AimKey } from "./types"

/**
 * PTZ auto-track shared state. usePtz publishes the camera's PTZ status here so the
 * stage (crosshair, framing window, chips, context menu) and the PTZ panel read one
 * source without prop drilling. Writes are optimistic: a pending patch wins over
 * pushed status for a moment so sliders do not jump back while the server applies it.
 */

export interface AutoTrackShared {
  cameraId: string | null
  /** PTZ connected and ready to move. */
  ready: boolean
  /** Camera has optical zoom. */
  zoom: boolean
  /** null: backend without auto-track (hide the UI). */
  autotrack: AutoTrack | null
}

const EMPTY: AutoTrackShared = { cameraId: null, ready: false, zoom: false, autotrack: null }
const PENDING_MS = 1200

let current: AutoTrackShared = EMPTY
let pending: { cameraId: string; patch: Partial<AutoTrackConfig>; until: number } | null = null
const listeners = new Set<() => void>()

function emit(next: AutoTrackShared) {
  current = next
  listeners.forEach((l) => l())
}

function withPending(cameraId: string | null, at: AutoTrack | null): AutoTrack | null {
  if (!at || !pending || pending.cameraId !== cameraId) return at
  if (Date.now() > pending.until) {
    pending = null
    return at
  }
  return { ...at, config: { ...at.config, ...pending.patch } }
}

/** Called by usePtz whenever the status of the active camera changes. */
export function publishPtz(cameraId: string | null, status: PtzStatus | null) {
  const next: AutoTrackShared = {
    cameraId,
    ready: status?.state === "ready",
    zoom: Boolean(status?.caps?.zoom),
    autotrack: withPending(cameraId, status?.autotrack ?? null),
  }
  const c = current
  if (c.cameraId === next.cameraId && c.ready === next.ready && c.zoom === next.zoom && c.autotrack === next.autotrack) return
  emit(next)
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => listeners.delete(l)
}

export function useAutoTrack(): AutoTrackShared {
  return useSyncExternalStore(subscribe, () => current, () => EMPTY)
}

function applyServer(cameraId: string, at: AutoTrack | undefined) {
  if (!at || current.cameraId !== cameraId) return
  emit({ ...current, autotrack: withPending(cameraId, at) })
}

/** PATCH the config of the active camera (optimistic). Debounce sliders before calling. */
export async function patchAutoTrack(patch: Partial<AutoTrackConfig>): Promise<string | null> {
  const { cameraId, autotrack } = current
  if (!cameraId || !autotrack) return "Auto-track is not available."
  pending = { cameraId, patch: { ...(pending?.cameraId === cameraId ? pending.patch : {}), ...patch }, until: Date.now() + PENDING_MS }
  emit({ ...current, autotrack: { ...autotrack, config: { ...autotrack.config, ...patch } } })
  try {
    const r = await opsApi.autotrackPatch(cameraId, patch)
    applyServer(cameraId, r.autotrack)
    return null
  } catch (e) {
    pending = null
    return errorText(e)
  }
}

/** Lock + track a target with the PTZ camera ("locked" mode); null stops tracking. */
export async function trackTarget(targetId: number | null): Promise<string | null> {
  const { cameraId, autotrack } = current
  if (!cameraId || !autotrack) return "Auto-track is not available."
  if (targetId === null) {
    emit({ ...current, autotrack: { ...autotrack, config: { ...autotrack.config, enabled: false } } })
  } else {
    emit({
      ...current,
      autotrack: {
        config: { ...autotrack.config, enabled: true, mode: "locked" },
        status: { ...autotrack.status, target_id: targetId },
      },
    })
  }
  try {
    const r = await opsApi.autotrackTarget(cameraId, targetId)
    pending = null
    applyServer(cameraId, r.autotrack)
    return null
  } catch (e) {
    emit({ ...current, autotrack })
    return errorText(e)
  }
}

/** Shift+T: off -> track the locked target (or auto pick); on -> off. */
export function toggleAutoTrack(lockedId: number | null): Promise<string | null> {
  const at = current.autotrack
  if (!at) return Promise.resolve("Auto-track is not available.")
  if (at.config.enabled) return patchAutoTrack({ enabled: false })
  if (lockedId !== null) return trackTarget(lockedId)
  return patchAutoTrack({ enabled: true, mode: "auto" })
}

function errorText(e: unknown): string {
  const err = e as { status?: number; body?: { error?: string; detail?: string }; message?: string }
  const code = err.body?.error ?? err.message ?? ""
  if (code === "target_not_visible" || err.status === 409) return "That target is not visible right now."
  if (err.body?.detail) return err.body.detail
  if (err.status === 404) return "This server does not support PTZ auto-track."
  return code ? `Auto-track: ${code}` : "Auto-track request failed."
}

// ------------------------------------------------------------------ framing window preview
let preview: number | null = null
const previewListeners = new Set<() => void>()

/** Show the framing window on the stage while its slider is dragged (null hides it). */
export function setDeadzonePreview(v: number | null) {
  if (preview === v) return
  preview = v
  previewListeners.forEach((l) => l())
}

export function useDeadzonePreview(): number | null {
  return useSyncExternalStore(
    (l) => {
      previewListeners.add(l)
      return () => previewListeners.delete(l)
    },
    () => preview,
    () => null,
  )
}

// ------------------------------------------------------------------ open panels from anywhere
export const OPEN_AUTOTRACK_EVENT = "securitycam:open-autotrack"
export const OPEN_CROSSHAIR_EVENT = "securitycam:open-crosshair"

let focusPending: "autotrack" | "crosshair" | null = null

/** Open the ESP menu on PTZ and scroll to the auto-track section. */
export function requestOpenAutoTrack() {
  focusPending = "autotrack"
  window.dispatchEvent(new Event(OPEN_AUTOTRACK_EVENT))
}

/** Open the ESP menu on Overlays and expand the crosshair settings. */
export function requestOpenCrosshair() {
  focusPending = "crosshair"
  window.dispatchEvent(new Event(OPEN_CROSSHAIR_EVENT))
}

/** True once for the section that was asked to scroll into view. */
export function consumeFocus(which: "autotrack" | "crosshair"): boolean {
  if (focusPending !== which) return false
  focusPending = null
  return true
}

// ------------------------------------------------------------------ labels
export const AIM_KEYS: AimKey[] = ["head", "face", "neck", "chest", "belly", "groin", "feet", "center"]

export const AIM_LABEL: Record<AimKey, string> = {
  head: "Head",
  face: "Face",
  neck: "Neck",
  chest: "Chest",
  belly: "Belly",
  groin: "Groin",
  feet: "Feet",
  center: "Center",
}

export const STATE_META: Record<AutoTrackState, { label: string; color: string; live: boolean }> = {
  off: { label: "Off", color: "#8b8f96", live: false },
  waiting_ptz: { label: "Waiting For PTZ", color: "#8b8f96", live: false },
  searching: { label: "Searching", color: "#f4f4f2", live: true },
  tracking: { label: "Tracking", color: "#ffd21f", live: true },
  holding: { label: "Holding", color: "#7ee0a1", live: true },
  lost: { label: "Lost", color: "#ff5a4e", live: true },
  paused: { label: "Paused", color: "#ffa64d", live: false },
}

/** Camera currently moving because of auto-track (pan / tilt / zoom speed above noise). */
export function isMoving(at: AutoTrack | null): boolean {
  const v = at?.status.velocity
  return Boolean(v && Math.hypot(v[0], v[1], v[2]) > 0.02)
}
