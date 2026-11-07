/** Depth package API: /api/cameras/<id>/depth (status) and .../depth/calibrate. */
import type { DepthStatus } from "./types"

const base = (id: string) => `/api/cameras/${encodeURIComponent(id)}/depth`

interface DepthReply {
  ok: boolean
  error?: string
  depth?: DepthStatus
}

async function call(url: string, body: unknown): Promise<DepthReply> {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
  try {
    return (await r.json()) as DepthReply
  } catch {
    return { ok: false, error: `http_${r.status}` }
  }
}

export const depthApi = {
  status: (id: string, signal?: AbortSignal) =>
    fetch(base(id), { signal }).then(async (r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      return (await r.json()) as { ok: boolean; depth: DepthStatus }
    }),
  /** "The locked target is `meters` away" (or a specific target id). */
  calibrate: (id: string, meters: number, targetId?: number) =>
    call(`${base(id)}/calibrate`, { distance_m: meters, ...(targetId !== undefined ? { target_id: targetId } : {}) }),
  resetCalibration: (id: string) => call(`${base(id)}/calibrate`, { reset: true }),
}

export const DEPTH_ERRORS: Record<string, string> = {
  no_locked_target: "Lock a target first (click it in the picture).",
  depth_not_running: "Depth is still starting. Try again in a moment.",
  unknown_target: "The locked target is no longer visible.",
  no_depth_at_target: "No depth at the target yet. Try again in a moment.",
  invalid_distance: "Enter a distance between 0.3 and 200 m.",
}
