/** Backends of the stage's right-click menu: tags, solo / hidden, freeze frame, manual plates, unknown faces. */

import { ApiError, type FaceCandidate, type Person } from "./recognition"
import type { PlateLogEntry } from "./platelog"
import type { NBox, TargetTag, ViewFilter } from "./types"

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) throw new ApiError((body as { error?: string } | null)?.error ?? `http_${res.status}`, res.status)
  return body as T
}

const json = (method: string, body: unknown = {}): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
})
const cam = (id: string) => `/api/cameras/${encodeURIComponent(id)}`

export interface TagCatalogItem extends TargetTag {
  count: number
}

export interface FrozenFrame {
  id: string
  camera: string
  w: number
  h: number
  url: string
  vehicles: { id: number; label: string; box: NBox }[]
}

export interface PlateCandidate {
  text: string
  conf: number
}

export interface ManualRead {
  text: string
  conf: number
  candidates: PlateCandidate[]
  crop: string | null
  refined_box: [number, number, number, number] | null
  vehicle: { id: number; label: string; box: NBox } | null
  tried: number
}

export interface ReadPlateResult {
  found: boolean
  text?: string
  conf?: number
  box?: NBox
  crop?: string | null
  freeze: FrozenFrame
}

export interface UnknownFace {
  uid: number
  label: string
  first_seen: number
  last_seen: number
  sightings: number
  hits: number
  samples: number
  thumb: string | null
}

export type TagOwner = "person" | "face" | "plate" | "track"

export const contextApi = {
  tagCatalog: () => call<{ tags: TagCatalogItem[]; palette: string[]; max_label: number }>("/api/tags"),
  addTag: (cameraId: string, targetId: number, label: string, color?: string) =>
    call<{ ok: true; tags: TargetTag[]; owner: TagOwner }>(`${cam(cameraId)}/targets/${targetId}/tags`, json("POST", { label, color })),
  removeTag: (cameraId: string, targetId: number, label: string) =>
    call<{ ok: true; tags: TargetTag[] }>(`${cam(cameraId)}/targets/${targetId}/tags`, json("DELETE", { label })),
  view: (cameraId: string, action: "solo" | "unsolo" | "hide" | "unhide" | "clear", targetId?: number) =>
    call<{ ok: true; view: ViewFilter }>(`${cam(cameraId)}/view`, json("POST", { action, target_id: targetId })),
  freeze: (cameraId: string) => call<FrozenFrame & { ok: true }>(`${cam(cameraId)}/freeze`, json("POST")),
  readBox: (cameraId: string, freezeId: string, box: NBox) =>
    call<ManualRead & { ok: true }>(`${cam(cameraId)}/manual_plate/read`, json("POST", { freeze_id: freezeId, box })),
  confirmPlate: (cameraId: string, freezeId: string, box: NBox, plate: string, ocrConf: number) =>
    call<{ ok: true; entry: PlateLogEntry }>(
      `${cam(cameraId)}/manual_plate`,
      json("POST", { freeze_id: freezeId, box, plate, ocr_conf: ocrConf }),
    ),
  readPlate: (cameraId: string, targetId: number) =>
    call<ReadPlateResult & { ok: true }>(`${cam(cameraId)}/targets/${targetId}/read_plate`, json("POST")),
  targetFace: (cameraId: string, targetId: number) =>
    call<{ ok: true; candidate: FaceCandidate }>(`${cam(cameraId)}/targets/${targetId}/face`, json("POST")),
  reocr: (entryId: string) =>
    call<{ ok: true } & Omit<ManualRead, "crop" | "vehicle">>(`/api/recognition/platelog/${encodeURIComponent(entryId)}/reocr`, json("POST")),
  setPlate: (entryId: string, plate: string) =>
    call<{ ok: true; entry: PlateLogEntry }>(`/api/recognition/platelog/${encodeURIComponent(entryId)}`, json("PATCH", { plate })),
  unknownFaces: () => call<{ faces: UnknownFace[] }>("/api/recognition/unknown"),
  nameFace: (uid: number, name: string) => call<{ ok: true; person: Person }>(`/api/recognition/unknown/${uid}/name`, json("POST", { name })),
  mergeFace: (uid: number, into: number) => call<{ ok: true }>(`/api/recognition/unknown/${uid}/merge`, json("POST", { into })),
  deleteFace: (uid: number) => call<{ ok: true }>(`/api/recognition/unknown/${uid}`, json("DELETE")),
}

const ERRORS: Record<string, string> = {
  invalid_label: "Tags are 1 to 24 characters.",
  invalid_color: "Pick a color from the list.",
  too_many_tags: "That target already has the maximum number of tags.",
  unknown_target: "That target is gone.",
  unknown_tag: "That tag no longer exists.",
  no_frame: "No frame yet. Start the stream first.",
  unknown_frame: "The frozen frame expired. Freeze it again.",
  invalid_box: "Draw a box around the plate.",
  box_too_small: "That box is too small. Zoom in and draw around the whole plate.",
  invalid_plate: "Plates are 2 to 16 letters or digits.",
  models_loading: "The models are still downloading. Try again in a moment.",
  ocr_unavailable: "The plate reader can't run here.",
  no_face: "No clear face on that person right now.",
  unknown_face: "That face no longer exists.",
  same_face: "Pick a different face to merge into.",
  invalid_name: "Enter a name (up to 40 characters).",
  cross_origin_request: "Blocked by the server's cross-site protection.",
}

export function contextError(e: unknown): string {
  if (e instanceof ApiError) return ERRORS[e.code] ?? `Request failed (${e.code}).`
  return "Request failed. Is the server running?"
}
