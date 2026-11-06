/** Recognition package API (/api/recognition): people gallery, plates, models, TensorRT. */

export type AssetState = "missing" | "downloading" | "ready" | "failed"

export interface RecognitionStatus {
  faces: {
    available: boolean
    assets: Record<string, AssetState>
    error: string | null
    install_cmd: string | null
    people: number
  }
  plates: {
    available: boolean
    backend: string | null
    assets: Record<string, AssetState>
    error: string | null
    install_cmd: string | null
  }
  tensorrt: TensorRtCaps
  reid: { available: boolean; reentries: number }
  errors: Record<string, string>
}

export interface TensorRtCaps {
  available: boolean
  tensorrt_installed: boolean
  cuda: boolean
  reason: "no_cuda" | "tensorrt_missing" | null
  install_cmd: string | null
}

export interface Person {
  id: string
  name: string
  samples: number
  created: number
  updated: number
  thumb: string
}

export interface FaceCandidate {
  id: string
  box: [number, number, number, number]
  thumb: string | null
  match: string | null
  score: number
}

export interface PlateEntry {
  key: string
  plate: string
  conf: number
  reads: number
  track_id: number | null
  vehicle: string
  first_seen: number
  last_seen: number
  thumb: string | null
}

export interface EngineJob {
  model: string
  status: "running" | "done" | "failed"
  stage: string
  progress: number
  started: number
  finished: number | null
  error: string | null
  elapsed_s?: number
}

export interface ModelInfo {
  name: string
  family: "YOLOv8" | "YOLO11" | "RT-DETR" | "Other"
  task: "detect" | "pose"
  size: string
  label: string
  tracking: boolean
  keypoints: boolean
  weights_downloaded: boolean
  engine: boolean
  engine_size_mb: number | null
  engine_error: string | null
  engine_active: boolean
  job: EngineJob | null
  in_use: boolean
}

export class ApiError extends Error {
  code: string
  status: number
  constructor(code: string, status: number) {
    super(code)
    this.code = code
    this.status = status
  }
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) {
    const code = (body as { error?: string } | null)?.error ?? `http_${res.status}`
    throw new ApiError(code, res.status)
  }
  return body as T
}

const base = "/api/recognition"
const json = (method: string, body?: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: body === undefined ? undefined : JSON.stringify(body),
})
const cam = (id: string) => `${base}/cameras/${encodeURIComponent(id)}`

export const recognitionApi = {
  status: (cameraId?: string) =>
    call<RecognitionStatus>(`${base}/status${cameraId ? `?camera=${encodeURIComponent(cameraId)}` : ""}`),
  people: () => call<{ people: Person[] }>(`${base}/people`),
  addFromCandidate: (name: string, candidateId: string) =>
    call<{ ok: true; person: Person }>(`${base}/people`, json("POST", { name, candidate_id: candidateId })),
  addFromPhoto: (name: string, photo: File) => {
    const form = new FormData()
    form.append("name", name)
    form.append("photo", photo)
    return call<{ ok: true; person: Person }>(`${base}/people`, { method: "POST", body: form })
  },
  rename: (id: string, name: string) =>
    call<{ ok: true; person: Person }>(`${base}/people/${encodeURIComponent(id)}`, json("PATCH", { name })),
  remove: (id: string) => call<{ ok: true }>(`${base}/people/${encodeURIComponent(id)}`, json("DELETE")),
  candidates: (cameraId: string) => call<{ ok: true; faces: FaceCandidate[] }>(`${cam(cameraId)}/face_candidates`),
  plates: (cameraId: string) => call<{ plates: PlateEntry[] }>(`${cam(cameraId)}/plates`),
  clearPlates: (cameraId: string) => call<{ ok: true }>(`${cam(cameraId)}/plates`, json("DELETE")),
  models: () => call<{ models: ModelInfo[]; tensorrt: TensorRtCaps }>(`${base}/models`),
  buildEngine: (model: string) =>
    call<{ ok: true; job: EngineJob }>(`${base}/models/${encodeURIComponent(model)}/engine`, json("POST")),
  deleteEngine: (model: string) =>
    call<{ ok: true; deleted: boolean }>(`${base}/models/${encodeURIComponent(model)}/engine`, json("DELETE")),
}

const ERRORS: Record<string, string> = {
  no_face: "No face found in that photo. Try a sharper, front-facing picture.",
  invalid_image: "That file is not an image (max 12 MB).",
  invalid_name: "Enter a name (up to 40 characters).",
  name_taken: "Someone with that name already exists.",
  models_loading: "Face models are still downloading. Try again in a moment.",
  unknown_candidate: "That face expired. Capture the frame again.",
  no_frame: "No live frame yet. Start the stream first.",
  unknown_person: "This person no longer exists.",
  busy: "Another optimization is already running.",
  tensorrt_missing: "TensorRT is not installed.",
  no_cuda: "GPU optimization needs an NVIDIA GPU with CUDA.",
  invalid_model: "Unknown model.",
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError) return ERRORS[e.code] ?? `Request failed (${e.code}).`
  return "Request failed. Is the server running?"
}

// ---------------------------------------------------------------------------- model picker groups
export interface ModelOption {
  name: string
  short: string
  pose: boolean
}

export interface ModelGroup {
  family: string
  options: ModelOption[]
}

const FAMILY_ORDER = ["YOLOv8", "YOLO11", "RT-DETR", "Other"]

export function modelFamily(name: string): string {
  if (name.startsWith("rtdetr")) return "RT-DETR"
  if (name.startsWith("yolo11")) return "YOLO11"
  if (name.startsWith("yolov8")) return "YOLOv8"
  return "Other"
}

/** Groups model file names for the picker: YOLOv8 / YOLO11 / RT-DETR, pose variants flagged. */
export function groupModels(models: string[]): ModelGroup[] {
  const groups = new Map<string, ModelOption[]>()
  for (const name of models) {
    const fam = modelFamily(name)
    const stem = name.replace(/\.(pt|engine)$/, "")
    const pose = stem.endsWith("-pose")
    const opts = groups.get(fam) ?? []
    opts.push({ name, short: stem.replace(/-pose$/, ""), pose })
    groups.set(fam, opts)
  }
  return FAMILY_ORDER.filter((f) => groups.has(f)).map((family) => ({ family, options: groups.get(family)! }))
}
