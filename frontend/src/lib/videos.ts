import type { Camera } from "./types"

/** A clip in the video library (data/videos/<id>). */
export interface Video {
  id: string
  name: string
  original_name: string
  size: number
  duration: number
  fps: number
  frames: number
  width: number
  height: number
  codec: string
  ext: string
  created: number
  thumb_url: string | null
  /** Virtual camera id while the clip is open as a source. */
  camera_id: string | null
}

export interface VideoSourceInfo {
  camera_id: string
  kind: "file" | "url"
  label: string
  video_id: string | null
  url?: string
  opened_at: number
}

export interface VideoLibrary {
  videos: Video[]
  total_bytes: number
  max_total_bytes: number
  max_upload_bytes: number
  extensions: string[]
  allow_urls: boolean
  sources: VideoSourceInfo[]
}

export interface Playback {
  kind: "file" | "url"
  seekable: boolean
  playing: boolean
  ended: boolean
  speed: number
  loop: boolean
  fps: number
  frame: number
  frames: number
  /** Seconds. */
  position: number
  duration: number
  server_time: number
}

export type PlaybackCommand =
  | { action: "play" | "pause" | "toggle" | "restart" }
  | { action: "seek"; seconds: number }
  | { action: "seek"; fraction: number }
  | { action: "seek"; delta: number }
  | { action: "step"; frames: number }
  | { speed: number }
  | { loop: boolean }

export interface OpenResult {
  ok: boolean
  camera_id: string
  camera: Camera
  playback: Playback
}

const ERRORS: Record<string, string> = {
  too_large: "That file is larger than the upload limit.",
  unsupported_type: "Unsupported file type. Use MP4, MKV, AVI, MOV or WebM.",
  invalid_video: "This file could not be decoded as a video.",
  empty_file: "The file is empty.",
  library_full: "The library is full. Delete a clip and try again.",
  multipart_required: "Upload failed: unexpected request format.",
  missing_file: "No file was sent.",
  invalid_url: "That does not look like a valid URL.",
  unsupported_scheme: "Only http, https and rtsp URLs are supported.",
  urls_disabled: "Stream URLs are disabled on this server.",
  unresolvable_host: "The host name could not be resolved.",
  blocked_address: "Blocked: the host points to a loopback, link-local or reserved address.",
  private_address: "Blocked: private network addresses must be allowed in .env first.",
  cross_origin_request: "Request blocked by the server's cross-site protection.",
  invalid_name: "Enter a name.",
}

export class VideoApiError extends Error {
  constructor(public code: string, public status: number) {
    super(ERRORS[code] ?? `Request failed (${code || status}).`)
  }
}

async function call<T>(url: string, method = "GET", body?: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method,
    signal,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const data = (await res.json().catch(() => ({}))) as { error?: string }
  if (!res.ok) throw new VideoApiError(data.error ?? "", res.status)
  return data as T
}

const vid = (id: string) => `/api/videos/${encodeURIComponent(id)}`
const src = (cameraId: string) => `/api/sources/${encodeURIComponent(cameraId)}`

export interface UploadHandle {
  promise: Promise<Video>
  abort: () => void
}

export const videosApi = {
  list: (signal?: AbortSignal) => call<VideoLibrary>("/api/videos", "GET", undefined, signal),
  rename: (id: string, name: string) => call<{ ok: boolean; video: Video }>(vid(id), "PATCH", { name }),
  remove: (id: string) => call<{ ok: boolean; closed: string | null }>(vid(id), "DELETE"),
  open: (id: string, copyFrom?: string | null) =>
    call<OpenResult>(`${vid(id)}/open`, "POST", copyFrom ? { copy_settings_from: copyFrom } : {}),
  openUrl: (url: string, label?: string, copyFrom?: string | null) =>
    call<OpenResult>("/api/sources/url", "POST", { url, ...(label ? { label } : {}), ...(copyFrom ? { copy_settings_from: copyFrom } : {}) }),
  close: (cameraId: string) => call<{ ok: boolean }>(`${src(cameraId)}/close`, "POST", {}),
  playback: (cameraId: string, signal?: AbortSignal) => call<Playback>(`${src(cameraId)}/playback`, "GET", undefined, signal),
  control: (cameraId: string, cmd: PlaybackCommand) =>
    call<{ ok: boolean; playback: Playback }>(`${src(cameraId)}/playback`, "POST", cmd),

  /** Multipart upload with real progress (XHR) and cancel. */
  upload(file: File, onProgress: (loaded: number, total: number) => void): UploadHandle {
    const xhr = new XMLHttpRequest()
    const promise = new Promise<Video>((resolve, reject) => {
      xhr.open("POST", "/api/videos")
      xhr.responseType = "json"
      xhr.upload.onprogress = (e) => onProgress(e.loaded, e.lengthComputable ? e.total : file.size)
      xhr.onload = () => {
        const data = (xhr.response ?? {}) as { video?: Video; error?: string }
        if (xhr.status >= 200 && xhr.status < 300 && data.video) resolve(data.video)
        else reject(new VideoApiError(data.error ?? (xhr.status === 413 ? "too_large" : ""), xhr.status))
      }
      xhr.onerror = () => reject(new VideoApiError("network", 0))
      xhr.onabort = () => reject(new DOMException("Upload cancelled", "AbortError"))
      const form = new FormData()
      form.append("file", file, file.name)
      xhr.send(form)
    })
    return { promise, abort: () => xhr.abort() }
  },
}

export const isVideoCamera = (c: Pick<Camera, "source_kind"> | undefined | null) =>
  Boolean(c && (c.source_kind === "file" || c.source_kind === "url"))

export function formatTime(s: number, withFraction = false) {
  if (!Number.isFinite(s) || s < 0) s = 0
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  const ss = withFraction ? sec.toFixed(1).padStart(4, "0") : String(Math.floor(sec)).padStart(2, "0")
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`
}

export function formatBytes(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 / 1024).toFixed(1)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}
