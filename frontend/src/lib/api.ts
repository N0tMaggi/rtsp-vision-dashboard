import type { CameraStatus, CamerasResponse, EspSettings, NBox, TargetsResponse, Zone } from "./types"

const cam = (id: string) => `/api/cameras/${encodeURIComponent(id)}`

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return (await res.json()) as T
}

function post<T>(url: string, body?: unknown, method = "POST"): Promise<T> {
  return fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => json<T>(r))
}

export const api = {
  cameras: () => fetch("/api/cameras").then((r) => json<CamerasResponse>(r)),
  status: (id: string, signal?: AbortSignal) =>
    fetch(`${cam(id)}/status`, { signal }).then((r) => json<CameraStatus>(r)),
  toggleYolo: (id: string) => post<{ yolo_enabled: boolean }>(`${cam(id)}/toggle_yolo`),
  setModel: (id: string, model_name: string) =>
    post<{ ok: boolean; yolo_model?: string; error?: string }>(`${cam(id)}/set_model`, { model_name }),
  updateEsp: (id: string, patch: EspSettings) =>
    post<{ ok: boolean; esp_settings: EspSettings }>(`${cam(id)}/update_esp`, patch),
  streamUrl: (id: string, nonce: number) => `/video_feed/${encodeURIComponent(id)}?v=${nonce}`,
  targets: (id: string, signal?: AbortSignal) =>
    fetch(`${cam(id)}/targets`, { signal }).then((r) => json<TargetsResponse>(r)),
  lock: (id: string, targetId: number | null) =>
    post<{ ok: boolean; locked: number | null }>(`${cam(id)}/lock`, { id: targetId }),
  addZone: (id: string, box: NBox, label?: string) =>
    post<{ ok: boolean; zone: Zone }>(`${cam(id)}/zones`, { box, ...(label ? { label } : {}) }),
  updateZone: (id: string, zoneId: string, patch: Partial<Pick<Zone, "label" | "color" | "box">>) =>
    post<{ ok: boolean; zone: Zone }>(`${cam(id)}/zones/${encodeURIComponent(zoneId)}`, patch, "PATCH"),
  heatmapUrl: (id: string, window: string, width = 480, nonce = 0) =>
    `${cam(id)}/heatmap.png?window=${encodeURIComponent(window)}&w=${width}&v=${nonce}`,
  resetHeatmap: (id: string) => post<{ ok: boolean }>(`${cam(id)}/heatmap/reset`),
  deleteZone: (id: string, zoneId: string) =>
    post<{ ok: boolean }>(`${cam(id)}/zones/${encodeURIComponent(zoneId)}`, undefined, "DELETE"),
}
