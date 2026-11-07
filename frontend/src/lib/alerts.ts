/** Types and API calls for alarms & events (backend: securitycam/alerts). */

/** Normalized point [x, y], 0..1 relative to the source frame. */
export type NPoint = [number, number]

export type Schedule = { mode: "always" } | { mode: "window"; start: string; end: string }

export interface ZoneRule {
  enabled: boolean
  classes: string[]
  min_dwell_s: number
  schedule: Schedule
  notify: boolean
}

export type WireTrigger = "both" | "in" | "out"

export interface Tripwire {
  id: string
  label: string
  /** Line from a to b; the "in" arrow is a→b rotated 90° clockwise on screen. */
  a: NPoint
  b: NPoint
  trigger: WireTrigger
  alarm: boolean
  notify: boolean
  classes: string[]
  color: string
  count_in: number
  count_out: number
  /** Newer backends (PTZ world anchoring): a/b are in the current view; these describe the drawn line. */
  in_view?: boolean
  anchored?: boolean
  drawn_a?: NPoint
  drawn_b?: NPoint
  anchor?: { pan: number; tilt: number; zoom: number }
  world?: { a: NPoint; b: NPoint }
}

export interface AlertsConfig {
  zone_rules: Record<string, ZoneRule>
  tripwires: Tripwire[]
  abandoned_notify: boolean
  /** Pose ESP pack: notify on fall alarms (newer backends). */
  fall_notify?: boolean
  abandon_classes: string[]
  channels: { ntfy: boolean; telegram: boolean; discord: boolean }
  clip: { pre_s: number; post_s: number }
}

/** Sound ESP events: "sound:<class>" (e.g. sound:glass_break) and "sound:loud". */
export type SoundEventKind = `sound:${string}`

export type EventKind = "zone" | "tripwire" | "abandoned" | "plate" | "fall" | "gesture" | SoundEventKind

export interface AlarmEvent {
  id: string
  camera: string
  camera_label?: string
  ts: number
  kind: EventKind
  source_id: string
  source_label: string
  cls: string
  track_id: number | null
  message: string
  direction?: "in" | "out"
  thumb: boolean
  thumb_url: string | null
  snapshot_url: string | null
  clip: {
    status: "recording" | "ready" | "failed"
    type?: "video" | "jpeg_seq"
    mime?: string
    frames?: number
    fps?: number
    duration?: number
    error?: string
  }
  clip_url?: string
  frames_url?: string
}

export interface ActiveAlarm {
  camera: string
  kind: EventKind
  source_id: string
}

export interface EventsResponse {
  events: AlarmEvent[]
  active: ActiveAlarm[]
  now: number
}

const cam = (id: string) => `/api/cameras/${encodeURIComponent(id)}`

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let code = `HTTP ${res.status}`
    try {
      const body = (await res.json()) as { error?: string }
      if (body.error) code = body.error
    } catch {
      /* not json */
    }
    throw new Error(code)
  }
  return (await res.json()) as T
}

function send<T>(url: string, method: string, body?: unknown): Promise<T> {
  return fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => json<T>(r))
}

export const alertsApi = {
  config: (id: string, signal?: AbortSignal) => fetch(`${cam(id)}/alerts`, { signal }).then((r) => json<AlertsConfig>(r)),
  setOptions: (id: string, patch: { abandoned_notify?: boolean; fall_notify?: boolean }) => send<AlertsConfig>(`${cam(id)}/alerts`, "PATCH", patch),
  setRule: (id: string, zoneId: string, patch: Partial<ZoneRule>) =>
    send<{ ok: boolean; rule: ZoneRule }>(`${cam(id)}/zones/${encodeURIComponent(zoneId)}/rule`, "PUT", patch),
  deleteRule: (id: string, zoneId: string) => send<{ ok: boolean }>(`${cam(id)}/zones/${encodeURIComponent(zoneId)}/rule`, "DELETE"),
  addTripwire: (id: string, a: NPoint, b: NPoint) => send<{ ok: boolean; tripwire: Tripwire }>(`${cam(id)}/tripwires`, "POST", { a, b }),
  updateTripwire: (id: string, wireId: string, patch: Partial<Omit<Tripwire, "id">> & { flip?: boolean }) =>
    send<{ ok: boolean; tripwire: Tripwire }>(`${cam(id)}/tripwires/${encodeURIComponent(wireId)}`, "PATCH", patch),
  deleteTripwire: (id: string, wireId: string) => send<{ ok: boolean }>(`${cam(id)}/tripwires/${encodeURIComponent(wireId)}`, "DELETE"),
  resetTripwire: (id: string, wireId: string) => send<{ ok: boolean }>(`${cam(id)}/tripwires/${encodeURIComponent(wireId)}/reset`, "POST"),
  events: (signal?: AbortSignal) => fetch(`/api/events?limit=300`, { signal }).then((r) => json<EventsResponse>(r)),
  deleteEvent: (e: AlarmEvent) => send<{ ok: boolean }>(`/api/events/${encodeURIComponent(e.camera)}/${encodeURIComponent(e.id)}`, "DELETE"),
  testNotify: (camera: string) =>
    send<{ ok: boolean; result?: Record<string, string>; error?: string }>(`/api/notify/test`, "POST", { camera }),
  /** Saves a snapshot on the server and returns the image for a browser download. */
  snapshot: async (id: string) => {
    const res = await fetch(`${cam(id)}/snapshot`, { method: "POST" })
    if (!res.ok) throw new Error(res.status === 409 ? "no_frame" : `HTTP ${res.status}`)
    const name = res.headers.get("X-Snapshot-File") ?? `snapshot-${Date.now()}.jpg`
    return { blob: await res.blob(), name }
  },
}

export const KIND_LABEL: Record<EventKind, string> = {
  zone: "Zone",
  tripwire: "Tripwire",
  abandoned: "Abandoned",
  plate: "Watchlist Plate",
  fall: "Fall",
  gesture: "Gesture",
}

/** Label of any event kind (sound kinds have no fixed entry in KIND_LABEL). */
export const kindLabel = (kind: EventKind): string => KIND_LABEL[kind] ?? (kind.startsWith("sound:") ? "Sound" : kind)
