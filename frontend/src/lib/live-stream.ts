/**
 * One Server-Sent Events connection per tab (backend: securitycam/live.py).
 *
 * The server pushes status, targets, playback, events, plate summary, grid
 * overview and PTZ state as named events; each payload has the shape of the
 * matching GET endpoint. Hooks subscribe to topics and fall back to their old
 * polling while the stream is not healthy (proxy blocks SSE, server restarting).
 */
import type { EventsResponse } from "./alerts"
import type { CameraOverview, PtzStatus } from "./ops-api"
import type { CameraStatus, TargetsResponse } from "./types"
import type { Playback } from "./videos"
import type { SoundPayload } from "./sound" // Sound ESP

export interface PlatesSummary {
  total: number
  new: number
  latest: number
  version: number
  enabled: boolean
  /** first_seen of the newest entries (newest first), for the unread badge */
  recent?: number[]
}

export interface LiveTopics {
  status: CameraStatus
  targets: TargetsResponse
  playback: Playback
  events: EventsResponse
  plates: PlatesSummary
  overview: { cameras: CameraOverview[]; ts: number }
  ptz: PtzStatus
  /** Sound ESP: only sent while the camera's `sound` setting is on */
  sound: SoundPayload
}

export type LiveTopic = keyof LiveTopics

/**
 * idle: no camera / no EventSource support (hooks poll)
 * connecting: first attempt, waiting for the first event (hooks wait briefly)
 * open: healthy, data comes from the stream (polling off)
 * reconnecting: lost or never reached; retrying with backoff (hooks poll)
 * paused: tab hidden for a while, stream closed (hooks pause too)
 */
export type LiveState = "idle" | "connecting" | "open" | "reconnecting" | "paused"

const TOPICS: LiveTopic[] = ["status", "targets", "playback", "events", "plates", "overview", "ptz", "sound"]
const PER_CAMERA = new Set<LiveTopic>(["status", "targets", "playback", "ptz", "sound"])
/** No first event within this time (e.g. a buffering proxy) counts as a failed connect. */
const FIRST_EVENT_MS = 4000
const BACKOFF_MIN_MS = 1000
const BACKOFF_MAX_MS = 30000
/** A hidden tab keeps its stream this long (quick tab switches don't reconnect). */
const HIDDEN_CLOSE_MS = 15000

type Listener = (data: unknown) => void

class LiveStream {
  private es: EventSource | null = null
  private camera: string | null = null
  private overviewRefs = 0
  private _state: LiveState = "idle"
  private last = new Map<LiveTopic, { camera: string | null; data: unknown }>()
  private listeners = new Map<LiveTopic, Set<Listener>>()
  private stateListeners = new Set<() => void>()
  private attempt = 0
  private retryTimer: number | undefined
  private firstTimer: number | undefined
  private hiddenTimer: number | undefined
  private visibilityBound = false

  get state(): LiveState {
    return this._state
  }

  get healthy(): boolean {
    return this._state === "open"
  }

  setCamera(id: string | null) {
    if (id === this.camera) return
    this.camera = id
    for (const t of PER_CAMERA) this.last.delete(t)
    this.attempt = 0
    this.restart()
  }

  /** Grid view asks for the overview topic (reference counted; changes reconnect). */
  wantOverview(on: boolean) {
    const before = this.overviewRefs > 0
    this.overviewRefs = Math.max(0, this.overviewRefs + (on ? 1 : -1))
    if (before !== this.overviewRefs > 0 && this.camera) {
      if (!this.overviewRefs) this.last.delete("overview")
      this.restart()
    }
  }

  subscribe(topic: LiveTopic, fn: Listener): () => void {
    let set = this.listeners.get(topic)
    if (!set) this.listeners.set(topic, (set = new Set()))
    set.add(fn)
    // replay the newest value so a late subscriber does not wait for the next change
    const hit = this.last.get(topic)
    if (hit && (!PER_CAMERA.has(topic) || hit.camera === this.camera)) {
      queueMicrotask(() => set.has(fn) && fn(hit.data))
    }
    return () => set.delete(fn)
  }

  onState(fn: () => void): () => void {
    this.stateListeners.add(fn)
    return () => this.stateListeners.delete(fn)
  }

  private setState(s: LiveState) {
    if (s === this._state) return
    this._state = s
    this.stateListeners.forEach((fn) => fn())
  }

  private bindVisibility() {
    if (this.visibilityBound || typeof document === "undefined") return
    this.visibilityBound = true
    document.addEventListener("visibilitychange", () => {
      window.clearTimeout(this.hiddenTimer)
      if (document.hidden) {
        this.hiddenTimer = window.setTimeout(() => {
          if (!document.hidden || !this.es) return
          this.close()
          this.setState("paused")
        }, HIDDEN_CLOSE_MS)
      } else if (this._state === "paused") {
        this.attempt = 0
        this.connect()
      }
    })
  }

  private close() {
    window.clearTimeout(this.retryTimer)
    window.clearTimeout(this.firstTimer)
    this.es?.close()
    this.es = null
  }

  private restart() {
    this.close()
    this.connect()
  }

  private connect() {
    this.close()
    if (!this.camera || typeof EventSource === "undefined") {
      this.setState("idle")
      return
    }
    this.bindVisibility()
    if (document.hidden && this._state === "paused") return
    const camera = this.camera
    const url = `/api/stream?camera=${encodeURIComponent(camera)}${this.overviewRefs > 0 ? "&overview=1" : ""}`
    const es = new EventSource(url)
    this.es = es
    if (this._state !== "reconnecting") this.setState("connecting")
    let got = false
    this.firstTimer = window.setTimeout(() => !got && this.fail(es), FIRST_EVENT_MS)

    for (const topic of TOPICS) {
      es.addEventListener(topic, (ev) => {
        if (this.es !== es) return
        let data: unknown
        try {
          data = JSON.parse((ev as MessageEvent<string>).data)
        } catch {
          return
        }
        if (!got) {
          got = true
          this.attempt = 0
          window.clearTimeout(this.firstTimer)
          this.setState("open")
        }
        this.last.set(topic, { camera, data })
        this.listeners.get(topic)?.forEach((fn) => fn(data))
      })
    }
    // virtual camera closed on the server: the camera list will move on; retry later
    es.addEventListener("closed", () => this.fail(es))
    es.onerror = () => this.fail(es)
  }

  private fail(es: EventSource) {
    if (this.es !== es) return
    this.close()
    this.attempt += 1
    this.setState("reconnecting")
    const base = Math.min(BACKOFF_MAX_MS, BACKOFF_MIN_MS * 2 ** (this.attempt - 1))
    this.retryTimer = window.setTimeout(() => this.connect(), base * (0.8 + Math.random() * 0.4))
  }
}

/** The tab's single live connection. */
export const liveStream = new LiveStream()
