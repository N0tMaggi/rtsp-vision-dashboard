/**
 * Sound ESP (backend: securitycam/audio). Live levels / spectrogram / labels come
 * from the SSE "sound" topic (only while the camera's `sound` setting is on);
 * rules live at /api/sound/config. Audio itself never reaches the browser.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import {
  AudioLines,
  Baby,
  BellRing,
  Bomb,
  Car,
  Cat,
  DoorOpen,
  Dog,
  Footprints,
  GlassWater,
  Hand,
  HandMetal,
  Megaphone,
  Mic,
  Music,
  Siren,
  AlarmSmoke,
  Volume2,
  type LucideIcon,
} from "lucide-react"
import { useLiveTopic, useShouldPoll } from "@/hooks/use-live-stream"

export const SOUND_PRIVACY = "Audio stays on this computer: it is analysed locally and never recorded, stored or uploaded."

export interface SoundLabel {
  id: string
  label: string
  icon: string
  score: number
}

export interface SoundPing {
  id: number
  ts: number
  /** curated class id or "loud" */
  kind: string
  label: string
  score: number | null
  /** true when an alarm rule fired (an event was created) */
  alarm: boolean
}

export type SoundState =
  | "idle"
  | "starting"
  | "live"
  | "no_audio"
  | "no_audio_file"
  | "error"
  | "auth"
  | "waiting_video"
  | "paused"
  | "unsupported"
  | "no_ffmpeg"

export interface ClassifierInfo {
  status: "idle" | "downloading" | "preparing" | "ready" | "unavailable"
  reason: string
  device: string | null
  ms: number | null
}

export interface SoundPayload {
  camera: string
  on: boolean
  state: SoundState
  message: string
  source: "rtsp" | "file" | "url"
  level: number
  peak: number
  meter: number
  floor: number | null
  loud: boolean
  /** sequence number of the newest spectrogram column */
  seq: number
  bands: number
  /** spectrogram columns per second */
  rate: number
  /** base64 uint8, newest columns last, `bands` values each (0 = quiet, 255 = loud) */
  cols: string
  labels: SoundLabel[]
  heard: { label: string; score: number } | null
  events: SoundPing[]
  classifier: ClassifierInfo
}

export interface SoundClassRule {
  enabled: boolean
  threshold: number
  cooldown_s: number
  notify: boolean
}

export interface SoundRules {
  classes: Record<string, SoundClassRule>
  loud: { enabled: boolean; db_above_floor: number; min_ms: number; cooldown_s: number; notify: boolean }
  display_threshold: number
}

export interface SoundClassMeta {
  id: string
  label: string
  icon: string
  notable: boolean
}

export interface SoundConfig {
  ok: boolean
  rules: SoundRules
  classes: SoundClassMeta[]
  model: { name: string; license: string; status: string; url: string }
  classifier: ClassifierInfo
  privacy: string
}

export type SoundRulesPatch = {
  classes?: Record<string, Partial<SoundClassRule>>
  loud?: Partial<SoundRules["loud"]>
  display_threshold?: number
}

const ICONS: Record<string, LucideIcon> = {
  mic: Mic,
  megaphone: Megaphone,
  glass: GlassWater,
  dog: Dog,
  cat: Cat,
  knock: Hand,
  door: DoorOpen,
  footprints: Footprints,
  siren: Siren,
  smoke: AlarmSmoke,
  baby: Baby,
  bang: Bomb,
  music: Music,
  car: Car,
  bell: BellRing,
  clap: HandMetal,
  loud: Volume2,
}

export function soundIcon(name: string | undefined): LucideIcon {
  return (name && ICONS[name]) || AudioLines
}

/** Icon of a sound event kind ("sound:glass_break") or ping kind ("glass_break" / "loud"). */
const KIND_TO_ICON: Record<string, string> = {
  speech: "mic", scream: "megaphone", glass_break: "glass", dog: "dog", cat: "cat", knock: "knock", door: "door",
  footsteps: "footprints", alarm: "siren", smoke_alarm: "smoke", baby_cry: "baby", bang: "bang", music: "music",
  vehicle: "car", doorbell: "bell", clap: "clap", loud: "loud",
}
export function soundKindIcon(kind: string): LucideIcon {
  return soundIcon(KIND_TO_ICON[kind.replace(/^sound:/, "")])
}

export const isSoundKind = (kind: string) => kind.startsWith("sound:")

/** Spectrogram columns of a payload: Uint8Array per column, oldest first. */
export function decodeColumns(p: Pick<SoundPayload, "cols" | "bands">): Uint8Array[] {
  if (!p.cols || !p.bands) return []
  let raw: string
  try {
    raw = atob(p.cols)
  } catch {
    return []
  }
  const n = Math.floor(raw.length / p.bands)
  const out: Uint8Array[] = []
  for (let c = 0; c < n; c++) {
    const col = new Uint8Array(p.bands)
    for (let b = 0; b < p.bands; b++) col[b] = raw.charCodeAt(c * p.bands + b)
    out.push(col)
  }
  return out
}

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

export const soundApi = {
  config: (signal?: AbortSignal) => fetch("/api/sound/config", { signal }).then((r) => json<SoundConfig>(r)),
  patch: (patch: SoundRulesPatch) =>
    fetch("/api/sound/config", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }).then((r) => json<SoundConfig>(r)),
  live: (camera: string, signal?: AbortSignal) =>
    fetch(`/api/sound/${encodeURIComponent(camera)}`, { signal }).then((r) => json<SoundPayload | { on: false }>(r)),
}

/**
 * Newest sound payload for ``cameraId`` while ``on`` (null otherwise). Uses the
 * SSE topic; polls /api/sound/<id> while the stream is down.
 */
export function useSound(cameraId: string | null | undefined, on: boolean): SoundPayload | null {
  const [data, setData] = useState<SoundPayload | null>(null)
  const poll = useShouldPoll()
  useLiveTopic(
    "sound",
    (d) => {
      if (d && d.camera === cameraId && d.on) setData(d)
    },
    on && Boolean(cameraId),
  )
  useEffect(() => {
    setData(null)
  }, [cameraId, on])
  useEffect(() => {
    if (!on || !cameraId || !poll) return
    const ctl = new AbortController()
    const tick = () =>
      soundApi
        .live(cameraId, ctl.signal)
        .then((d) => "camera" in d && d.on && setData(d as SoundPayload))
        .catch(() => {})
    tick()
    const t = window.setInterval(tick, 500)
    return () => {
      ctl.abort()
      window.clearInterval(t)
    }
  }, [cameraId, on, poll])
  return on ? data : null
}

/** Sound rules + class list (loaded once per mount; patches apply optimistically). */
export function useSoundConfig() {
  const [config, setConfig] = useState<SoundConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const seq = useRef(0)
  useEffect(() => {
    const ctl = new AbortController()
    soundApi
      .config(ctl.signal)
      .then(setConfig)
      .catch((e: Error) => e.name !== "AbortError" && setError(e.message))
    return () => ctl.abort()
  }, [])
  const update = useCallback((patch: SoundRulesPatch) => {
    const mine = ++seq.current
    setConfig((c) => (c ? { ...c, rules: mergeRules(c.rules, patch) } : c))
    soundApi
      .patch(patch)
      .then((c) => mine === seq.current && setConfig(c))
      .catch((e: Error) => setError(e.message))
  }, [])
  return { config, error, update }
}

function mergeRules(r: SoundRules, p: SoundRulesPatch): SoundRules {
  const classes = { ...r.classes }
  for (const [k, v] of Object.entries(p.classes ?? {})) classes[k] = { ...classes[k], ...v }
  return {
    classes,
    loud: { ...r.loud, ...(p.loud ?? {}) },
    display_threshold: p.display_threshold ?? r.display_threshold,
  }
}

// ------------------------------------------------------------------ radar ping store
/** A sound ping as the radar draws it: `at` is the browser time it first showed up. */
export interface RadarPing extends SoundPing {
  key: string
  at: number
}

const PING_SHOW_MS = 4000
let pings: RadarPing[] = []
const pingSubs = new Set<() => void>()
const seenPings = new Set<string>()

function notifyPings() {
  pingSubs.forEach((fn) => fn())
}

/** Shared by the Sound HUD (writes) and the radar (reads); no direction - mono audio. */
export const soundPings = {
  get: () => pings,
  subscribe(fn: () => void) {
    pingSubs.add(fn)
    return () => pingSubs.delete(fn)
  },
  /** Feed the events of a payload; `primed` false = first payload, old events are not pinged. */
  ingest(events: SoundPing[], primed: boolean) {
    const now = performance.now()
    const wallNow = Date.now() / 1000
    let changed = false
    for (const e of events) {
      const key = `${e.ts}-${e.id}`
      if (seenPings.has(key)) continue
      seenPings.add(key)
      if (!primed && wallNow - e.ts > 3) continue
      pings = [...pings, { ...e, key, at: now }]
      changed = true
    }
    const fresh = pings.filter((p) => now - p.at < PING_SHOW_MS)
    if (fresh.length !== pings.length) {
      pings = fresh
      changed = true
    }
    if (seenPings.size > 200) {
      // keep only what the server still lists, so nothing old pings again
      const keep = new Set(events.map((e) => `${e.ts}-${e.id}`))
      for (const k of seenPings) if (!keep.has(k)) seenPings.delete(k)
    }
    if (changed) notifyPings()
  },
  clear() {
    if (!pings.length) return
    pings = []
    notifyPings()
  },
  showMs: PING_SHOW_MS,
}
