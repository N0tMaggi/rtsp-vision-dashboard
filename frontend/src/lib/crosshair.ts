import { useSyncExternalStore } from "react"
import type { AimKey, Target } from "./types"

/**
 * Crosshair drawn on the locked / PTZ-tracked target (frontend only, this browser).
 * Settings + user templates persist in localStorage; every read is validated, so
 * a broken or hand-edited value falls back to defaults instead of breaking the HUD.
 */

export const CROSSHAIR_STYLES = ["cross", "dot", "circle", "cross-dot", "t", "brackets", "chevron", "sniper"] as const
export type CrosshairStyle = (typeof CROSSHAIR_STYLES)[number]

export const STYLE_LABEL: Record<CrosshairStyle, string> = {
  cross: "Cross",
  dot: "Dot",
  circle: "Circle",
  "cross-dot": "Cross + Dot",
  t: "T",
  brackets: "Brackets",
  chevron: "Chevron",
  sniper: "Sniper",
}

const AIMS: AimKey[] = ["head", "face", "neck", "chest", "belly", "groin", "feet", "center"]

/** Look of a crosshair (what templates store). */
export interface CrosshairLook {
  /** Body point, or "autotrack" = same as the PTZ auto-track setting. */
  aim: AimKey | "autotrack"
  style: CrosshairStyle
  color: string
  /** Outer size in screen px. */
  size: number
  thickness: number
  /** Free space around the center in px. */
  gap: number
  outline: boolean
  opacity: number
  /** Degrees. */
  rotation: number
  spin: boolean
  pulse: boolean
  label: boolean
  allPersons: boolean
}

export interface CrosshairSettings extends CrosshairLook {
  enabled: boolean
}

export interface CrosshairTemplate {
  id: string
  name: string
  builtin?: boolean
  look: CrosshairLook
}

export const PALETTE = ["#ffd21f", "#f4f4f2", "#ff5a4e", "#7ee0a1", "#4dd2ff", "#ff6bd6", "#ffa64d", "#b48cff"] as const

export const LIMITS = {
  size: [12, 120],
  thickness: [1, 6],
  gap: [0, 24],
  opacity: [0.2, 1],
  rotation: [0, 359],
} as const

const DEFAULT_LOOK: CrosshairLook = {
  aim: "autotrack",
  style: "cross-dot",
  color: "#ffd21f",
  size: 34,
  thickness: 2,
  gap: 5,
  outline: true,
  opacity: 0.95,
  rotation: 0,
  spin: false,
  pulse: false,
  label: true,
  allPersons: false,
}

export const DEFAULT_SETTINGS: CrosshairSettings = { enabled: true, ...DEFAULT_LOOK }

export const BUILTIN_TEMPLATES: CrosshairTemplate[] = [
  { id: "builtin:classic", name: "Classic", builtin: true, look: { ...DEFAULT_LOOK } },
  {
    id: "builtin:dot",
    name: "Dot",
    builtin: true,
    look: { ...DEFAULT_LOOK, style: "dot", size: 12, thickness: 2, gap: 0, label: false },
  },
  {
    id: "builtin:sniper",
    name: "Sniper",
    builtin: true,
    look: { ...DEFAULT_LOOK, aim: "head", style: "sniper", color: "#ff5a4e", size: 64, thickness: 1, gap: 8, opacity: 0.9 },
  },
  {
    id: "builtin:brackets",
    name: "Brackets",
    builtin: true,
    look: { ...DEFAULT_LOOK, aim: "chest", style: "brackets", color: "#f4f4f2", size: 44, thickness: 2, gap: 0, outline: true },
  },
  {
    id: "builtin:minimal",
    name: "Minimal",
    builtin: true,
    look: { ...DEFAULT_LOOK, style: "cross", color: "#f4f4f2", size: 18, thickness: 1, gap: 4, outline: false, opacity: 0.7, label: false },
  },
  {
    id: "builtin:below-the-belt",
    name: "Below the Belt",
    builtin: true,
    look: { ...DEFAULT_LOOK, aim: "groin", style: "chevron", color: "#ff6bd6", size: 30, thickness: 3, gap: 3, pulse: true },
  },
]

// ------------------------------------------------------------------ validation
const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v))
const HEX = /^#[0-9a-f]{6}$/i

export function normalizeHex(v: string): string | null {
  const s = v.trim().toLowerCase()
  if (/^#?[0-9a-f]{3}$/.test(s)) {
    const h = s.replace("#", "")
    return `#${h[0]}${h[0]}${h[1]}${h[1]}${h[2]}${h[2]}`
  }
  if (/^#?[0-9a-f]{6}$/.test(s)) return s.startsWith("#") ? s : `#${s}`
  return null
}

function num(v: unknown, fallback: number, lim: readonly [number, number]) {
  return typeof v === "number" && Number.isFinite(v) ? clamp(v, lim) : fallback
}

const bool = (v: unknown, fallback: boolean) => (typeof v === "boolean" ? v : fallback)

/** Any object -> a valid look (unknown keys dropped, numbers clamped). */
export function sanitizeLook(raw: unknown, base: CrosshairLook = DEFAULT_LOOK): CrosshairLook {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>
  const aim = r.aim === "autotrack" || AIMS.includes(r.aim as AimKey) ? (r.aim as CrosshairLook["aim"]) : base.aim
  const style = CROSSHAIR_STYLES.includes(r.style as CrosshairStyle) ? (r.style as CrosshairStyle) : base.style
  const color = typeof r.color === "string" && normalizeHex(r.color) ? normalizeHex(r.color)! : base.color
  return {
    aim,
    style,
    color: HEX.test(color) ? color : DEFAULT_LOOK.color,
    size: num(r.size, base.size, LIMITS.size),
    thickness: num(r.thickness, base.thickness, LIMITS.thickness),
    gap: num(r.gap, base.gap, LIMITS.gap),
    outline: bool(r.outline, base.outline),
    opacity: num(r.opacity, base.opacity, LIMITS.opacity),
    rotation: Math.round(num(r.rotation, base.rotation, LIMITS.rotation)),
    spin: bool(r.spin, base.spin),
    pulse: bool(r.pulse, base.pulse),
    label: bool(r.label, base.label),
    allPersons: bool(r.allPersons, base.allPersons),
  }
}

export function lookOf(s: CrosshairSettings): CrosshairLook {
  const { enabled: _enabled, ...look } = s
  void _enabled
  return look
}

// ------------------------------------------------------------------ store
interface Stored {
  settings: CrosshairSettings
  templates: CrosshairTemplate[]
  /** Template the current settings came from (cleared by any edit). */
  activeId: string | null
}

const KEY = "securitycam.crosshair.v1"
const MAX_TEMPLATES = 50
const MAX_NAME = 32
const INITIAL: Stored = { settings: DEFAULT_SETTINGS, templates: [], activeId: "builtin:classic" }

function cleanName(v: unknown, fallback: string) {
  const s = typeof v === "string" ? v.trim().slice(0, MAX_NAME) : ""
  return s || fallback
}

function sanitizeTemplates(raw: unknown): CrosshairTemplate[] {
  if (!Array.isArray(raw)) return []
  const out: CrosshairTemplate[] = []
  for (const t of raw.slice(0, MAX_TEMPLATES)) {
    if (!t || typeof t !== "object") continue
    const r = t as Record<string, unknown>
    const id = typeof r.id === "string" && r.id && !r.id.startsWith("builtin:") ? r.id : newId()
    out.push({ id, name: cleanName(r.name, "Template"), look: sanitizeLook(r.look ?? r.settings) })
  }
  return out
}

function load(): Stored {
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return INITIAL
    const p = JSON.parse(raw) as Partial<Stored>
    const look = sanitizeLook(p.settings)
    const enabled = bool((p.settings as Partial<CrosshairSettings> | undefined)?.enabled, DEFAULT_SETTINGS.enabled)
    return {
      settings: { enabled, ...look },
      templates: sanitizeTemplates(p.templates),
      activeId: typeof p.activeId === "string" ? p.activeId : null,
    }
  } catch {
    return INITIAL
  }
}

let state: Stored = load()
const listeners = new Set<() => void>()

function commit(next: Stored) {
  state = next
  try {
    window.localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    // private mode / blocked storage: keep the in-memory value
  }
  listeners.forEach((l) => l())
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => listeners.delete(l)
}

export function useCrosshair(): Stored {
  return useSyncExternalStore(subscribe, () => state, () => INITIAL)
}

function newId() {
  return `user:${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`
}

export function setCrosshair(patch: Partial<CrosshairSettings>) {
  const settings = { ...state.settings, ...patch }
  const onlyEnabled = Object.keys(patch).every((k) => k === "enabled")
  commit({ ...state, settings, activeId: onlyEnabled ? state.activeId : null })
}

export function allTemplates(s: Stored = state): CrosshairTemplate[] {
  return [...BUILTIN_TEMPLATES, ...s.templates]
}

export function applyTemplate(id: string) {
  const t = allTemplates().find((x) => x.id === id)
  if (!t) return
  commit({ ...state, settings: { ...state.settings, ...t.look, enabled: true }, activeId: id })
}

function uniqueName(name: string, except?: string, extra: string[] = []) {
  const taken = new Set([...allTemplates().filter((t) => t.id !== except).map((t) => t.name), ...extra].map((n) => n.toLowerCase()))
  if (!taken.has(name.toLowerCase())) return name
  for (let i = 2; ; i++) {
    const n = `${name.slice(0, MAX_NAME - 4)} ${i}`
    if (!taken.has(n.toLowerCase())) return n
  }
}

export function saveTemplate(name: string): string | null {
  if (state.templates.length >= MAX_TEMPLATES) return null
  const t: CrosshairTemplate = { id: newId(), name: uniqueName(cleanName(name, "My Crosshair")), look: lookOf(state.settings) }
  commit({ ...state, templates: [...state.templates, t], activeId: t.id })
  return t.id
}

export function renameTemplate(id: string, name: string) {
  commit({
    ...state,
    templates: state.templates.map((t) => (t.id === id ? { ...t, name: uniqueName(cleanName(name, t.name), id) } : t)),
  })
}

export function overwriteTemplate(id: string) {
  commit({ ...state, templates: state.templates.map((t) => (t.id === id ? { ...t, look: lookOf(state.settings) } : t)), activeId: id })
}

export function deleteTemplate(id: string) {
  commit({ ...state, templates: state.templates.filter((t) => t.id !== id), activeId: state.activeId === id ? null : state.activeId })
}

// ------------------------------------------------------------------ export / import
const EXPORT_KIND = "securitycam.crosshair"

export function exportJson(): string {
  return JSON.stringify(
    {
      kind: EXPORT_KIND,
      version: 1,
      current: lookOf(state.settings),
      templates: state.templates.map(({ name, look }) => ({ name, look })),
    },
    null,
    2,
  )
}

/**
 * Accepts an export of this app ({kind, current, templates}), a list of templates,
 * a single template ({name, look}) or a bare look. Returns a summary or throws a
 * readable error.
 */
export function importJson(text: string): string {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    throw new Error("That is not valid JSON.")
  }
  if (!data || typeof data !== "object") throw new Error("Expected a JSON object or list.")
  const rec = data as Record<string, unknown>
  let rawTemplates: unknown[] = []
  let current: CrosshairLook | null = null
  if (Array.isArray(data)) rawTemplates = data
  else if (rec.kind !== undefined || "templates" in rec || "current" in rec) {
    if (rec.kind !== undefined && rec.kind !== EXPORT_KIND) throw new Error("This JSON is not a crosshair export.")
    if (Array.isArray(rec.templates)) rawTemplates = rec.templates
    if (rec.current && typeof rec.current === "object") current = sanitizeLook(rec.current)
  } else if ("look" in rec || "settings" in rec) rawTemplates = [rec]
  else if ("style" in rec || "color" in rec || "aim" in rec) rawTemplates = [{ name: "Imported", look: rec }]
  else throw new Error("No crosshair settings found in this JSON.")

  const valid = rawTemplates.filter((t) => t && typeof t === "object" && ("look" in (t as object) || "settings" in (t as object)))
  if (!valid.length && !current) throw new Error("No crosshair templates found in this JSON.")
  const room = MAX_TEMPLATES - state.templates.length
  const added: CrosshairTemplate[] = []
  for (const t of sanitizeTemplates(valid).slice(0, Math.max(0, room))) {
    // unique against existing templates and the ones of this batch
    const name = uniqueName(t.name, undefined, added.map((a) => a.name))
    added.push({ id: newId(), name, look: t.look })
  }
  const next: Stored = { ...state, templates: [...state.templates, ...added] }
  if (current) {
    next.settings = { ...state.settings, ...current }
    next.activeId = null
  }
  commit(next)
  const parts: string[] = []
  if (added.length) parts.push(`${added.length} template${added.length === 1 ? "" : "s"} added`)
  if (current) parts.push("current crosshair applied")
  if (valid.length > added.length) parts.push(`${valid.length - added.length} skipped (limit ${MAX_TEMPLATES})`)
  return parts.join(", ")
}

// ------------------------------------------------------------------ aim points
/** Box fractions from the top for backends without `aim` (x = box center). */
const FALLBACK_Y: Record<AimKey, number> = {
  head: 0.07,
  face: 0.1,
  neck: 0.17,
  chest: 0.3,
  belly: 0.45,
  groin: 0.53,
  feet: 0.97,
  center: 0.5,
}

/** Normalized point of `key` on a target (server pose point, else box estimate). */
export function aimPoint(t: Target, key: AimKey): [number, number] {
  const isPerson = t.label.toLowerCase() === "person"
  // non-persons only know "center"
  const k = isPerson ? key : "center"
  const p = t.aim?.[k]
  if (p && Number.isFinite(p[0]) && Number.isFinite(p[1])) return p
  const [x1, y1, x2, y2] = t.box
  return [(x1 + x2) / 2, y1 + (y2 - y1) * FALLBACK_Y[k]]
}
