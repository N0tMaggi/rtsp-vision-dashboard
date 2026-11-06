import { useSyncExternalStore } from "react"

/** Per-viewer HUD toggles (frontend only), persisted in localStorage. */
export interface VisualPrefs {
  radar: boolean
  arrows: boolean
  infoCard: boolean
}

const KEY = "securitycam.visual-prefs.v1"
const DEFAULTS: VisualPrefs = { radar: true, arrows: true, infoCard: true }

function load(): VisualPrefs {
  try {
    const raw = window.localStorage.getItem(KEY)
    if (!raw) return DEFAULTS
    const parsed = JSON.parse(raw) as Partial<VisualPrefs>
    const out = { ...DEFAULTS }
    for (const k of Object.keys(DEFAULTS) as (keyof VisualPrefs)[]) {
      if (typeof parsed[k] === "boolean") out[k] = parsed[k]
    }
    return out
  } catch {
    return DEFAULTS
  }
}

let current: VisualPrefs = load()
const listeners = new Set<() => void>()

export function setVisualPref<K extends keyof VisualPrefs>(key: K, value: VisualPrefs[K]) {
  current = { ...current, [key]: value }
  try {
    window.localStorage.setItem(KEY, JSON.stringify(current))
  } catch {
    // private mode / blocked storage: keep the in-memory value
  }
  listeners.forEach((l) => l())
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => listeners.delete(l)
}

export function useVisualPrefs(): VisualPrefs {
  return useSyncExternalStore(subscribe, () => current, () => DEFAULTS)
}
