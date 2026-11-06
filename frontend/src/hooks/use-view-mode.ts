import { useCallback, useEffect, useState } from "react"

export type ViewMode = "single" | "grid"

const STORAGE_KEY = "securitycam.view-mode.v1"

function load(): ViewMode {
  try {
    return localStorage.getItem(STORAGE_KEY) === "grid" ? "grid" : "single"
  } catch {
    return "single"
  }
}

/**
 * Single camera vs. multi-camera grid, remembered per browser.
 * The grid only exists with 2+ cameras; with one camera the mode is always single.
 */
export function useViewMode(cameraCount: number) {
  const [stored, setStored] = useState<ViewMode>(load)
  const gridAllowed = cameraCount > 1
  const mode: ViewMode = gridAllowed ? stored : "single"

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, stored)
    } catch {
      /* not remembered */
    }
  }, [stored])

  const setMode = useCallback((m: ViewMode) => setStored(m), [])
  const toggle = useCallback(() => setStored((m) => (m === "grid" ? "single" : "grid")), [])

  return { mode, setMode, toggle, gridAllowed }
}
