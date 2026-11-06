import { useCallback, useEffect, useRef, useState } from "react"
import { api } from "@/lib/api"
import type { EspSettings } from "@/lib/types"

const DEBOUNCE_MS = 150

function normalize(s: EspSettings | undefined): EspSettings {
  if (!s) return {}
  // Backend renamed gay_mode -> rgb_mode; tolerate both.
  if (!("rgb_mode" in s) && "gay_mode" in s) return { ...s, rgb_mode: s.gay_mode }
  return s
}

/**
 * Server settings overlaid with local, not-yet-acknowledged edits so the
 * 1s status poll never snaps a control back while the user is dragging it.
 */
export function useEspSettings(cameraId: string | null, server: EspSettings | undefined) {
  const [pending, setPending] = useState<EspSettings>({})
  const queued = useRef<EspSettings>({})
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    setPending({})
    queued.current = {}
    window.clearTimeout(timer.current)
  }, [cameraId])

  const flush = useCallback(async () => {
    if (!cameraId) return
    const patch = queued.current
    queued.current = {}
    if (Object.keys(patch).length === 0) return
    try {
      await api.updateEsp(cameraId, patch)
    } finally {
      setPending((prev) => {
        const next = { ...prev }
        for (const k of Object.keys(patch)) if (next[k] === patch[k]) delete next[k]
        return next
      })
    }
  }, [cameraId])

  const update = useCallback(
    (key: string, value: boolean | number | string, immediate = true) => {
      setPending((p) => ({ ...p, [key]: value }))
      queued.current[key] = value
      window.clearTimeout(timer.current)
      if (immediate) void flush()
      else timer.current = window.setTimeout(() => void flush(), DEBOUNCE_MS)
    },
    [flush],
  )

  return { settings: { ...normalize(server), ...pending }, update }
}
