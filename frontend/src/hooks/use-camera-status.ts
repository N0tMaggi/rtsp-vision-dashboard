import { useCallback, useEffect, useRef, useState } from "react"
import { api } from "@/lib/api"
import type { CameraStatus } from "@/lib/types"

const INTERVAL_MS = 1000

/** Polls /status for the active camera while the page is visible. */
export function useCameraStatus(cameraId: string | null) {
  const [status, setStatus] = useState<CameraStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cache = useRef(new Map<string, CameraStatus>())
  const ctrl = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    if (!cameraId || document.hidden) return
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    try {
      const s = await api.status(cameraId, c.signal)
      cache.current.set(cameraId, s)
      setStatus(s)
      setError(null)
    } catch (e) {
      if ((e as Error).name === "AbortError") return
      setError("Status request failed")
    }
  }, [cameraId])

  useEffect(() => {
    setStatus(cameraId ? (cache.current.get(cameraId) ?? null) : null)
    setError(null)
    void refresh()
    const t = window.setInterval(() => void refresh(), INTERVAL_MS)
    const onVis = () => !document.hidden && void refresh()
    document.addEventListener("visibilitychange", onVis)
    return () => {
      window.clearInterval(t)
      document.removeEventListener("visibilitychange", onVis)
      ctrl.current?.abort()
    }
  }, [cameraId, refresh])

  return { status, error, refresh }
}
