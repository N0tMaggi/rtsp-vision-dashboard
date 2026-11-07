import { useCallback, useEffect, useRef, useState } from "react"
import { useLiveTopic, useShouldPoll } from "@/hooks/use-live-stream"
import { api } from "@/lib/api"
import type { CameraStatus } from "@/lib/types"

const INTERVAL_MS = 1000

/**
 * Status of the active camera: pushed by the live stream; polls /status
 * only while the stream is down. ``refresh`` fetches once (after commands).
 */
export function useCameraStatus(cameraId: string | null) {
  const [status, setStatus] = useState<CameraStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cache = useRef(new Map<string, CameraStatus>())
  const ctrl = useRef<AbortController | null>(null)
  const poll = useShouldPoll()

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

  useLiveTopic("status", (s) => {
    if (s.camera?.id !== cameraId) return
    cache.current.set(s.camera.id, s)
    setStatus(s)
    setError(null)
  })

  useEffect(() => {
    setStatus(cameraId ? (cache.current.get(cameraId) ?? null) : null)
    setError(null)
  }, [cameraId])

  useEffect(() => {
    if (!poll) return
    void refresh()
    const t = window.setInterval(() => void refresh(), INTERVAL_MS)
    const onVis = () => !document.hidden && void refresh()
    document.addEventListener("visibilitychange", onVis)
    return () => {
      window.clearInterval(t)
      document.removeEventListener("visibilitychange", onVis)
      ctrl.current?.abort()
    }
  }, [poll, refresh])

  return { status, error, refresh }
}
