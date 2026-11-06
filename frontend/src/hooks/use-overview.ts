import { useEffect, useState } from "react"
import { opsApi, type CameraOverview } from "@/lib/ops-api"

const INTERVAL_MS = 1500

/** fps / detections / live flag of every camera, polled while the grid is visible. */
export function useOverview(enabled: boolean) {
  const [cameras, setCameras] = useState<Map<string, CameraOverview>>(new Map())

  useEffect(() => {
    if (!enabled) return
    let ctrl: AbortController | null = null
    const poll = async () => {
      if (document.hidden) return
      ctrl?.abort()
      ctrl = new AbortController()
      try {
        const r = await opsApi.overview(ctrl.signal)
        setCameras(new Map(r.cameras.map((c) => [c.id, c])))
      } catch {
        /* keep the last values; tiles show their own stream state */
      }
    }
    void poll()
    const t = window.setInterval(() => void poll(), INTERVAL_MS)
    return () => {
      window.clearInterval(t)
      ctrl?.abort()
    }
  }, [enabled])

  return cameras
}
