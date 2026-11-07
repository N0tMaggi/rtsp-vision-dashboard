import { useEffect, useState } from "react"
import { useLiveOverview, useLiveTopic, useShouldPoll } from "@/hooks/use-live-stream"
import { opsApi, type CameraOverview } from "@/lib/ops-api"

const INTERVAL_MS = 1500

/**
 * fps / detections / live flag of every camera while the grid is visible:
 * pushed by the live stream (it adds the overview topic on request), polled
 * only while the stream is down.
 */
export function useOverview(enabled: boolean) {
  const [cameras, setCameras] = useState<Map<string, CameraOverview>>(new Map())
  const poll = useShouldPoll()
  useLiveOverview(enabled)
  useLiveTopic("overview", (r) => setCameras(new Map(r.cameras.map((c) => [c.id, c]))), enabled)

  useEffect(() => {
    if (!enabled || !poll) return
    let ctrl: AbortController | null = null
    const tick = async () => {
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
    void tick()
    const t = window.setInterval(() => void tick(), INTERVAL_MS)
    return () => {
      window.clearInterval(t)
      ctrl?.abort()
    }
  }, [enabled, poll])

  return cameras
}
