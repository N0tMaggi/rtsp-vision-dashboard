import { useEffect, useRef, useState } from "react"
import { useLiveTopic, useShouldPoll } from "@/hooks/use-live-stream"
import { liveStream } from "@/lib/live-stream"
import { api } from "@/lib/api"
import type { TargetsResponse } from "@/lib/types"

const INTERVAL_MS = 120

/**
 * Live target boxes for hover / click-to-lock / follow-zoom.
 * Pushed by the live stream once per rendered frame (<= 12 Hz); polls only
 * while the stream is down. An older backend without the endpoint yields null.
 */
export function useTargets(cameraId: string | null, enabled: boolean) {
  const [data, setData] = useState<TargetsResponse | null>(null)
  const unsupported = useRef(false)
  const poll = useShouldPoll()

  useEffect(() => {
    setData(null)
    unsupported.current = false
  }, [cameraId])

  // the stream is per camera: a payload belongs to the camera it was opened for
  useLiveTopic(
    "targets",
    (d) => {
      if (cameraId && liveStream.healthy) setData(d)
    },
    Boolean(cameraId) && enabled,
  )

  useEffect(() => {
    if (!cameraId || !enabled) {
      setData((d) => (d && d.targets.length ? { ...d, targets: [] } : d))
      return
    }
    if (!poll) return
    let stopped = false
    let ctrl: AbortController | null = null
    let timer: number | undefined

    const tick = async () => {
      if (stopped || unsupported.current) return
      if (!document.hidden) {
        ctrl = new AbortController()
        try {
          setData(await api.targets(cameraId, ctrl.signal))
        } catch (e) {
          if ((e as Error).message === "HTTP 404") unsupported.current = true
        }
      }
      if (!stopped) timer = window.setTimeout(tick, INTERVAL_MS)
    }
    void tick()
    return () => {
      stopped = true
      window.clearTimeout(timer)
      ctrl?.abort()
    }
  }, [cameraId, enabled, poll])

  return data
}
