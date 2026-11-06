import { useCallback, useEffect, useRef, useState } from "react"
import { alertsApi, type AlertsConfig } from "@/lib/alerts"

const INTERVAL_MS = 1500

/** Zone rules, tripwires (with live counters) and channel status of one camera. */
export function useAlertsConfig(cameraId: string | null) {
  const [config, setConfig] = useState<AlertsConfig | null>(null)
  const ctrl = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    if (!cameraId || document.hidden) return
    ctrl.current?.abort()
    const c = new AbortController()
    ctrl.current = c
    try {
      setConfig(await alertsApi.config(cameraId, c.signal))
    } catch {
      /* keep last */
    }
  }, [cameraId])

  useEffect(() => {
    setConfig(null)
    void refresh()
    const t = window.setInterval(() => void refresh(), INTERVAL_MS)
    return () => {
      window.clearInterval(t)
      ctrl.current?.abort()
    }
  }, [refresh])

  /** Run a mutation, then reload (also on failure, so the UI never drifts). */
  const act = useCallback(
    (fn: () => Promise<unknown>) => {
      void fn().then(refresh, refresh)
    },
    [refresh],
  )

  return { config, refresh, act }
}
