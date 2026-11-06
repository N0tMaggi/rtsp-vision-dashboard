import { useCallback, useEffect, useState } from "react"
import { opsApi, type PresetsResponse } from "@/lib/ops-api"

const INTERVAL_MS = 15000

/** Presets + schedules for a camera (global ones included). Refreshes while `active`. */
export function usePresets(cameraId: string | null, active: boolean) {
  const [data, setData] = useState<PresetsResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    if (!cameraId) return
    try {
      setData(await opsApi.presets(cameraId))
      setError(null)
    } catch {
      setError("Could not load presets.")
    }
  }, [cameraId])

  useEffect(() => {
    if (!active) return
    void refresh()
    const t = window.setInterval(() => !document.hidden && void refresh(), INTERVAL_MS)
    return () => window.clearInterval(t)
  }, [active, refresh])

  /** Runs a mutation, then reloads; returns false (and sets an error) on failure. */
  const mutate = useCallback(
    async (fn: () => Promise<unknown>, failMessage: string) => {
      try {
        await fn()
        setError(null)
        return true
      } catch {
        setError(failMessage)
        return false
      } finally {
        await refresh()
      }
    },
    [refresh],
  )

  return { data, error, refresh, mutate }
}
