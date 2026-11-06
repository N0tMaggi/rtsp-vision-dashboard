import { useCallback, useEffect, useRef, useState } from "react"
import { api } from "@/lib/api"
import { opsApi, type PtzSettings, type PtzStatus } from "@/lib/ops-api"

const POLL_MS = 1500
/** While the camera moves or calibrates, the anchoring status line follows closely. */
const POLL_FAST_MS = 400
/** Re-send a held move well inside the server watchdog (1.5 s). */
const HOLD_REPEAT_MS = 500

export interface Vector {
  x: number
  y: number
  zoom: number
}

const ZERO: Vector = { x: 0, y: 0, zoom: 0 }
const isZero = (v: Vector) => v.x === 0 && v.y === 0 && v.zoom === 0

/**
 * ONVIF PTZ state for one camera.
 * - Status is cheap and never logs in; connecting happens on first use of the
 *   panel / PTZ mode (once per camera and page load) or via the retry button,
 *   so a wrong password cannot lock the camera account through polling.
 * - Held moves are re-sent every 500 ms; letting go (or losing focus) stops.
 */
export function usePtz(cameraId: string | null, active: boolean, keyboard: boolean) {
  const [status, setStatus] = useState<PtzStatus | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const tried = useRef(new Set<string>())
  const vector = useRef<Vector>(ZERO)
  const repeat = useRef<number | undefined>(undefined)

  const refresh = useCallback(async () => {
    if (!cameraId || document.hidden) return
    try {
      setStatus(await opsApi.ptz(cameraId))
    } catch {
      /* server unreachable: keep last state */
    }
  }, [cameraId])

  useEffect(() => {
    setStatus(null)
    setMessage(null)
    void refresh()
  }, [refresh])

  const fast = Boolean(status?.pose?.moving || status?.pose?.calibration?.state === "running")
  useEffect(() => {
    if (!active) return
    const t = window.setInterval(() => void refresh(), fast ? POLL_FAST_MS : POLL_MS)
    return () => window.clearInterval(t)
  }, [active, fast, refresh])

  const connect = useCallback(async () => {
    if (!cameraId) return
    tried.current.add(cameraId)
    try {
      const r = await opsApi.ptzConnect(cameraId)
      setStatus(r.ptz)
    } catch (e) {
      const body = (e as { body?: { ptz?: PtzStatus } }).body
      if (body?.ptz) setStatus(body.ptz)
    }
  }, [cameraId])

  // First use per camera connects automatically; failures wait for the user.
  useEffect(() => {
    if (!active || !cameraId || !status) return
    if (status.available && status.configured && status.state === "idle" && !tried.current.has(cameraId)) void connect()
  }, [active, cameraId, status, connect])

  const ready = status?.state === "ready"

  const send = useCallback(
    (v: Vector) => {
      if (!cameraId) return
      const p = isZero(v) ? opsApi.ptzStop(cameraId) : opsApi.ptzMove(cameraId, v.x, v.y, v.zoom)
      p.catch(() => void refresh())
    },
    [cameraId, refresh],
  )

  const hold = useCallback(
    (v: Vector) => {
      const prev = vector.current
      vector.current = v
      window.clearInterval(repeat.current)
      if (isZero(v)) {
        if (!isZero(prev)) send(ZERO)
        return
      }
      send(v)
      repeat.current = window.setInterval(() => send(vector.current), HOLD_REPEAT_MS)
    },
    [send],
  )

  // Stop on camera switch / unmount / tab hidden.
  useEffect(() => {
    const stopAll = () => hold(ZERO)
    window.addEventListener("blur", stopAll)
    document.addEventListener("visibilitychange", stopAll)
    return () => {
      window.removeEventListener("blur", stopAll)
      document.removeEventListener("visibilitychange", stopAll)
      stopAll()
    }
  }, [hold])

  // Arrow keys (and PageUp/PageDown for zoom) while PTZ mode is on.
  useEffect(() => {
    if (!keyboard || !ready) return
    const pressed = new Set<string>()
    const keys = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "PageUp", "PageDown"]
    const apply = () =>
      hold({
        x: (pressed.has("ArrowRight") ? 1 : 0) - (pressed.has("ArrowLeft") ? 1 : 0),
        y: (pressed.has("ArrowUp") ? 1 : 0) - (pressed.has("ArrowDown") ? 1 : 0),
        zoom: (pressed.has("PageUp") ? 1 : 0) - (pressed.has("PageDown") ? 1 : 0),
      })
    const down = (e: KeyboardEvent) => {
      if (!keys.includes(e.key)) return
      if ((e.target as HTMLElement).closest("input, textarea, [role=combobox], [role=listbox], [role=slider]")) return
      e.preventDefault()
      if (e.repeat || pressed.has(e.key)) return
      pressed.add(e.key)
      apply()
    }
    const up = (e: KeyboardEvent) => {
      if (!pressed.delete(e.key)) return
      apply()
    }
    window.addEventListener("keydown", down)
    window.addEventListener("keyup", up)
    return () => {
      window.removeEventListener("keydown", down)
      window.removeEventListener("keyup", up)
      if (pressed.size) hold(ZERO)
    }
  }, [keyboard, ready, hold])

  const act = useCallback(
    async (fn: () => Promise<unknown>, fail: string) => {
      try {
        await fn()
        setMessage(null)
      } catch {
        setMessage(fail)
      }
      await refresh()
    },
    [refresh],
  )

  const center = useCallback(
    (u: number, v: number) => cameraId && act(() => opsApi.ptzCenter(cameraId, u, v), "The camera could not be moved."),
    [cameraId, act],
  )

  const updateSettings = useCallback(
    async (patch: Partial<PtzSettings>) => {
      if (!cameraId) return
      setStatus((s) => (s ? { ...s, settings: { ...s.settings, ...patch } } : s))
      await opsApi.ptzSettings(cameraId, patch).catch(() => setMessage("Calibration could not be saved."))
    },
    [cameraId],
  )

  const setWorldAnchor = useCallback(
    async (on: boolean) => {
      if (!cameraId) return
      setStatus((s) => (s?.pose ? { ...s, pose: { ...s.pose, world_anchor: on } } : s))
      await api.updateEsp(cameraId, { world_anchor: on }).catch(() => setMessage("The setting could not be saved."))
      await refresh()
    },
    [cameraId, refresh],
  )

  return {
    status,
    message,
    ready,
    setWorldAnchor,
    reanchor: () => cameraId && act(() => opsApi.ptzReanchor(cameraId), "Could not re-anchor."),
    syncOnvif: () => cameraId && act(() => opsApi.ptzSyncOnvif(cameraId), "The camera reports no usable position."),
    calibrate: (degrees = 20) =>
      cameraId && act(() => opsApi.ptzCalibrate(cameraId, degrees), "Calibration could not start (camera busy or not ready)."),
    connect,
    hold,
    center,
    home: () => cameraId && act(() => opsApi.ptzHome(cameraId), "Home position failed."),
    gotoPreset: (token: string) => cameraId && act(() => opsApi.ptzGotoPreset(cameraId, token), "Could not go to that position."),
    savePreset: (name: string) => cameraId && act(() => opsApi.ptzSavePreset(cameraId, name), "Could not save the position."),
    deletePreset: (token: string) => cameraId && act(() => opsApi.ptzDeletePreset(cameraId, token), "Could not delete the position."),
    updateSettings,
  }
}

export type PtzControls = ReturnType<typeof usePtz>
