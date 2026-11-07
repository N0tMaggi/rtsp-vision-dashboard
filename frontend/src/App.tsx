import { useCallback, useEffect, useMemo, useState } from "react"
import { MotionConfig, motion } from "motion/react"
import { Camera as CameraIcon, SlidersHorizontal } from "lucide-react"
import { AlarmToasts, SnapshotFlash, useToasts } from "@/components/alarm-toasts"
import { ClipPlayer } from "@/components/clip-player"
import { EventsButton, EventsDrawer } from "@/components/events-drawer"
import { CameraSwitcher } from "@/components/camera-switcher"
import { ComputeChip } from "@/components/compute-panel"
import { EspMenu } from "@/components/esp-menu"
import { PlateLog, PlateLogButton } from "@/components/plate-log"
import { useOperation } from "@/components/operation"
import { useVideoAnalysis } from "@/components/video-analysis"
import { Stage } from "@/components/stage"
import type { Tool } from "@/components/stage-toolbar"
import { LiveReadout, TelemetryBar } from "@/components/telemetry"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useCameraStatus } from "@/hooks/use-camera-status"
import { useEspSettings } from "@/hooks/use-esp-settings"
import { useEvents } from "@/hooks/use-events"
import { useLiveStream } from "@/hooks/use-live-stream"
import { usePlateLog } from "@/hooks/use-plate-log"
import { useTargets } from "@/hooks/use-targets"
import { useWorkspace } from "@/hooks/use-workspace"
import { alertsApi, type AlarmEvent } from "@/lib/alerts"
import { OPEN_COMPUTE_EVENT } from "@/lib/compute"
import { api } from "@/lib/api"
import { OPEN_PLATE_LOG_EVENT, plateLogApi, type PlateLogEntry } from "@/lib/platelog"
import type { Camera, NBox } from "@/lib/types"
import { cn } from "@/lib/utils"

export default function App() {
  const [cameras, setCameras] = useState<Camera[]>([])
  const [defaultId, setDefaultId] = useState<string | null>(null)
  const [models, setModels] = useState<string[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [menuOpen, setMenuOpen] = useState(() => window.matchMedia("(min-width: 768px)").matches)
  const [desktop, setDesktop] = useState(() => window.matchMedia("(min-width: 768px)").matches)
  const [tool, setTool] = useState<Tool>("select")
  const [follow, setFollow] = useState(false)
  const [lockOverride, setLockOverride] = useState<{ id: number | null; at: number } | null>(null)
  // alarms & events
  const [eventsOpen, setEventsOpen] = useState(false)
  const [playing, setPlaying] = useState<AlarmEvent | null>(null)
  const [flash, setFlash] = useState(0)
  const toast = useToasts()
  const { push: pushToast } = toast
  // license plate log (P)
  const [plateLogOpen, setPlateLogOpen] = useState(false)
  const plateLog = usePlateLog(plateLogOpen)
  const ev = useEvents((fresh) => fresh.slice(0, 3).reverse().forEach((e) => toast.push({ key: `${e.camera}/${e.id}`, type: "alarm", event: e })))

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)")
    const on = () => setDesktop(mq.matches)
    mq.addEventListener("change", on)
    return () => mq.removeEventListener("change", on)
  }, [])

  const loadCameras = useCallback(
    () =>
      api.cameras().then((r) => {
        setCameras(r.cameras)
        setDefaultId(r.default_camera_id)
        setModels(r.available_models ?? [])
      }),
    [],
  )
  useEffect(() => {
    loadCameras().catch(() => setLoadError("The SecurityCam server is not responding. Start it with python main.py, then reload this page."))
  }, [loadCameras])

  // M toggles the menu, like an in-game overlay.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.closest("input, textarea, [role=combobox], [role=listbox]")) return
      if (e.key === "m" || e.key === "M" || e.key === "Insert") setMenuOpen((o) => !o)
      if (e.key === "Escape") setMenuOpen(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const cameraMap = useMemo(() => new Map(cameras.map((c) => [c.id, c])), [cameras])
  const { activeId, open } = useWorkspace(cameras, defaultId)
  // one SSE connection per tab feeds status, targets, events, plates, playback, grid and PTZ
  const liveStream = useLiveStream(activeId)
  const { status, error, refresh } = useCameraStatus(activeId)
  const { settings, update } = useEspSettings(activeId, status?.esp_settings)
  const aiOn = status?.yolo_enabled ?? false
  const targets = useTargets(activeId, aiOn)

  // Optimistic lock state until the next targets poll confirms it.
  const serverLocked = targets?.locked ?? status?.locked ?? null
  const lockedId = lockOverride && Date.now() - lockOverride.at < 1500 ? lockOverride.id : serverLocked

  const lock = useCallback(
    (id: number | null) => {
      if (!activeId) return
      setLockOverride({ id, at: Date.now() })
      if (id === null) setFollow(false)
      api.lock(activeId, id).catch(() => setLockOverride(null))
    },
    [activeId],
  )

  const { markSeen } = ev
  useEffect(() => {
    if (eventsOpen) markSeen()
  }, [eventsOpen, ev.events, markSeen])

  // Plate log and events share the left side: opening one closes the other.
  const { markSeen: markPlatesSeen } = plateLog
  useEffect(() => {
    if (plateLogOpen) markPlatesSeen()
  }, [plateLogOpen, plateLog.unread, markPlatesSeen])
  useEffect(() => {
    if (eventsOpen) setPlateLogOpen(false)
  }, [eventsOpen])
  useEffect(() => {
    if (plateLogOpen) setEventsOpen(false)
  }, [plateLogOpen])

  // P toggles the plate log; the Recognition panel opens it through a window event.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.closest("input, textarea, select, [role=combobox], [role=listbox]") || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === "p" || e.key === "P") setPlateLogOpen((o) => !o)
      else if (e.key === "Escape") setPlateLogOpen(false)
    }
    const onOpen = () => setPlateLogOpen(true)
    window.addEventListener("keydown", onKey)
    window.addEventListener(OPEN_PLATE_LOG_EVENT, onOpen)
    return () => {
      window.removeEventListener("keydown", onKey)
      window.removeEventListener(OPEN_PLATE_LOG_EVENT, onOpen)
    }
  }, [])

  const snapshot = useCallback(async () => {
    if (!activeId) return
    setFlash((f) => f + 1)
    try {
      const { blob, name } = await alertsApi.snapshot(activeId)
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = name
      a.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 2000)
      pushToast({ key: `snap-${Date.now()}`, type: "info", title: "Snapshot saved", detail: name })
    } catch (e) {
      pushToast({
        key: `snap-${Date.now()}`,
        type: "info",
        tone: "error",
        title: (e as Error).message === "no_frame" ? "No frame to save yet" : "Snapshot failed",
      })
    }
  }, [activeId, pushToast])

  // S saves a snapshot of the rendered frame, E toggles the events drawer.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t.closest("input, textarea, [role=combobox], [role=listbox]") || e.ctrlKey || e.metaKey || e.altKey) return
      if (e.key === "s" || e.key === "S") void snapshot()
      else if (e.key === "e" || e.key === "E") setEventsOpen((o) => !o)
      else if (e.key === "Escape") setEventsOpen(false)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [snapshot])

  const addTripwire = (a: [number, number], b: [number, number]) =>
    activeId &&
    void alertsApi.addTripwire(activeId, a, b).catch(() =>
      toast.push({ key: `wire-${Date.now()}`, type: "info", tone: "error", title: "Could not create tripwire" }),
    )

  const zoneAction = (fn: () => Promise<unknown>) => void fn().then(refresh, refresh)
  const zones = {
    list: status?.zones ?? [],
    onDraw: () => {
      setTool("zone")
      if (!desktop) setMenuOpen(false)
    },
    onRename: (id: string, label: string) => activeId && zoneAction(() => api.updateZone(activeId, id, { label })),
    onRecolor: (id: string, color: string) => activeId && zoneAction(() => api.updateZone(activeId, id, { color })),
    onDelete: (id: string) => activeId && zoneAction(() => api.deleteZone(activeId, id)),
  }
  const addZone = (box: NBox) => activeId && zoneAction(() => api.addZone(activeId, box))

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    try {
      await fn()
      await refresh()
    } finally {
      setBusy(false)
    }
  }

  const camera = activeId ? cameraMap.get(activeId) : undefined
  const streamError = status?.stream_error || error || null
  const live = Boolean(camera?.has_stream && status && !streamError)
  const modelList = status?.available_models?.length ? status.available_models : models
  const rightInset = menuOpen && desktop ? 420 : undefined
  const ops = useOperation({ cameras, activeId, open, refresh, menuOpen, rightInset: rightInset ?? 24 })
  const video = useVideoAnalysis({ cameras, activeId, open, reloadCameras: loadCameras, rightInset: rightInset ?? 24, single: ops.mode === "single" })

  // "Show Live": lock the logged vehicle on its camera, switch there and follow it.
  const { openSingle } = ops
  const showLive = useCallback(
    async (entry: PlateLogEntry) => {
      const r = await plateLogApi.showLive(entry.id)
      openSingle(r.camera)
      setLockOverride({ id: r.target_id, at: Date.now() })
      setFollow(true)
      setPlateLogOpen(false)
    },
    [openSingle],
  )

  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider delayDuration={300}>
        <div className="relative h-dvh overflow-hidden">
          {loadError ? (
            <p role="alert" className="grid h-full place-items-center px-6 text-center text-sm text-destructive">
              {loadError}
            </p>
          ) : (
            <>
              {ops.grid}
              {ops.mode === "single" && (
              <motion.div layoutId={activeId ? `cam-${activeId}` : undefined} transition={{ type: "spring", stiffness: 300, damping: 34 }} className="absolute inset-0">
              <Stage
                camera={camera}
                error={camera?.has_stream ? streamError : null}
                aiOn={aiOn}
                targets={targets}
                lockedId={lockedId}
                tool={tool}
                onToolChange={setTool}
                follow={follow}
                onFollowChange={setFollow}
                onLock={lock}
                onZoneDrawn={addZone}
                onTripwireDrawn={addTripwire}
                rightInset={rightInset}
                onAimAt={ops.onAimAt}
                bottomInset={video.bottomInset}
              />
              </motion.div>
              )}
              {ops.overlay}

              <motion.header
                initial={{ opacity: 0, y: -12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="absolute inset-x-0 top-0 z-20 flex items-center gap-3 p-4 md:p-5"
              >
                <span className="hidden shrink-0 text-sm font-semibold sm:block">
                  SecurityCam<span className="text-primary">_</span>
                </span>
                <div className="flex min-w-0 flex-1 items-center gap-2">
                  <CameraSwitcher cameras={cameras} activeId={activeId} onSelect={open} onClose={(id) => void video.closeCamera(id)} />
                  {video.entry}
                </div>
                {ops.viewToggle}
                <ComputeChip
                  cameraId={activeId}
                  onOpen={() => {
                    setMenuOpen(true)
                    window.setTimeout(() => window.dispatchEvent(new Event(OPEN_COMPUTE_EVENT)), 0)
                  }}
                />
                <LiveReadout status={status} live={live} reconnecting={liveStream.state === "reconnecting"} />
                <button
                  type="button"
                  onClick={() => void snapshot()}
                  aria-label="Snapshot (S)"
                  title="Snapshot (S)"
                  disabled={!live}
                  className="hidden size-9 shrink-0 place-items-center rounded-full border border-border/70 bg-background/70 text-foreground backdrop-blur-md transition-colors duration-300 hover:border-primary/60 active:scale-95 disabled:opacity-40 sm:grid"
                >
                  <CameraIcon className="size-4" />
                </button>
                <PlateLogButton open={plateLogOpen} unread={plateLogOpen ? 0 : plateLog.unread} onClick={() => setPlateLogOpen((o) => !o)} />
                <EventsButton
                  open={eventsOpen}
                  unread={eventsOpen ? 0 : ev.unread}
                  alarm={ev.active.length > 0}
                  onClick={() => setEventsOpen((o) => !o)}
                />
                <button
                  type="button"
                  onClick={() => setMenuOpen((o) => !o)}
                  aria-expanded={menuOpen}
                  aria-label="ESP menu"
                  className={cn(
                    "grid size-9 shrink-0 place-items-center rounded-full border backdrop-blur-md transition-colors duration-300",
                    menuOpen
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border/70 bg-background/70 text-foreground hover:border-primary/60",
                  )}
                >
                  <SlidersHorizontal className="size-4" />
                </button>
              </motion.header>

              <EspMenu
                open={menuOpen}
                onClose={() => setMenuOpen(false)}
                status={status}
                models={modelList}
                settings={settings}
                onUpdate={update}
                busy={busy}
                onToggleAi={() => activeId && void run(() => api.toggleYolo(activeId))}
                onModelChange={(m) => activeId && void run(() => api.setModel(activeId, m))}
                zones={zones}
                onDrawTripwire={() => {
                  setTool("tripwire")
                  if (!desktop) setMenuOpen(false)
                }}
                extraCategories={ops.extraCategories}
                onCategoryChange={ops.onCategoryChange}
              />

              <EventsDrawer
                open={eventsOpen}
                onClose={() => setEventsOpen(false)}
                events={ev.events}
                active={ev.active}
                cameras={cameras}
                seenTs={ev.seen}
                onPlay={setPlaying}
                onDelete={(e) => void ev.remove(e)}
              />

              <PlateLog
                open={plateLogOpen}
                onClose={() => setPlateLogOpen(false)}
                cameras={cameras}
                log={plateLog}
                onShowLive={showLive}
              />

              <motion.footer
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="absolute inset-x-0 bottom-0 z-10"
              >
                <TelemetryBar status={status} />
              </motion.footer>
              {video.playbackBar}
              {video.library}
              <SnapshotFlash flashKey={flash} />
              <AlarmToasts
                toasts={toast.toasts}
                onDismiss={toast.dismiss}
                onOpen={(e) => {
                  setPlaying(e)
                  toast.dismiss(`${e.camera}/${e.id}`)
                }}
              />
              <ClipPlayer
                event={playing ? (ev.events.find((x) => x.id === playing.id && x.camera === playing.camera) ?? playing) : null}
                onClose={() => setPlaying(null)}
                onDelete={(e) => {
                  setPlaying(null)
                  void ev.remove(e)
                }}
              />
            </>
          )}
        </div>
      </TooltipProvider>
    </MotionConfig>
  )
}
