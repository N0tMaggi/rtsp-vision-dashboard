import { useCallback, useEffect, useMemo, useState } from "react"
import { MotionConfig, motion } from "motion/react"
import { SlidersHorizontal } from "lucide-react"
import { CameraSwitcher } from "@/components/camera-switcher"
import { EspMenu } from "@/components/esp-menu"
import { Stage } from "@/components/stage"
import type { Tool } from "@/components/stage-toolbar"
import { LiveReadout, TelemetryBar } from "@/components/telemetry"
import { TooltipProvider } from "@/components/ui/tooltip"
import { useCameraStatus } from "@/hooks/use-camera-status"
import { useEspSettings } from "@/hooks/use-esp-settings"
import { useTargets } from "@/hooks/use-targets"
import { useWorkspace } from "@/hooks/use-workspace"
import { api } from "@/lib/api"
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

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)")
    const on = () => setDesktop(mq.matches)
    mq.addEventListener("change", on)
    return () => mq.removeEventListener("change", on)
  }, [])

  useEffect(() => {
    api
      .cameras()
      .then((r) => {
        setCameras(r.cameras)
        setDefaultId(r.default_camera_id)
        setModels(r.available_models ?? [])
      })
      .catch(() => setLoadError("The SecurityCam server is not responding. Start it with python main.py, then reload this page."))
  }, [])

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
                rightInset={menuOpen && desktop ? 420 : undefined}
              />

              <motion.header
                initial={{ opacity: 0, y: -12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="absolute inset-x-0 top-0 z-20 flex items-center gap-3 p-4 md:p-5"
              >
                <span className="hidden shrink-0 text-sm font-semibold sm:block">
                  SecurityCam<span className="text-primary">_</span>
                </span>
                <div className="min-w-0 flex-1">
                  <CameraSwitcher cameras={cameras} activeId={activeId} onSelect={open} />
                </div>
                <LiveReadout status={status} live={live} />
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
              />

              <motion.footer
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
                className="absolute inset-x-0 bottom-0 z-10"
              >
                <TelemetryBar status={status} />
              </motion.footer>
            </>
          )}
        </div>
      </TooltipProvider>
    </MotionConfig>
  )
}
