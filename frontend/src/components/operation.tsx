import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ArchiveRestore, Bookmark, Move } from "lucide-react"
import { BackupPanel } from "@/components/backup-panel"
import { CameraGrid } from "@/components/camera-grid"
import type { ExtraCategory } from "@/components/esp-menu"
import { PresetsPanel } from "@/components/presets-panel"
import { PtzPad } from "@/components/ptz-pad"
import { PtzPanel } from "@/components/ptz-panel"
import { ViewToggle } from "@/components/view-toggle"
import { useOverview } from "@/hooks/use-overview"
import { usePresets } from "@/hooks/use-presets"
import { usePtz } from "@/hooks/use-ptz"
import { useViewMode } from "@/hooks/use-view-mode"
import type { Camera } from "@/lib/types"

interface Options {
  cameras: Camera[]
  activeId: string | null
  open: (id: string) => void
  /** Reload the active camera's status (after a preset / import changed settings). */
  refresh: () => unknown
  menuOpen: boolean
  rightInset: number
}

const isTyping = (e: KeyboardEvent) =>
  Boolean((e.target as HTMLElement).closest("input, textarea, select, [role=combobox], [role=listbox], [role=slider]"))

/**
 * Operation features wired into the app shell: grid view, presets &
 * schedules, PTZ and backup. App only places the returned pieces.
 */
export function useOperation({ cameras, activeId, open, refresh, menuOpen, rightInset }: Options) {
  const { mode, setMode, toggle, gridAllowed } = useViewMode(cameras.length)
  const [category, setCategory] = useState("visual")
  const [ptzMode, setPtzMode] = useState(false)
  const [aim, setAim] = useState<{ x: number; y: number; key: number } | null>(null)
  const single = mode === "single"

  const overview = useOverview(mode === "grid")
  const presets = usePresets(activeId, menuOpen && category === "presets")
  const ptzPanelOpen = menuOpen && category === "ptz"
  const ptz = usePtz(activeId, ptzPanelOpen || ptzMode, ptzMode && single)

  // Turning PTZ mode off or switching camera ends any held move (handled in usePtz).
  useEffect(() => setPtzMode(false), [activeId])

  const openSingle = useCallback(
    (id: string) => {
      open(id)
      setMode("single")
    },
    [open, setMode],
  )

  // G toggles the grid, 1-9 pick a camera (and open it from the grid).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return
      if ((e.key === "g" || e.key === "G") && gridAllowed) {
        toggle()
        return
      }
      if (/^[1-9]$/.test(e.key)) {
        const cam = cameras[Number(e.key) - 1]
        if (cam) openSingle(cam.id)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [cameras, gridAllowed, toggle, openSingle])

  const ptzReady = ptz.ready
  const centerEnabled = ptzMode && ptzReady && Boolean(ptz.status?.caps.relative)
  const onAimAt = useMemo(
    () =>
      centerEnabled
        ? (u: number, v: number, x: number, y: number) => {
            setAim({ x, y, key: Date.now() })
            void ptz.center(u, v)
          }
        : undefined,
    [centerEnabled, ptz],
  )

  const extraCategories: ExtraCategory[] = [
    {
      id: "presets",
      label: "Presets",
      icon: Bookmark,
      body: <PresetsPanel cameraId={activeId} cameras={cameras} presets={presets} onApplied={() => void refresh()} />,
    },
    { id: "ptz", label: "PTZ", icon: Move, body: <PtzPanel ptz={ptz} mode={ptzMode} onModeChange={setPtzMode} /> },
    {
      id: "backup",
      label: "Backup",
      icon: ArchiveRestore,
      body: (
        <BackupPanel
          onImported={() => {
            void refresh()
            void presets.refresh()
          }}
        />
      ),
    },
  ]

  const viewToggle: ReactNode = gridAllowed ? <ViewToggle mode={mode} onChange={setMode} /> : null

  const grid: ReactNode =
    mode === "grid" ? (
      <CameraGrid cameras={cameras} activeId={activeId} overview={overview} onOpen={openSingle} rightInset={rightInset} />
    ) : null

  const overlay: ReactNode = (
    <>
      <AnimatePresence>{single && ptzMode && ptzReady && <PtzPad key="ptz-pad" ptz={ptz} rightInset={rightInset} />}</AnimatePresence>
      <AnimatePresence>
        {aim && (
          <motion.span
            key={aim.key}
            aria-hidden
            className="pointer-events-none fixed z-30 size-10 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary"
            style={{ left: aim.x, top: aim.y }}
            initial={{ opacity: 1, scale: 0.3 }}
            animate={{ opacity: 0, scale: 1.8 }}
            transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
            onAnimationComplete={() => setAim((a) => (a?.key === aim.key ? null : a))}
          />
        )}
      </AnimatePresence>
    </>
  )

  return {
    mode,
    viewToggle,
    grid,
    overlay,
    onAimAt,
    extraCategories,
    onCategoryChange: setCategory,
  }
}
