import { useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion, useTransform } from "motion/react"
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Crosshair,
  Download,
  Eye,
  Maximize2,
  Radio,
  Trash2,
  X,
} from "lucide-react"
import { useViewTransform } from "@/hooks/use-view-transform"
import { plateLogApi, relativeTime, type LiveTarget, type PlateLogEntry } from "@/lib/platelog"
import type { NBox } from "@/lib/types"
import { cn } from "@/lib/utils"
import { formatDay, formatTime } from "./clip-player"
import { PlateBadge, WatchFlag } from "./plate-badge"

const spring = { type: "spring", stiffness: 380, damping: 34 } as const
const LIVE_POLL_MS = 1500
const LIVE_WINDOW_S = 600

type ViewKind = "frame" | "vehicle"

/** Plate box mapped into the vehicle crop's own coordinates. */
function plateInCrop(e: PlateLogEntry): NBox | null {
  const p = e.plate_box
  const c = e.crop_box
  if (!p || !c) return null
  const w = c[2] - c[0]
  const h = c[3] - c[1]
  if (w <= 0 || h <= 0) return null
  const k = (v: number) => Math.min(1, Math.max(0, v))
  return [k((p[0] - c[0]) / w), k((p[1] - c[1]) / h), k((p[2] - c[0]) / w), k((p[3] - c[1]) / h)]
}

/**
 * Full-screen detail view of one plate-log entry: the snapshot frame with
 * smooth zoom/pan (wheel, drag, double-click), auto-zoomed onto the plate.
 */
export function PlateViewer({
  list,
  index,
  onIndex,
  onClose,
  onDelete,
  onShowLive,
  watched,
  onToggleWatch,
}: {
  list: PlateLogEntry[]
  index: number | null
  onIndex: (i: number) => void
  onClose: () => void
  onDelete: (e: PlateLogEntry) => void
  onShowLive: (e: PlateLogEntry) => Promise<void>
  watched: (plate: string) => boolean
  onToggleWatch: (plate: string) => void
}) {
  const entry = index !== null ? (list[index] ?? null) : null
  return (
    <AnimatePresence>
      {entry && index !== null && (
        <motion.div
          key="plate-viewer"
          className="absolute inset-0 z-50 grid place-items-center bg-[#16171a]/82 p-2 backdrop-blur-md md:p-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onPointerDown={(e) => e.target === e.currentTarget && onClose()}
          role="dialog"
          aria-modal="true"
          aria-label={`License plate ${entry.plate}`}
        >
          <ViewerCard
            entry={entry}
            index={index}
            count={list.length}
            onIndex={onIndex}
            onClose={onClose}
            onDelete={onDelete}
            onShowLive={onShowLive}
            watched={watched(entry.plate)}
            onToggleWatch={() => onToggleWatch(entry.plate)}
          />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function ViewerCard({
  entry,
  index,
  count,
  onIndex,
  onClose,
  onDelete,
  onShowLive,
  watched,
  onToggleWatch,
}: {
  entry: PlateLogEntry
  index: number
  count: number
  onIndex: (i: number) => void
  onClose: () => void
  onDelete: (e: PlateLogEntry) => void
  onShowLive: (e: PlateLogEntry) => Promise<void>
  watched: boolean
  onToggleWatch: () => void
}) {
  const [kind, setKind] = useState<ViewKind>("frame")
  const [copied, setCopied] = useState(false)
  const [live, setLive] = useState<LiveTarget | null>(null)
  const [liveBusy, setLiveBusy] = useState(false)
  const [liveErr, setLiveErr] = useState(false)
  const zoomRef = useRef<ZoomHandle | null>(null)

  const hasFrame = Boolean(entry.frame_url)
  const hasVehicle = Boolean(entry.vehicle_url)
  const view: ViewKind = kind === "vehicle" && hasVehicle ? "vehicle" : hasFrame ? "frame" : "vehicle"
  const src = view === "frame" ? entry.frame_url : entry.vehicle_url
  const box = view === "frame" ? (entry.plate_box ?? null) : plateInCrop(entry)

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(entry.plate)
    } catch {
      // insecure context (plain http on a LAN IP): fall back to a hidden textarea
      const ta = document.createElement("textarea")
      ta.value = entry.plate
      document.body.appendChild(ta)
      ta.select()
      document.execCommand("copy")
      ta.remove()
    }
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }, [entry.plate])

  // Is the vehicle still in view? Polled while the entry is recent.
  useEffect(() => {
    setLive(null)
    setLiveErr(false)
    if (entry.track_id === null || Date.now() / 1000 - entry.last_seen > LIVE_WINDOW_S) return
    let stopped = false
    let timer: number | undefined
    let ctrl: AbortController | null = null
    const tick = async () => {
      if (stopped) return
      ctrl = new AbortController()
      try {
        setLive(await plateLogApi.live(entry.id, ctrl.signal))
      } catch {
        setLive(null)
      }
      if (!stopped) timer = window.setTimeout(tick, LIVE_POLL_MS)
    }
    void tick()
    return () => {
      stopped = true
      window.clearTimeout(timer)
      ctrl?.abort()
    }
  }, [entry.id, entry.track_id, entry.last_seen])

  const showLive = useCallback(async () => {
    setLiveBusy(true)
    setLiveErr(false)
    try {
      await onShowLive(entry)
    } catch {
      setLiveErr(true)
      setLive(null)
    } finally {
      setLiveBusy(false)
    }
  }, [entry, onShowLive])

  // Keyboard: captured while the viewer is open so no other shortcut fires underneath.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      if ((e.target as HTMLElement).closest("input, textarea")) return
      e.stopImmediatePropagation()
      const k = e.key
      if (k === "Escape") onClose()
      else if (k === "ArrowLeft" || k === "ArrowUp" || k === "k" || k === "K") {
        e.preventDefault()
        if (index > 0) onIndex(index - 1)
      } else if (k === "ArrowRight" || k === "ArrowDown" || k === "j" || k === "J") {
        e.preventDefault()
        if (index < count - 1) onIndex(index + 1)
      } else if (k === "c" || k === "C") void copy()
      else if (k === "v" || k === "V" || k === "Tab") {
        if (k === "Tab") e.preventDefault()
        setKind((v) => (v === "frame" ? "vehicle" : "frame"))
      } else if (k === "f" || k === "F") zoomRef.current?.focusPlate()
      else if (k === "0") zoomRef.current?.reset()
      else if (k === "+" || k === "=") zoomRef.current?.zoom(1.4)
      else if (k === "-") zoomRef.current?.zoom(1 / 1.4)
      else if ((k === "l" || k === "L") && live?.live) void showLive()
      else if (k === "Delete") onDelete(entry)
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [index, count, onIndex, onClose, copy, live, showLive, onDelete, entry])

  const isLive = Boolean(live?.live)

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 10, scale: 0.98 }}
      transition={spring}
      className="flex h-full max-h-[920px] w-full max-w-6xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-[0_32px_120px_-24px_rgba(0,0,0,0.8)]"
    >
      {/* header */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-4 py-3">
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.div
            key={entry.id}
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={spring}
          >
            <PlateBadge plate={entry.plate} size="lg" />
          </motion.div>
        </AnimatePresence>
        <button
          type="button"
          onClick={() => void copy()}
          aria-label="Copy plate text (C)"
          title="Copy plate text (C)"
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          {copied ? <Check className="size-4 text-primary" /> : <Copy className="size-4" />}
        </button>
        {entry.watch && <WatchFlag note={entry.watch.note} />}
        <div className="min-w-0 flex-1 text-xs leading-relaxed text-muted-foreground">
          <p className="truncate text-sm text-foreground">
            {entry.camera_label}
            <span className="text-muted-foreground"> · {entry.vehicle}{entry.track_id !== null ? ` #${entry.track_id}` : ""}</span>
          </p>
          <p className="truncate tabular-nums">
            {formatDay(entry.first_seen)} {formatTime(entry.first_seen)}
            {entry.last_seen - entry.first_seen >= 1 ? ` – ${formatTime(entry.last_seen)}` : ""} · {relativeTime(entry.last_seen)} ·{" "}
            {Math.round(entry.conf * 100)}% vote · {entry.reads} {entry.reads === 1 ? "read" : "reads"}
          </p>
        </div>
        <span className="hidden text-xs tabular-nums text-muted-foreground sm:block">
          {index + 1} / {count}
        </span>
        <div className="flex items-center gap-0.5">
          <IconButton label="Previous (←)" disabled={index <= 0} onClick={() => onIndex(index - 1)}>
            <ChevronLeft className="size-4" />
          </IconButton>
          <IconButton label="Next (→)" disabled={index >= count - 1} onClick={() => onIndex(index + 1)}>
            <ChevronRight className="size-4" />
          </IconButton>
          {src && (
            <a
              href={src}
              download={`plate-${entry.plate}-${entry.id}-${view}.jpg`}
              aria-label="Download image"
              title="Download image"
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <Download className="size-4" />
            </a>
          )}
          <IconButton label="Delete entry (Del)" danger onClick={() => onDelete(entry)}>
            <Trash2 className="size-4" />
          </IconButton>
          <IconButton label="Close (Esc)" onClick={onClose}>
            <X className="size-4" />
          </IconButton>
        </div>
      </div>

      {/* body */}
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="relative min-h-[260px] flex-1 bg-[#1b1c1f]">
          {src ? (
            <ZoomImage key={`${entry.id}-${view}`} ref={zoomRef} src={src} box={box} />
          ) : (
            <p className="grid h-full place-items-center text-xs text-muted-foreground">No image stored for this read</p>
          )}
          {/* view toggle */}
          <div className="absolute left-3 top-3 flex rounded-full border border-border bg-background/80 p-0.5 text-xs backdrop-blur-md">
            {(["frame", "vehicle"] as const).map((k) => {
              const on = view === k
              const disabled = k === "frame" ? !hasFrame : !hasVehicle
              return (
                <button
                  key={k}
                  type="button"
                  disabled={disabled}
                  onClick={() => setKind(k)}
                  className={cn(
                    "relative rounded-full px-3 py-1 transition-colors duration-200 disabled:opacity-40",
                    on ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {on && (
                    <motion.span
                      layoutId="plate-viewer-kind"
                      className="absolute inset-0 rounded-full bg-primary"
                      transition={{ type: "spring", stiffness: 450, damping: 34 }}
                    />
                  )}
                  <span className="relative">{k === "frame" ? "Frame" : "Vehicle"}</span>
                </button>
              )
            })}
          </div>
          <div className="absolute bottom-3 left-3 flex gap-1">
            <IconButton label="Zoom to plate (F)" pill onClick={() => zoomRef.current?.focusPlate()}>
              <Crosshair className="size-4" />
            </IconButton>
            <IconButton label="Fit (0)" pill onClick={() => zoomRef.current?.reset()}>
              <Maximize2 className="size-4" />
            </IconButton>
          </div>
        </div>

        {/* side rail */}
        <aside className="flex shrink-0 flex-col gap-4 border-t border-border p-4 md:w-[280px] md:border-l md:border-t-0">
          <div>
            <p className="mb-1.5 text-[11px] uppercase tracking-wider text-muted-foreground">Plate crop</p>
            <div className="overflow-hidden rounded-md bg-black ring-2 ring-[#f4f4f2]/85">
              {entry.plate_url ? (
                <motion.img
                  key={entry.plate_url}
                  src={entry.plate_url}
                  alt={`Plate ${entry.plate}`}
                  initial={{ opacity: 0, scale: 1.04 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                  className="block w-full [image-rendering:auto]"
                />
              ) : (
                <div className="h-16" />
              )}
            </div>
          </div>

          <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
            <Stat label="Vote" value={`${Math.round(entry.conf * 100)}%`} bar={entry.conf} />
            <Stat label="OCR" value={`${Math.round((entry.ocr_conf ?? 0) * 100)}%`} bar={entry.ocr_conf ?? 0} />
            <Stat label="Reads" value={String(entry.reads)} />
            <Stat label="Seen for" value={duration(entry.last_seen - entry.first_seen)} />
          </dl>

          <div className="mt-auto flex flex-col gap-2">
            <AnimatePresence>
              {isLive && (
                <motion.button
                  key="live"
                  type="button"
                  onClick={() => void showLive()}
                  disabled={liveBusy}
                  initial={{ opacity: 0, y: 8, scale: 0.96 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 8, scale: 0.96 }}
                  transition={spring}
                  className="group flex items-center justify-center gap-2 rounded-lg bg-primary px-3 py-2.5 text-sm font-semibold text-primary-foreground transition-transform active:scale-[0.98] disabled:opacity-60"
                >
                  <span className="relative flex size-2">
                    <span className="absolute inset-0 animate-ping rounded-full bg-[#222427]" />
                    <span className="relative size-2 rounded-full bg-[#222427]" />
                  </span>
                  Show Live
                  <kbd className="ml-1 rounded border border-[#222427]/30 px-1 text-[10px] font-normal">L</kbd>
                </motion.button>
              )}
            </AnimatePresence>
            {!isLive && entry.track_id !== null && Date.now() / 1000 - entry.last_seen < LIVE_WINDOW_S && (
              <p className="flex items-center gap-2 text-[11px] text-muted-foreground">
                <Radio className="size-3.5" />
                {liveErr ? "The vehicle just left the view." : "Show Live appears while this vehicle is in view."}
              </p>
            )}
            <button
              type="button"
              onClick={onToggleWatch}
              className={cn(
                "flex items-center justify-center gap-2 rounded-lg border px-3 py-2 text-xs transition-colors",
                watched
                  ? "border-destructive/60 bg-destructive/10 text-destructive hover:bg-destructive/20"
                  : "border-border text-muted-foreground hover:border-primary/60 hover:text-foreground",
              )}
            >
              <Eye className="size-3.5" />
              {watched ? "On Watchlist" : "Add To Watchlist"}
            </button>
            <p className="hidden text-[10px] leading-relaxed text-muted-foreground md:block">
              Wheel zoom · drag to pan · double-click zoom · ← → browse · V view · F plate · C copy · Esc close
            </p>
          </div>
        </aside>
      </div>
    </motion.div>
  )
}

function duration(s: number) {
  if (s < 1) return "<1 s"
  if (s < 90) return `${Math.round(s)} s`
  if (s < 5400) return `${Math.round(s / 60)} min`
  return `${(s / 3600).toFixed(1)} h`
}

function Stat({ label, value, bar }: { label: string; value: string; bar?: number }) {
  return (
    <div>
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="tabular-nums text-foreground">{value}</dd>
      {bar !== undefined && (
        <div className="mt-1 h-0.5 overflow-hidden rounded-full bg-muted">
          <motion.div
            className="h-full rounded-full bg-primary"
            initial={{ width: 0 }}
            animate={{ width: `${Math.round(Math.min(1, Math.max(0, bar)) * 100)}%` }}
            transition={{ type: "spring", stiffness: 160, damping: 26 }}
          />
        </div>
      )}
    </div>
  )
}

function IconButton({
  label,
  onClick,
  disabled,
  danger,
  pill,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  danger?: boolean
  pill?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className={cn(
        "p-1.5 text-muted-foreground transition-colors disabled:opacity-30",
        pill ? "rounded-full border border-border bg-background/80 backdrop-blur-md hover:text-primary" : "rounded-md",
        danger ? "hover:bg-destructive/15 hover:text-destructive" : "hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------- zoomable image
interface ZoomHandle {
  focusPlate: () => void
  reset: () => void
  zoom: (factor: number) => void
}

function ZoomImage({ src, box, ref }: { src: string; box: NBox | null; ref: React.Ref<ZoomHandle> }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [natural, setNatural] = useState({ w: 0, h: 0 })
  const [loaded, setLoaded] = useState(false)
  const [dragging, setDragging] = useState(false)
  const drag = useRef<{ x: number; y: number; id: number } | null>(null)

  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const fit = useMemo(() => {
    if (!natural.w || !natural.h || !size.w || !size.h) return { left: 0, top: 0, w: 0, h: 0 }
    const k = Math.min(size.w / natural.w, size.h / natural.h)
    const w = natural.w * k
    const h = natural.h * k
    return { left: (size.w - w) / 2, top: (size.h - h) / 2, w, h }
  }, [natural, size])

  const view = useViewTransform(fit.w, fit.h)
  const { focus, reset, zoomAt } = view
  const stroke = useTransform(view.s, (s) => 2.5 / s)

  const focusPlate = useCallback(() => {
    if (box) focus(box, 0.34, 8)
  }, [box, focus])

  // Auto-zoom onto the plate once the image is laid out.
  const zoomedFor = useRef<string | null>(null)
  useEffect(() => {
    if (!loaded || !fit.w || zoomedFor.current === src) return
    zoomedFor.current = src
    const t = window.setTimeout(focusPlate, 260)
    return () => window.clearTimeout(t)
  }, [loaded, fit.w, src, focusPlate])

  useImperativeHandle(ref, () => ({ focusPlate, reset, zoom: (f: number) => zoomAt(fit.w / 2, fit.h / 2, f) }), [
    focusPlate,
    reset,
    zoomAt,
    fit.w,
    fit.h,
  ])

  const local = useCallback(
    (e: { clientX: number; clientY: number }) => {
      const b = containerRef.current!.getBoundingClientRect()
      return [e.clientX - b.left - fit.left, e.clientY - b.top - fit.top] as const
    },
    [fit.left, fit.top],
  )

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const [px, py] = local(e)
      zoomAt(px, py, Math.exp(-e.deltaY * 0.0015))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [local, zoomAt])

  return (
    <div
      ref={containerRef}
      className={cn("absolute inset-0 touch-none select-none overflow-hidden", dragging ? "cursor-grabbing" : "cursor-grab")}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.currentTarget.setPointerCapture(e.pointerId)
        drag.current = { x: e.clientX, y: e.clientY, id: e.pointerId }
        setDragging(true)
      }}
      onPointerMove={(e) => {
        const d = drag.current
        if (!d || d.id !== e.pointerId) return
        view.panBy(e.clientX - d.x, e.clientY - d.y)
        d.x = e.clientX
        d.y = e.clientY
      }}
      onPointerUp={() => {
        drag.current = null
        setDragging(false)
      }}
      onPointerCancel={() => {
        drag.current = null
        setDragging(false)
      }}
      onDoubleClick={(e) => {
        const [px, py] = local(e)
        if (view.view.current.s > 1.3) reset()
        else zoomAt(px, py, 3)
      }}
    >
      <div className="absolute overflow-hidden" style={{ left: fit.left, top: fit.top, width: fit.w, height: fit.h }}>
        <motion.div
          className="absolute inset-0 origin-top-left will-change-transform"
          style={{ x: view.x, y: view.y, scale: view.s }}
        >
          <motion.img
            src={src}
            alt=""
            draggable={false}
            onLoad={(e) => {
              const img = e.currentTarget
              setNatural({ w: img.naturalWidth, h: img.naturalHeight })
              setLoaded(true)
            }}
            initial={{ opacity: 0, filter: "blur(6px)" }}
            animate={loaded ? { opacity: 1, filter: "blur(0px)" } : { opacity: 0 }}
            transition={{ duration: 0.35 }}
            className={cn("block", fit.w ? "size-full" : "invisible absolute")}
          />
          {box && loaded && fit.w > 0 && (
            <motion.div
              className="pointer-events-none absolute"
              style={{
                left: `${box[0] * 100}%`,
                top: `${box[1] * 100}%`,
                width: `${(box[2] - box[0]) * 100}%`,
                height: `${(box[3] - box[1]) * 100}%`,
              }}
              initial={{ opacity: 0, scale: 2.2 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", stiffness: 220, damping: 20, delay: 0.45 }}
            >
              {/* spotlight: dims everything but the plate */}
              <motion.span
                className="absolute inset-0 rounded-[2px]"
                initial={{ boxShadow: "0 0 0 9999px rgba(22,23,26,0)" }}
                animate={{ boxShadow: "0 0 0 9999px rgba(22,23,26,0.38)" }}
                transition={{ duration: 0.6, delay: 0.5 }}
              />
              <motion.span
                className="absolute inset-0 rounded-[2px] border-[#ffd21f]"
                style={{ borderWidth: stroke }}
                animate={{ opacity: [1, 0.45, 1] }}
                transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
              />
              {/* sonar ring */}
              <motion.span
                className="absolute inset-0 rounded-[3px] border-[#f4f4f2]"
                style={{ borderWidth: stroke }}
                initial={{ opacity: 0.8, scale: 1 }}
                animate={{ opacity: 0, scale: 1.6 }}
                transition={{ duration: 1.6, repeat: Infinity, ease: "easeOut", delay: 0.9 }}
              />
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  )
}

