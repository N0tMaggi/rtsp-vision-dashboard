import { useEffect, useRef, useState, type ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ChevronLeft, ChevronRight, Pause, Play, Radio, Repeat, RotateCcw, Sparkles } from "lucide-react"
import { openEnhance, setEnhanceVideoPaused } from "@/lib/enhance"
import { formatTime, type Playback, type PlaybackCommand } from "@/lib/videos"
import { cn } from "@/lib/utils"

export const SPEEDS = [0.25, 0.5, 1, 1.5, 2, 4] as const
const spring = { type: "spring", stiffness: 420, damping: 34 } as const

interface Props {
  playback: Playback | null
  position: () => number
  onCommand: (cmd: PlaybackCommand) => void
  /** Live URL sources have no timeline. */
  live?: { label: string } | null
  rightInset?: number
}

/** Animation-frame clock while playing, so the time readout and scrubber glide. */
function useTicker(active: boolean) {
  const [, setTick] = useState(0)
  useEffect(() => {
    if (!active) return
    let raf = 0
    let last = 0
    const loop = (t: number) => {
      if (t - last > 33) {
        last = t
        setTick((n) => n + 1)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [active])
}

/**
 * Transport controls for a video source, floating at the bottom of the stage
 * above the telemetry line. Space/K play-pause, J/L -10s/+10s, ,/. frame step,
 * [/] slower/faster (shortcuts live in useVideoAnalysis).
 */
export function PlaybackBar({ playback, position, onCommand, live, rightInset = 24 }: Props) {
  const pb = playback
  const [scrub, setScrub] = useState<number | null>(null) // 0..1 while dragging
  useTicker(Boolean(pb?.playing) && scrub === null)

  const duration = pb?.duration ?? 0
  const pos = scrub !== null ? scrub * duration : position()
  const frac = duration > 0 ? Math.min(1, Math.max(0, pos / duration)) : 0

  // Paused video -> the stage toolbar offers "Enhance" too.
  const paused = Boolean(pb && !pb.playing && !live)
  useEffect(() => {
    setEnhanceVideoPaused(paused)
    return () => setEnhanceVideoPaused(false)
  }, [paused])

  // Lift the stage toolbar (zoom / tools) above the bar while it is shown.
  useEffect(() => {
    const root = document.documentElement
    root.style.setProperty("--stage-bottom-extra", "36px")
    return () => {
      root.style.removeProperty("--stage-bottom-extra")
    }
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 16, scale: 0.98 }}
      transition={{ type: "spring", stiffness: 320, damping: 32 }}
      className="pointer-events-none absolute bottom-[52px] left-6 z-10 flex justify-center transition-[right] duration-500"
      style={{ right: rightInset }}
    >
      <div
        role="toolbar"
        aria-label="Video playback"
        className="pointer-events-auto flex w-full max-w-[860px] items-center gap-1.5 rounded-full border border-border/70 bg-background/80 py-1.5 pl-1.5 pr-3 text-[13px] shadow-[0_8px_32px_-12px_rgba(0,0,0,0.6)] backdrop-blur-md"
      >
        {live ? (
          <LiveSource label={live.label} />
        ) : (
          <>
            <motion.button
              type="button"
              whileTap={{ scale: 0.9 }}
              onClick={() => onCommand({ action: pb?.playing ? "pause" : "play" })}
              aria-label={pb?.playing ? "Pause (Space)" : "Play (Space)"}
              title={pb?.playing ? "Pause (Space)" : "Play (Space)"}
              disabled={!pb}
              className="grid size-8 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
            >
              <AnimatePresence mode="popLayout" initial={false}>
                <motion.span
                  key={pb?.playing ? "pause" : "play"}
                  initial={{ opacity: 0, scale: 0.5, rotate: -30 }}
                  animate={{ opacity: 1, scale: 1, rotate: 0 }}
                  exit={{ opacity: 0, scale: 0.5, rotate: 30 }}
                  transition={spring}
                  className="grid place-items-center"
                >
                  {pb?.playing ? <Pause className="size-3.5 fill-current" /> : <Play className="ml-0.5 size-3.5 fill-current" />}
                </motion.span>
              </AnimatePresence>
            </motion.button>
            <IconButton label="Previous frame (,)" onClick={() => onCommand({ action: "step", frames: -1 })} disabled={!pb}>
              <ChevronLeft className="size-4" />
            </IconButton>
            <IconButton label="Next frame (.)" onClick={() => onCommand({ action: "step", frames: 1 })} disabled={!pb}>
              <ChevronRight className="size-4" />
            </IconButton>

            <span className="hidden shrink-0 px-1 tabular-nums sm:inline" aria-live="off">
              {formatTime(pos)}
              <span className="text-muted-foreground"> / {formatTime(duration)}</span>
            </span>

            <Scrubber
              fraction={frac}
              duration={duration}
              disabled={!pb || duration <= 0}
              onScrub={setScrub}
              onSeek={(f) => onCommand({ action: "seek", fraction: f })}
            />

            <SpeedPicker speed={pb?.speed ?? 1} onChange={(s) => onCommand({ speed: s })} />
            <IconButton
              label={pb?.loop ? "Loop on" : "Loop off"}
              pressed={Boolean(pb?.loop)}
              onClick={() => onCommand({ loop: !pb?.loop })}
              disabled={!pb}
            >
              <Repeat className="size-3.5" />
            </IconButton>
            <IconButton label="Restart" onClick={() => onCommand({ action: "restart" })} disabled={!pb}>
              <RotateCcw className="size-3.5" />
            </IconButton>
            <AnimatePresence initial={false}>
              {paused && (
                <motion.span initial={{ width: 0, opacity: 0 }} animate={{ width: "auto", opacity: 1 }} exit={{ width: 0, opacity: 0 }} className="overflow-hidden">
                  <IconButton label="Enhance this frame (X)" onClick={() => openEnhance()}>
                    <Sparkles className="size-3.5 text-primary" />
                  </IconButton>
                </motion.span>
              )}
            </AnimatePresence>
          </>
        )}
      </div>
    </motion.div>
  )
}

function LiveSource({ label }: { label: string }) {
  return (
    <div className="flex min-w-0 items-center gap-3 py-1 pl-2.5">
      <span className="relative flex size-2">
        <span className="absolute inset-0 animate-ping rounded-full bg-primary opacity-60" />
        <span className="relative size-2 rounded-full bg-primary" />
      </span>
      <Radio className="size-3.5 shrink-0 text-muted-foreground" />
      <span className="truncate">{label}</span>
      <span className="shrink-0 text-muted-foreground">Live stream · no timeline</span>
    </div>
  )
}

function IconButton({
  label,
  onClick,
  disabled,
  pressed,
  children,
}: {
  label: string
  onClick: () => void
  disabled?: boolean
  pressed?: boolean
  children: ReactNode
}) {
  return (
    <motion.button
      type="button"
      whileTap={{ scale: 0.88 }}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      aria-pressed={pressed}
      title={label}
      className={cn(
        "grid size-7 shrink-0 place-items-center rounded-full transition-colors duration-200 disabled:opacity-40",
        pressed ? "bg-primary/15 text-primary" : "text-muted-foreground hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </motion.button>
  )
}

function Scrubber({
  fraction,
  duration,
  disabled,
  onScrub,
  onSeek,
}: {
  fraction: number
  duration: number
  disabled: boolean
  onScrub: (f: number | null) => void
  onSeek: (f: number) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [dragging, setDragging] = useState(false)
  const lastSent = useRef(0)

  const at = (clientX: number) => {
    const r = ref.current!.getBoundingClientRect()
    return Math.min(1, Math.max(0, (clientX - r.left) / Math.max(1, r.width)))
  }

  return (
    <div
      ref={ref}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label="Seek"
      aria-valuemin={0}
      aria-valuemax={Math.round(duration)}
      aria-valuenow={Math.round(fraction * duration)}
      aria-valuetext={formatTime(fraction * duration)}
      aria-disabled={disabled}
      className={cn("group relative mx-1 flex h-7 min-w-24 flex-1 cursor-pointer items-center outline-none", disabled && "pointer-events-none opacity-40")}
      onPointerDown={(e) => {
        if (disabled) return
        e.currentTarget.setPointerCapture(e.pointerId)
        setDragging(true)
        const f = at(e.clientX)
        onScrub(f)
        onSeek(f)
        lastSent.current = performance.now()
      }}
      onPointerMove={(e) => {
        const f = at(e.clientX)
        setHover(f)
        if (!dragging) return
        onScrub(f)
        // live scrubbing: seek at most every 120 ms while dragging
        if (performance.now() - lastSent.current > 120) {
          lastSent.current = performance.now()
          onSeek(f)
        }
      }}
      onPointerUp={(e) => {
        if (!dragging) return
        setDragging(false)
        onSeek(at(e.clientX))
        // keep the dragged position until the server confirms it
        window.setTimeout(() => onScrub(null), 250)
      }}
      onPointerLeave={() => !dragging && setHover(null)}
      onKeyDown={(e) => {
        const step = duration > 0 ? 5 / duration : 0
        if (e.key === "ArrowLeft") onSeek(Math.max(0, fraction - step))
        else if (e.key === "ArrowRight") onSeek(Math.min(1, fraction + step))
        else return
        e.preventDefault()
        e.stopPropagation()
      }}
    >
      <div className="relative h-1 w-full overflow-hidden rounded-full bg-muted transition-[height] duration-200 group-hover:h-1.5">
        {hover !== null && (
          <div className="absolute inset-y-0 left-0 bg-foreground/15" style={{ width: `${hover * 100}%` }} />
        )}
        <div className="absolute inset-y-0 left-0 rounded-full bg-primary" style={{ width: `${fraction * 100}%` }} />
      </div>
      <motion.span
        aria-hidden
        className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-foreground shadow"
        style={{ left: `${fraction * 100}%` }}
        animate={{ scale: dragging ? 1.35 : hover !== null ? 1.1 : 0.85 }}
        transition={spring}
      />
      <AnimatePresence>
        {hover !== null && duration > 0 && (
          <motion.span
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: 0.15 }}
            className="pointer-events-none absolute -top-8 -translate-x-1/2 rounded-md border border-border/70 bg-background/95 px-2 py-0.5 text-xs tabular-nums text-foreground shadow"
            style={{ left: `${hover * 100}%` }}
          >
            {formatTime(hover * duration, duration < 60)}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  )
}

function SpeedPicker({ speed, onChange }: { speed: number; onChange: (s: number) => void }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const off = (e: PointerEvent) => !ref.current?.contains(e.target as Node) && setOpen(false)
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false)
    window.addEventListener("pointerdown", off)
    window.addEventListener("keydown", esc)
    return () => {
      window.removeEventListener("pointerdown", off)
      window.removeEventListener("keydown", esc)
    }
  }, [open])

  return (
    <div ref={ref} className="relative shrink-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Playback speed ${speed}x ([ / ])`}
        title="Playback speed ([ / ])"
        className={cn(
          "h-7 min-w-12 rounded-full border px-2 text-xs tabular-nums transition-colors duration-200",
          speed !== 1 ? "border-primary/60 text-primary" : "border-border/70 text-muted-foreground hover:text-foreground",
        )}
      >
        {speed}×
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            role="listbox"
            aria-label="Playback speed"
            initial={{ opacity: 0, y: 6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.96 }}
            transition={spring}
            className="absolute bottom-9 right-0 z-20 flex origin-bottom-right flex-col rounded-lg border border-border/70 bg-background/95 p-1 shadow-xl backdrop-blur-md"
          >
            {SPEEDS.map((s) => (
              <li key={s}>
                <button
                  type="button"
                  role="option"
                  aria-selected={s === speed}
                  onClick={() => {
                    onChange(s)
                    setOpen(false)
                  }}
                  className={cn(
                    "relative w-16 rounded-md px-2 py-1 text-right text-xs tabular-nums transition-colors",
                    s === speed ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {s === speed && (
                    <motion.span layoutId="speed-indicator" className="absolute inset-0 rounded-md bg-primary" transition={spring} />
                  )}
                  <span className="relative">{s}×</span>
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}
