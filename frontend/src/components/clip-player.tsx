import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Download, Pause, Play, Trash2, X } from "lucide-react"
import { kindLabel, type AlarmEvent } from "@/lib/alerts"
import { cn } from "@/lib/utils"

const spring = { type: "spring", stiffness: 380, damping: 34 } as const

export function formatTime(ts: number, seconds = true) {
  return new Date(ts * 1000).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    ...(seconds ? { second: "2-digit" } : {}),
  })
}

export function formatDay(ts: number) {
  const d = new Date(ts * 1000)
  const today = new Date()
  const yesterday = new Date(Date.now() - 86400000)
  if (d.toDateString() === today.toDateString()) return "Today"
  if (d.toDateString() === yesterday.toDateString()) return "Yesterday"
  return d.toLocaleDateString("en-GB", { weekday: "short", day: "2-digit", month: "short" })
}

/** Full-screen overlay that plays an event clip (video or JPEG sequence). */
export function ClipPlayer({
  event,
  onClose,
  onDelete,
}: {
  event: AlarmEvent | null
  onClose: () => void
  onDelete: (e: AlarmEvent) => void
}) {
  useEffect(() => {
    if (!event) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopImmediatePropagation()
        onClose()
      }
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [event, onClose])

  return (
    <AnimatePresence>
      {event && (
        <motion.div
          key="clip-player"
          className="absolute inset-0 z-50 grid place-items-center bg-[#16171a]/80 p-4 backdrop-blur-md md:p-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onPointerDown={(e) => e.target === e.currentTarget && onClose()}
          role="dialog"
          aria-modal="true"
          aria-label={`Event clip: ${event.message}`}
        >
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.98 }}
            transition={spring}
            className="flex max-h-full w-full max-w-5xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-[0_32px_120px_-24px_rgba(0,0,0,0.8)]"
          >
            <div className="flex items-center gap-3 border-b border-border px-4 py-3">
              <span className="rounded-full bg-destructive/15 px-2 py-0.5 text-xs text-destructive">{kindLabel(event.kind)}</span>
              <p className="min-w-0 flex-1 truncate text-sm">
                {event.message}
                <span className="ml-2 text-muted-foreground">
                  {event.camera_label ?? event.camera} · {formatDay(event.ts)} {formatTime(event.ts)}
                </span>
              </p>
              {event.snapshot_url && (
                <a
                  href={event.snapshot_url}
                  download={`${event.camera}-${event.id}.jpg`}
                  aria-label="Download snapshot"
                  title="Download snapshot"
                  className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <Download className="size-4" />
                </a>
              )}
              <button
                type="button"
                onClick={() => onDelete(event)}
                aria-label="Delete event"
                title="Delete event"
                className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-destructive/15 hover:text-destructive"
              >
                <Trash2 className="size-4" />
              </button>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close player"
                className="-mr-1 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="relative grid min-h-0 flex-1 place-items-center bg-[#1b1c1f]">
              <ClipBody event={event} />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function ClipBody({ event }: { event: AlarmEvent }) {
  if (event.clip.status === "ready" && event.clip_url) {
    return (
      <video
        key={event.clip_url}
        src={event.clip_url}
        controls
        autoPlay
        loop
        muted
        playsInline
        className="max-h-[72dvh] w-full object-contain"
      />
    )
  }
  if (event.clip.status === "ready" && event.frames_url && event.clip.frames) {
    return <SequencePlayer base={event.frames_url} frames={event.clip.frames} fps={event.clip.fps ?? 10} />
  }
  return (
    <div className="relative w-full">
      {event.snapshot_url && <img src={event.snapshot_url} alt="" className="max-h-[72dvh] w-full object-contain opacity-80" />}
      <p className="absolute inset-x-0 bottom-4 mx-auto w-fit rounded-full bg-background/80 px-3 py-1.5 text-xs text-muted-foreground backdrop-blur-md">
        {event.clip.status === "recording" ? "The clip is still being recorded" : "No clip available for this event"}
      </p>
    </div>
  )
}

/** Plays frames/00000.jpg … as an animation (fallback when no video encoder is available). */
function SequencePlayer({ base, frames, fps }: { base: string; frames: number; fps: number }) {
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(true)
  const cache = useRef<HTMLImageElement[]>([])
  const url = (i: number) => `${base}${i}.jpg`

  // Preload everything once so playback does not stutter.
  useEffect(() => {
    cache.current = Array.from({ length: frames }, (_, i) => {
      const img = new Image()
      img.src = `${base}${i}.jpg`
      return img
    })
  }, [base, frames])

  useEffect(() => {
    if (!playing) return
    const t = window.setInterval(() => setIndex((i) => (i + 1) % frames), 1000 / Math.max(1, Math.min(30, fps)))
    return () => window.clearInterval(t)
  }, [playing, frames, fps])

  return (
    <div className="flex w-full flex-col">
      <img src={url(index)} alt="" className="max-h-[66dvh] w-full object-contain" />
      <div className="flex items-center gap-3 border-t border-border px-4 py-2">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? "Pause" : "Play"}
          className="grid size-8 place-items-center rounded-full bg-primary text-primary-foreground transition-transform active:scale-95"
        >
          {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
        </button>
        <input
          type="range"
          min={0}
          max={frames - 1}
          value={index}
          aria-label="Position"
          onChange={(e) => {
            setPlaying(false)
            setIndex(Number(e.target.value))
          }}
          className={cn("h-1 flex-1 cursor-pointer accent-[#ffd21f]")}
        />
        <span className="w-20 text-right text-xs tabular-nums text-muted-foreground">
          {(index / fps).toFixed(1)} / {(frames / fps).toFixed(1)} s
        </span>
      </div>
    </div>
  )
}
