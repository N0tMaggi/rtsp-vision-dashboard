import { useCallback, useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Camera, Siren, X } from "lucide-react"
import { kindLabel, type AlarmEvent } from "@/lib/alerts"
import { formatTime } from "./clip-player"

const TOAST_MS = 7000
const MAX_TOASTS = 3

export type Toast =
  | { key: string; type: "alarm"; event: AlarmEvent }
  | { key: string; type: "info"; title: string; detail?: string; tone?: "ok" | "error" }

/** Toast queue: newest on top, auto-dismissed. */
export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([])
  const timers = useRef(new Map<string, number>())

  const dismiss = useCallback((key: string) => {
    setToasts((t) => t.filter((x) => x.key !== key))
    window.clearTimeout(timers.current.get(key))
    timers.current.delete(key)
  }, [])

  const push = useCallback(
    (toast: Toast) => {
      setToasts((t) => [toast, ...t.filter((x) => x.key !== toast.key)].slice(0, MAX_TOASTS))
      window.clearTimeout(timers.current.get(toast.key))
      timers.current.set(toast.key, window.setTimeout(() => dismiss(toast.key), toast.type === "alarm" ? TOAST_MS : 3500))
    },
    [dismiss],
  )

  useEffect(() => {
    const map = timers.current
    return () => map.forEach((t) => window.clearTimeout(t))
  }, [])

  return { toasts, push, dismiss }
}

/** Top-center alarm / info toasts. */
export function AlarmToasts({
  toasts,
  onOpen,
  onDismiss,
}: {
  toasts: Toast[]
  onOpen: (e: AlarmEvent) => void
  onDismiss: (key: string) => void
}) {
  return (
    <div
      className="pointer-events-none absolute inset-x-0 top-16 z-40 flex flex-col items-center gap-2 px-3 md:top-5"
      aria-live="assertive"
    >
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <motion.div
            key={t.key}
            layout
            initial={{ opacity: 0, y: -24, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.96, transition: { duration: 0.18 } }}
            transition={{ type: "spring", stiffness: 420, damping: 30 }}
            className="pointer-events-auto w-full max-w-md"
          >
            {t.type === "alarm" ? (
              <AlarmToast event={t.event} onOpen={() => onOpen(t.event)} onDismiss={() => onDismiss(t.key)} />
            ) : (
              <div className="mx-auto flex w-fit items-center gap-2.5 rounded-full border border-border bg-card/95 py-2 pl-3 pr-4 text-sm shadow-xl backdrop-blur-xl">
                <span className={t.tone === "error" ? "text-destructive" : "text-primary"}>
                  <Camera className="size-4" />
                </span>
                <span>{t.title}</span>
                {t.detail && <span className="max-w-[16rem] truncate text-xs text-muted-foreground">{t.detail}</span>}
              </div>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

function AlarmToast({ event, onOpen, onDismiss }: { event: AlarmEvent; onOpen: () => void; onDismiss: () => void }) {
  return (
    <div className="relative flex items-center gap-3 overflow-hidden rounded-xl border border-destructive/60 bg-card/95 p-2 pr-2 shadow-[0_18px_60px_-18px_rgba(255,90,78,0.55)] backdrop-blur-xl">
      {/* draining timer bar */}
      <motion.span
        className="absolute inset-x-0 bottom-0 h-0.5 origin-left bg-destructive"
        initial={{ scaleX: 1 }}
        animate={{ scaleX: 0 }}
        transition={{ duration: TOAST_MS / 1000, ease: "linear" }}
      />
      <button type="button" onClick={onOpen} className="flex min-w-0 flex-1 items-center gap-3 text-left">
        <span className="relative grid size-11 shrink-0 place-items-center overflow-hidden rounded-lg bg-destructive/15 text-destructive">
          {event.thumb_url ? <img src={event.thumb_url} alt="" className="size-full object-cover" /> : <Siren className="size-5" />}
          <motion.span
            className="absolute inset-0 rounded-lg ring-2 ring-destructive"
            animate={{ opacity: [1, 0.2, 1] }}
            transition={{ duration: 1.2, repeat: Infinity }}
          />
        </span>
        <span className="min-w-0">
          <span className="flex items-center gap-2 text-xs">
            <span className="font-semibold tracking-wide text-destructive">ALARM</span>
            <span className="text-muted-foreground">
              {kindLabel(event.kind)} · {event.camera_label ?? event.camera} · {formatTime(event.ts)}
            </span>
          </span>
          <span className="block truncate text-sm">{event.message}</span>
        </span>
      </button>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss alarm"
        className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
      >
        <X className="size-4" />
      </button>
    </div>
  )
}

/** Short white flash over the stage after a snapshot. */
export function SnapshotFlash({ flashKey }: { flashKey: number }) {
  return (
    <AnimatePresence>
      {flashKey > 0 && (
        <motion.div
          key={flashKey}
          aria-hidden
          className="pointer-events-none absolute inset-0 z-40 bg-[#f4f4f2]"
          initial={{ opacity: 0.55 }}
          animate={{ opacity: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.45, ease: "easeOut" }}
        />
      )}
    </AnimatePresence>
  )
}
