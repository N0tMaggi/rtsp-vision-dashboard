import { useEffect, useRef } from "react"
import { motion } from "motion/react"
import { ScanEye, X } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { AIM_LABEL, isMoving, patchAutoTrack, requestOpenAutoTrack, STATE_META, toggleAutoTrack, trackTarget, useAutoTrack } from "@/lib/autotrack"
import type { AutoTrackStatus } from "@/lib/ops-api"
import type { Notice } from "./stage-context"

const spring = { type: "spring", stiffness: 420, damping: 32 } as const
const cap = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s)

export function targetText(st: AutoTrackStatus) {
  if (st.target_id === null) return null
  return `${cap(st.target_label ?? "Target")} #${st.target_id}`
}

/** Compact stage chip while PTZ auto-track is on: state, target and aim point; click opens the settings. */
export function AutoTrackChip({ notify }: { notify?: (n: Notice) => void }) {
  const { autotrack } = useAutoTrack()
  if (!autotrack) return null
  const { config, status } = autotrack
  const meta = STATE_META[status.state]
  const who = targetText(status)
  const moving = isMoving(autotrack)
  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.96 }}
      transition={spring}
      className="flex max-w-[calc(100vw-5rem)] items-center gap-0.5 rounded-full border bg-background/80 py-1 pl-1 pr-1 text-xs backdrop-blur-md"
      style={{ borderColor: `${meta.color}80` }}
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={requestOpenAutoTrack}
            className="flex min-w-0 items-center gap-2 rounded-full py-0.5 pl-2 pr-2 transition-colors hover:bg-accent"
          >
            <span className="relative flex size-2 shrink-0">
              {meta.live && (moving || status.state === "searching") && (
                <span className="absolute inset-0 animate-ping rounded-full opacity-70" style={{ background: meta.color }} />
              )}
              <span className="relative size-2 rounded-full" style={{ background: meta.color }} />
            </span>
            <span className="font-semibold tracking-widest" style={{ color: meta.color }}>
              AUTO-TRACK
            </span>
            <span className="truncate text-foreground">
              <span className="text-muted-foreground">·</span> {meta.label}
              {who && <> {who}</>}
              <span className="hidden sm:inline">
                {" "}
                <span className="text-muted-foreground">·</span> {AIM_LABEL[config.aim]}
              </span>
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-64">
          {status.reason ? `${status.reason} · ` : ""}
          {config.mode === "auto" ? "Auto pick" : "Locked target"} · click for settings · Shift+T toggles
        </TooltipContent>
      </Tooltip>
      <button
        type="button"
        aria-label="Stop auto-track"
        onClick={() => void patchAutoTrack({ enabled: false }).then((err) => err && notify?.({ title: err, tone: "error" }))}
        className="rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
      >
        <X className="size-3.5" />
      </button>
    </motion.div>
  )
}

/** Small "PTZ Track" toggle for the locked-target chip. */
export function PtzTrackButton({ lockedId, notify }: { lockedId: number; notify?: (n: Notice) => void }) {
  const { ready, autotrack } = useAutoTrack()
  if (!ready || !autotrack) return null
  const on = autotrack.config.enabled && autotrack.config.mode === "locked" && autotrack.status.target_id === lockedId
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={on}
          onClick={() =>
            void trackTarget(on ? null : lockedId).then((err) => err && notify?.({ title: err, tone: "error" }))
          }
          className={
            on
              ? "flex items-center gap-1.5 rounded-full bg-primary px-2.5 py-1 text-xs text-primary-foreground transition-colors duration-200"
              : "flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs text-muted-foreground transition-colors duration-200 hover:bg-accent hover:text-foreground"
          }
        >
          <ScanEye className="size-3.5" /> PTZ Track
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">{on ? "Stop moving the camera (Shift+T)" : "Move the PTZ camera with this target (Shift+T)"}</TooltipContent>
    </Tooltip>
  )
}

/** Shift+T toggles auto-track (locked target if any, else auto pick). */
export function useAutoTrackShortcut(lockedId: number | null, notify?: (n: Notice) => void) {
  const { autotrack } = useAutoTrack()
  const supported = Boolean(autotrack)
  const ref = useRef({ lockedId, notify })
  useEffect(() => {
    ref.current = { lockedId, notify }
  })
  useEffect(() => {
    if (!supported) return
    const onKey = (e: KeyboardEvent) => {
      if (!e.shiftKey || e.ctrlKey || e.metaKey || e.altKey || e.key.toLowerCase() !== "t") return
      if ((e.target as HTMLElement).closest("input, textarea, select, [role=combobox], [role=listbox], [role=slider]")) return
      e.preventDefault()
      void toggleAutoTrack(ref.current.lockedId).then((err) => err && ref.current.notify?.({ title: err, tone: "error" }))
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [supported])
}
