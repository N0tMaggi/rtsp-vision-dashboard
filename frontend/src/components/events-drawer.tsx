import { useMemo, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { AudioLines, Film, Footprints, Hand, Luggage, PersonStanding, RectangleHorizontal, Siren, SquareDashed, Trash2, X } from "lucide-react"
import { kindLabel, type ActiveAlarm, type AlarmEvent, type EventKind } from "@/lib/alerts"
import { isSoundKind, soundKindIcon } from "@/lib/sound" // Sound ESP
import type { Camera } from "@/lib/types"
import { cn } from "@/lib/utils"
import { formatDay, formatTime } from "./clip-player"

const ease = [0.22, 1, 0.36, 1] as const
const KIND_ICON: Record<EventKind, typeof Siren> = { zone: SquareDashed, tripwire: Footprints, abandoned: Luggage, plate: RectangleHorizontal, fall: PersonStanding, gesture: Hand }

/** Header button: unread badge, pulses red while any alarm is active. */
export function EventsButton({
  open,
  unread,
  alarm,
  onClick,
}: {
  open: boolean
  unread: number
  alarm: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={open}
      aria-label={alarm ? "Events (alarm active)" : unread ? `Events (${unread} unread)` : "Events"}
      className={cn(
        "relative grid size-9 shrink-0 place-items-center rounded-full border backdrop-blur-md transition-colors duration-300",
        alarm
          ? "border-destructive bg-destructive/20 text-destructive"
          : open
            ? "border-primary bg-primary text-primary-foreground"
            : "border-border/70 bg-background/70 text-foreground hover:border-primary/60",
      )}
    >
      {alarm && <span className="absolute inset-0 animate-ping rounded-full border border-destructive opacity-70" />}
      <Siren className="size-4" />
      <AnimatePresence>
        {unread > 0 && (
          <motion.span
            key="badge"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            transition={{ type: "spring", stiffness: 500, damping: 26 }}
            className="absolute -right-1 -top-1 grid h-[18px] min-w-[18px] place-items-center rounded-full bg-destructive px-1 text-[10px] font-semibold tabular-nums text-[#222427]"
          >
            <motion.span key={unread} initial={{ y: -6, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
              {unread > 99 ? "99+" : unread}
            </motion.span>
          </motion.span>
        )}
      </AnimatePresence>
    </button>
  )
}

/** Left drawer with the event timeline. */
export function EventsDrawer({
  open,
  onClose,
  events,
  active,
  cameras,
  seenTs,
  onPlay,
  onDelete,
}: {
  open: boolean
  onClose: () => void
  events: AlarmEvent[]
  active: ActiveAlarm[]
  cameras: Camera[]
  seenTs: number
  onPlay: (e: AlarmEvent) => void
  onDelete: (e: AlarmEvent) => void
}) {
  const [camera, setCamera] = useState<string>("all")
  const [soundOnly, setSoundOnly] = useState(false) // Sound ESP: sound event list
  const labels = useMemo(() => new Map(cameras.map((c) => [c.id, c.label])), [cameras])
  const hasSound = useMemo(() => events.some((e) => isSoundKind(e.kind)), [events])
  const list = (camera === "all" ? events : events.filter((e) => e.camera === camera)).filter(
    (e) => !soundOnly || isSoundKind(e.kind),
  )
  const groups = useMemo(() => {
    const out: { day: string; items: AlarmEvent[] }[] = []
    for (const e of list) {
      const day = formatDay(e.ts)
      if (!out.length || out[out.length - 1].day !== day) out.push({ day, items: [] })
      out[out.length - 1].items.push(e)
    }
    return out
  }, [list])

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          key="events-drawer"
          aria-label="Events"
          initial={{ opacity: 0, x: -24, scale: 0.98 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={{ opacity: 0, x: -24, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 340, damping: 32 }}
          className={cn(
            "absolute z-30 flex flex-col overflow-hidden border border-border bg-card/92 shadow-[0_24px_80px_-24px_rgba(0,0,0,0.6)] backdrop-blur-xl",
            "inset-x-2 bottom-2 max-h-[72dvh] rounded-xl",
            "md:inset-x-auto md:bottom-20 md:left-5 md:top-20 md:max-h-none md:w-[360px]",
          )}
        >
          <div className="flex items-center gap-3 border-b border-border px-4 py-3">
            <h2 className="flex-1 text-sm font-medium">
              Events
              <span className="ml-2 text-xs text-muted-foreground">{list.length}</span>
            </h2>
            {(hasSound || soundOnly) && (
              <button
                type="button"
                onClick={() => setSoundOnly((v) => !v)}
                aria-pressed={soundOnly}
                aria-label="Show sound events only"
                title="Sound events only"
                className={cn(
                  "flex items-center gap-1 rounded-full px-2.5 py-1 text-xs transition-colors duration-200",
                  soundOnly ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <AudioLines className="size-3.5" />
                Sound
              </button>
            )}
            {active.length > 0 && (
              <span className="flex items-center gap-1.5 rounded-full bg-destructive/15 px-2.5 py-1 text-xs text-destructive">
                <span className="relative flex size-1.5">
                  <span className="absolute inset-0 animate-ping rounded-full bg-destructive" />
                  <span className="relative size-1.5 rounded-full bg-destructive" />
                </span>
                Alarm active
              </span>
            )}
            <button
              type="button"
              onClick={onClose}
              aria-label="Close events"
              className="-mr-1 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <X className="size-4" />
            </button>
          </div>

          {cameras.length > 1 && (
            <div role="radiogroup" aria-label="Filter by camera" className="flex gap-1 overflow-x-auto border-b border-border px-3 py-2">
              {[{ id: "all", label: "All" }, ...cameras].map((c) => {
                const on = camera === c.id
                return (
                  <button
                    key={c.id}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    onClick={() => setCamera(c.id)}
                    className={cn(
                      "relative shrink-0 rounded-full px-3 py-1 text-xs transition-colors duration-200",
                      on ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {on && (
                      <motion.span
                        layoutId="events-camera-filter"
                        className="absolute inset-0 rounded-full bg-primary"
                        transition={{ type: "spring", stiffness: 450, damping: 34 }}
                      />
                    )}
                    <span className="relative">{c.label}</span>
                  </button>
                )
              })}
            </div>
          )}

          <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
            {list.length === 0 && soundOnly ? (
              <p className="px-1 py-6 text-xs leading-relaxed text-muted-foreground">
                No sound events. Arm a sound in the ESP menu under Sound - audio is analysed locally and never stored.
              </p>
            ) : list.length === 0 ? (
              <p className="px-1 py-6 text-xs leading-relaxed text-muted-foreground">
                No events yet. Add an alarm rule to a zone, arm a tripwire or turn on Abandoned Objects in the ESP menu
                under Alarms. Every alarm shows up here with a clip.
              </p>
            ) : (
              groups.map((g) => (
                <section key={g.day} className="mb-3">
                  <h3 className="sticky top-0 z-10 mb-1 bg-card/95 px-1 py-1 text-xs text-muted-foreground backdrop-blur">{g.day}</h3>
                  {/* timeline rail */}
                  <ol className="relative ml-[7px] border-l border-border pl-4">
                    <AnimatePresence initial={false}>
                      {g.items.map((e) => (
                        <motion.li
                          key={`${e.camera}/${e.id}`}
                          layout
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.25, ease }}
                          className="overflow-hidden"
                        >
                          <EventRow
                            event={e}
                            unread={e.ts > seenTs}
                            cameraLabel={cameras.length > 1 ? (labels.get(e.camera) ?? e.camera) : null}
                            onPlay={() => onPlay(e)}
                            onDelete={() => onDelete(e)}
                          />
                        </motion.li>
                      ))}
                    </AnimatePresence>
                  </ol>
                </section>
              ))
            )}
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

function EventRow({
  event,
  unread,
  cameraLabel,
  onPlay,
  onDelete,
}: {
  event: AlarmEvent
  unread: boolean
  cameraLabel: string | null
  onPlay: () => void
  onDelete: () => void
}) {
  const Icon = isSoundKind(event.kind) ? soundKindIcon(event.kind) : (KIND_ICON[event.kind] ?? Siren)
  const recording = event.clip.status === "recording"
  return (
    <div className="group relative py-1.5">
      <span
        className={cn(
          "absolute -left-[21px] top-4 size-2.5 rounded-full ring-4 ring-card",
          unread ? "bg-destructive" : "bg-muted-foreground/50",
        )}
      />
      <button
        type="button"
        onClick={onPlay}
        className="flex w-full items-center gap-3 rounded-lg p-1.5 text-left transition-colors duration-150 hover:bg-accent/70"
      >
        <span className="relative aspect-video w-24 shrink-0 overflow-hidden rounded-md bg-muted">
          {event.thumb_url && (
            <img src={event.thumb_url} alt="" loading="lazy" className="size-full object-cover transition-transform duration-500 group-hover:scale-105" />
          )}
          <span className="absolute bottom-1 right-1 grid size-5 place-items-center rounded-full bg-background/80 text-foreground backdrop-blur">
            {recording ? <span className="size-1.5 animate-pulse rounded-full bg-destructive" /> : <Film className="size-3" />}
          </span>
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Icon className="size-3.5" />
            {kindLabel(event.kind)}
            <span className="ml-auto tabular-nums">{formatTime(event.ts)}</span>
          </span>
          <span className="mt-0.5 block truncate text-sm">{event.source_label}</span>
          <span className="block truncate text-xs text-muted-foreground">
            {event.cls}
            {event.direction ? ` · ${event.direction}` : ""}
            {cameraLabel ? ` · ${cameraLabel}` : ""}
          </span>
        </span>
      </button>
      <button
        type="button"
        onClick={onDelete}
        aria-label="Delete event"
        className="absolute right-1 top-1 rounded-md p-1 text-muted-foreground opacity-0 transition-all hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
      >
        <Trash2 className="size-3.5" />
      </button>
    </div>
  )
}
