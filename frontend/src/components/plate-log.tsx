import { useEffect, useMemo, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { CalendarDays, ChevronDown, Download, Eye, Plus, Search, Trash2, X } from "lucide-react"
import type { usePlateLog } from "@/hooks/use-plate-log"
import {
  plateLogApi,
  relativeTime,
  type PlateGroup,
  type PlateLogEntry,
  type PlateLogFilters,
} from "@/lib/platelog"
import type { Camera } from "@/lib/types"
import { cn } from "@/lib/utils"
import { formatDay, formatTime } from "./clip-player"
import { PlateBadge, PlateIcon, WatchFlag } from "./plate-badge"
import { PlateViewer } from "./plate-viewer"

const ease = [0.22, 1, 0.36, 1] as const
const spring = { type: "spring", stiffness: 450, damping: 34 } as const

type PlateLogState = ReturnType<typeof usePlateLog>
type Tab = "reads" | "plates" | "watch"

// ---------------------------------------------------------------------------- small pieces
/** Header button: plate glyph with a badge for new reads. */
export function PlateLogButton({ open, unread, onClick }: { open: boolean; unread: number; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={open}
      aria-label={unread ? `Plate Log (${unread} new) (P)` : "Plate Log (P)"}
      title="Plate Log (P)"
      className={cn(
        "relative grid size-9 shrink-0 place-items-center rounded-full border backdrop-blur-md transition-colors duration-300",
        open
          ? "border-primary bg-primary text-primary-foreground"
          : "border-border/70 bg-background/70 text-foreground hover:border-primary/60",
      )}
    >
      <PlateIcon className="size-[18px]" />
      <AnimatePresence>
        {unread > 0 && (
          <motion.span
            key="badge"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            transition={{ type: "spring", stiffness: 500, damping: 26 }}
            className="absolute -right-1 -top-1 grid h-[18px] min-w-[18px] place-items-center rounded-full bg-primary px-1 text-[10px] font-semibold tabular-nums text-[#222427] ring-2 ring-background"
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

function Segmented<T extends string>({
  id,
  value,
  options,
  onChange,
}: {
  id: string
  value: T
  options: { value: T; label: React.ReactNode }[]
  onChange: (v: T) => void
}) {
  return (
    <div role="radiogroup" className="flex gap-1 overflow-x-auto">
      {options.map((o) => {
        const on = value === o.value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(o.value)}
            className={cn(
              "relative shrink-0 rounded-full px-3 py-1 text-xs transition-colors duration-200",
              on ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {on && <motion.span layoutId={id} className="absolute inset-0 rounded-full bg-primary" transition={spring} />}
            <span className="relative">{o.label}</span>
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------- date filter
type Range = "any" | "1h" | "24h" | "7d" | "day"

function rangeFilter(range: Range, day: string): Pick<PlateLogFilters, "since" | "until"> {
  const now = Date.now() / 1000
  if (range === "1h") return { since: Math.floor(now - 3600) }
  if (range === "24h") return { since: Math.floor(now - 86400) }
  if (range === "7d") return { since: Math.floor(now - 7 * 86400) }
  if (range === "day" && day) {
    const start = new Date(`${day}T00:00:00`).getTime() / 1000
    if (Number.isFinite(start)) return { since: start, until: start + 86400 - 0.001 }
  }
  return {}
}

// ---------------------------------------------------------------------------- the drawer + viewer
export function PlateLog({
  open,
  onClose,
  cameras,
  log,
  onShowLive,
}: {
  open: boolean
  onClose: () => void
  cameras: Camera[]
  log: PlateLogState
  onShowLive: (e: PlateLogEntry) => Promise<void>
}) {
  const [tab, setTab] = useState<Tab>("reads")
  const [range, setRange] = useState<Range>("any")
  const [day, setDay] = useState("")
  const [viewer, setViewer] = useState<{ list: PlateLogEntry[]; id: string } | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const { filters, setFilters } = log
  const watchedPlates = useMemo(() => new Set(log.watch.map((w) => w.plate)), [log.watch])

  // date range -> filters (relative ranges are re-anchored every minute)
  useEffect(() => {
    const apply = () => setFilters((f) => ({ ...f, since: undefined, until: undefined, ...rangeFilter(range, day) }))
    apply()
    if (range === "any" || range === "day") return
    const t = window.setInterval(apply, 60000)
    return () => window.clearInterval(t)
  }, [range, day, setFilters])

  useEffect(() => {
    if (!confirmClear) return
    const t = window.setTimeout(() => setConfirmClear(false), 3000)
    return () => window.clearTimeout(t)
  }, [confirmClear])

  // keep the viewer's list fresh while the log updates underneath
  const viewerList = useMemo(() => {
    if (!viewer) return []
    const fresh = new Map(log.items.map((e) => [e.id, e]))
    return viewer.list.map((e) => fresh.get(e.id) ?? e)
  }, [viewer, log.items])
  const viewerIndex = viewer ? viewerList.findIndex((e) => e.id === viewer.id) : -1

  const openEntry = (list: PlateLogEntry[], e: PlateLogEntry) => setViewer({ list, id: e.id })

  const remove = (e: PlateLogEntry) => {
    if (viewer) {
      const i = viewerList.findIndex((x) => x.id === e.id)
      const rest = viewerList.filter((x) => x.id !== e.id)
      setViewer(rest.length ? { list: rest, id: rest[Math.min(i, rest.length - 1)].id } : null)
    }
    void log.remove(e)
  }

  const labels = useMemo(() => new Map(cameras.map((c) => [c.id, c.label])), [cameras])
  const items = log.items
  const dayGroups = useMemo(() => {
    const out: { day: string; items: PlateLogEntry[] }[] = []
    for (const e of items) {
      const d = formatDay(e.last_seen)
      if (!out.length || out[out.length - 1].day !== d) out.push({ day: d, items: [] })
      out[out.length - 1].items.push(e)
    }
    return out
  }, [items])
  const searching = Boolean(filters.q?.trim())

  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.aside
            key="plate-log"
            aria-label="Plate Log"
            initial={{ opacity: 0, x: -24, scale: 0.98 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: -24, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 340, damping: 32 }}
            className={cn(
              "absolute z-30 flex flex-col overflow-hidden border border-border bg-card/92 shadow-[0_24px_80px_-24px_rgba(0,0,0,0.6)] backdrop-blur-xl",
              "inset-x-2 bottom-2 max-h-[78dvh] rounded-xl",
              "md:inset-x-auto md:bottom-20 md:left-5 md:top-20 md:max-h-none md:w-[420px]",
            )}
          >
            {/* title row */}
            <div className="flex items-center gap-2 border-b border-border px-4 py-3">
              <PlateIcon className="size-4 text-primary" />
              <h2 className="flex-1 text-sm font-medium">
                Plate Log
                <span className="ml-2 text-xs tabular-nums text-muted-foreground">{log.total}</span>
              </h2>
              <a
                href={plateLogApi.exportUrl(filters, "csv")}
                download
                title="Export the filtered log as CSV"
                className="flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <Download className="size-3.5" />
                CSV
              </a>
              <a
                href={plateLogApi.exportUrl(filters, "json")}
                download
                title="Export the filtered log as JSON"
                className="rounded-md px-1.5 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                JSON
              </a>
              <button
                type="button"
                onClick={() => {
                  if (!confirmClear) return setConfirmClear(true)
                  setConfirmClear(false)
                  void log.clear(filters.camera)
                }}
                disabled={log.total === 0}
                title={filters.camera ? "Delete all reads of this camera" : "Delete all reads"}
                className={cn(
                  "rounded-md px-1.5 py-1 text-xs transition-colors disabled:opacity-40",
                  confirmClear ? "bg-destructive/15 text-destructive" : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                {confirmClear ? "Confirm?" : "Clear"}
              </button>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close plate log (Esc)"
                className="-mr-1 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
              >
                <X className="size-4" />
              </button>
            </div>

            {/* search + filters */}
            <div className="space-y-2 border-b border-border px-3 py-2.5">
              <label className="flex items-center gap-2 rounded-lg border border-input bg-background/60 px-2.5 py-1.5 focus-within:border-primary/70">
                <Search className="size-3.5 text-muted-foreground" />
                <input
                  value={filters.q ?? ""}
                  onChange={(e) => setFilters((f) => ({ ...f, q: e.target.value.toUpperCase() }))}
                  onKeyDown={(e) => e.key === "Escape" && (e.currentTarget.value ? setFilters((f) => ({ ...f, q: "" })) : e.currentTarget.blur())}
                  placeholder="Search plate (0/O, 1/I, 8/B tolerant)"
                  aria-label="Search plates"
                  spellCheck={false}
                  maxLength={16}
                  className="min-w-0 flex-1 bg-transparent text-sm uppercase tracking-wider outline-none placeholder:normal-case placeholder:tracking-normal placeholder:text-muted-foreground"
                />
                {filters.q && (
                  <button type="button" aria-label="Clear search" onClick={() => setFilters((f) => ({ ...f, q: "" }))} className="text-muted-foreground hover:text-foreground">
                    <X className="size-3.5" />
                  </button>
                )}
              </label>
              {cameras.length > 1 && (
                <Segmented
                  id="plate-log-camera"
                  value={filters.camera ?? "all"}
                  onChange={(v) => setFilters((f) => ({ ...f, camera: v === "all" ? undefined : v }))}
                  options={[{ value: "all", label: "All" }, ...cameras.map((c) => ({ value: c.id, label: c.label }))]}
                />
              )}
              <div className="flex items-center gap-1">
                <div className="min-w-0 flex-1">
                  <Segmented
                    id="plate-log-range"
                    value={range === "day" ? ("day" as Range) : range}
                    onChange={(v) => setRange(v)}
                    options={[
                      { value: "any", label: "Any Time" },
                      { value: "1h", label: "1 h" },
                      { value: "24h", label: "24 h" },
                      { value: "7d", label: "7 d" },
                    ]}
                  />
                </div>
                <label
                  className={cn(
                    "relative flex shrink-0 items-center gap-1 rounded-full px-2 py-1 text-xs transition-colors",
                    range === "day" ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                  title="Pick a day"
                >
                  <CalendarDays className="size-3.5" />
                  {range === "day" && day ? new Date(`${day}T00:00:00`).toLocaleDateString("en-GB", { day: "2-digit", month: "short" }) : "Day"}
                  <input
                    type="date"
                    value={day}
                    aria-label="Filter by day"
                    onChange={(e) => {
                      setDay(e.target.value)
                      setRange(e.target.value ? "day" : "any")
                    }}
                    className="absolute inset-0 cursor-pointer opacity-0 [color-scheme:dark]"
                  />
                </label>
              </div>
            </div>

            {/* tabs */}
            <div className="flex items-center border-b border-border px-3 py-1.5">
              <Segmented
                id="plate-log-tab"
                value={tab}
                onChange={setTab}
                options={[
                  { value: "reads", label: "Reads" },
                  { value: "plates", label: `Plates ${log.groups.length ? `· ${log.groups.length}` : ""}` },
                  { value: "watch", label: `Watchlist ${log.watch.length ? `· ${log.watch.length}` : ""}` },
                ]}
              />
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={tab}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.18, ease }}
                >
                  {tab === "reads" &&
                    (items.length === 0 ? (
                      <Empty loaded={log.loaded} filtered={searching || Boolean(filters.camera || filters.since)} />
                    ) : (
                      dayGroups.map((g) => (
                        <section key={g.day} className="mb-3">
                          <h3 className="sticky top-0 z-10 mb-1 bg-card/95 px-1 py-1 text-xs text-muted-foreground backdrop-blur">{g.day}</h3>
                          <ol className="space-y-1.5">
                            <AnimatePresence initial={false}>
                              {g.items.map((e) => (
                                <motion.li
                                  key={e.id}
                                  layout
                                  initial={{ opacity: 0, height: 0, scale: 0.98 }}
                                  animate={{ opacity: 1, height: "auto", scale: 1 }}
                                  exit={{ opacity: 0, height: 0 }}
                                  transition={{ duration: 0.28, ease }}
                                  className="overflow-hidden"
                                >
                                  <ReadRow
                                    entry={e}
                                    fresh={e.first_seen > log.seen}
                                    cameraLabel={cameras.length > 1 ? (labels.get(e.camera) ?? e.camera_label) : null}
                                    onOpen={() => openEntry(items, e)}
                                    onDelete={() => remove(e)}
                                  />
                                </motion.li>
                              ))}
                            </AnimatePresence>
                          </ol>
                        </section>
                      ))
                    ))}
                  {tab === "plates" &&
                    (log.groups.length === 0 ? (
                      <Empty loaded={log.loaded} filtered={searching || Boolean(filters.camera || filters.since)} />
                    ) : (
                      <ol className="space-y-1">
                        {log.groups.map((g) => (
                          <GroupRow
                            key={g.plate_key}
                            group={g}
                            labels={labels}
                            items={items}
                            watched={watchedPlates.has(g.plate)}
                            onOpen={(list, e) => openEntry(list, e)}
                          />
                        ))}
                      </ol>
                    ))}
                  {tab === "watch" && <WatchTab log={log} />}
                </motion.div>
              </AnimatePresence>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      <PlateViewer
        list={viewerList}
        index={viewerIndex >= 0 ? viewerIndex : null}
        onIndex={(i) => viewerList[i] && setViewer((v) => (v ? { ...v, id: viewerList[i].id } : v))}
        onClose={() => setViewer(null)}
        onDelete={remove}
        onShowLive={async (e) => {
          await onShowLive(e)
          setViewer(null)
        }}
        watched={(p) => watchedPlates.has(p)}
        onToggleWatch={(p) => void log.toggleWatch(p)}
      />
    </>
  )
}

function Empty({ loaded, filtered }: { loaded: boolean; filtered: boolean }) {
  return (
    <p className="px-1 py-6 text-xs leading-relaxed text-muted-foreground">
      {!loaded
        ? "Loading…"
        : filtered
          ? "Nothing matches these filters."
          : "No plates logged yet. Turn on License Plates (and Tracking) in the ESP menu under Recognition. Every confirmed read lands here with a plate crop, the vehicle and the full frame."}
    </p>
  )
}

function ReadRow({
  entry,
  fresh,
  cameraLabel,
  onOpen,
  onDelete,
}: {
  entry: PlateLogEntry
  fresh: boolean
  cameraLabel: string | null
  onOpen: () => void
  onDelete: () => void
}) {
  const pct = Math.round(entry.conf * 100)
  return (
    <div className="group relative">
      <button
        type="button"
        onClick={onOpen}
        className={cn(
          "flex w-full gap-3 rounded-lg border p-2 text-left transition-colors duration-150",
          fresh ? "border-primary/40 bg-primary/[0.04]" : "border-transparent",
          "hover:border-border hover:bg-accent/70",
        )}
      >
        {/* vehicle */}
        <span className="relative aspect-[4/3] w-[74px] shrink-0 self-start overflow-hidden rounded-md bg-muted">
          {entry.vehicle_url && (
            <img src={entry.vehicle_url} alt="" loading="lazy" className="size-full object-cover transition-transform duration-500 group-hover:scale-105" />
          )}
          {fresh && <span className="absolute left-1 top-1 size-1.5 rounded-full bg-primary shadow-[0_0_8px_#ffd21f]" />}
        </span>
        <span className="min-w-0 flex-1">
          {/* plate crop, framed like a plate */}
          <span className="block overflow-hidden rounded-[5px] bg-black ring-2 ring-[#f4f4f2]/85">
            {entry.plate_url ? (
              <img src={entry.plate_url} alt={`Plate ${entry.plate}`} loading="lazy" className="h-11 w-full object-cover" />
            ) : (
              <span className="block h-11" />
            )}
          </span>
          <span className="mt-1.5 flex items-center gap-2">
            <PlateBadge plate={entry.plate} size="sm" />
            {entry.watch && <WatchFlag note={entry.watch.note} />}
            <span className="ml-auto text-[11px] tabular-nums text-muted-foreground" title={new Date(entry.last_seen * 1000).toLocaleString("en-GB")}>
              {relativeTime(entry.last_seen)}
            </span>
          </span>
          <span className="mt-1 flex items-center gap-2 text-[11px] text-muted-foreground">
            <span className="min-w-0 flex-1 truncate">
              {formatTime(entry.first_seen)}
              {cameraLabel ? ` · ${cameraLabel}` : ""} · {entry.vehicle}
              {entry.track_id !== null ? ` #${entry.track_id}` : ""} · {entry.reads} {entry.reads === 1 ? "read" : "reads"}
            </span>
            <span className="flex shrink-0 items-center gap-1 tabular-nums" title="Share of votes for this reading">
              <span className="h-1 w-8 overflow-hidden rounded-full bg-muted">
                <span className="block h-full rounded-full bg-primary" style={{ width: `${pct}%` }} />
              </span>
              {pct}%
            </span>
          </span>
        </span>
      </button>
      <button
        type="button"
        onClick={onDelete}
        aria-label={`Delete read ${entry.plate}`}
        className="absolute left-1 top-1 rounded-md bg-background/80 p-1 text-muted-foreground opacity-0 backdrop-blur transition-all hover:bg-destructive/20 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
      >
        <Trash2 className="size-3.5" />
      </button>
    </div>
  )
}

function GroupRow({
  group,
  labels,
  items,
  watched,
  onOpen,
}: {
  group: PlateGroup
  labels: Map<string, string>
  items: PlateLogEntry[]
  watched: boolean
  onOpen: (list: PlateLogEntry[], e: PlateLogEntry) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const [sightings, setSightings] = useState<PlateLogEntry[] | null>(null)
  const cover = items.find((e) => e.id === group.best_id)

  useEffect(() => {
    if (!expanded) return
    let alive = true
    plateLogApi
      .sightings(group.plate)
      .then((r) => alive && setSightings(r.items))
      .catch(() => alive && setSightings([]))
    return () => {
      alive = false
    }
  }, [expanded, group.plate, group.count, group.last_seen])

  return (
    <li className="rounded-lg border border-transparent transition-colors hover:border-border">
      <button type="button" onClick={() => setExpanded((x) => !x)} aria-expanded={expanded} className="flex w-full items-center gap-3 rounded-lg p-2 text-left hover:bg-accent/60">
        <span className="w-[112px] shrink-0 overflow-hidden rounded-[5px] bg-black ring-2 ring-[#f4f4f2]/85">
          {cover?.plate_url ? <img src={cover.plate_url} alt="" loading="lazy" className="h-8 w-full object-cover" /> : <span className="block h-8" />}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2">
            <PlateBadge plate={group.plate} size="sm" />
            {(group.watch || watched) && <Eye className="size-3.5 text-destructive" />}
          </span>
          <span className="mt-1 block truncate text-[11px] text-muted-foreground">
            {group.cameras.map((c) => labels.get(c) ?? c).join(", ")} · last {relativeTime(group.last_seen)}
            {group.variants.length > 1 ? ` · also read ${group.variants.filter((v) => v !== group.plate).join(", ")}` : ""}
          </span>
        </span>
        <span className="rounded-full bg-muted px-2 py-0.5 text-xs tabular-nums">×{group.count}</span>
        <motion.span animate={{ rotate: expanded ? 180 : 0 }} transition={spring} className="text-muted-foreground">
          <ChevronDown className="size-4" />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {expanded && (
          <motion.ol
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease }}
            className="relative ml-[22px] overflow-hidden border-l border-border pl-3"
          >
            {sightings === null ? (
              <li className="py-2 text-[11px] text-muted-foreground">Loading sightings…</li>
            ) : (
              sightings.map((e) => (
                <li key={e.id} className="relative">
                  <span className="absolute -left-[17px] top-3.5 size-2 rounded-full bg-muted-foreground/60 ring-4 ring-card" />
                  <button
                    type="button"
                    onClick={() => onOpen(sightings, e)}
                    className="flex w-full items-center gap-2 rounded-md px-1.5 py-1 text-left text-xs transition-colors hover:bg-accent/70"
                  >
                    <span className="aspect-[4/3] w-10 shrink-0 overflow-hidden rounded bg-muted">
                      {e.vehicle_url && <img src={e.vehicle_url} alt="" loading="lazy" className="size-full object-cover" />}
                    </span>
                    <span className="min-w-0 flex-1 truncate">
                      {formatDay(e.first_seen)} {formatTime(e.first_seen, false)}
                      <span className="text-muted-foreground"> · {labels.get(e.camera) ?? e.camera_label}</span>
                    </span>
                    <span className="tabular-nums text-muted-foreground">{Math.round(e.conf * 100)}%</span>
                  </button>
                </li>
              ))
            )}
          </motion.ol>
        )}
      </AnimatePresence>
    </li>
  )
}

function WatchTab({ log }: { log: PlateLogState }) {
  const [plate, setPlate] = useState("")
  const [note, setNote] = useState("")
  const [err, setErr] = useState<string | null>(null)
  const submit = async () => {
    const p = plate.replace(/[^A-Za-z0-9]/g, "").toUpperCase()
    if (p.length < 2) return setErr("Enter at least 2 letters or digits.")
    try {
      await log.addWatch(p, note.trim())
      setPlate("")
      setNote("")
      setErr(null)
    } catch {
      setErr("Could not add this plate.")
    }
  }
  return (
    <div>
      <p className="mb-3 text-xs leading-relaxed text-muted-foreground">
        A logged read that matches a watched plate (OCR-tolerant) raises an alarm event with notifications, like a zone alarm.
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void submit()
        }}
        className="mb-4 flex flex-col gap-2"
      >
        <div className="flex gap-2">
          <input
            value={plate}
            onChange={(e) => setPlate(e.target.value.toUpperCase())}
            placeholder="Plate"
            aria-label="Plate to watch"
            maxLength={16}
            spellCheck={false}
            className="w-32 rounded-md border border-input bg-background/60 px-2 py-1.5 text-sm uppercase tracking-wider outline-none focus:border-primary/70"
          />
          <input
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Note (optional)"
            aria-label="Note"
            maxLength={80}
            className="min-w-0 flex-1 rounded-md border border-input bg-background/60 px-2 py-1.5 text-sm outline-none focus:border-primary/70"
          />
          <button type="submit" aria-label="Add to watchlist" className="grid size-9 shrink-0 place-items-center rounded-md bg-primary text-primary-foreground transition-transform active:scale-95">
            <Plus className="size-4" />
          </button>
        </div>
        {err && <p className="text-[11px] text-destructive">{err}</p>}
      </form>
      {log.watch.length === 0 ? (
        <p className="text-xs text-muted-foreground">No plates on the watchlist.</p>
      ) : (
        <ul className="space-y-1">
          <AnimatePresence initial={false}>
            {log.watch.map((w) => (
              <motion.li
                key={w.id}
                layout
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.22, ease }}
                className="group flex items-center gap-3 rounded-md px-1.5 py-1.5 hover:bg-accent/70"
              >
                <PlateBadge plate={w.plate} size="sm" />
                <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground">{w.note || "—"}</span>
                <button
                  type="button"
                  onClick={() => void log.removeWatch(w.id)}
                  aria-label={`Remove ${w.plate} from the watchlist`}
                  className="rounded-md p-1 text-muted-foreground opacity-0 transition-all hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <Trash2 className="size-3.5" />
                </button>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </div>
  )
}
