import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  Camera as CameraIcon,
  Check,
  Copy,
  Crosshair,
  EyeOff,
  Focus,
  Loader2,
  Lock,
  LockOpen,
  Navigation,
  Pause,
  Play,
  RectangleHorizontal,
  ScanFace,
  ScanText,
  Slash,
  SquareDashed,
  Tag,
  Users,
  X,
} from "lucide-react"
import { ContextMenu, type MenuEntry, type PanelApi } from "@/components/context-menu"
import { contextApi, contextError, type FrozenFrame, type ManualRead, type TagCatalogItem, type TagOwner } from "@/lib/context-api"
import { recognitionApi, errorText, type FaceCandidate, type Person } from "@/lib/recognition"
import type { Camera, NBox, Target, TargetTag, ViewFilter } from "@/lib/types"
import { videosApi } from "@/lib/videos"
import { cn } from "@/lib/utils"

const VEHICLES = new Set(["car", "truck", "bus", "motorcycle"])
const isVehicle = (t: Target | null | undefined) => Boolean(t && VEHICLES.has(t.label.toLowerCase()))
const isPerson = (t: Target | null | undefined) => t?.label.toLowerCase() === "person"
const spring = { type: "spring", stiffness: 420, damping: 32 } as const
const VISUAL_ONLY = "Visual only: alarms, zone counts and recordings still see everyone."

export interface Notice {
  title: string
  detail?: string
  tone?: "error"
}

interface Options {
  camera: Camera | undefined
  /** All targets of the camera, including ones hidden by the view filter. */
  targets: Target[]
  view: ViewFilter | undefined
  lockedId: number | null
  onLock: (id: number | null) => void
  follow: boolean
  onFollowChange: (f: boolean) => void
  /** Start drawing a zone / tripwire anchored at this point (click places the other end). */
  onDrawAt: (kind: "zone" | "tripwire", u: number, v: number) => void
  onAimAt?: (u: number, v: number, clientX: number, clientY: number) => void
  onSnapshot?: () => void
  /** Video file source on stage: play state + toggle. */
  video?: { playing: boolean | null; toggle: () => void } | null
  notify?: (n: Notice) => void
  /** Pan / zoom the stage onto a normalized box. */
  focusBox: (box: NBox) => void
}

interface MenuState {
  x: number
  y: number
  u: number
  v: number
  target: Target | null
  panel?: string | null
}

interface Annotate {
  freeze: FrozenFrame
  /** Drawn plate box (normalized), once the user let go. */
  box: NBox | null
  at: { x: number; y: number }
  read: ManualRead | null
  reading: boolean
  error: string | null
}

export function targetTitle(t: Target) {
  const who = t.identity?.name ?? (t.identity?.face_id !== undefined ? `Face #${t.identity.face_id}` : null)
  return `${t.label} #${t.id}${who ? ` · ${who}` : ""}`
}

function infoText(t: Target, camera: Camera | undefined) {
  const parts = [`${t.label} #${t.id}`]
  if (t.identity?.name) parts.push(t.identity.name)
  if (t.identity?.face_id !== undefined) parts.push(`Face #${t.identity.face_id}`)
  if (t.identity?.plate) parts.push(`plate ${t.identity.plate}`)
  if (t.tags?.length) parts.push(`tags ${t.tags.map((x) => x.label).join(", ")}`)
  parts.push(`confidence ${Math.round(t.conf * 100)}%`)
  if (t.distance !== undefined) parts.push(`~${t.distance.toFixed(1)} m`)
  if (t.dwell !== undefined) parts.push(`dwell ${Math.round(t.dwell)} s`)
  if (t.first_seen !== undefined) parts.push(`first seen ${new Date(t.first_seen * 1000).toLocaleTimeString("en-GB")}`)
  if (camera) parts.push(`camera ${camera.label}`)
  parts.push(`box ${t.box.map((n) => n.toFixed(3)).join(", ")}`)
  return parts.join(" · ")
}

/**
 * The stage's right-click menu and everything it starts: tags, solo / hide,
 * face to gallery, plate reading and the "Mark Plate" draw mode on a frozen frame.
 * Stage keeps the pointer handling; this hook owns the menu state and overlays.
 */
export function useStageContext(o: Options) {
  const [menu, setMenu] = useState<MenuState | null>(null)
  const [annotate, setAnnotate] = useState<Annotate | null>(null)
  const cameraId = o.camera?.id ?? null
  const isFile = o.camera?.source_kind === "file"
  const { notify } = o

  // Camera switch: drop menu + annotation.
  useEffect(() => {
    setMenu(null)
    setAnnotate(null)
  }, [cameraId])

  const live = useMemo(() => new Map(o.targets.map((t) => [t.id, t])), [o.targets])
  // the menu keeps showing the target it was opened on, refreshed with live data while visible
  const target = menu?.target ? (live.get(menu.target.id) ?? menu.target) : null

  const open = useCallback((m: MenuState) => setMenu(m), [])
  const close = useCallback(() => setMenu(null), [])

  const view = useCallback(
    (action: "solo" | "unsolo" | "hide" | "unhide" | "clear", id?: number) => {
      if (!cameraId) return
      contextApi.view(cameraId, action, id).catch((e) => notify?.({ title: contextError(e), tone: "error" }))
    },
    [cameraId, notify],
  )

  const startMarkPlate = useCallback(
    async (t: Target | null, at: { x: number; y: number }) => {
      if (!cameraId) return
      try {
        if (isFile && o.video?.playing) await videosApi.control(cameraId, { action: "pause" })
        const freeze = await contextApi.freeze(cameraId)
        setAnnotate({ freeze, box: null, at, read: null, reading: false, error: null })
        if (t) o.focusBox(t.box)
      } catch (e) {
        notify?.({ title: contextError(e), tone: "error" })
      }
    },
    [cameraId, isFile, o, notify],
  )

  const cancelAnnotate = useCallback(() => setAnnotate(null), [])

  const onPlateBox = useCallback(
    async (box: NBox, clientX: number, clientY: number) => {
      if (!cameraId || !annotate) return
      const freeze = annotate.freeze
      setAnnotate({ ...annotate, box, at: { x: clientX, y: clientY }, reading: true, read: null, error: null })
      try {
        const read = await contextApi.readBox(cameraId, freeze.id, box)
        setAnnotate((a) => (a && a.freeze.id === freeze.id ? { ...a, read, reading: false } : a))
      } catch (e) {
        setAnnotate((a) => (a && a.freeze.id === freeze.id ? { ...a, reading: false, error: contextError(e) } : a))
      }
    },
    [annotate, cameraId],
  )

  // Esc leaves the plate mode (the panel handles its own Esc first).
  useEffect(() => {
    if (!annotate) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !(e.target as HTMLElement).closest("input")) setAnnotate(null)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [annotate])

  const entries: MenuEntry[] = useMemo(() => {
    if (!menu) return []
    const at = { x: menu.x, y: menu.y }
    const out: MenuEntry[] = []
    const viewActive = Boolean(o.view?.solo || o.view?.hidden.length)
    if (target) {
      const locked = target.id === o.lockedId
      const solo = o.view?.solo?.id === target.id
      out.push({ kind: "header", id: "head", content: <TargetHeader target={target} /> })
      out.push({
        kind: "item",
        id: "lock",
        label: locked ? "Unlock" : "Lock",
        icon: locked ? LockOpen : Lock,
        shortcut: "Click",
        onSelect: () => o.onLock(locked ? null : target.id),
      })
      out.push({
        kind: "item",
        id: "follow",
        label: locked && o.follow ? "Stop Following" : "Follow",
        icon: Crosshair,
        shortcut: "F",
        active: locked && o.follow,
        onSelect: () => {
          if (locked && o.follow) return o.onFollowChange(false)
          if (!locked) o.onLock(target.id)
          window.setTimeout(() => o.onFollowChange(true), 0)
        },
      })
      out.push({ kind: "separator", id: "s1" })
      out.push({
        kind: "item",
        id: "solo",
        label: solo ? "Show Everyone" : "Only Show This",
        icon: Focus,
        active: solo,
        hint: VISUAL_ONLY,
        onSelect: () => view(solo ? "unsolo" : "solo", target.id),
      })
      out.push({ kind: "item", id: "hide", label: "Hide This", icon: EyeOff, hint: VISUAL_ONLY, onSelect: () => view("hide", target.id) })
      out.push({ kind: "separator", id: "s2" })
      out.push({
        kind: "item",
        id: "tag",
        label: target.tags?.length ? "Tags…" : "Add Tag…",
        icon: Tag,
        panel: (api) => <TagPanel cameraId={cameraId!} target={target} api={api} />,
      })
      if (target.tags?.length)
        out.push({
          kind: "item",
          id: "untag",
          label: "Remove Tag…",
          icon: X,
          panel: (api) => <RemoveTagPanel cameraId={cameraId!} target={target} api={api} />,
        })
      out.push({
        kind: "item",
        id: "copy",
        label: "Copy Info",
        icon: Copy,
        onSelect: () => {
          const text = infoText(target, o.camera)
          void navigator.clipboard
            ?.writeText(text)
            .then(() => notify?.({ title: "Copied target info", detail: text.slice(0, 80) }))
            .catch(() => notify?.({ title: "Clipboard not available", tone: "error" }))
        },
      })
      if (isPerson(target)) {
        out.push({ kind: "separator", id: "s3" })
        out.push({
          kind: "item",
          id: "face",
          label: "Add Face To People…",
          icon: ScanFace,
          disabled: !target.identity?.face,
          hint: target.identity?.face ? "Save this face in the Known Faces gallery." : "No face seen on this person right now (needs Known Faces on).",
          panel: (api) => <FacePanel cameraId={cameraId!} target={target} api={api} notify={notify} />,
        })
      }
      if (isVehicle(target)) {
        out.push({ kind: "separator", id: "s3" })
        out.push({
          kind: "item",
          id: "read",
          label: "Read Plate…",
          icon: ScanText,
          panel: (api) => (
            <ReadPlatePanel
              cameraId={cameraId!}
              target={target}
              api={api}
              notify={notify}
              onMark={() => {
                api.close()
                void startMarkPlate(target, at)
              }}
            />
          ),
        })
        out.push({
          kind: "item",
          id: "mark",
          label: "Mark Plate…",
          icon: RectangleHorizontal,
          hint: isFile ? "Pauses the video, then drag a box around the plate." : "Freezes the frame, then drag a box around the plate.",
          onSelect: () => void startMarkPlate(target, at),
        })
      }
    } else {
      out.push({ kind: "item", id: "zone", label: "Draw Zone Here", icon: SquareDashed, shortcut: "Z", onSelect: () => o.onDrawAt("zone", menu.u, menu.v) })
      out.push({ kind: "item", id: "wire", label: "Draw Tripwire", icon: Slash, shortcut: "T", onSelect: () => o.onDrawAt("tripwire", menu.u, menu.v) })
      if (o.onAimAt)
        out.push({
          kind: "item",
          id: "aim",
          label: "Aim Camera Here",
          icon: Navigation,
          shortcut: "Alt+Right",
          onSelect: () => o.onAimAt?.(menu.u, menu.v, menu.x, menu.y),
        })
      out.push({ kind: "separator", id: "s1" })
      out.push({
        kind: "item",
        id: "mark",
        label: isFile ? "Mark Plate…" : "Freeze & Mark Plate…",
        icon: RectangleHorizontal,
        onSelect: () => void startMarkPlate(null, at),
      })
      if (o.onSnapshot) out.push({ kind: "item", id: "snap", label: "Snapshot", icon: CameraIcon, shortcut: "S", onSelect: o.onSnapshot })
      if (isFile && o.video)
        out.push({
          kind: "item",
          id: "play",
          label: o.video.playing ? "Pause" : "Play",
          icon: o.video.playing ? Pause : Play,
          shortcut: "Space",
          onSelect: o.video.toggle,
        })
    }
    if (viewActive) {
      out.push({ kind: "separator", id: "s9" })
      out.push({ kind: "item", id: "clear", label: "Show Everyone", icon: Users, hint: VISUAL_ONLY, onSelect: () => view("clear") })
    }
    return out
  }, [menu, target, o, cameraId, view, notify, isFile, startMarkPlate])

  const overlay: ReactNode = (
    <>
      <ContextMenu
        open={Boolean(menu)}
        x={menu?.x ?? 0}
        y={menu?.y ?? 0}
        entries={entries}
        onClose={close}
        label={target ? `${target.label} #${target.id}` : "Stage"}
        initialPanel={menu?.panel ?? null}
      />
      <ViewChips view={o.view} onClear={view} />
      <AnimatePresence>
        {annotate && (
          <motion.div
            key="plate-hint"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={spring}
            onPointerDown={(e) => e.stopPropagation()}
            className="absolute left-1/2 top-20 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full border border-primary/50 bg-background/85 py-1 pl-3 pr-1 text-xs backdrop-blur-md"
          >
            <span className="size-1.5 animate-pulse rounded-full bg-primary" />
            <span>{annotate.box ? "Check the reading, or drag again" : "Frozen · drag a box around the plate"}</span>
            <button
              type="button"
              onClick={cancelAnnotate}
              className="rounded-full px-2 py-0.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              Esc
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {annotate?.box && cameraId && (
          <PlatePanel
            key={`${annotate.freeze.id}-${annotate.box.join(",")}`}
            cameraId={cameraId}
            a={annotate}
            onDone={(plate) => {
              setAnnotate(null)
              notify?.({ title: `Plate ${plate} saved`, detail: "Added to the Plate Log as Manual" })
            }}
            onCancel={cancelAnnotate}
          />
        )}
      </AnimatePresence>
    </>
  )

  // Inside the zoom / pan layer: the frozen still and the chosen box.
  const frozenLayer: ReactNode = annotate ? (
    <>
      <img src={annotate.freeze.url} alt="Frozen frame" draggable={false} className="absolute inset-0 size-full object-contain" />
      {annotate.box && (
        <svg className="pointer-events-none absolute inset-0 size-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
          <rect
            x={annotate.box[0]}
            y={annotate.box[1]}
            width={annotate.box[2] - annotate.box[0]}
            height={annotate.box[3] - annotate.box[1]}
            fill="rgba(255,210,31,0.10)"
            stroke="#ffd21f"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
          />
        </svg>
      )}
    </>
  ) : null

  return { open, close, menuOpen: Boolean(menu), overlay, frozenLayer, annotating: Boolean(annotate), onPlateBox, cancelAnnotate }
}

// ---------------------------------------------------------------------------- menu pieces
function TagChip({ tag, onRemove }: { tag: TargetTag; onRemove?: () => void }) {
  return (
    <span
      className="inline-flex max-w-full items-center gap-1 rounded-full border px-1.5 py-px text-[10px] font-semibold uppercase tracking-wide"
      style={{ borderColor: `${tag.color}66`, color: tag.color, background: `${tag.color}14` }}
    >
      <span className="truncate">{tag.label}</span>
      {onRemove && (
        <button type="button" aria-label={`Remove tag ${tag.label}`} onClick={onRemove} className="-mr-0.5 rounded-full opacity-70 hover:opacity-100">
          <X className="size-3" />
        </button>
      )}
    </span>
  )
}

export function TagChips({ tags, className }: { tags?: TargetTag[]; className?: string }) {
  if (!tags?.length) return null
  return (
    <div className={cn("flex flex-wrap gap-1", className)}>
      {tags.map((t) => (
        <TagChip key={t.label} tag={t} />
      ))}
    </div>
  )
}

function TargetHeader({ target }: { target: Target }) {
  const id = target.identity
  return (
    <div className="space-y-1">
      <div className="flex items-baseline gap-2">
        <span className="font-semibold uppercase tracking-wide">{target.label}</span>
        <span className="text-muted-foreground">#{target.id}</span>
        <span className="ml-auto text-[11px] tabular-nums text-muted-foreground">{Math.round(target.conf * 100)}%</span>
      </div>
      {(id?.name || id?.face_id !== undefined || id?.plate) && (
        <div className="flex flex-wrap gap-x-2 text-[11px]">
          {id?.name && <span className="text-[#7ee0a1]">{id.name}</span>}
          {!id?.name && id?.face_id !== undefined && <span className="text-[#ff8a80]">Face #{id.face_id}</span>}
          {id?.plate && <span className="text-foreground/80">{id.plate}</span>}
        </div>
      )}
      <TagChips tags={target.tags} />
    </div>
  )
}

const OWNER_TEXT: Record<TagOwner, string> = {
  person: "Saved to this person: comes back whenever they are recognized.",
  face: "Saved to this face id: comes back whenever the face is seen.",
  plate: "Saved to this plate: comes back whenever it is read.",
  track: "Saved to this track. Moves to the person or plate once recognized.",
}

let catalogCache: { tags: TagCatalogItem[]; palette: string[] } | null = null

function TagPanel({ cameraId, target, api }: { cameraId: string; target: Target; api: PanelApi }) {
  const [label, setLabel] = useState("")
  const [color, setColor] = useState<string | null>(null)
  const [catalog, setCatalog] = useState(catalogCache)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<{ text: string; error?: boolean } | null>(null)
  const [tags, setTags] = useState<TargetTag[]>(target.tags ?? [])
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    input.current?.focus()
    contextApi
      .tagCatalog()
      .then((c) => {
        catalogCache = c
        setCatalog(c)
      })
      .catch(() => undefined)
  }, [])

  const palette = catalog?.palette ?? ["#ffd21f", "#ff5a5a", "#7ee0a1", "#5ad1ff", "#c792ea", "#ff9f43", "#f4f4f2"]
  const q = label.trim().toLowerCase()
  const have = new Set(tags.map((t) => t.label.toLowerCase()))
  const suggestions = (catalog?.tags ?? []).filter((t) => !have.has(t.label.toLowerCase()) && (!q || t.label.toLowerCase().includes(q))).slice(0, 6)
  const known = catalog?.tags.find((t) => t.label.toLowerCase() === q)

  const add = async (text: string, c: string | null) => {
    const v = text.trim()
    if (!v) return
    setBusy(true)
    try {
      const r = await contextApi.addTag(cameraId, target.id, v, c ?? undefined)
      setTags(r.tags)
      setLabel("")
      setMsg({ text: OWNER_TEXT[r.owner] })
      catalogCache = null
    } catch (e) {
      setMsg({ text: contextError(e), error: true })
    } finally {
      setBusy(false)
      input.current?.focus()
    }
  }

  const remove = async (t: TargetTag) => {
    try {
      setTags((await contextApi.removeTag(cameraId, target.id, t.label)).tags)
    } catch (e) {
      setMsg({ text: contextError(e), error: true })
    }
  }

  return (
    <div className="space-y-2.5">
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {tags.map((t) => (
            <TagChip key={t.label} tag={t} onRemove={() => void remove(t)} />
          ))}
        </div>
      )}
      <form
        className="flex gap-1.5"
        onSubmit={(e) => {
          e.preventDefault()
          void add(label, color ?? known?.color ?? null)
        }}
      >
        <input
          ref={input}
          value={label}
          maxLength={24}
          placeholder="Tag, e.g. Mail Carrier"
          aria-label="Tag label"
          onChange={(e) => setLabel(e.target.value)}
          className="min-w-0 flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm outline-none transition-colors placeholder:text-muted-foreground/60 focus:border-primary/60"
        />
        <button
          type="submit"
          disabled={busy || !label.trim()}
          aria-label="Add tag"
          className="grid size-8 shrink-0 place-items-center rounded-md bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
        >
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
        </button>
      </form>
      <div role="radiogroup" aria-label="Tag color" className="flex gap-1.5">
        {palette.map((c) => {
          const on = (color ?? known?.color) === c
          return (
            <button
              key={c}
              type="button"
              role="radio"
              aria-checked={on}
              aria-label={`Color ${c}`}
              onClick={() => setColor(on ? null : c)}
              className={cn("size-5 rounded-full ring-offset-2 ring-offset-[#26282c] transition-transform hover:scale-110", on && "ring-2 ring-foreground")}
              style={{ background: c }}
            />
          )
        })}
      </div>
      {suggestions.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {suggestions.map((s) => (
            <button
              key={s.label}
              type="button"
              onClick={() => void add(s.label, s.color)}
              className="rounded-full border border-white/10 px-2 py-0.5 text-[11px] transition-colors hover:border-white/30"
              style={{ color: s.color }}
              title={`Used ${s.count}×`}
            >
              + {s.label}
            </button>
          ))}
        </div>
      )}
      {msg && <p className={cn("text-[11px] leading-snug", msg.error ? "text-destructive" : "text-muted-foreground")}>{msg.text}</p>}
      <button type="button" onClick={api.close} className="w-full rounded-md py-1 text-xs text-muted-foreground transition-colors hover:bg-white/6 hover:text-foreground">
        Done
      </button>
    </div>
  )
}

function RemoveTagPanel({ cameraId, target, api }: { cameraId: string; target: Target; api: PanelApi }) {
  const [tags, setTags] = useState<TargetTag[]>(target.tags ?? [])
  const [err, setErr] = useState<string | null>(null)
  const first = useRef<HTMLButtonElement>(null)
  useEffect(() => first.current?.focus(), [])
  return (
    <div className="space-y-1">
      {tags.length === 0 && <p className="px-1 py-2 text-xs text-muted-foreground">No tags left.</p>}
      {tags.map((t, i) => (
        <button
          key={t.label}
          ref={i === 0 ? first : undefined}
          type="button"
          onClick={async () => {
            try {
              const r = await contextApi.removeTag(cameraId, target.id, t.label)
              setTags(r.tags)
              if (!r.tags.length) api.close()
            } catch (e) {
              setErr(contextError(e))
            }
          }}
          className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-left transition-colors hover:bg-white/8 focus-visible:bg-white/8 focus-visible:outline-none"
        >
          <TagChip tag={t} />
          <X className="size-3.5 text-muted-foreground" />
        </button>
      ))}
      {err && <p className="text-[11px] text-destructive">{err}</p>}
    </div>
  )
}

function FacePanel({ cameraId, target, api, notify }: { cameraId: string; target: Target; api: PanelApi; notify?: (n: Notice) => void }) {
  const [cand, setCand] = useState<FaceCandidate | null>(null)
  const [people, setPeople] = useState<Person[]>([])
  const [name, setName] = useState("")
  const [state, setState] = useState<"loading" | "ready" | "saving" | "error">("loading")
  const [err, setErr] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let alive = true
    contextApi
      .targetFace(cameraId, target.id)
      .then((r) => {
        if (!alive) return
        setCand(r.candidate)
        setName(r.candidate.match ?? "")
        setState("ready")
        window.setTimeout(() => input.current?.focus(), 0)
      })
      .catch((e) => {
        if (!alive) return
        setErr(contextError(e))
        setState("error")
      })
    recognitionApi
      .people()
      .then((r) => alive && setPeople(r.people))
      .catch(() => undefined)
    return () => {
      alive = false
    }
  }, [cameraId, target.id])

  const save = async () => {
    if (!cand || !name.trim()) return
    setState("saving")
    try {
      const r = await recognitionApi.addFromCandidate(name.trim(), cand.id)
      notify?.({ title: `Face added to ${r.person.name}`, detail: `${r.person.samples} samples` })
      api.close()
    } catch (e) {
      setErr(errorText(e))
      setState("ready")
    }
  }

  if (state === "loading")
    return (
      <p className="flex items-center gap-2 py-3 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin text-primary" /> Looking for the face…
      </p>
    )
  if (state === "error" || !cand) return <p className="py-2 text-xs text-destructive">{err ?? "No face found."}</p>
  const q = name.trim().toLowerCase()
  const matches = people.filter((p) => !q || p.name.toLowerCase().includes(q)).slice(0, 5)
  return (
    <div className="space-y-2.5">
      <div className="flex items-center gap-3">
        {cand.thumb && <img src={cand.thumb} alt="" className="size-14 rounded-lg object-cover ring-1 ring-white/15" />}
        <p className="text-[11px] leading-snug text-muted-foreground">
          {cand.match ? `Looks like ${cand.match} (${Math.round(cand.score * 100)} %). Add another sample or pick someone else.` : "Unknown face. Name a new person or pick an existing one."}
        </p>
      </div>
      <form
        className="flex gap-1.5"
        onSubmit={(e) => {
          e.preventDefault()
          void save()
        }}
      >
        <input
          ref={input}
          value={name}
          maxLength={40}
          placeholder="Name"
          aria-label="Person name"
          onChange={(e) => setName(e.target.value)}
          className="min-w-0 flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm outline-none focus:border-primary/60"
        />
        <button type="submit" disabled={state === "saving" || !name.trim()} aria-label="Add face" className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground disabled:opacity-40">
          {state === "saving" ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
        </button>
      </form>
      {matches.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {matches.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setName(p.name)}
              className="flex items-center gap-1 rounded-full border border-white/10 py-0.5 pl-0.5 pr-2 text-[11px] transition-colors hover:border-[#7ee0a1]/60"
            >
              <img src={p.thumb} alt="" className="size-4 rounded-full object-cover" /> {p.name}
            </button>
          ))}
        </div>
      )}
      {err && <p className="text-[11px] text-destructive">{err}</p>}
    </div>
  )
}

function ReadPlatePanel({
  cameraId,
  target,
  api,
  notify,
  onMark,
}: {
  cameraId: string
  target: Target
  api: PanelApi
  notify?: (n: Notice) => void
  onMark: () => void
}) {
  const [res, setRes] = useState<Awaited<ReturnType<typeof contextApi.readPlate>> | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  useEffect(() => {
    let alive = true
    contextApi
      .readPlate(cameraId, target.id)
      .then((r) => alive && setRes(r))
      .catch((e) => alive && setErr(contextError(e)))
    return () => {
      alive = false
    }
  }, [cameraId, target.id])

  if (err) return <p className="py-2 text-xs text-destructive">{err}</p>
  if (!res)
    return (
      <p className="flex items-center gap-2 py-3 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin text-primary" /> Reading the plate…
      </p>
    )
  const markBtn = (
    <button type="button" onClick={onMark} className="flex-1 rounded-md border border-white/10 py-1.5 text-xs transition-colors hover:border-primary/60">
      Mark Manually
    </button>
  )
  if (!res.found)
    return (
      <div className="space-y-2.5">
        <p className="text-xs text-muted-foreground">No readable plate on this vehicle right now. Draw the plate yourself instead.</p>
        <div className="flex gap-1.5">{markBtn}</div>
      </div>
    )
  return (
    <div className="space-y-2.5">
      {res.crop && <img src={res.crop} alt="Plate crop" className="w-full rounded-md ring-1 ring-white/10" />}
      <div className="flex items-baseline justify-between">
        <span className="text-lg font-semibold tracking-widest">{res.text}</span>
        <span className="text-xs tabular-nums text-muted-foreground">{Math.round((res.conf ?? 0) * 100)}%</span>
      </div>
      <div className="flex gap-1.5">
        {markBtn}
        <button
          type="button"
          disabled={saving}
          onClick={async () => {
            if (!res.box || !res.text) return
            setSaving(true)
            try {
              await contextApi.confirmPlate(cameraId, res.freeze.id, res.box, res.text, res.conf ?? 0)
              notify?.({ title: `Plate ${res.text} saved`, detail: "Added to the Plate Log as Manual" })
              api.close()
            } catch (e) {
              setErr(contextError(e))
            } finally {
              setSaving(false)
            }
          }}
          className="flex-1 rounded-md bg-primary py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
        >
          Save To Log
        </button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------- plate annotation panel
function PlatePanel({ cameraId, a, onDone, onCancel }: { cameraId: string; a: Annotate; onDone: (plate: string) => void; onCancel: () => void }) {
  const [text, setText] = useState("")
  const [touched, setTouched] = useState(false)
  const [saving, setSaving] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ left: a.at.x + 12, top: a.at.y + 12 })

  const read = a.read
  const shown = touched ? text : (read?.text ?? "")
  useEffect(() => {
    if (!a.reading) input.current?.focus()
  }, [a.reading])
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const w = el.offsetWidth
    const h = el.offsetHeight
    setPos({
      left: Math.max(8, Math.min(a.at.x + 14, window.innerWidth - w - 8)),
      top: Math.max(8, Math.min(a.at.y + 14, window.innerHeight - h - 8)),
    })
  }, [a.at.x, a.at.y, a.reading])

  const confirm = async () => {
    const plate = shown.trim()
    if (!a.box || plate.replace(/[^a-z0-9]/gi, "").length < 2) return setErr("Type the plate (at least 2 letters or digits).")
    setSaving(true)
    try {
      const r = await contextApi.confirmPlate(cameraId, a.freeze.id, a.box, plate, read?.conf ?? 0)
      onDone(r.entry.plate)
    } catch (e) {
      setErr(contextError(e))
      setSaving(false)
    }
  }

  return (
    <motion.div
      ref={ref}
      role="dialog"
      aria-label="Mark plate"
      initial={{ opacity: 0, scale: 0.94, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={spring}
      style={pos}
      onPointerDown={(e) => e.stopPropagation()}
      onPointerUp={(e) => e.stopPropagation()}
      onPointerMove={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        e.stopPropagation()
        if (e.key === "Escape") onCancel()
      }}
      className="fixed z-[55] w-72 overflow-hidden rounded-xl border border-white/10 bg-[#26282c]/95 p-3 text-[13px] shadow-[0_24px_60px_-18px_rgba(0,0,0,0.75)] backdrop-blur-xl"
    >
      <div className="mb-2 flex items-center gap-2">
        <RectangleHorizontal className="size-4 text-primary" />
        <span className="flex-1 font-medium">Mark Plate</span>
        {read?.vehicle && (
          <span className="text-[11px] text-muted-foreground">
            {read.vehicle.label} #{read.vehicle.id}
          </span>
        )}
      </div>
      {read?.crop && <img src={read.crop} alt="Selected plate" className="mb-2 w-full rounded-md ring-1 ring-white/10" />}
      {a.reading ? (
        <p className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin text-primary" /> Reading…
        </p>
      ) : (
        <>
          <form
            className="flex items-center gap-1.5"
            onSubmit={(e) => {
              e.preventDefault()
              void confirm()
            }}
          >
            <input
              ref={input}
              value={shown}
              maxLength={20}
              aria-label="Plate text"
              placeholder="Type the plate"
              onChange={(e) => {
                setTouched(true)
                setText(e.target.value.toUpperCase())
                setErr(null)
              }}
              className="min-w-0 flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-base font-semibold tracking-widest outline-none focus:border-primary/60"
            />
            {read && read.text && !touched && (
              <span
                className={cn("text-xs tabular-nums", read.conf >= 0.8 ? "text-[#7ee0a1]" : read.conf >= 0.6 ? "text-primary" : "text-[#ff8a80]")}
                title="OCR confidence"
              >
                {Math.round(read.conf * 100)}%
              </span>
            )}
          </form>
          {read && !read.text && <p className="mt-1.5 text-[11px] text-muted-foreground">Couldn't read it. Type the plate yourself.</p>}
          {read && read.candidates.length > 1 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {read.candidates.slice(1).map((c) => (
                <button
                  key={c.text}
                  type="button"
                  onClick={() => {
                    setTouched(true)
                    setText(c.text)
                  }}
                  className="rounded-full border border-white/10 px-2 py-0.5 text-[11px] tracking-wider transition-colors hover:border-primary/60"
                >
                  {c.text} <span className="text-muted-foreground">{Math.round(c.conf * 100)}%</span>
                </button>
              ))}
            </div>
          )}
        </>
      )}
      {(err || a.error) && <p className="mt-2 text-[11px] text-destructive">{err ?? a.error}</p>}
      <div className="mt-3 flex gap-1.5">
        <button type="button" onClick={onCancel} className="flex-1 rounded-md border border-white/10 py-1.5 text-xs transition-colors hover:border-white/30">
          Cancel
        </button>
        <button
          type="button"
          disabled={saving || a.reading}
          onClick={() => void confirm()}
          className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-primary py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-50"
        >
          {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Check className="size-3.5" />} Confirm
        </button>
      </div>
    </motion.div>
  )
}

// ---------------------------------------------------------------------------- solo / hidden chips
function ViewChips({ view, onClear }: { view: ViewFilter | undefined; onClear: (a: "unsolo" | "unhide" | "clear", id?: number) => void }) {
  const [open, setOpen] = useState(false)
  const solo = view?.solo
  const hidden = view?.hidden ?? []
  useEffect(() => {
    if (!hidden.length) setOpen(false)
  }, [hidden.length])
  const name = (e: { label: string; id: number; name: string | null }) => `${e.label} #${e.id}${e.name ? ` · ${e.name}` : ""}`
  return (
    <div
      className="absolute left-1/2 top-[68px] z-[15] flex -translate-x-1/2 flex-col items-center gap-1.5 md:top-[76px]"
      onPointerDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <AnimatePresence>
        {solo && (
          <motion.div
            key="solo"
            layout
            initial={{ opacity: 0, y: -8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.95 }}
            transition={spring}
            title={VISUAL_ONLY}
            className="flex items-center gap-1.5 rounded-full border border-primary/60 bg-background/85 py-1 pl-3 pr-1 text-xs backdrop-blur-md"
          >
            <Focus className="size-3.5 text-primary" />
            <span>
              Solo: <span className="font-semibold">{name(solo)}</span>
            </span>
            <button type="button" aria-label="Clear solo" onClick={() => onClear("unsolo")} className="rounded-full p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">
              <X className="size-3.5" />
            </button>
          </motion.div>
        )}
        {hidden.length > 0 && (
          <motion.div key="hidden" layout initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={spring} className="flex flex-col items-center">
            <div className="flex items-center gap-1 rounded-full border border-border/70 bg-background/85 py-1 pl-3 pr-1 text-xs backdrop-blur-md" title={VISUAL_ONLY}>
              <EyeOff className="size-3.5 text-muted-foreground" />
              <button type="button" aria-expanded={open} onClick={() => setOpen((v) => !v)} className="px-1 hover:text-primary">
                {hidden.length} Hidden
              </button>
              <button type="button" aria-label="Show all hidden targets" onClick={() => onClear("unhide")} className="rounded-full p-1 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">
                <X className="size-3.5" />
              </button>
            </div>
            <AnimatePresence>
              {open && (
                <motion.ul
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
                  className="mt-1 w-60 overflow-hidden rounded-lg border border-white/10 bg-[#26282c]/95 p-1 text-xs backdrop-blur-xl"
                >
                  {hidden.map((h) => (
                    <li key={`${h.id}-${h.epoch}`} className="flex items-center justify-between gap-2 rounded-md px-2 py-1 hover:bg-white/6">
                      <span className="truncate">{name(h)}</span>
                      <button type="button" onClick={() => onClear("unhide", h.id)} className="shrink-0 text-muted-foreground hover:text-primary">
                        Show
                      </button>
                    </li>
                  ))}
                  <li className="px-2 pb-1 pt-1.5 text-[10px] leading-snug text-muted-foreground">{VISUAL_ONLY}</li>
                </motion.ul>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
