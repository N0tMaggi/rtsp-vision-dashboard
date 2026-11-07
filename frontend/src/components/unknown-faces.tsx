import { useCallback, useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Check, GitMerge, Loader2, Pencil, ScanFace, Trash2, UserRound, X } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { contextApi, contextError, type UnknownFace } from "@/lib/context-api"
import type { Person } from "@/lib/recognition"
import type { EspSettings } from "@/lib/types"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const ease = [0.22, 1, 0.36, 1] as const
const POLL_MS = 4000

function ago(epoch: number) {
  const s = Math.max(0, Date.now() / 1000 - epoch)
  if (s < 60) return "just now"
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} d ago`
}

/**
 * Automatic face ids: unknown faces numbered "Face #1, #2, …" by the backend
 * (clustered SFace embeddings, data/faces/unknown). Name one to move it into
 * the known gallery, merge duplicates, or delete it.
 */
export function UnknownFaces({
  settings,
  onUpdate,
  people,
  onPeopleChanged,
}: {
  settings: EspSettings
  onUpdate: Update
  people: Person[] | null
  onPeopleChanged: () => Promise<void>
}) {
  const knownOn = Boolean(settings.known_faces)
  const autoOn = settings.face_auto_ids === undefined ? true : Boolean(settings.face_auto_ids)
  const threshold = Number(settings.face_cluster_threshold ?? 0.45)
  const [faces, setFaces] = useState<UnknownFace[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      setFaces((await contextApi.unknownFaces()).faces)
    } catch {
      /* older backend or server busy: keep the last list */
    }
  }, [])

  useEffect(() => {
    let alive = true
    let timer: number | undefined
    const tick = async () => {
      if (!document.hidden) await load()
      if (alive) timer = window.setTimeout(tick, POLL_MS)
    }
    void tick()
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [load])

  const act = async (fn: () => Promise<unknown>, peopleChanged = false) => {
    setError(null)
    try {
      await fn()
      await load()
      if (peopleChanged) await onPeopleChanged().catch(() => undefined)
    } catch (e) {
      setError(contextError(e))
    }
  }

  return (
    <section className="mt-5 border-t border-border pt-4">
      <div className="mb-2 flex items-center gap-2">
        <ScanFace className="size-4 text-primary" />
        <h3 className="flex-1 text-sm font-medium">Unknown Faces</h3>
        <span className="text-xs tabular-nums text-muted-foreground">{faces?.length ?? 0}</span>
      </div>
      <label
        htmlFor="recog-face_auto_ids"
        title="Numbers every unknown face as Face #1, #2, … and recognizes it again later. Needs Known Faces."
        className="-mx-2 flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors duration-150 hover:bg-accent/70"
      >
        <span className={cn(autoOn && knownOn ? "text-foreground" : "text-muted-foreground")}>
          Auto Face IDs
          {!knownOn && autoOn && <span className="ml-2 text-[11px] text-muted-foreground">needs Known Faces</span>}
        </span>
        <Switch id="recog-face_auto_ids" checked={autoOn} onCheckedChange={(v) => onUpdate("face_auto_ids", v)} />
      </label>
      <div className="mt-1 space-y-2.5">
        <div className="flex justify-between text-sm">
          <span className="text-muted-foreground">Same Face</span>
          <span className="tabular-nums">{Math.round(threshold * 100)}%</span>
        </div>
        <Slider
          min={0.3}
          max={0.8}
          step={0.01}
          value={[threshold]}
          aria-label="Same Face threshold"
          onValueChange={([v]) => onUpdate("face_cluster_threshold", v, false)}
        />
        <p className="text-xs leading-relaxed text-muted-foreground">
          How alike two faces must be to share a number. Higher splits one person into several ids, lower may merge
          strangers. 45 % is a good start.
        </p>
      </div>

      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      {faces && faces.length === 0 ? (
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          No unknown faces yet. With Known Faces and Auto Face IDs on, every face that isn't in your gallery gets a
          number after a few clear sightings. Everything stays in data/faces/unknown on this machine.
        </p>
      ) : (
        <ul className="mt-2 space-y-0.5">
          <AnimatePresence initial={false}>
            {(faces ?? []).map((f) => (
              <motion.li
                key={f.uid}
                layout
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease }}
                className="overflow-hidden"
              >
                <FaceRow
                  face={f}
                  others={(faces ?? []).filter((x) => x.uid !== f.uid)}
                  people={people ?? []}
                  onName={(name) => act(() => contextApi.nameFace(f.uid, name), true)}
                  onMerge={(into) => act(() => contextApi.mergeFace(f.uid, into))}
                  onDelete={() => act(() => contextApi.deleteFace(f.uid))}
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </section>
  )
}

function FaceRow({
  face,
  others,
  people,
  onName,
  onMerge,
  onDelete,
}: {
  face: UnknownFace
  others: UnknownFace[]
  people: Person[]
  onName: (name: string) => Promise<void>
  onMerge: (into: number) => Promise<void>
  onDelete: () => Promise<void>
}) {
  const [mode, setMode] = useState<null | "name" | "merge" | "delete">(null)
  const [name, setName] = useState("")
  const [busy, setBusy] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (mode === "name") input.current?.focus()
  }, [mode])

  const run = async (fn: () => Promise<void>) => {
    setBusy(true)
    try {
      await fn()
      setMode(null)
      setName("")
    } finally {
      setBusy(false)
    }
  }
  const q = name.trim().toLowerCase()
  const matches = q ? people.filter((p) => p.name.toLowerCase().includes(q)).slice(0, 4) : []

  return (
    <div className="rounded-md px-1.5 py-1 transition-colors hover:bg-accent/70">
      <div className="group flex items-center gap-2.5">
        {face.thumb ? (
          <img src={face.thumb} alt="" className="size-8 shrink-0 rounded-full object-cover ring-1 ring-[#ff5a5a]/50" />
        ) : (
          <span className="grid size-8 shrink-0 place-items-center rounded-full bg-background/60 ring-1 ring-border">
            <UserRound className="size-4 text-muted-foreground" />
          </span>
        )}
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">{face.label}</p>
          <p className="truncate text-[11px] text-muted-foreground">
            {face.sightings} {face.sightings === 1 ? "sighting" : "sightings"} · {ago(face.last_seen)}
          </p>
        </div>
        <div className={cn("flex items-center gap-0.5 transition-opacity", mode ? "opacity-100" : "opacity-60 group-hover:opacity-100")}>
          <RowButton label={`Name ${face.label}`} active={mode === "name"} onClick={() => setMode(mode === "name" ? null : "name")} icon={Pencil} />
          {others.length > 0 && (
            <RowButton label={`Merge ${face.label} into another face`} active={mode === "merge"} onClick={() => setMode(mode === "merge" ? null : "merge")} icon={GitMerge} />
          )}
          <RowButton label={`Delete ${face.label}`} active={mode === "delete"} danger onClick={() => setMode(mode === "delete" ? null : "delete")} icon={Trash2} />
        </div>
      </div>
      <AnimatePresence initial={false}>
        {mode && (
          <motion.div
            key={mode}
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease }}
            className="overflow-hidden"
          >
            <div className="pb-1 pl-[42px] pt-1.5">
              {mode === "name" && (
                <>
                  <form
                    className="flex gap-1.5"
                    onSubmit={(e) => {
                      e.preventDefault()
                      if (name.trim()) void run(() => onName(name.trim()))
                    }}
                  >
                    <input
                      ref={input}
                      value={name}
                      maxLength={40}
                      placeholder="Name This Person"
                      aria-label={`Name for ${face.label}`}
                      onChange={(e) => setName(e.target.value)}
                      onKeyDown={(e) => e.key === "Escape" && setMode(null)}
                      className="min-w-0 flex-1 rounded-md border border-border bg-background/60 px-2 py-1 text-sm outline-none focus:border-primary/60"
                    />
                    <button type="submit" disabled={busy || !name.trim()} aria-label="Save name" className="grid size-7 place-items-center rounded-md bg-primary text-primary-foreground disabled:opacity-40">
                      {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Check className="size-3.5" />}
                    </button>
                  </form>
                  {matches.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {matches.map((p) => (
                        <button
                          key={p.id}
                          type="button"
                          onClick={() => setName(p.name)}
                          className="flex items-center gap-1 rounded-full border border-white/10 py-0.5 pl-0.5 pr-2 text-[11px] hover:border-[#7ee0a1]/60"
                        >
                          <img src={p.thumb} alt="" className="size-4 rounded-full object-cover" /> {p.name}
                        </button>
                      ))}
                    </div>
                  )}
                  <p className="mt-1 text-[11px] text-muted-foreground">Moves this face into People (an existing name adds samples).</p>
                </>
              )}
              {mode === "merge" && (
                <div className="flex flex-wrap gap-1">
                  {others.slice(0, 12).map((o) => (
                    <button
                      key={o.uid}
                      type="button"
                      disabled={busy}
                      onClick={() => void run(() => onMerge(o.uid))}
                      className="flex items-center gap-1 rounded-full border border-white/10 py-0.5 pl-0.5 pr-2 text-[11px] transition-colors hover:border-primary/60 disabled:opacity-40"
                    >
                      {o.thumb ? <img src={o.thumb} alt="" className="size-4 rounded-full object-cover" /> : <UserRound className="size-4" />}
                      {o.label}
                    </button>
                  ))}
                </div>
              )}
              {mode === "delete" && (
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="flex-1 text-muted-foreground">Forget {face.label} and its samples?</span>
                  <button type="button" onClick={() => setMode(null)} className="rounded px-1.5 py-0.5 text-muted-foreground hover:text-foreground">
                    <X className="size-3.5" />
                  </button>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void run(onDelete)}
                    className="rounded-md bg-destructive/90 px-2 py-0.5 font-medium text-white disabled:opacity-40"
                  >
                    Delete
                  </button>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function RowButton({
  label,
  icon: Icon,
  active,
  danger,
  onClick,
}: {
  label: string
  icon: typeof Pencil
  active: boolean
  danger?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-pressed={active}
      title={label}
      className={cn(
        "rounded-md p-1.5 transition-colors",
        active ? "bg-accent text-foreground" : "text-muted-foreground hover:text-foreground",
        danger && "hover:text-destructive",
      )}
    >
      <Icon className="size-3.5" />
    </button>
  )
}
