import { useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Camera, Check, Copy, Cpu, ImageUp, Loader2, ScanFace, Trash2, X, Zap } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { useRecognition } from "@/hooks/use-recognition"
import {
  errorText,
  recognitionApi,
  type FaceCandidate,
  type ModelInfo,
  type Person,
  type PlateEntry,
  type TensorRtCaps,
} from "@/lib/recognition"
import type { EspSettings } from "@/lib/types"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const ease = [0.22, 1, 0.36, 1] as const

const TOGGLES = [
  { key: "reid", label: "ReID", hint: "Keeps IDs through occlusions and when people come back. Needs Tracking." },
  { key: "known_faces", label: "Known Faces", hint: "Labels people from your gallery, unknown faces in red." },
  { key: "plates", label: "License Plates", hint: "Reads plates on cars, trucks, buses and motorcycles." },
  { key: "pets_show", label: "Show Pets", hint: "Cats and dogs get their own color and a PET tag." },
  { key: "pets_ignore_alarms", label: "Pets Ignore Alarms", hint: "Cats and dogs never trigger alarms." },
] as const

const COLOR_KEYS = [
  { key: "color_known", label: "Known" },
  { key: "color_unknown", label: "Unknown" },
  { key: "color_pet", label: "Pet" },
] as const

interface Props {
  cameraId: string | null
  currentModel: string | null
  settings: EspSettings
  onUpdate: Update
}

export function RecognitionPanel({ cameraId, currentModel, settings, onUpdate }: Props) {
  const platesOn = Boolean(settings.plates)
  const rec = useRecognition(cameraId, platesOn)
  const faceAssets = rec.status ? Object.values(rec.status.faces.assets) : []
  const plateAssets = rec.status ? Object.values(rec.status.plates.assets) : []
  const loading = (a: string[]) => a.some((s) => s === "downloading")

  return (
    <>
      <ul className="-mx-2">
        {TOGGLES.map((t) => {
          const on = Boolean(settings[t.key])
          const id = `recog-${t.key}`
          const needsTracking = t.key === "reid" && on && !settings.tracking
          return (
            <li key={t.key}>
              <label
                htmlFor={id}
                title={t.hint}
                className="flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors duration-150 hover:bg-accent/70"
              >
                <span className={cn("transition-colors duration-200", on ? "text-foreground" : "text-muted-foreground")}>
                  {t.label}
                  {needsTracking && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.preventDefault()
                        onUpdate("tracking", true)
                      }}
                      className="ml-2 rounded border border-primary/40 px-1.5 text-[11px] text-primary transition-colors hover:bg-primary/10"
                    >
                      Turn on Tracking
                    </button>
                  )}
                </span>
                <Switch id={id} checked={on} onCheckedChange={(v) => onUpdate(t.key, v)} />
              </label>
            </li>
          )
        })}
      </ul>

      <AnimatePresence initial={false}>
        {(Boolean(settings.known_faces) && loading(faceAssets)) || (platesOn && loading(plateAssets)) ? (
          <motion.p
            key="dl"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-2 overflow-hidden text-xs text-muted-foreground"
          >
            <Loader2 className="size-3.5 animate-spin text-primary" /> Downloading models into models/ …
          </motion.p>
        ) : null}
      </AnimatePresence>
      {rec.status?.plates.error && platesOn && (
        <InstallHint text="Plate models can't run here." cmd={rec.status.plates.install_cmd} />
      )}
      {rec.status?.faces.error && (
        <InstallHint text="This OpenCV build has no YuNet/SFace." cmd={rec.status.faces.install_cmd} />
      )}

      <div className="mt-4 space-y-4 border-t border-border pt-4">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-muted-foreground">Face Detector</span>
          <Segment
            value={String(settings.face_detector ?? "yunet")}
            options={[
              { value: "yunet", label: "YuNet" },
              { value: "haar", label: "Haar" },
            ]}
            onChange={(v) => onUpdate("face_detector", v)}
          />
        </div>
        <div className="space-y-2.5">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Face Match</span>
            <span className="tabular-nums">{Math.round(Number(settings.face_match_threshold ?? 0.36) * 100)}%</span>
          </div>
          <Slider
            min={0.2}
            max={0.8}
            step={0.01}
            value={[Number(settings.face_match_threshold ?? 0.36)]}
            aria-label="Face Match"
            onValueChange={([v]) => onUpdate("face_match_threshold", v, false)}
          />
          <p className="text-xs leading-relaxed text-muted-foreground">
            Higher means fewer false names but more faces stay unknown. 36 % is a good start.
          </p>
        </div>
        <div className="flex gap-1">
          {COLOR_KEYS.map((c) => (
            <ColorDot key={c.key} label={c.label} value={String(settings[c.key] ?? "#ffffff")} onChange={(v) => onUpdate(c.key, v, false)} />
          ))}
        </div>
      </div>

      <PeopleSection cameraId={cameraId} people={rec.people} setPeople={rec.setPeople} reload={rec.reloadPeople} />

      {(platesOn || rec.plates.length > 0) && (
        <PlatesSection
          plates={rec.plates}
          onClear={async () => {
            if (!cameraId) return
            await recognitionApi.clearPlates(cameraId).catch(() => undefined)
            rec.setPlates([])
          }}
        />
      )}

      <EngineSection model={rec.models.find((m) => m.name === currentModel) ?? null} trt={rec.trt} reload={rec.reloadModels} />
    </>
  )
}

// ---------------------------------------------------------------------------- small controls
function Segment({
  value,
  options,
  onChange,
}: {
  value: string
  options: { value: string; label: string }[]
  onChange: (v: string) => void
}) {
  return (
    <div role="radiogroup" aria-label="Face Detector" className="flex gap-1 rounded-lg bg-background/60 p-1">
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={cn(
              "relative rounded-md px-3 py-1 text-xs transition-colors duration-200",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="face-detector-indicator"
                className="absolute inset-0 rounded-md bg-primary"
                transition={{ type: "spring", stiffness: 450, damping: 34 }}
              />
            )}
            <span className="relative">{o.label}</span>
          </button>
        )
      })}
    </div>
  )
}

function ColorDot({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  const [local, setLocal] = useState(value)
  const [prev, setPrev] = useState(value)
  if (value !== prev) {
    setPrev(value)
    setLocal(value)
  }
  return (
    <label className="group flex flex-1 cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-accent/70">
      <span className="relative size-4 shrink-0 overflow-hidden rounded-full ring-1 ring-white/15 transition-transform duration-200 group-hover:scale-110">
        <span className="absolute inset-0" style={{ background: local }} />
        <input
          type="color"
          value={local}
          aria-label={`${label} color`}
          onChange={(e) => {
            setLocal(e.target.value)
            onChange(e.target.value)
          }}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
      </span>
      <span className="truncate text-muted-foreground group-hover:text-foreground">{label}</span>
    </label>
  )
}

function SectionTitle({ icon: Icon, title, aside }: { icon: typeof ScanFace; title: string; aside?: React.ReactNode }) {
  return (
    <div className="mb-2 flex items-center gap-2">
      <Icon className="size-4 text-primary" />
      <h3 className="flex-1 text-sm font-medium">{title}</h3>
      {aside}
    </div>
  )
}

function InstallHint({ text, cmd }: { text: string; cmd: string | null }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="mt-2 rounded-lg border border-border bg-background/50 p-2.5 text-xs text-muted-foreground">
      <p>{text}{cmd ? " Install with:" : ""}</p>
      {cmd && (
        <div className="mt-1.5 flex items-center gap-2">
          <code className="min-w-0 flex-1 truncate rounded bg-background px-1.5 py-1 text-[11px] text-foreground" title={cmd}>
            {cmd}
          </code>
          <button
            type="button"
            aria-label="Copy install command"
            onClick={() => {
              void navigator.clipboard?.writeText(cmd).then(() => {
                setCopied(true)
                window.setTimeout(() => setCopied(false), 1500)
              })
            }}
            className="rounded-md p-1 transition-colors hover:bg-accent hover:text-foreground"
          >
            {copied ? <Check className="size-3.5 text-primary" /> : <Copy className="size-3.5" />}
          </button>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------- people
function PeopleSection({
  cameraId,
  people,
  setPeople,
  reload,
}: {
  cameraId: string | null
  people: Person[] | null
  setPeople: (p: Person[]) => void
  reload: () => Promise<void>
}) {
  const [name, setName] = useState("")
  const [busy, setBusy] = useState<null | "upload" | "capture" | "add">(null)
  const [error, setError] = useState<string | null>(null)
  const [candidates, setCandidates] = useState<FaceCandidate[] | null>(null)
  const file = useRef<HTMLInputElement>(null)

  const trimmed = name.trim()
  const done = async () => {
    setName("")
    setCandidates(null)
    setError(null)
    await reload().catch(() => undefined)
  }

  const upload = async (f: File) => {
    if (!trimmed) return setError("Enter a name first.")
    setBusy("upload")
    try {
      await recognitionApi.addFromPhoto(trimmed, f)
      await done()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(null)
    }
  }

  const capture = async () => {
    if (!cameraId) return
    setBusy("capture")
    setError(null)
    try {
      const r = await recognitionApi.candidates(cameraId)
      setCandidates(r.faces)
      if (r.faces.length === 0) setError("No clear face in the current frame. Faces need to be at least 36 px wide.")
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(null)
    }
  }

  const pick = async (c: FaceCandidate) => {
    const n = trimmed || c.match || ""
    if (!n) return setError("Enter a name first, then pick the face.")
    setBusy("add")
    try {
      await recognitionApi.addFromCandidate(n, c.id)
      await done()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <section className="mt-5 border-t border-border pt-4">
      <SectionTitle
        icon={ScanFace}
        title="People"
        aside={<span className="text-xs tabular-nums text-muted-foreground">{people?.length ?? 0}</span>}
      />
      <div className="flex gap-1.5">
        <input
          value={name}
          maxLength={40}
          placeholder="Name"
          aria-label="Name of the person"
          onChange={(e) => setName(e.target.value)}
          className="min-w-0 flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm outline-none transition-colors placeholder:text-muted-foreground/70 focus:border-primary/60"
        />
        <ActionButton label="Upload photo" busy={busy === "upload"} onClick={() => file.current?.click()} icon={ImageUp} />
        <ActionButton label="Capture face from frame" busy={busy === "capture"} onClick={() => void capture()} icon={Camera} disabled={!cameraId} />
        <input
          ref={file}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            e.target.value = ""
            if (f) void upload(f)
          }}
        />
      </div>

      <AnimatePresence initial={false}>
        {candidates && candidates.length > 0 && (
          <motion.div
            key="cands"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
              <span>Pick a face to add it{trimmed ? ` as ${trimmed}` : ""}</span>
              <button type="button" aria-label="Close face picker" onClick={() => setCandidates(null)} className="rounded p-0.5 hover:text-foreground">
                <X className="size-3.5" />
              </button>
            </div>
            <div className="mt-1.5 flex gap-2 overflow-x-auto pb-1">
              {candidates.map((c, i) => (
                <motion.button
                  key={c.id}
                  type="button"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: i * 0.04, duration: 0.2, ease }}
                  onClick={() => void pick(c)}
                  disabled={busy !== null}
                  className="group relative shrink-0 overflow-hidden rounded-lg ring-1 ring-border transition-all duration-200 hover:ring-2 hover:ring-primary disabled:opacity-50"
                  title={c.match ? `Looks like ${c.match} (${Math.round(c.score * 100)} %)` : "Unknown face"}
                >
                  {c.thumb && <img src={c.thumb} alt="" className="size-16 object-cover" />}
                  <span
                    className={cn(
                      "absolute inset-x-0 bottom-0 truncate px-1 py-0.5 text-[10px] leading-tight",
                      c.match ? "bg-[#7ee0a1]/85 text-[#222427]" : "bg-background/80 text-muted-foreground",
                    )}
                  >
                    {c.match ?? "Unknown"}
                  </span>
                </motion.button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      {people && people.length === 0 ? (
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          Nobody added yet. Type a name, then upload a photo or capture the face from the live image. Two or three
          samples per person (different angles) work best. Everything stays in data/faces on this machine.
        </p>
      ) : (
        <ul className="mt-2 space-y-0.5">
          <AnimatePresence initial={false}>
            {(people ?? []).map((p) => (
              <motion.li
                key={p.id}
                layout
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease }}
                className="overflow-hidden"
              >
                <PersonRow
                  person={p}
                  onRenamed={(np) => setPeople((people ?? []).map((x) => (x.id === np.id ? np : x)))}
                  onDeleted={() => setPeople((people ?? []).filter((x) => x.id !== p.id))}
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </section>
  )
}

function ActionButton({
  label,
  icon: Icon,
  busy,
  disabled,
  onClick,
}: {
  label: string
  icon: typeof Camera
  busy: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy || disabled}
      aria-label={label}
      title={label}
      className="grid size-8 shrink-0 place-items-center rounded-md border border-border text-muted-foreground transition-colors duration-200 hover:border-primary/60 hover:text-primary disabled:opacity-40"
    >
      {busy ? <Loader2 className="size-4 animate-spin" /> : <Icon className="size-4" />}
    </button>
  )
}

function PersonRow({ person, onRenamed, onDeleted }: { person: Person; onRenamed: (p: Person) => void; onDeleted: () => void }) {
  const [label, setLabel] = useState(person.name)
  const [prev, setPrev] = useState(person.name)
  const [err, setErr] = useState<string | null>(null)
  if (person.name !== prev) {
    setPrev(person.name)
    setLabel(person.name)
  }
  const commit = async () => {
    const v = label.trim()
    if (!v || v === person.name) return setLabel(person.name)
    try {
      onRenamed((await recognitionApi.rename(person.id, v)).person)
      setErr(null)
    } catch (e) {
      setErr(errorText(e))
      setLabel(person.name)
    }
  }
  return (
    <div className="group flex items-center gap-2.5 rounded-md px-1.5 py-1 transition-colors hover:bg-accent/70">
      <img src={person.thumb} alt="" className="size-8 shrink-0 rounded-full object-cover ring-1 ring-[#7ee0a1]/50" />
      <div className="min-w-0 flex-1">
        <input
          value={label}
          maxLength={40}
          aria-label={`Rename ${person.name}`}
          onChange={(e) => setLabel(e.target.value)}
          onBlur={() => void commit()}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          className="w-full rounded bg-transparent px-1 py-0.5 text-sm outline-none focus:bg-background/60 focus:ring-1 focus:ring-ring"
        />
        <p className={cn("px-1 text-[11px]", err ? "text-destructive" : "text-muted-foreground")}>
          {err ?? `${person.samples} ${person.samples === 1 ? "sample" : "samples"}`}
        </p>
      </div>
      <button
        type="button"
        onClick={async () => {
          try {
            await recognitionApi.remove(person.id)
            onDeleted()
          } catch (e) {
            setErr(errorText(e))
          }
        }}
        aria-label={`Delete ${person.name}`}
        className="rounded-md p-1 text-muted-foreground opacity-0 transition-all hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
      >
        <Trash2 className="size-3.5" />
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------- plates
function ago(ts: number) {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return `${Math.round(s)} s ago`
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  return `${Math.round(s / 3600)} h ago`
}

function PlatesSection({ plates, onClear }: { plates: PlateEntry[]; onClear: () => void }) {
  return (
    <section className="mt-5 border-t border-border pt-4">
      <SectionTitle
        icon={Zap}
        title="Recent Plates"
        aside={
          plates.length > 0 && (
            <button type="button" onClick={onClear} className="text-xs text-muted-foreground transition-colors hover:text-foreground">
              Clear
            </button>
          )
        }
      />
      {plates.length === 0 ? (
        <p className="text-xs leading-relaxed text-muted-foreground">
          No plates read yet. Plates need a vehicle detection and are read best when the car is close and the plate is
          at least ~80 px wide. Turn on Tracking so reads are voted per vehicle.
        </p>
      ) : (
        <ul className="space-y-1">
          <AnimatePresence initial={false}>
            {plates.slice(0, 12).map((p) => (
              <motion.li
                key={p.key}
                layout
                initial={{ opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.22, ease }}
                className="flex items-center gap-2.5 rounded-md px-1.5 py-1 hover:bg-accent/70"
              >
                {p.thumb ? (
                  <img src={p.thumb} alt="" className="h-6 w-16 shrink-0 rounded-sm object-cover ring-1 ring-border" />
                ) : (
                  <span className="h-6 w-16 shrink-0 rounded-sm bg-muted" />
                )}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium tracking-wider text-primary">{p.plate}</span>
                  <span className="block text-[11px] text-muted-foreground">
                    {p.vehicle}
                    {p.track_id !== null ? ` #${p.track_id}` : ""} · {p.reads} {p.reads === 1 ? "read" : "reads"} · {ago(p.last_seen)}
                  </span>
                </span>
                <span className="text-xs tabular-nums text-muted-foreground" title="Share of votes for this reading">
                  {Math.round(p.conf * 100)}%
                </span>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </section>
  )
}

// ---------------------------------------------------------------------------- model / TensorRT
const STAGES: Record<string, string> = {
  queued: "Starting…",
  downloading: "Downloading weights…",
  loading: "Loading model…",
  exporting: "Building TensorRT engine (takes a few minutes)…",
}

function EngineSection({ model, trt, reload }: { model: ModelInfo | null; trt: TensorRtCaps | null; reload: () => Promise<void> }) {
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  const job = model?.job
  const running = job?.status === "running"
  const canBuild = Boolean(trt?.available) && !running && !pending && Boolean(model)

  const build = async () => {
    if (!model) return
    setPending(true)
    setError(null)
    try {
      await recognitionApi.buildEngine(model.name)
      await reload()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setPending(false)
    }
  }

  const remove = async () => {
    if (!model) return
    try {
      await recognitionApi.deleteEngine(model.name)
      await reload()
    } catch (e) {
      setError(errorText(e))
    }
  }

  return (
    <section className="mt-5 border-t border-border pt-4">
      <SectionTitle icon={Cpu} title="Model" />
      {!model ? (
        <p className="text-xs text-muted-foreground">No model selected.</p>
      ) : (
        <>
          <div className="flex items-baseline justify-between gap-2 text-sm">
            <span>
              {model.label}
              {model.keypoints && <span className="ml-1.5 rounded bg-primary/15 px-1 text-[10px] text-primary">POSE</span>}
            </span>
            <span className={cn("text-xs", model.engine_active ? "text-primary" : "text-muted-foreground")}>
              {model.engine_active ? "TensorRT active" : model.engine ? "Engine ready" : "PyTorch"}
            </span>
          </div>
          {model.family === "RT-DETR" && (
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              RT-DETR is slower than YOLO but strong on crowded scenes. ReID uses a small extra classifier with it.
            </p>
          )}

          <AnimatePresence initial={false}>
            {running && job && (
              <motion.div
                key="progress"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-2 overflow-hidden"
              >
                <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                  <motion.div
                    className="h-full rounded-full bg-primary"
                    animate={{ width: `${Math.round(job.progress * 100)}%` }}
                    transition={{ duration: 0.6, ease }}
                  />
                </div>
                <p className="mt-1 flex justify-between text-[11px] text-muted-foreground">
                  <span>{STAGES[job.stage] ?? job.stage}</span>
                  <span className="tabular-nums">{Math.round(job.elapsed_s ?? 0)} s</span>
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={() => void build()}
              disabled={!canBuild}
              className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-primary/50 py-2 text-sm text-primary transition-colors duration-200 hover:border-primary hover:bg-primary/10 disabled:cursor-not-allowed disabled:border-border disabled:text-muted-foreground disabled:hover:bg-transparent"
            >
              {running || pending ? <Loader2 className="size-4 animate-spin" /> : <Zap className="size-4" />}
              {model.engine ? "Rebuild for GPU" : "Optimize for GPU"}
            </button>
            {model.engine && !running && (
              <button
                type="button"
                onClick={() => void remove()}
                aria-label="Delete TensorRT engine"
                title="Delete engine and go back to PyTorch"
                className="grid w-9 place-items-center rounded-lg border border-border text-muted-foreground transition-colors hover:border-destructive/60 hover:text-destructive"
              >
                <Trash2 className="size-4" />
              </button>
            )}
          </div>
          {job?.status === "failed" && <p className="mt-2 text-xs text-destructive">Optimization failed: {job.error}</p>}
          {model.engine_error && <p className="mt-2 text-xs text-destructive">Engine could not be loaded, using PyTorch: {model.engine_error}</p>}
          {error && <p className="mt-2 text-xs text-destructive">{error}</p>}
          {trt && !trt.available && (
            <InstallHint
              text={
                trt.reason === "no_cuda"
                  ? "GPU optimization needs an NVIDIA GPU with CUDA."
                  : "GPU optimization needs TensorRT (large download, NVIDIA only)."
              }
              cmd={trt.install_cmd}
            />
          )}
          {trt?.available && !model.engine && !running && (
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
              Builds an FP16 TensorRT engine into models/. Detection keeps running meanwhile and switches over when it's done.
            </p>
          )}
        </>
      )}
    </section>
  )
}
