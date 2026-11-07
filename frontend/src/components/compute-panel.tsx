import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Check, ChevronDown, Copy, Gauge, Info, Loader2, Microchip, OctagonAlert, RotateCcw, TriangleAlert } from "lucide-react"
import {
  DEVICE_SHORT,
  computeApi,
  computeErrorText,
  setComputeState,
  useCompute,
  type BenchRow,
  type ComputeCamera,
  type ComputeState,
  type DeviceId,
  type DeviceOption,
  type Notice,
  type Workload,
} from "@/lib/compute"
import { modelLabel } from "@/lib/models"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const
const spring = { type: "spring", stiffness: 450, damping: 34 } as const

const STAGES: Record<string, string> = {
  queued: "Waiting…",
  preparing: "Preparing…",
  downloading: "Downloading weights…",
  exporting: "Exporting for OpenVINO (one time)…",
  compiling: "Compiling for the device…",
  loading: "Loading model…",
}

/** Compute tab: which hardware runs which workload. Everything defaults to Auto. */
export function ComputePanel({ cameraId }: { cameraId: string | null }) {
  const s = useCompute(true)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  const apply = async (fn: () => Promise<ComputeState>) => {
    setPending(true)
    setError(null)
    try {
      setComputeState(await fn())
    } catch (e) {
      setError(computeErrorText(e))
    } finally {
      setPending(false)
    }
  }

  if (!s) {
    return (
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <Loader2 className="size-3.5 animate-spin text-primary" /> Reading hardware…
      </p>
    )
  }

  return (
    <>
      <p className="text-xs leading-relaxed text-muted-foreground">
        Auto is recommended and picks the fastest backend known to work. Change a workload only if you want a specific
        chip to do it, for example to keep the GPU free while gaming.
      </p>

      <AnimatePresence initial={false}>
        {s.non_default && (
          <motion.div
            key="reset"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            <ResetButton disabled={pending} onClick={() => void apply(computeApi.reset)} className="mt-3" />
          </motion.div>
        )}
      </AnimatePresence>
      {error && <p className="mt-2 text-xs text-destructive">{error}</p>}

      <Hardware s={s} />

      {s.workloads.map((w) => (
        <WorkloadSection
          key={w.id}
          w={w}
          s={s}
          cameraId={cameraId}
          disabled={pending}
          onPick={(dev) => void apply(() => computeApi.patch({ workloads: { [w.id]: dev } }))}
          onCamera={(cam, dev) => void apply(() => computeApi.patch({ cameras: { [cam]: dev } }))}
        />
      ))}

      {s.non_default && (
        <ResetButton disabled={pending} onClick={() => void apply(computeApi.reset)} className="mt-5" />
      )}
    </>
  )
}

function ResetButton({ onClick, disabled, className }: { onClick: () => void; disabled?: boolean; className?: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "flex w-full items-center justify-center gap-2 rounded-lg border border-primary/50 py-2 text-sm text-primary transition-colors duration-200 hover:border-primary hover:bg-primary/10 disabled:opacity-50",
        className,
      )}
    >
      <RotateCcw className="size-4" /> Reset to recommended
    </button>
  )
}

// ---------------------------------------------------------------------------- hardware overview
function Hardware({ s }: { s: ComputeState }) {
  const ov = s.hardware.openvino
  return (
    <section className="mt-4 border-t border-border pt-4">
      <Title text="Hardware" right={!s.hardware.ready ? <Loader2 className="size-3.5 animate-spin text-primary" /> : null} />
      <ul className="space-y-1.5">
        {s.hardware.devices.map((d) => (
          <li key={d.id} className="flex items-start gap-2.5 text-sm" title={d.reason ?? undefined}>
            <span
              className={cn(
                "mt-1.5 size-1.5 shrink-0 rounded-full",
                d.available ? "bg-primary shadow-[0_0_8px_var(--color-primary)]" : "bg-muted-foreground/40",
              )}
            />
            <span className="min-w-0 flex-1">
              <span className={cn("flex items-baseline justify-between gap-2", !d.available && "text-muted-foreground")}>
                <span className="truncate">{d.label}</span>
                <span className="shrink-0 text-[11px] text-muted-foreground">{d.backend}</span>
              </span>
              <span className="block truncate text-[11px] text-muted-foreground">
                {d.available ? d.name : (d.reason ?? "Not available")}
              </span>
            </span>
          </li>
        ))}
      </ul>
      {!ov.installed && ov.install_cmd && (
        <InstallHint text="Intel NPU and Intel GPU need OpenVINO (optional, ~60 MB):" cmd={ov.install_cmd} />
      )}
      {ov.installed && ov.version && <p className="mt-2 text-[11px] text-muted-foreground">OpenVINO {ov.version}</p>}
    </section>
  )
}

// ---------------------------------------------------------------------------- one workload
function WorkloadSection({
  w,
  s,
  cameraId,
  disabled,
  onPick,
  onCamera,
}: {
  w: Workload
  s: ComputeState
  cameraId: string | null
  disabled: boolean
  onPick: (dev: DeviceId) => void
  onCamera: (cam: string, dev: DeviceId | null) => void
}) {
  const sel = w.options.find((o) => o.id === w.assignment)
  const unavailable = w.options.filter((o) => !o.available && o.reason)
  const activeCam = w.id === "yolo" ? s.cameras.find((c) => c.id === cameraId) : undefined
  return (
    <section className="mt-5 border-t border-border pt-4">
      <Title text={w.label} right={<RunBadge w={w} cam={activeCam} />} />
      <p className="mb-2.5 text-xs text-muted-foreground">{w.description}</p>
      <DevicePicker id={`cmp-${w.id}`} options={w.options} value={w.assignment} onChange={(v) => v && onPick(v)} disabled={disabled} />
      <p className="mt-1.5 text-[11px] text-muted-foreground">
        {w.assignment === "auto" ? (
          <>
            Auto runs on <span className="text-foreground">{w.auto.label}</span>
          </>
        ) : (
          <>
            Using <span className="text-foreground">{sel?.label}</span>
            {w.effective.device === "auto" && <span className="text-destructive"> · fell back to Auto</span>}
          </>
        )}
        {w.id === "yolo" && w.model && <> · {modelLabel(w.model)}</>}
      </p>
      {unavailable.length > 0 && <UnavailableList options={unavailable} />}

      <Notices items={w.warnings} />

      <SpeedTest w={w} s={s} cameraId={cameraId} />

      {w.per_camera && s.cameras.length > 0 && (
        <PerCamera cameras={s.cameras} options={w.options} global={w.assignment} disabled={disabled} onChange={onCamera} />
      )}
    </section>
  )
}

function RunBadge({ w, cam }: { w: Workload; cam?: ComputeCamera }) {
  const st = cam?.state ?? w.state
  if (w.effective.fallback_reason && (cam ? cam.effective.requested === w.assignment : true)) {
    return <span className="text-[11px] text-destructive">Fallback: Auto</span>
  }
  if (st === "preparing" || st === "switching") {
    const job = cam?.job
    return (
      <span className="flex items-center gap-1.5 text-[11px] text-primary">
        <Loader2 className="size-3 animate-spin" />
        {job ? (STAGES[job.stage] ?? "Preparing…") : "Preparing…"}
        {job?.elapsed_s ? <span className="tabular-nums text-muted-foreground">{Math.round(job.elapsed_s)} s</span> : null}
      </span>
    )
  }
  const req = cam?.effective.requested ?? w.assignment
  if (req === "auto") return <span className="text-[11px] text-muted-foreground">Recommended</span>
  if (st === "active") return <span className="text-[11px] text-primary">Running on {DEVICE_SHORT[req]}</span>
  if (st === "pending" || st === "idle") {
    return <span className="text-[11px] text-muted-foreground">{w.id === "yolo" ? "Starts with detection" : "Starts on first use"}</span>
  }
  return null
}

function UnavailableList({ options }: { options: DeviceOption[] }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="mt-1">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex items-center gap-1 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
      >
        Why {options.length === 1 ? "is" : "are"} {options.map((o) => o.short ?? o.label).join(", ")} greyed out?
        <ChevronDown className={cn("size-3 transition-transform", open && "rotate-180")} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2, ease }}
            className="overflow-hidden"
          >
            {options.map((o) => (
              <li key={o.id} className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                <span className="text-foreground">{o.label}:</span> {o.reason}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------- picker
function DevicePicker({
  id,
  options,
  value,
  onChange,
  disabled,
  inherit,
  small,
}: {
  id: string
  options: DeviceOption[]
  value: DeviceId | null
  onChange: (v: DeviceId | null) => void
  disabled?: boolean
  inherit?: string
  small?: boolean
}) {
  const items: { id: DeviceId | null; label: string; available: boolean; reason: string | null }[] = [
    ...(inherit ? [{ id: null, label: inherit, available: true, reason: "Use the default above" }] : []),
    ...options.map((o) => ({ id: o.id, label: o.id === "auto" ? "Auto" : (o.short ?? o.label), available: o.available, reason: o.reason })),
  ]
  return (
    <div role="radiogroup" aria-label="Device" className={cn("flex gap-1 rounded-lg bg-background/60 p-1", small && "p-0.5")}>
      {items.map((o) => {
        const active = o.id === value
        return (
          <button
            key={o.id ?? "inherit"}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled || !o.available}
            title={o.available ? (o.id === "auto" ? "Recommended" : undefined) : (o.reason ?? undefined)}
            onClick={() => onChange(o.id)}
            className={cn(
              "relative flex-1 rounded-md text-xs transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-30",
              small ? "py-1 text-[11px]" : "py-1.5",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && <motion.span layoutId={`${id}-ind`} className="absolute inset-0 rounded-md bg-primary" transition={spring} />}
            <span className="relative">{o.label}</span>
            {o.id === "auto" && !active && !small && (
              <span className="absolute right-1.5 top-1.5 size-1 rounded-full bg-primary" aria-label="recommended" />
            )}
          </button>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------- notices
function Notices({ items }: { items: Notice[] }) {
  if (!items.length) return null
  return (
    <ul className="mt-3 space-y-1.5">
      <AnimatePresence initial={false}>
        {items.map((n) => {
          const Icon = n.level === "error" ? OctagonAlert : n.level === "warn" ? TriangleAlert : Info
          return (
            <motion.li
              key={n.text}
              layout
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease }}
              className={cn(
                "flex gap-2 rounded-md border px-2.5 py-2 text-xs leading-relaxed",
                n.level === "error" && "border-destructive/60 text-destructive",
                n.level === "warn" && "border-primary/60 text-foreground",
                n.level === "info" && "border-border text-muted-foreground",
              )}
            >
              <Icon className={cn("mt-0.5 size-3.5 shrink-0", n.level === "warn" && "text-primary")} />
              <span>{n.text}</span>
            </motion.li>
          )
        })}
      </AnimatePresence>
    </ul>
  )
}

// ---------------------------------------------------------------------------- speed test
function SpeedTest({ w, s, cameraId }: { w: Workload; s: ComputeState; cameraId: string | null }) {
  const [error, setError] = useState<string | null>(null)
  const job = s.benchmark_job
  const mine = job && job.workload === w.id && (w.id !== "yolo" || job.model === w.model || job.status === "running")
  const running = job?.status === "running"
  const rows: BenchRow[] | undefined = mine && job ? job.rows : w.benchmark?.rows
  const finished = mine && job ? job.finished : w.benchmark?.finished
  const best = Math.max(1, ...(rows ?? []).map((r) => r.ms ?? 0))

  const start = async () => {
    setError(null)
    try {
      const r = await computeApi.benchmark(w.id, w.id === "yolo" ? cameraId : null)
      if (s) setComputeState({ ...s, benchmark_job: r.job })
    } catch (e) {
      setError(computeErrorText(e))
    }
  }

  return (
    <div className="mt-3">
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => void start()}
          disabled={running}
          className="flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-md border border-border px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground disabled:opacity-50"
        >
          {running && mine ? <Loader2 className="size-3.5 animate-spin text-primary" /> : <Gauge className="size-3.5" />}
          Test speed
        </button>
        <span className="text-[11px] text-muted-foreground">
          {running && mine
            ? `Measuring… ${Math.round(job?.elapsed_s ?? 0)} s`
            : finished
              ? `Measured ${ago(finished)}${w.id === "yolo" ? " · ms per frame, end to end" : " · ms per call"}`
              : "about 5–15 s, first run longer"}
        </span>
      </div>
      {error && <p className="mt-1.5 text-xs text-destructive">{error}</p>}
      {mine && job?.status === "failed" && <p className="mt-1.5 text-xs text-destructive">{job.error}</p>}
      {rows && rows.length > 0 && (
        <ul className="mt-2 space-y-1">
          {rows.map((r) => (
            <li key={r.device} className="grid grid-cols-[minmax(0,1fr)_64px_52px] items-center gap-2 text-[11px]">
              <span className={cn("truncate", r.device === w.assignment ? "text-foreground" : "text-muted-foreground")} title={r.label}>
                {r.label}
                {r.auto && <span className="ml-1 text-primary">· Auto</span>}
              </span>
              <span className="h-1 overflow-hidden rounded-full bg-muted">
                {r.ms != null && (
                  <motion.span
                    className="block h-full rounded-full bg-primary"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(6, (r.ms / best) * 100)}%` }}
                    transition={{ type: "spring", stiffness: 200, damping: 26 }}
                  />
                )}
              </span>
              <span className="text-right tabular-nums">
                {r.ms != null ? (
                  `${r.ms < 10 ? r.ms.toFixed(1) : Math.round(r.ms)} ms`
                ) : r.error ? (
                  <span className="text-destructive" title={r.error}>failed</span>
                ) : r.status === "running" ? (
                  <Loader2 className="ml-auto size-3 animate-spin text-primary" />
                ) : (
                  <span className="text-muted-foreground">–</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------- per camera
function PerCamera({
  cameras,
  options,
  global,
  disabled,
  onChange,
}: {
  cameras: ComputeCamera[]
  options: DeviceOption[]
  global: DeviceId
  disabled: boolean
  onChange: (cam: string, dev: DeviceId | null) => void
}) {
  const overrides = cameras.filter((c) => c.override !== null).length
  const [open, setOpen] = useState(overrides > 0)
  return (
    <div className="mt-4">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-xs text-muted-foreground transition-colors hover:text-foreground"
      >
        <span>
          Per camera
          {overrides > 0 && <span className="ml-1.5 rounded bg-primary/15 px-1.5 text-[10px] text-primary">{overrides} custom</span>}
        </span>
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ul
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            {cameras.map((c) => (
              <li key={c.id} className="mt-3">
                <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
                  <span className="truncate">
                    {c.label}
                    {c.source_kind !== "rtsp" && <span className="ml-1 text-[10px] text-muted-foreground">VIDEO</span>}
                  </span>
                  <span className="shrink-0 text-[11px] text-muted-foreground">
                    {modelLabel(c.model)}
                    {c.override === null && ` · ${global === "auto" ? "Auto" : DEVICE_SHORT[global]}`}
                  </span>
                </div>
                <DevicePicker
                  id={`cmp-cam-${c.id}`}
                  options={options}
                  value={c.override}
                  inherit="Default"
                  small
                  disabled={disabled}
                  onChange={(v) => onChange(c.id, v)}
                />
                {c.override && <Notices items={c.warnings.filter((n) => n.level !== "info")} />}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------- bits
function Title({ text, right }: { text: string; right?: React.ReactNode }) {
  return (
    <div className="mb-1.5 flex items-center justify-between gap-2">
      <h3 className="text-sm font-medium">{text}</h3>
      {right}
    </div>
  )
}

function InstallHint({ text, cmd }: { text: string; cmd: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="mt-2 rounded-md border border-border bg-background/60 p-2 text-xs text-muted-foreground">
      <p>{text}</p>
      <div className="mt-1.5 flex items-center gap-2">
        <code className="min-w-0 flex-1 truncate font-mono text-[11px] text-foreground">{cmd}</code>
        <button
          type="button"
          aria-label="Copy command"
          onClick={() => {
            void navigator.clipboard?.writeText(cmd).then(() => {
              setCopied(true)
              window.setTimeout(() => setCopied(false), 1500)
            })
          }}
          className="rounded p-1 transition-colors hover:bg-accent hover:text-foreground"
        >
          {copied ? <Check className="size-3.5 text-primary" /> : <Copy className="size-3.5" />}
        </button>
      </div>
    </div>
  )
}

function ago(ts: number): string {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return "just now"
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return `${Math.round(s / 86400)} d ago`
}

// ---------------------------------------------------------------------------- header chip
/** Small chip in the header, only while something is not on Auto. Click opens the Compute tab. */
export function ComputeChip({ cameraId, onOpen }: { cameraId: string | null; onOpen: () => void }) {
  const s = useCompute()
  if (!s?.non_default) return null
  const cam = s.cameras.find((c) => c.id === cameraId)
  const parts: { text: string; bad: boolean }[] = []
  if (cam && cam.effective.requested !== "auto") {
    parts.push({ text: DEVICE_SHORT[cam.effective.requested], bad: cam.effective.device !== cam.effective.requested })
  }
  for (const w of s.workloads) {
    if (w.id !== "yolo" && w.assignment !== "auto") {
      parts.push({ text: `${w.id === "faces" ? "Faces" : "Plates"} ${DEVICE_SHORT[w.assignment]}`, bad: w.effective.device !== w.assignment })
    }
  }
  if (!parts.length) parts.push({ text: "Custom", bad: false })
  const bad = parts.some((p) => p.bad)
  const title = [
    ...s.workloads.filter((w) => w.assignment !== "auto").map((w) => `${w.label}: ${DEVICE_SHORT[w.assignment]}`),
    ...s.cameras.filter((c) => c.override && c.override !== "auto").map((c) => `${c.label}: ${DEVICE_SHORT[c.override as DeviceId]}`),
  ].join(" · ")
  return (
    <motion.button
      type="button"
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      onClick={onOpen}
      title={`Not on Auto - ${title || "custom compute devices"}${bad ? " (a device failed, running on Auto)" : ""}`}
      aria-label="Compute devices are not on Auto"
      className={cn(
        "hidden h-7 shrink-0 items-center gap-1.5 rounded-full border bg-background/70 px-2.5 text-[11px] backdrop-blur-md transition-colors sm:flex",
        bad ? "border-destructive/60 text-destructive" : "border-primary/60 text-primary hover:bg-primary/10",
      )}
    >
      <Microchip className="size-3.5" />
      {parts.map((p) => p.text).join(" · ")}
    </motion.button>
  )
}
