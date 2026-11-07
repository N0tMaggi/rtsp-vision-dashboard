import { useEffect, useMemo, useState } from "react"
import { motion } from "motion/react"
import { Check, ChevronDown, Cpu, Download, Sparkles, Zap } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import {
  FAMILIES,
  SIZE_META,
  SIZE_ORDER,
  TASKS,
  TASK_META,
  findModel,
  modelLabel,
  parseModel,
  recommendedModel,
  sizesFor,
  tasksFor,
  type ModelFamily,
  type ModelTask,
} from "@/lib/models"
import { recognitionApi, type ModelInfo } from "@/lib/recognition"
import { cn } from "@/lib/utils"

const spring = { type: "spring", stiffness: 450, damping: 34 } as const

interface Props {
  models: string[]
  value: string | null
  onChange: (model: string) => void
  disabled?: boolean
  /** CUDA GPU available: drives the recommendation (M on GPU, N on CPU). */
  cuda: boolean
  /** Compute device when not on Auto (Compute tab), shown as a chip. */
  device?: { label: string; fallback: boolean; state: string } | null
}

/**
 * Model picker: three small decisions (family, task, size) instead of one long
 * list. Nothing changes until "Use Model" is pressed, so browsing never
 * triggers a weight download or a model reload.
 */
export function ModelPicker({ models, value, onChange, disabled, cuda, device }: Props) {
  const [open, setOpen] = useState(false)
  const current = value ? parseModel(value) : null
  const [family, setFamily] = useState<ModelFamily>(current?.family ?? "YOLO26")
  const [task, setTask] = useState<ModelTask>(current?.task ?? "detect")
  const [size, setSize] = useState<string>(current?.size ?? (cuda ? "M" : "N"))
  const [info, setInfo] = useState<Record<string, ModelInfo>>({})

  // Re-sync the draft with the active model whenever the picker opens.
  useEffect(() => {
    if (!open) return
    if (current) {
      setFamily(current.family)
      setTask(current.task)
      setSize(current.size)
    }
    recognitionApi
      .models()
      .then((r) => setInfo(Object.fromEntries(r.models.map((m) => [m.name, m]))))
      .catch(() => setInfo({}))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const families = useMemo(() => FAMILIES.filter((f) => models.some((n) => parseModel(n).family === f.id)), [models])
  const tasks = tasksFor(models, family)
  const effTask = tasks.includes(task) ? task : tasks[0] ?? "detect"
  const sizes = sizesFor(models, family, effTask)
  const effSize = sizes.includes(size) ? size : (sizes.find((s) => s === (cuda ? "M" : "N")) ?? sizes[0])
  const draft = findModel(models, family, effTask, effSize ?? "")
  const rec = recommendedModel(models, cuda, effTask)
  const recSize = rec ? parseModel(rec).size : null
  const d = draft ? info[draft] : undefined
  const changed = draft !== null && draft !== value
  const curInfo = value ? info[value] : undefined

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          aria-label="Model"
          className={cn(
            "flex h-8 min-w-0 max-w-[170px] items-center gap-1.5 rounded-md border border-input bg-background/60 px-2.5 text-xs transition-colors",
            "hover:border-primary/60 disabled:cursor-not-allowed disabled:opacity-50",
            open && "border-primary",
          )}
        >
          <Cpu className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="truncate">{modelLabel(value)}</span>
          {curInfo?.engine_active && !device && <Zap className="size-3 shrink-0 text-primary" aria-label="GPU engine active" />}
          {device && <DeviceChip device={device} />}
          <ChevronDown className={cn("ml-auto size-3.5 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} />
        </button>
      </PopoverTrigger>

      <PopoverContent align="end" sideOffset={8} collisionPadding={12} className="max-h-[var(--radix-popover-content-available-height)] w-[340px] overflow-y-auto border-border bg-card/95 p-0 backdrop-blur-xl">
        <div className="border-b border-border px-4 py-3">
          <div className="text-sm font-medium">Model</div>
          <div className="mt-0.5 text-xs text-muted-foreground">
            Active: <span className="text-foreground">{modelLabel(value)}</span>
            {rec && value !== rec && (
              <button
                type="button"
                onClick={() => {
                  const r = parseModel(rec)
                  setFamily(r.family)
                  setTask(r.task)
                  setSize(r.size)
                }}
                className="ml-2 inline-flex items-center gap-1 text-primary hover:underline"
              >
                <Sparkles className="size-3" /> Suggest best
              </button>
            )}
          </div>
        </div>

        {/* Family */}
        <Section label="Family">
          <div className="space-y-1" role="radiogroup" aria-label="Model family">
            {families.map((f) => {
              const active = f.id === family
              return (
                <button
                  key={f.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() => setFamily(f.id)}
                  className={cn(
                    "relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left transition-colors duration-200",
                    active ? "text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                  )}
                >
                  {active && (
                    <motion.span layoutId="mp-family" className="absolute inset-0 rounded-lg bg-primary/10 ring-1 ring-primary/50" transition={spring} />
                  )}
                  <span className="relative min-w-0 flex-1">
                    <span className="flex items-center gap-2 text-sm">
                      {f.label}
                      {f.recommended && (
                        <span className="rounded bg-primary px-1.5 py-px text-[10px] font-medium text-primary-foreground">Recommended</span>
                      )}
                    </span>
                    <span className="block truncate text-xs text-muted-foreground">{f.blurb}</span>
                  </span>
                  {active && <Check className="relative size-4 text-primary" />}
                </button>
              )
            })}
          </div>
        </Section>

        {/* Task */}
        <Section label="Task">
          <Segmented
            id="mp-task"
            options={TASKS.map((t) => ({ value: t, label: TASK_META[t].label, disabled: !tasks.includes(t) }))}
            value={effTask}
            onChange={(v) => setTask(v as ModelTask)}
          />
          <p className="mt-2 text-xs text-muted-foreground">{TASK_META[effTask].hint}</p>
        </Section>

        {/* Size */}
        <Section label="Size">
          <Segmented
            id="mp-size"
            options={SIZE_ORDER.map((s) => ({ value: s, label: s, disabled: !sizes.includes(s), dot: s === recSize && family === "YOLO26" }))}
            value={effSize ?? ""}
            onChange={setSize}
          />
          {effSize && (
            <div className="mt-2.5 space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-foreground">
                  {SIZE_META[effSize]?.label} <span className="text-muted-foreground">· {SIZE_META[effSize]?.hint}</span>
                </span>
                {effSize === recSize && family === "YOLO26" && (
                  <span className="text-primary">Best for this {cuda ? "GPU" : "CPU"}</span>
                )}
              </div>
              <Meter label="Speed" value={1 - SIZE_ORDER.indexOf(effSize as (typeof SIZE_ORDER)[number]) / 4} />
              <Meter label="Accuracy" value={0.2 + (SIZE_ORDER.indexOf(effSize as (typeof SIZE_ORDER)[number]) / 4) * 0.8} />
            </div>
          )}
        </Section>

        {/* Footer */}
        <div className="flex items-center gap-2 border-t border-border px-4 py-3">
          <div className="min-w-0 flex-1 text-xs">
            <div className="truncate text-foreground">{draft ?? "Not available"}</div>
            <div className="flex flex-wrap items-center gap-x-2 text-muted-foreground">
              {d?.engine ? (
                <span className="inline-flex items-center gap-1 text-primary">
                  <Zap className="size-3" /> GPU engine
                </span>
              ) : null}
              {d && !d.weights_downloaded && (
                <span className="inline-flex items-center gap-1">
                  <Download className="size-3" /> Downloads on first use
                </span>
              )}
              {draft === value && <span>In use</span>}
              {device && (
                <span className={device.fallback ? "text-destructive" : "text-primary"}>
                  {device.fallback ? `${device.label} failed, on Auto` : `Runs on ${device.label}`}
                </span>
              )}
            </div>
          </div>
          <button
            type="button"
            disabled={!changed}
            onClick={() => {
              if (draft) onChange(draft)
              setOpen(false)
            }}
            className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-opacity disabled:opacity-40"
          >
            Use Model
          </button>
        </div>
      </PopoverContent>
    </Popover>
  )
}

function DeviceChip({ device }: { device: { label: string; fallback: boolean; state: string } }) {
  return (
    <span
      title={device.fallback ? `${device.label} failed - running on Auto (see Compute tab)` : `Detection runs on ${device.label} (Compute tab)`}
      className={cn(
        "shrink-0 rounded border px-1 text-[10px] leading-4",
        device.fallback ? "border-destructive/60 text-destructive" : "border-primary/60 text-primary",
        device.state === "preparing" && "animate-pulse",
      )}
    >
      {device.label}
    </span>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border-b border-border px-4 py-3">
      <div className="mb-2 text-xs text-muted-foreground">{label}</div>
      {children}
    </div>
  )
}

function Segmented({
  id,
  options,
  value,
  onChange,
}: {
  id: string
  options: { value: string; label: string; disabled?: boolean; dot?: boolean }[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div role="radiogroup" className="flex gap-1 rounded-lg bg-background/60 p-1">
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            className={cn(
              "relative flex-1 rounded-md py-1.5 text-xs transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-30",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && <motion.span layoutId={`${id}-ind`} className="absolute inset-0 rounded-md bg-primary" transition={spring} />}
            <span className="relative">{o.label}</span>
            {o.dot && !active && <span className="absolute right-1.5 top-1.5 size-1 rounded-full bg-primary" aria-label="recommended" />}
          </button>
        )
      })}
    </div>
  )
}

function Meter({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
      <span className="w-14">{label}</span>
      <span className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
        <motion.span
          className="block h-full rounded-full bg-primary"
          animate={{ width: `${Math.round(Math.max(0.08, Math.min(1, value)) * 100)}%` }}
          transition={{ type: "spring", stiffness: 200, damping: 26 }}
        />
      </span>
    </div>
  )
}
