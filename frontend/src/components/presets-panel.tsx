import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Check, Clock, Plus, RefreshCw, Trash2 } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import type { usePresets } from "@/hooks/use-presets"
import { DAY_LABELS, opsApi, type Preset, type Schedule } from "@/lib/ops-api"
import type { Camera } from "@/lib/types"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const
const spring = { type: "spring", stiffness: 420, damping: 30 } as const
const ALL = "__all__"

interface Props {
  cameraId: string | null
  cameras: Camera[]
  presets: ReturnType<typeof usePresets>
  /** Called after a preset changed the live settings (refresh status). */
  onApplied: () => void
}

export function PresetsPanel({ cameraId, cameras, presets, onApplied }: Props) {
  const { data, error, mutate } = presets
  const list = data?.presets ?? []
  const activeId = cameraId ? data?.last_applied[cameraId]?.preset_id : undefined
  const cameraLabel = (id: string | null) => (id ? (cameras.find((c) => c.id === id)?.label ?? id) : "All cameras")

  return (
    <>
      <p className="-mt-1 mb-3 text-xs leading-relaxed text-muted-foreground">
        A preset stores the ESP settings. Apply it by hand or let a schedule switch it, for example Night at 22:00.
      </p>

      {!data && !error && <p className="text-xs text-muted-foreground">Loading presets…</p>}
      {data && list.length === 0 && <p className="text-xs text-muted-foreground">No presets yet. Save the current look below.</p>}

      <ul className="-mx-2 space-y-0.5">
        <AnimatePresence initial={false}>
          {list.map((p) => (
            <motion.li
              key={p.id}
              layout
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.22, ease }}
              className="overflow-hidden"
            >
              <PresetRow
                preset={p}
                active={p.id === activeId}
                scopeLabel={p.camera_id ? "This camera" : "All"}
                onApply={async () => {
                  const ok = await mutate(() => opsApi.applyPreset(p.id, cameraId), "Preset could not be applied.")
                  if (ok) onApplied()
                  return ok
                }}
                onRename={(name) => void mutate(() => opsApi.updatePreset(p.id, { name }), "Preset could not be renamed.")}
                onCapture={() =>
                  cameraId && void mutate(() => opsApi.updatePreset(p.id, { capture_from: cameraId }), "Preset could not be updated.")
                }
                onDelete={() => void mutate(() => opsApi.deletePreset(p.id), "Preset could not be deleted.")}
              />
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>

      <SaveCurrent
        disabled={!cameraId}
        onSave={(name, scope) =>
          cameraId
            ? mutate(
                () => opsApi.createPreset({ name, camera_id: scope === "camera" ? cameraId : null, from_camera: cameraId }),
                "Preset could not be saved.",
              )
            : Promise.resolve(false)
        }
      />

      <Schedules
        schedules={data?.schedules ?? []}
        presets={list}
        cameras={cameras}
        cameraLabel={cameraLabel}
        mutate={mutate}
      />

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            role="alert"
            className="mt-3 text-xs text-destructive"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </>
  )
}

function PresetRow({
  preset,
  active,
  scopeLabel,
  onApply,
  onRename,
  onCapture,
  onDelete,
}: {
  preset: Preset
  active: boolean
  scopeLabel: string
  onApply: () => Promise<boolean>
  onRename: (name: string) => void
  onCapture: () => void
  onDelete: () => void
}) {
  const [name, setName] = useState(preset.name)
  const [prev, setPrev] = useState(preset.name)
  const [phase, setPhase] = useState<"idle" | "applying" | "done">("idle")
  if (preset.name !== prev) {
    setPrev(preset.name)
    setName(preset.name)
  }
  const commit = () => {
    const v = name.trim().slice(0, 32)
    if (v && v !== preset.name) onRename(v)
    else setName(preset.name)
  }
  const apply = async () => {
    if (phase !== "idle") return
    setPhase("applying")
    const ok = await onApply()
    setPhase(ok ? "done" : "idle")
    if (ok) window.setTimeout(() => setPhase("idle"), 1400)
  }

  return (
    <div className="group relative flex items-center gap-2 rounded-md px-2 py-1.5">
      {active && (
        <motion.span
          layoutId="preset-active"
          className="absolute inset-0 rounded-md bg-primary/10 ring-1 ring-primary/35"
          transition={spring}
        />
      )}
      <span className="relative flex min-w-0 flex-1 flex-col">
        <input
          value={name}
          maxLength={32}
          aria-label="Preset name"
          onChange={(e) => setName(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          className="min-w-0 rounded bg-transparent px-1 py-0.5 text-sm outline-none focus:bg-background/60 focus:ring-1 focus:ring-ring"
        />
        <span className="px-1 text-[11px] text-muted-foreground">
          {scopeLabel}
          {preset.yolo_enabled === true && " · detection on"}
          {preset.yolo_enabled === false && " · detection off"}
        </span>
      </span>
      <span className="relative flex items-center opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        <IconBtn label={`Overwrite ${preset.name} with the current settings`} onClick={onCapture}>
          <RefreshCw className="size-3.5" />
        </IconBtn>
        <IconBtn label={`Delete ${preset.name}`} onClick={onDelete} danger>
          <Trash2 className="size-3.5" />
        </IconBtn>
      </span>
      <motion.button
        type="button"
        onClick={() => void apply()}
        aria-label={`Apply ${preset.name}`}
        layout
        transition={spring}
        className={cn(
          "relative h-7 overflow-hidden rounded-full border px-3 text-xs transition-colors duration-300",
          phase === "done"
            ? "border-primary bg-primary text-primary-foreground"
            : "border-primary/50 text-primary hover:bg-primary/12",
        )}
      >
        {phase === "applying" && (
          <motion.span
            className="absolute inset-y-0 left-0 bg-primary/30"
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ duration: 0.6, ease }}
          />
        )}
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.span
            key={phase === "done" ? "done" : "apply"}
            initial={{ opacity: 0, y: 8, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8, scale: 0.8 }}
            transition={spring}
            className="relative flex items-center gap-1"
          >
            {phase === "done" ? (
              <>
                <Check className="size-3.5" /> Applied
              </>
            ) : (
              "Apply"
            )}
          </motion.span>
        </AnimatePresence>
      </motion.button>
    </div>
  )
}

function SaveCurrent({
  disabled,
  onSave,
}: {
  disabled: boolean
  onSave: (name: string, scope: "camera" | "all") => Promise<boolean>
}) {
  const [name, setName] = useState("")
  const [scope, setScope] = useState<"camera" | "all">("all")
  const [saving, setSaving] = useState(false)
  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    const v = name.trim()
    if (!v || saving) return
    setSaving(true)
    const ok = await onSave(v.slice(0, 32), scope)
    setSaving(false)
    if (ok) setName("")
  }
  return (
    <form onSubmit={(e) => void submit(e)} className="mt-4 space-y-2 border-t border-border pt-4">
      <p className="text-sm">Save current settings</p>
      <div className="flex gap-2">
        <input
          value={name}
          maxLength={32}
          placeholder="Preset name"
          aria-label="New preset name"
          onChange={(e) => setName(e.target.value)}
          className="h-8 min-w-0 flex-1 rounded-md border border-input bg-input/30 px-2.5 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50"
        />
        <button
          type="submit"
          disabled={disabled || saving || !name.trim()}
          className="flex h-8 items-center gap-1.5 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground transition-opacity disabled:opacity-40"
        >
          <Plus className="size-3.5" /> Save
        </button>
      </div>
      <div role="radiogroup" aria-label="Preset scope" className="grid grid-cols-2 gap-1 rounded-lg bg-background/60 p-1">
        {(
          [
            ["all", "All Cameras"],
            ["camera", "This Camera"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={scope === value}
            onClick={() => setScope(value)}
            className={cn(
              "relative rounded-md py-1 text-xs transition-colors duration-200",
              scope === value ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {scope === value && <motion.span layoutId="preset-scope" className="absolute inset-0 rounded-md bg-primary" transition={spring} />}
            <span className="relative">{label}</span>
          </button>
        ))}
      </div>
    </form>
  )
}

function Schedules({
  schedules,
  presets,
  cameras,
  cameraLabel,
  mutate,
}: {
  schedules: Schedule[]
  presets: Preset[]
  cameras: Camera[]
  cameraLabel: (id: string | null) => string
  mutate: (fn: () => Promise<unknown>, fail: string) => Promise<boolean>
}) {
  const add = () => {
    const preset = presets[0]
    if (!preset) return
    void mutate(
      () => opsApi.createSchedule({ preset_id: preset.id, time: "22:00", days: [0, 1, 2, 3, 4, 5, 6], camera_id: preset.camera_id, enabled: true }),
      "Schedule could not be created.",
    )
  }
  return (
    <section className="mt-5 border-t border-border pt-4" aria-label="Schedules">
      <div className="mb-2 flex items-center justify-between">
        <p className="flex items-center gap-2 text-sm">
          <Clock className="size-3.5 text-muted-foreground" /> Schedules
        </p>
        <button
          type="button"
          onClick={add}
          disabled={presets.length === 0}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-primary transition-colors hover:bg-primary/10 disabled:opacity-40"
        >
          <Plus className="size-3.5" /> Add
        </button>
      </div>
      {schedules.length === 0 && (
        <p className="text-xs leading-relaxed text-muted-foreground">No schedules. Add one to switch presets automatically (local time).</p>
      )}
      <ul className="space-y-2">
        <AnimatePresence initial={false}>
          {schedules.map((s) => (
            <motion.li
              key={s.id}
              layout
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.22, ease }}
              className="overflow-hidden"
            >
              <ScheduleRow
                schedule={s}
                presets={presets}
                cameras={cameras}
                cameraLabel={cameraLabel}
                onChange={(patch) => void mutate(() => opsApi.updateSchedule(s.id, patch), "Schedule could not be saved.")}
                onDelete={() => void mutate(() => opsApi.deleteSchedule(s.id), "Schedule could not be deleted.")}
              />
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </section>
  )
}

function formatNext(iso: string | null | undefined) {
  if (!iso) return "Off"
  const d = new Date(iso)
  return `Next: ${DAY_LABELS[(d.getDay() + 6) % 7]} ${iso.slice(11, 16)}`
}

function ScheduleRow({
  schedule,
  presets,
  cameras,
  cameraLabel,
  onChange,
  onDelete,
}: {
  schedule: Schedule
  presets: Preset[]
  cameras: Camera[]
  cameraLabel: (id: string | null) => string
  onChange: (patch: Partial<Omit<Schedule, "id" | "next_run">>) => void
  onDelete: () => void
}) {
  const [time, setTime] = useState(schedule.time)
  const [prevTime, setPrevTime] = useState(schedule.time)
  if (schedule.time !== prevTime) {
    setPrevTime(schedule.time)
    setTime(schedule.time)
  }
  const preset = presets.find((p) => p.id === schedule.preset_id)
  const cameraChoices = preset?.camera_id ? cameras.filter((c) => c.id === preset.camera_id) : cameras
  const toggleDay = (d: number) => {
    const days = schedule.days.includes(d) ? schedule.days.filter((x) => x !== d) : [...schedule.days, d].sort()
    if (days.length) onChange({ days })
  }

  return (
    <div className={cn("group rounded-lg border border-border/80 bg-background/40 p-2.5 transition-opacity", !schedule.enabled && "opacity-60")}>
      <div className="flex items-center gap-2">
        <input
          type="time"
          value={time}
          aria-label="Time"
          onChange={(e) => setTime(e.target.value)}
          onBlur={() => time && time !== schedule.time && onChange({ time })}
          className="h-8 w-[92px] rounded-md border border-input bg-input/30 px-2 text-sm tabular-nums outline-none [color-scheme:dark] focus-visible:border-ring"
        />
        <Select value={schedule.preset_id} onValueChange={(v) => onChange({ preset_id: v })}>
          <SelectTrigger size="sm" className="min-w-0 flex-1 text-xs" aria-label="Preset">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {presets.map((p) => (
              <SelectItem key={p.id} value={p.id} className="text-xs">
                {p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Switch checked={schedule.enabled} onCheckedChange={(v) => onChange({ enabled: v })} aria-label="Schedule enabled" />
      </div>
      <div className="mt-2 flex items-center gap-1" role="group" aria-label="Days">
        {DAY_LABELS.map((label, d) => {
          const on = schedule.days.includes(d)
          return (
            <button
              key={label}
              type="button"
              aria-pressed={on}
              onClick={() => toggleDay(d)}
              className={cn(
                "h-6 flex-1 rounded text-[11px] transition-colors duration-200",
                on ? "bg-primary/90 text-primary-foreground" : "bg-muted/60 text-muted-foreground hover:text-foreground",
              )}
            >
              {label.slice(0, 2)}
            </button>
          )
        })}
      </div>
      <div className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
        <Select
          value={schedule.camera_id ?? ALL}
          onValueChange={(v) => onChange({ camera_id: v === ALL ? null : v })}
        >
          <SelectTrigger size="sm" className="h-6! min-w-0 border-transparent px-1.5 text-[11px]" aria-label="Cameras">
            <SelectValue>{cameraLabel(schedule.camera_id)}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {!preset?.camera_id && (
              <SelectItem value={ALL} className="text-xs">
                All cameras
              </SelectItem>
            )}
            {cameraChoices.map((c) => (
              <SelectItem key={c.id} value={c.id} className="text-xs">
                {c.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <span className="ml-auto tabular-nums">{schedule.enabled ? formatNext(schedule.next_run) : "Paused"}</span>
        <button
          type="button"
          onClick={onDelete}
          aria-label="Delete schedule"
          className="rounded p-1 opacity-0 transition-opacity hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
        >
          <Trash2 className="size-3.5" />
        </button>
      </div>
    </div>
  )
}

function IconBtn({ label, onClick, danger, children }: { label: string; onClick: () => void; danger?: boolean; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={cn(
        "rounded-md p-1.5 text-muted-foreground transition-colors",
        danger ? "hover:bg-destructive/15 hover:text-destructive" : "hover:bg-accent hover:text-foreground",
      )}
    >
      {children}
    </button>
  )
}
