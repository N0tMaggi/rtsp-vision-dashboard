import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ArrowUpDown, ChevronDown, RotateCcw, Send, Slash, Trash2 } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { useAlertsConfig } from "@/hooks/use-alerts-config"
import { alertsApi, type AlertsConfig, type Tripwire, type WireTrigger, type ZoneRule } from "@/lib/alerts"
import type { SliderDef, ToggleDef } from "@/lib/esp-schema"
import type { EspSettings, Zone } from "@/lib/types"
import { cn } from "@/lib/utils"
import { SliderList, ToggleList } from "./esp-menu"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const ease = [0.22, 1, 0.36, 1] as const

export const ALERT_TOGGLES: ToggleDef[] = [
  { key: "alarms", label: "Alarms" },
  { key: "tripwires", label: "Show Tripwires" },
  { key: "abandoned", label: "Abandoned Objects" },
]

export const ALERT_SLIDERS: SliderDef[] = [
  { key: "abandoned_seconds", label: "Abandoned After", min: 5, max: 600, step: 5, format: (v) => `${Math.round(v)}s` },
]

/** Classes offered as quick filters (COCO names). */
const CLASS_PRESETS = ["person", "car", "truck", "bicycle", "motorcycle", "dog", "cat"]

const TRIGGERS: { value: WireTrigger; label: string }[] = [
  { value: "both", label: "Both" },
  { value: "in", label: "↑ In" },
  { value: "out", label: "↓ Out" },
]

const DEFAULT_RULE: ZoneRule = { enabled: true, classes: [], min_dwell_s: 0, schedule: { mode: "always" }, notify: true }

export interface AlertsPanelProps {
  cameraId: string | null
  settings: EspSettings
  onUpdate: Update
  zones: Zone[]
  onDrawTripwire: () => void
}

/** "Alarms" category of the ESP menu: zone rules, tripwires, abandoned objects, notifications. */
export function AlertsPanel({ cameraId, settings, onUpdate, zones, onDrawTripwire }: AlertsPanelProps) {
  const { config, act } = useAlertsConfig(cameraId)

  return (
    <>
      <ToggleList items={ALERT_TOGGLES} settings={settings} onUpdate={onUpdate} />
      {Boolean(settings.abandoned) && <SliderList items={ALERT_SLIDERS} settings={settings} onUpdate={onUpdate} />}
      {Boolean(settings.abandoned) && (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Bags, suitcases and similar objects that appear and stay put this long get marked and raise an event.
        </p>
      )}

      <Section title="Zone Rules">
        {zones.length === 0 ? (
          <p className="text-xs leading-relaxed text-muted-foreground">Draw a zone first (Zones category or Z), then arm it here.</p>
        ) : (
          <ul className="-mx-2 space-y-0.5">
            {zones.map((z) => (
              <ZoneRuleRow
                key={z.id}
                zone={z}
                rule={config?.zone_rules[z.id]}
                onChange={(patch) => cameraId && act(() => alertsApi.setRule(cameraId, z.id, patch))}
                onRemove={() => cameraId && act(() => alertsApi.deleteRule(cameraId, z.id))}
              />
            ))}
          </ul>
        )}
      </Section>

      <Section title="Tripwires">
        <button
          type="button"
          onClick={onDrawTripwire}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-primary/50 py-2.5 text-sm text-primary transition-colors duration-200 hover:border-primary hover:bg-primary/10"
        >
          <Slash className="size-4" /> Draw Tripwire
          <kbd className="rounded border border-primary/40 px-1.5 text-[11px]">T</kbd>
        </button>
        {config && config.tripwires.length === 0 && (
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            Drag a line across a path. The arrow in the video marks the "in" direction; crossings are counted both ways.
          </p>
        )}
        <ul className="-mx-2 mt-2 space-y-0.5">
          <AnimatePresence initial={false}>
            {config?.tripwires.map((w) => (
              <motion.li
                key={w.id}
                layout
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease }}
                className="overflow-hidden"
              >
                <TripwireRow
                  wire={w}
                  onChange={(patch) => cameraId && act(() => alertsApi.updateTripwire(cameraId, w.id, patch))}
                  onReset={() => cameraId && act(() => alertsApi.resetTripwire(cameraId, w.id))}
                  onDelete={() => cameraId && act(() => alertsApi.deleteTripwire(cameraId, w.id))}
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      </Section>

      <Section title="Notifications">
        <Notifications cameraId={cameraId} config={config} onAbandonedNotify={(v) => cameraId && act(() => alertsApi.setOptions(cameraId, { abandoned_notify: v }))} />
      </Section>
    </>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-5 border-t border-border pt-4">
      <h3 className="mb-2 text-sm text-muted-foreground">{title}</h3>
      {children}
    </section>
  )
}

function Segments<T extends string>({
  label,
  options,
  value,
  onChange,
}: {
  label: string
  options: { value: T; label: string }[]
  value: T
  onChange: (v: T) => void
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex gap-1 rounded-lg bg-background/60 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={cn(
            "flex-1 rounded-md px-2 py-1 text-xs transition-colors duration-200",
            o.value === value ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

function ClassChips({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const toggle = (c: string) => onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c])
  return (
    <div className="flex flex-wrap gap-1" role="group" aria-label="Classes">
      <button
        type="button"
        onClick={() => onChange([])}
        aria-pressed={value.length === 0}
        className={cn(
          "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
          value.length === 0 ? "border-primary bg-primary/15 text-primary" : "border-border text-muted-foreground hover:text-foreground",
        )}
      >
        Any
      </button>
      {CLASS_PRESETS.map((c) => (
        <button
          key={c}
          type="button"
          onClick={() => toggle(c)}
          aria-pressed={value.includes(c)}
          className={cn(
            "rounded-full border px-2 py-0.5 text-[11px] capitalize transition-colors",
            value.includes(c) ? "border-primary bg-primary/15 text-primary" : "border-border text-muted-foreground hover:text-foreground",
          )}
        >
          {c}
        </button>
      ))}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="text-xs text-muted-foreground">{label}</p>
      {children}
    </div>
  )
}

function ZoneRuleRow({
  zone,
  rule,
  onChange,
  onRemove,
}: {
  zone: Zone
  rule: ZoneRule | undefined
  onChange: (patch: Partial<ZoneRule>) => void
  onRemove: () => void
}) {
  const armed = Boolean(rule?.enabled)
  const [open, setOpen] = useState(false)
  const r = rule ?? DEFAULT_RULE
  const [dwell, setDwell] = useState(r.min_dwell_s)
  const [prevDwell, setPrevDwell] = useState(r.min_dwell_s)
  if (r.min_dwell_s !== prevDwell) {
    setPrevDwell(r.min_dwell_s)
    setDwell(r.min_dwell_s)
  }
  const sched = r.schedule.mode === "window" ? r.schedule : { mode: "window" as const, start: "22:00", end: "06:00" }
  const summary = [
    r.classes.length ? r.classes.join(", ") : "any class",
    r.min_dwell_s > 0 ? `${r.min_dwell_s}s` : null,
    r.schedule.mode === "window" ? `${r.schedule.start}–${r.schedule.end}` : null,
  ]
    .filter(Boolean)
    .join(" · ")

  return (
    <li className="rounded-md transition-colors hover:bg-accent/40">
      <div className="flex items-center gap-2 px-2 py-1.5">
        <span className="size-3 shrink-0 rounded-full ring-1 ring-white/15" style={{ background: zone.color }} />
        <button type="button" onClick={() => setOpen((o) => !o)} className="flex min-w-0 flex-1 items-center gap-1.5 text-left" aria-expanded={open}>
          <span className="min-w-0">
            <span className={cn("block truncate text-sm", armed ? "text-foreground" : "text-muted-foreground")}>{zone.label}</span>
            {rule && <span className="block truncate text-[11px] text-muted-foreground">{summary}</span>}
          </span>
          <ChevronDown className={cn("ml-auto size-3.5 shrink-0 text-muted-foreground transition-transform duration-200", open && "rotate-180")} />
        </button>
        <Switch
          checked={armed}
          aria-label={`Alarm for ${zone.label}`}
          onCheckedChange={(v) => {
            onChange({ enabled: v })
            if (v && !rule) setOpen(true)
          }}
        />
      </div>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            <div className="space-y-3 px-2 pb-3 pt-1">
              <Field label="Classes">
                <ClassChips value={r.classes} onChange={(classes) => onChange({ classes })} />
              </Field>
              <Field label={`Min Dwell · ${dwell.toFixed(dwell % 1 ? 1 : 0)}s`}>
                <Slider
                  min={0}
                  max={60}
                  step={0.5}
                  value={[dwell]}
                  aria-label="Min Dwell"
                  onValueChange={([v]) => setDwell(v)}
                  onValueCommit={([v]) => onChange({ min_dwell_s: v })}
                />
              </Field>
              <Field label="Schedule">
                <Segments
                  label="Schedule"
                  options={[
                    { value: "always", label: "Always" },
                    { value: "window", label: "Time Window" },
                  ]}
                  value={r.schedule.mode}
                  onChange={(m) => onChange({ schedule: m === "always" ? { mode: "always" } : sched })}
                />
                {r.schedule.mode === "window" && (
                  <div className="flex items-center gap-2 text-sm">
                    <TimeInput label="From" value={r.schedule.start} onChange={(start) => onChange({ schedule: { ...sched, start } })} />
                    <span className="text-muted-foreground">–</span>
                    <TimeInput label="To" value={r.schedule.end} onChange={(end) => onChange({ schedule: { ...sched, end } })} />
                  </div>
                )}
              </Field>
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Switch checked={r.notify} onCheckedChange={(notify) => onChange({ notify })} aria-label="Notify" />
                  Notify
                </label>
                {rule && (
                  <button type="button" onClick={onRemove} className="text-xs text-muted-foreground transition-colors hover:text-destructive">
                    Remove rule
                  </button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  )
}

function TimeInput({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <input
      type="time"
      aria-label={label}
      value={value}
      onChange={(e) => /^\d{2}:\d{2}$/.test(e.target.value) && onChange(e.target.value)}
      className="flex-1 rounded-md border border-border bg-background/60 px-2 py-1 text-sm tabular-nums outline-none focus:ring-1 focus:ring-ring"
    />
  )
}

function TripwireRow({
  wire,
  onChange,
  onReset,
  onDelete,
}: {
  wire: Tripwire
  onChange: (patch: Partial<Omit<Tripwire, "id">> & { flip?: boolean }) => void
  onReset: () => void
  onDelete: () => void
}) {
  const [open, setOpen] = useState(false)
  const [label, setLabel] = useState(wire.label)
  const [prev, setPrev] = useState(wire.label)
  if (wire.label !== prev) {
    setPrev(wire.label)
    setLabel(wire.label)
  }
  const commit = () => {
    const v = label.trim().slice(0, 32)
    if (v && v !== wire.label) onChange({ label: v })
    else setLabel(wire.label)
  }
  return (
    <div className="group rounded-md transition-colors hover:bg-accent/40">
      <div className="flex items-center gap-2 px-2 py-1.5">
        <span className="relative size-3 shrink-0 overflow-hidden rounded-full ring-1 ring-white/15">
          <span className="absolute inset-0" style={{ background: wire.color }} />
          <input
            type="color"
            value={wire.color}
            aria-label={`Color of ${wire.label}`}
            onChange={(e) => onChange({ color: e.target.value })}
            className="absolute inset-0 cursor-pointer opacity-0"
          />
        </span>
        <input
          value={label}
          maxLength={32}
          aria-label="Tripwire name"
          onChange={(e) => setLabel(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
          className="min-w-0 flex-1 rounded bg-transparent px-1 py-0.5 text-sm outline-none focus:bg-background/60 focus:ring-1 focus:ring-ring"
        />
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground" title="Crossings in / out">
          ↑{wire.count_in} ↓{wire.count_out}
        </span>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          aria-label={`Settings for ${wire.label}`}
          className="rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground"
        >
          <ChevronDown className={cn("size-3.5 transition-transform duration-200", open && "rotate-180")} />
        </button>
        <Switch checked={wire.alarm} aria-label={`Alarm for ${wire.label}`} onCheckedChange={(alarm) => onChange({ alarm })} />
      </div>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            <div className="space-y-3 px-2 pb-3 pt-1">
              <Field label="Alarm Direction">
                <Segments label="Alarm Direction" options={TRIGGERS} value={wire.trigger} onChange={(trigger) => onChange({ trigger })} />
              </Field>
              <Field label="Classes">
                <ClassChips value={wire.classes} onChange={(classes) => onChange({ classes })} />
              </Field>
              <div className="flex flex-wrap items-center gap-1">
                <label className="mr-auto flex items-center gap-2 text-sm text-muted-foreground">
                  <Switch checked={wire.notify} onCheckedChange={(notify) => onChange({ notify })} aria-label="Notify" />
                  Notify
                </label>
                <IconText onClick={() => onChange({ flip: true })} icon={ArrowUpDown} label="Flip" />
                <IconText onClick={onReset} icon={RotateCcw} label="Reset" />
                <IconText onClick={onDelete} icon={Trash2} label="Delete" danger />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function IconText({ onClick, icon: Icon, label, danger }: { onClick: () => void; icon: typeof Trash2; label: string; danger?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors",
        danger ? "hover:bg-destructive/15 hover:text-destructive" : "hover:bg-accent hover:text-foreground",
      )}
    >
      <Icon className="size-3.5" /> {label}
    </button>
  )
}

function Notifications({
  cameraId,
  config,
  onAbandonedNotify,
}: {
  cameraId: string | null
  config: AlertsConfig | null
  onAbandonedNotify: (v: boolean) => void
}) {
  const [result, setResult] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const channels = config?.channels ?? { ntfy: false, telegram: false, discord: false }
  const any = Object.values(channels).some(Boolean)
  const test = async () => {
    if (!cameraId) return
    setSending(true)
    setResult(null)
    try {
      const r = await alertsApi.testNotify(cameraId)
      setResult(
        Object.entries(r.result ?? {})
          .map(([k, v]) => `${k}: ${v === "ok" ? "sent" : v}`)
          .join(" · ") || "Nothing sent",
      )
    } catch (e) {
      setResult((e as Error).message === "no_channels" ? "No channel configured" : "Test failed")
    } finally {
      setSending(false)
    }
  }
  return (
    <>
      <div className="flex flex-wrap gap-1.5">
        {(["ntfy", "telegram", "discord"] as const).map((c) => (
          <span
            key={c}
            className={cn(
              "flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs capitalize",
              channels[c] ? "border-primary/50 text-primary" : "border-border text-muted-foreground",
            )}
          >
            <span className={cn("size-1.5 rounded-full", channels[c] ? "bg-primary" : "bg-muted-foreground/40")} />
            {c}
          </span>
        ))}
      </div>
      {!any && (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Set SECURITYCAM_NTFY_URL, the Telegram bot token and chat id or a Discord webhook in .env to get alarms with a snapshot on your phone.
        </p>
      )}
      <label className="mt-3 flex items-center justify-between gap-3 text-sm text-muted-foreground">
        Notify Abandoned Objects
        <Switch checked={config?.abandoned_notify ?? true} onCheckedChange={onAbandonedNotify} aria-label="Notify Abandoned Objects" />
      </label>
      <button
        type="button"
        onClick={() => void test()}
        disabled={!any || sending}
        className="mt-3 flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
      >
        <Send className="size-3.5" /> {sending ? "Sending…" : "Send Test"}
      </button>
      {result && <p className="mt-2 text-xs text-muted-foreground">{result}</p>}
    </>
  )
}
