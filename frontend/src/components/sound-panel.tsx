import { useEffect, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Lock, Volume2 } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import type { EspSettings } from "@/lib/types"
import { SOUND_PRIVACY, soundIcon, useSoundConfig, type SoundClassRule, type SoundRulesPatch } from "@/lib/sound"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const ease = [0.22, 1, 0.36, 1] as const
const DEVICE_LABEL: Record<string, string> = { npu: "Intel NPU", igpu: "Intel GPU", cpu: "CPU" }
const pct = (v: number) => `${Math.round(v * 100)}%`
const secs = (v: number) => (v >= 60 ? `${Math.round(v / 60)} min` : `${Math.round(v)} s`)

/** "Sound" category of the ESP menu: the per-camera toggle and the (global) sound alarm rules. */
export function SoundPanel({ settings, onUpdate }: { settings: EspSettings; onUpdate: Update }) {
  const on = Boolean(settings.sound)
  const { config, error, update } = useSoundConfig()
  const rules = config?.rules
  const clf = config?.classifier

  return (
    <div className="space-y-5">
      <section className="space-y-2">
        <label
          htmlFor="esp-sound"
          className="-mx-2 flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors duration-150 hover:bg-accent/70"
        >
          <span className={cn("transition-colors duration-200", on ? "text-foreground" : "text-muted-foreground")}>Sound ESP</span>
          <Switch id="esp-sound" checked={on} onCheckedChange={(v) => onUpdate("sound", v)} />
        </label>
        <p className="flex gap-2 rounded-lg border border-border bg-muted/40 p-2.5 text-xs leading-relaxed text-muted-foreground">
          <Lock className="mt-0.5 size-3.5 shrink-0 text-primary" />
          <span>
            {SOUND_PRIVACY} Listens only while this camera is on screen. Mono audio has no direction, so the radar shows sounds as
            rings around the camera.
          </span>
        </p>
        {config && (
          <p className="text-xs text-muted-foreground">
            {config.model.name} ·{" "}
            {clf?.status === "ready" && clf.device
              ? `${DEVICE_LABEL[clf.device] ?? clf.device}${clf.ms !== null ? ` · ${clf.ms.toFixed(1)} ms` : ""}`
              : config.model.status === "ready"
                ? "loads when sound is on"
                : "downloads when sound is on"}
          </p>
        )}
      </section>

      {error && <p className="text-xs text-destructive">Could not load sound settings ({error}).</p>}

      {rules && (
        <>
          <section className="space-y-2 border-t border-border pt-4">
            <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Labels</h3>
            <CommitRange
              label="Show labels from"
              min={0.05}
              max={0.9}
              step={0.05}
              value={rules.display_threshold}
              format={pct}
              onCommit={(v) => update({ display_threshold: v })}
            />
          </section>

          <section className="space-y-1 border-t border-border pt-4">
            <h3 className="text-xs uppercase tracking-widest text-muted-foreground">Sound alarms</h3>
            <p className="pb-1 text-xs leading-relaxed text-muted-foreground">
              Off by default. An armed sound creates an event (snapshot, clip, notification) on every camera with Sound ESP on.
            </p>
            <RuleRow
              id="loud"
              label="Loud sound"
              icon={Volume2}
              enabled={rules.loud.enabled}
              onToggle={(v) => update({ loud: { enabled: v } })}
            >
              <CommitRange label="Above background" min={6} max={50} step={1} value={rules.loud.db_above_floor}
                format={(v) => `${Math.round(v)} dB`} onCommit={(v) => update({ loud: { db_above_floor: v } })} />
              <CommitRange label="For at least" min={100} max={5000} step={100} value={rules.loud.min_ms}
                format={(v) => `${(v / 1000).toFixed(1)} s`} onCommit={(v) => update({ loud: { min_ms: v } })} />
              <CommitRange label="Cooldown" min={0} max={600} step={5} value={rules.loud.cooldown_s} format={secs}
                onCommit={(v) => update({ loud: { cooldown_s: v } })} />
              <NotifyToggle id="loud" on={rules.loud.notify} onChange={(v) => update({ loud: { notify: v } })} />
            </RuleRow>
            {config.classes.map((c) => {
              const rule = rules.classes[c.id]
              if (!rule) return null
              const set = (patch: Partial<SoundClassRule>) => update({ classes: { [c.id]: patch } } as SoundRulesPatch)
              return (
                <RuleRow key={c.id} id={c.id} label={c.label} icon={soundIcon(c.icon)} enabled={rule.enabled} onToggle={(v) => set({ enabled: v })}>
                  <CommitRange label="Confidence" min={0.05} max={0.95} step={0.05} value={rule.threshold} format={pct}
                    onCommit={(v) => set({ threshold: v })} />
                  <CommitRange label="Cooldown" min={0} max={600} step={5} value={rule.cooldown_s} format={secs}
                    onCommit={(v) => set({ cooldown_s: v })} />
                  <NotifyToggle id={c.id} on={rule.notify} onChange={(v) => set({ notify: v })} />
                </RuleRow>
              )
            })}
          </section>
        </>
      )}
    </div>
  )
}

function RuleRow({
  id,
  label,
  icon: Icon,
  enabled,
  onToggle,
  children,
}: {
  id: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  enabled: boolean
  onToggle: (v: boolean) => void
  children: React.ReactNode
}) {
  return (
    <div className={cn("-mx-2 rounded-lg transition-colors duration-200", enabled && "bg-primary/6 ring-1 ring-primary/25")}>
      <label htmlFor={`sound-rule-${id}`} className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2 py-2 text-sm hover:bg-accent/50">
        <Icon className={cn("size-4 transition-colors", enabled ? "text-primary" : "text-muted-foreground")} />
        <span className={cn("flex-1 transition-colors duration-200", enabled ? "text-foreground" : "text-muted-foreground")}>{label}</span>
        <Switch id={`sound-rule-${id}`} checked={enabled} onCheckedChange={onToggle} />
      </label>
      <AnimatePresence initial={false}>
        {enabled && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.26, ease }}
            className="overflow-hidden"
          >
            <div className="space-y-1 px-2 pb-2.5 pl-8">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function NotifyToggle({ id, on, onChange }: { id: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label htmlFor={`sound-notify-${id}`} className="flex cursor-pointer items-center justify-between py-1 text-xs text-muted-foreground">
      Send notification
      <Switch id={`sound-notify-${id}`} checked={on} onCheckedChange={onChange} />
    </label>
  )
}

/** Slider with local state; the value is sent once on release. */
function CommitRange({
  label,
  min,
  max,
  step,
  value,
  format,
  onCommit,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  format: (v: number) => string
  onCommit: (v: number) => void
}) {
  const [local, setLocal] = useState(value)
  useEffect(() => setLocal(value), [value])
  return (
    <div className="space-y-2 py-1">
      <div className="flex justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums">{format(local)}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[local]}
        aria-label={label}
        onValueChange={([v]) => setLocal(v)}
        onValueCommit={([v]) => onCommit(v)}
      />
    </div>
  )
}
