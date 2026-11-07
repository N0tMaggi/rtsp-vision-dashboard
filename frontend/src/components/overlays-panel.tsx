import { useEffect, useState } from "react"
import { motion } from "motion/react"
import { RotateCcw } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { CrosshairPanel } from "@/components/crosshair-panel"
import { fmtDuration } from "@/components/stage-visuals"
import { api } from "@/lib/api"
import type { EspSettings } from "@/lib/types"
import { setVisualPref, useVisualPrefs, type VisualPrefs } from "@/lib/visual-prefs"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const HEATMAP_WINDOWS = [
  { value: "5m", label: "5 Min" },
  { value: "1h", label: "1 Hour" },
  { value: "24h", label: "24 Hours" },
] as const

const NIGHT_MODES = [
  { value: "off", label: "Off" },
  { value: "on", label: "On" },
  { value: "auto", label: "Auto" },
] as const

const HUD_TOGGLES: { key: keyof VisualPrefs; label: string }[] = [
  { key: "radar", label: "Radar" },
  { key: "arrows", label: "Off-Screen Arrows" },
  { key: "infoCard", label: "Target Info Card" },
]

const ease = [0.22, 1, 0.36, 1] as const
const PREVIEW_REFRESH_MS = 5000

/** "Overlays" category: heatmap, loitering, night vision (server) and HUD toggles (this browser). */
export function OverlaysPanel({ cameraId, settings, onUpdate }: { cameraId?: string; settings: EspSettings; onUpdate: Update }) {
  const prefs = useVisualPrefs()
  const heatOn = Boolean(settings.heatmap)
  const loiterOn = Boolean(settings.loiter)
  const night = String(settings.night_vision ?? "off")

  return (
    <div className="space-y-5">
      <Section title="Heatmap">
        <Toggle id="esp-heatmap" label="Heatmap" on={heatOn} onChange={(v) => onUpdate("heatmap", v)} />
        <Segments
          label="Heatmap Window"
          id="heatmap-window"
          options={HEATMAP_WINDOWS}
          value={String(settings.heatmap_window ?? "1h")}
          onChange={(v) => onUpdate("heatmap_window", v)}
        />
        <Range
          label="Heatmap Opacity"
          min={0.1}
          max={1}
          step={0.05}
          value={Number(settings.heatmap_opacity ?? 0.55)}
          format={(v) => `${Math.round(v * 100)}%`}
          onChange={(v) => onUpdate("heatmap_opacity", v, false)}
        />
        {cameraId && <HeatmapPreview cameraId={cameraId} window={String(settings.heatmap_window ?? "1h")} />}
      </Section>

      <Section title="Loitering">
        <Toggle id="esp-loiter" label="Loiter Alert" on={loiterOn} onChange={(v) => onUpdate("loiter", v)} />
        <Range
          label="Loiter Time"
          min={10}
          max={900}
          step={10}
          value={Number(settings.loiter_seconds ?? 120)}
          format={fmtDuration}
          onChange={(v) => onUpdate("loiter_seconds", v, false)}
        />
        <p className="text-xs leading-relaxed text-muted-foreground">
          Boxes of targets that stay in place turn from yellow to red over this time. The label shows how long they have been there.
        </p>
      </Section>

      <Section title="Night Vision">
        <Segments label="Night Vision" id="night-vision" options={NIGHT_MODES} value={night} onChange={(v) => onUpdate("night_vision", v)} />
        <Toggle
          id="esp-night-scanlines"
          label="Scanlines"
          on={Boolean(settings.night_vision_scanlines)}
          onChange={(v) => onUpdate("night_vision_scanlines", v)}
        />
        <p className="text-xs leading-relaxed text-muted-foreground">
          Auto switches on when the camera delivers an infrared picture and off again in daylight.
        </p>
      </Section>

      <Section title="Crosshair">
        <CrosshairPanel />
      </Section>

      <Section title="HUD">
        {HUD_TOGGLES.map((t) => (
          <Toggle key={t.key} id={`hud-${t.key}`} label={t.label} on={prefs[t.key]} onChange={(v) => setVisualPref(t.key, v)} />
        ))}
        <p className="text-xs leading-relaxed text-muted-foreground">Saved in this browser only. Arrows appear while zoomed in.</p>
      </Section>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2 border-t border-border pt-4 first:border-t-0 first:pt-0">
      <h3 className="text-xs uppercase tracking-widest text-muted-foreground">{title}</h3>
      {children}
    </section>
  )
}

function Toggle({ id, label, on, onChange }: { id: string; label: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <label
      htmlFor={id}
      className="-mx-2 flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors duration-150 hover:bg-accent/70"
    >
      <span className={cn("transition-colors duration-200", on ? "text-foreground" : "text-muted-foreground")}>{label}</span>
      <Switch id={id} checked={on} onCheckedChange={onChange} />
    </label>
  )
}

function Segments({
  label,
  id,
  options,
  value,
  onChange,
}: {
  label: string
  id: string
  options: ReadonlyArray<{ value: string; label: string }>
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div role="radiogroup" aria-label={label} className="grid grid-cols-3 gap-1 rounded-lg bg-background/60 p-1">
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
              "relative rounded-md py-1.5 text-xs transition-colors duration-200",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId={`${id}-indicator`}
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

function Range({
  label,
  min,
  max,
  step,
  value,
  format,
  onChange,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  format: (v: number) => string
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-2.5 py-1.5">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums">{format(value)}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} aria-label={label} onValueChange={([v]) => onChange(v)} />
    </div>
  )
}

function HeatmapPreview({ cameraId, window: win }: { cameraId: string; window: string }) {
  const [nonce, setNonce] = useState(() => Date.now())
  const [empty, setEmpty] = useState(false)
  useEffect(() => {
    const t = setInterval(() => !document.hidden && setNonce(Date.now()), PREVIEW_REFRESH_MS)
    return () => clearInterval(t)
  }, [])
  const url = api.heatmapUrl(cameraId, win, 360, nonce)
  return (
    <div className="relative mt-1 overflow-hidden rounded-lg border border-border bg-[#1b1c1f]">
      <div className="aspect-video">
        <motion.img
          key={url}
          src={url}
          alt="Heatmap preview"
          initial={{ opacity: 0.4 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, ease }}
          onLoad={() => setEmpty(false)}
          onError={() => setEmpty(true)}
          className={cn("size-full object-contain", empty && "invisible")}
        />
      </div>
      {empty && (
        <p className="absolute inset-0 grid place-items-center px-4 text-center text-xs text-muted-foreground">
          No activity recorded yet
        </p>
      )}
      <button
        type="button"
        onClick={() => void api.resetHeatmap(cameraId).then(() => setNonce(Date.now()))}
        className="absolute right-1.5 top-1.5 flex items-center gap-1 rounded-full bg-background/80 px-2 py-1 text-[11px] text-muted-foreground backdrop-blur-md transition-colors hover:text-foreground"
      >
        <RotateCcw className="size-3" /> Reset
      </button>
    </div>
  )
}
