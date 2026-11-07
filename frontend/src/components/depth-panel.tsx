import { useEffect, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Crosshair, RotateCcw } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { depthApi, DEPTH_ERRORS } from "@/lib/depth"
import type { DepthStatus, EspSettings } from "@/lib/types"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const PALETTES = [
  { value: "turbo", label: "Turbo" },
  { value: "inferno", label: "Inferno" },
] as const

const ease = [0.22, 1, 0.36, 1] as const
const POLL_MS = 1500

/** Depth section of the Overlays category: Depth Anything on/off, depth view, rate, FOV, calibration, status. */
export function DepthPanel({ cameraId, settings, onUpdate }: { cameraId?: string; settings: EspSettings; onUpdate: Update }) {
  const on = Boolean(settings.depth)
  const status = useDepthStatus(cameraId, on)

  return (
    <div className="space-y-1">
      <Toggle id="esp-depth" label="Depth" on={on} onChange={(v) => onUpdate("depth", v)} />
      <AnimatePresence initial={false}>
        {on && (
          <motion.div
            key="depth-body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.32, ease }}
            className="space-y-1 overflow-hidden"
          >
            <StatusBlock status={status} />
            <Toggle id="esp-depth-view" label="Depth View" on={Boolean(settings.depth_view)} onChange={(v) => onUpdate("depth_view", v)} />
            {Boolean(settings.depth_view) && (
              <>
                <Segments
                  label="Depth Palette"
                  id="depth-palette"
                  options={PALETTES}
                  value={String(settings.depth_view_palette ?? "turbo")}
                  onChange={(v) => onUpdate("depth_view_palette", v)}
                />
                <Range
                  label="View Opacity"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={Number(settings.depth_view_opacity ?? 0.5)}
                  format={(v) => `${Math.round(v * 100)}%`}
                  onChange={(v) => onUpdate("depth_view_opacity", v, false)}
                />
              </>
            )}
            <Range
              label="Rate"
              min={0.5}
              max={10}
              step={0.5}
              value={Number(settings.depth_rate ?? 3)}
              format={(v) => `${v.toFixed(1)} / s`}
              onChange={(v) => onUpdate("depth_rate", v, false)}
            />
            <Range
              label="Field of View"
              min={30}
              max={170}
              step={5}
              value={Number(settings.depth_fov ?? 90)}
              format={(v) => `${Math.round(v)}°`}
              onChange={(v) => onUpdate("depth_fov", v, false)}
            />
            <Calibrate cameraId={cameraId} status={status} />
            <p className="pt-1 text-xs leading-relaxed text-muted-foreground">
              {status?.note ?? "Distances are estimated from a single camera."} Auto calibration learns the scale from standing
              people; enter the real distance of a locked target for better numbers. Field of view applies to fixed cameras, PTZ
              cameras use their lens calibration.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function useDepthStatus(cameraId: string | undefined, on: boolean) {
  const [status, setStatus] = useState<DepthStatus | null>(null)
  useEffect(() => {
    if (!cameraId || !on) {
      setStatus(null)
      return
    }
    let alive = true
    const ctrl = new AbortController()
    const tick = () => {
      if (document.hidden) return
      depthApi
        .status(cameraId, ctrl.signal)
        .then((r) => alive && setStatus(r.depth))
        .catch(() => {})
    }
    tick()
    const t = setInterval(tick, POLL_MS)
    return () => {
      alive = false
      ctrl.abort()
      clearInterval(t)
    }
  }, [cameraId, on])
  return status
}

const STATE_TEXT: Record<DepthStatus["state"], string> = {
  off: "Off",
  idle: "Starts when the camera is viewed",
  downloading: "Downloading model (99 MB)…",
  loading: "Preparing model…",
  running: "Running",
  failed: "Failed",
  stopped: "Paused (camera not viewed)",
}

function StatusBlock({ status }: { status: DepthStatus | null }) {
  const cal = status?.calibration
  const calText = !cal
    ? "–"
    : cal.mode === "user"
      ? "Manual"
      : cal.ready
        ? `Auto · ${Math.round(cal.confidence * 100)}%`
        : cal.samples
          ? `Learning (${cal.samples} samples)`
          : "Waiting for people"
  const running = status?.state === "running"
  const rows: [string, string, boolean?][] = [
    ["State", status ? (status.model.state === "downloading" ? STATE_TEXT.downloading : STATE_TEXT[status.state]) : "…", running],
    ["Device", status?.device_label ? `${status.device_label}${status.source === "auto" ? " · Auto" : ""}` : "–"],
    [
      "Speed",
      status?.latency_ms != null ? `${status.latency_ms.toFixed(0)} ms${status.fps != null ? ` · ${status.fps.toFixed(1)} / s` : ""}` : "–",
    ],
    ["Calibration", calText, cal?.ready],
  ]
  return (
    <div className="my-1 rounded-lg border border-border bg-background/50 px-3 py-2">
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        {rows.map(([k, v, hi]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className={cn("truncate text-right tabular-nums transition-colors duration-300", hi && "text-primary")}>{v}</dd>
          </div>
        ))}
      </dl>
      {status?.error && <p className="mt-1.5 line-clamp-2 text-[11px] text-destructive">{status.error}</p>}
    </div>
  )
}

function Calibrate({ cameraId, status }: { cameraId?: string; status: DepthStatus | null }) {
  const [meters, setMeters] = useState("")
  const [msg, setMsg] = useState<{ text: string; bad: boolean } | null>(null)
  const [busy, setBusy] = useState(false)
  const value = Number(meters.replace(",", "."))
  const valid = Number.isFinite(value) && value >= 0.3 && value <= 200

  const run = async (fn: () => Promise<{ ok: boolean; error?: string }>, done: string) => {
    if (!cameraId) return
    setBusy(true)
    try {
      const r = await fn()
      setMsg(r.ok ? { text: done, bad: false } : { text: DEPTH_ERRORS[r.error ?? ""] ?? "Calibration failed.", bad: true })
    } catch {
      setMsg({ text: "Calibration failed.", bad: true })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-2 py-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">Locked Target Distance</span>
        {status?.calibration.mode === "user" && (
          <button
            type="button"
            disabled={busy}
            onClick={() => void run(() => depthApi.resetCalibration(cameraId!), "Back to automatic calibration.")}
            className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <RotateCcw className="size-3" /> Auto
          </button>
        )}
      </div>
      <form
        className="flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          if (valid) void run(() => depthApi.calibrate(cameraId!, value), `Scale set: locked target = ${value} m.`)
        }}
      >
        <div className="relative flex-1">
          <input
            inputMode="decimal"
            value={meters}
            onChange={(e) => setMeters(e.target.value)}
            placeholder="e.g. 4.5"
            aria-label="Distance of the locked target in meters"
            className="h-8 w-full rounded-md border border-border bg-background/60 px-2.5 pr-7 text-sm tabular-nums outline-none transition-colors focus:border-primary"
          />
          <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">m</span>
        </div>
        <button
          type="submit"
          disabled={!valid || busy}
          className="flex h-8 items-center gap-1.5 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground transition-[opacity,transform] duration-200 active:scale-95 disabled:opacity-40"
        >
          <Crosshair className="size-3.5" /> Set
        </button>
      </form>
      <AnimatePresence>
        {msg && (
          <motion.p
            key={msg.text}
            initial={{ opacity: 0, y: -3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className={cn("text-[11px]", msg.bad ? "text-destructive" : "text-primary")}
          >
            {msg.text}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}

// ------------------------------------------------------------------ small controls (same look as the Overlays panel)
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
    <div role="radiogroup" aria-label={label} className="grid grid-cols-2 gap-1 rounded-lg bg-background/60 p-1">
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
