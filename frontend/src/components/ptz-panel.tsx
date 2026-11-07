import { useEffect, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Anchor, Copy, Crosshair, Navigation, Plus, Ruler, Trash2 } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { AutoTrackSection } from "@/components/autotrack-section"
import type { PtzControls } from "@/hooks/use-ptz"
import type { PoseStatus, PtzSettings, PtzStatus } from "@/lib/ops-api"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const

interface Props {
  ptz: PtzControls
  mode: boolean
  onModeChange: (on: boolean) => void
}

const CALIBRATION: { key: keyof PtzSettings; label: string; min: number; max: number; step: number; format: (v: number) => string }[] = [
  { key: "speed", label: "Speed", min: 0.05, max: 1, step: 0.05, format: (v) => `${Math.round(v * 100)}%` },
  { key: "fov_h", label: "FOV Horizontal", min: 10, max: 180, step: 1, format: (v) => `${Math.round(v)}°` },
  { key: "fov_v", label: "FOV Vertical", min: 5, max: 120, step: 1, format: (v) => `${Math.round(v)}°` },
]

function stateLine(s: PtzStatus): { text: string; tone: "ok" | "warn" | "muted" } {
  switch (s.state) {
    case "ready":
      return { text: `Connected via ONVIF (port ${s.port})`, tone: "ok" }
    case "connecting":
      return { text: "Connecting…", tone: "muted" }
    case "unsupported":
      return { text: "This camera reports no PTZ support.", tone: "warn" }
    case "backoff":
      return { text: s.error || "Connection failed.", tone: "warn" }
    default:
      return { text: s.error || "Not connected", tone: "muted" }
  }
}

export function PtzPanel({ ptz, mode, onModeChange }: Props) {
  const s = ptz.status
  if (!s) return <p className="text-xs text-muted-foreground">Checking PTZ…</p>

  if (!s.available) return <InstallHint status={s} />
  if (!s.configured)
    return (
      <p className="text-xs leading-relaxed text-muted-foreground">
        This camera has no IP or account in .env. PTZ logs in with the same camera account as the stream.
      </p>
    )

  const line = stateLine(s)
  const retry = s.state === "backoff" ? Math.ceil(s.retry_in) : 0

  return (
    <>
      <label
        htmlFor="ptz-mode"
        className="-mx-2 flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors hover:bg-accent/70"
      >
        <span className={mode ? "text-foreground" : "text-muted-foreground"}>PTZ Mode</span>
        <Switch id="ptz-mode" checked={mode} onCheckedChange={onModeChange} />
      </label>

      <div className="mt-1 flex items-start gap-2 text-xs">
        <span
          className={cn(
            "mt-1 size-1.5 shrink-0 rounded-full",
            line.tone === "ok" ? "bg-primary" : line.tone === "warn" ? "bg-destructive" : "bg-muted-foreground/60",
          )}
        />
        <span className={cn("leading-relaxed", line.tone === "warn" ? "text-destructive" : "text-muted-foreground")}>{line.text}</span>
      </div>
      {(s.state === "idle" || s.state === "backoff") && (
        <button
          type="button"
          onClick={() => void ptz.connect()}
          disabled={retry > 0}
          className="mt-2 w-full rounded-lg border border-primary/50 py-2 text-xs text-primary transition-colors hover:bg-primary/10 disabled:border-border disabled:text-muted-foreground"
        >
          {retry > 0 ? `Retry in ${retry} s` : "Connect"}
        </button>
      )}

      {s.state === "ready" && (
        <>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
            With PTZ Mode on, the arrow keys and the on-screen pad move the camera
            {s.caps.zoom ? ", Page Up/Down zoom" : ""}.
            {s.caps.relative ? " Right-click the video to center that point." : ""}
          </p>
          {s.caps.presets && <PtzPresets ptz={ptz} status={s} />}
          <div className="mt-4 space-y-5 border-t border-border pt-4">
            {CALIBRATION.filter((c) => c.key === "speed" || s.caps.relative).map((c) => (
              <CalibrationSlider key={c.key} def={c} value={s.settings[c.key] ?? c.min} onCommit={(v) => void ptz.updateSettings({ [c.key]: v })} />
            ))}
            {s.caps.relative && (
              <p className="text-xs leading-relaxed text-muted-foreground">
                If a right-click overshoots, raise the FOV; if it stops short, lower it. Anchored zones use the same
                FOV: after a change, pan a little and check that a zone stays on its spot.
              </p>
            )}
          </div>
        </>
      )}

      {s.autotrack && <AutoTrackSection status={s} />}

      {s.pose && (s.state === "ready" || s.pose.ptz_known) && <AnchorSection ptz={ptz} status={s} pose={s.pose} />}

      <AnimatePresence>
        {ptz.message && (
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert" className="mt-3 text-xs text-destructive">
            {ptz.message}
          </motion.p>
        )}
      </AnimatePresence>
    </>
  )
}

const SOURCE_LABEL: Record<PoseStatus["source"], string> = {
  both: "ONVIF + Vision",
  onvif: "ONVIF",
  vision: "Vision",
  none: "No Fix",
}

const fmtDeg = (v: number) => `${v >= 0 ? "" : "−"}${Math.abs(v).toFixed(1)}°`

/** World anchoring: zones, tripwires and the heatmap stay put in the room while the camera moves. */
function AnchorSection({ ptz, status, pose }: { ptz: PtzControls; status: PtzStatus; pose: PoseStatus }) {
  const on = pose.world_anchor
  const live = pose.active
  const calib = pose.calibration
  const calibrating = calib?.state === "running"
  const drift = pose.drift_deg
  const driftTone = drift < 0.5 ? "text-muted-foreground" : drift < 2 ? "text-primary" : "text-destructive"
  const onvifNote = pose.onvif?.stale
    ? "The camera's reported position never changes; using the picture only."
    : pose.onvif?.supported === false
      ? "The camera reports no absolute position; using the picture only."
      : pose.onvif?.inverted?.some(Boolean)
        ? "The camera reports an inverted axis; corrected automatically."
        : null
  return (
    <section className="mt-4 border-t border-border pt-4" aria-label="World anchoring">
      <label
        htmlFor="ptz-anchor"
        className="-mx-2 flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors hover:bg-accent/70"
      >
        <span className="flex items-center gap-2">
          <Anchor className={cn("size-4", live ? "text-primary" : "text-muted-foreground")} />
          <span className={on ? "text-foreground" : "text-muted-foreground"}>Anchor Overlays</span>
        </span>
        <Switch id="ptz-anchor" checked={on} onCheckedChange={(v) => void ptz.setWorldAnchor(v)} />
      </label>

      <div className="mt-1 flex items-center gap-2 text-xs" role="status">
        <span className="relative flex size-2">
          {live && pose.moving && <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary/70" />}
          <span className={cn("relative inline-flex size-2 rounded-full", live ? "bg-primary" : "bg-muted-foreground/50")} />
        </span>
        <span className={cn("font-medium tracking-wide", live ? "text-primary" : "text-muted-foreground")}>
          {live ? (pose.moving ? "Tracking Move" : "Anchored") : on ? "Waiting For PTZ" : "Off"}
        </span>
        {live && <span className="ml-auto text-muted-foreground">{SOURCE_LABEL[pose.source]}</span>}
      </div>

      {live && (
        <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
          <Stat label="Pan" value={fmtDeg(pose.pose.pan)} />
          <Stat label="Tilt" value={fmtDeg(pose.pose.tilt)} />
          <Stat label="FOV" value={pose.fov_h ? `${pose.fov_h.toFixed(0)}°` : "–"} />
          <div className="col-span-3 flex items-baseline justify-between rounded-md bg-background/50 px-2.5 py-1.5">
            <dt className="text-muted-foreground">Drift</dt>
            <dd className={cn("tabular-nums", driftTone)}>
              ±{drift.toFixed(2)}° <span className="text-muted-foreground">{pose.drift_kind === "measured" ? "measured" : "estimated"}</span>
            </dd>
          </div>
        </dl>
      )}
      {live && pose.vision === "lost" && (
        <p className="mt-2 text-xs leading-relaxed text-destructive">
          The picture has too little detail to track right now{pose.source === "onvif" ? "; following the camera's position." : "."}
        </p>
      )}
      {live && onvifNote && <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{onvifNote}</p>}

      {live && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => void ptz.reanchor()}
            className="flex items-center justify-center gap-1.5 rounded-lg border border-primary/50 py-2 text-xs text-primary transition-colors hover:bg-primary/10"
          >
            <Crosshair className="size-3.5" /> Re-anchor Here
          </button>
          {pose.onvif?.aligned && !pose.onvif.stale ? (
            <button
              type="button"
              onClick={() => void ptz.syncOnvif()}
              className="flex items-center justify-center gap-1.5 rounded-lg border border-border py-2 text-xs text-foreground transition-colors hover:bg-accent"
            >
              <Navigation className="size-3.5" /> Sync To Camera
            </button>
          ) : (
            <span />
          )}
          {status.state === "ready" && status.caps.relative && (
            <button
              type="button"
              disabled={calibrating}
              onClick={() => void ptz.calibrate(20)}
              className="col-span-2 flex items-center justify-center gap-1.5 rounded-lg border border-border py-2 text-xs text-foreground transition-colors hover:bg-accent disabled:text-muted-foreground"
            >
              <Ruler className="size-3.5" /> {calibrating ? "Calibrating…" : "Calibrate FOV"}
            </button>
          )}
        </div>
      )}
      <AnimatePresence>
        {calib && calib.state !== "idle" && calib.message && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className={cn("mt-2 text-xs leading-relaxed", calib.state === "failed" ? "text-destructive" : "text-muted-foreground")}
          >
            {calib.message}
            {calib.state === "done" && calib.fov_h ? ` FOV set to ${calib.fov_h.toFixed(0)}° × ${calib.fov_v?.toFixed(0)}°.` : ""}
          </motion.p>
        )}
      </AnimatePresence>
      {live && (
        <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
          Zones, tripwires, the heatmap and markers stay on their spot in the room while the camera moves. Overlays look
          right? Re-anchor Here locks that alignment and clears the drift. Calibrate FOV pans 20° and back to measure the lens.
        </p>
      )}
    </section>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-background/50 px-2.5 py-1.5">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="tabular-nums text-foreground">{value}</dd>
    </div>
  )
}

function InstallHint({ status }: { status: PtzStatus }) {
  const [copied, setCopied] = useState(false)
  if (!status.enabled)
    return <p className="text-xs leading-relaxed text-muted-foreground">PTZ is turned off (SECURITYCAM_PTZ=false in .env).</p>
  return (
    <div className="space-y-3 text-xs leading-relaxed text-muted-foreground">
      <p>Pan, tilt and zoom work over ONVIF and need one optional package. Install it in the app&apos;s Python environment:</p>
      <div className="flex items-center gap-2 rounded-md border border-border bg-background/70 py-1.5 pl-3 pr-1.5">
        <code className="flex-1 text-[12px] text-foreground">{status.install_hint}</code>
        <button
          type="button"
          aria-label="Copy command"
          onClick={() => {
            void navigator.clipboard?.writeText(status.install_hint).then(() => {
              setCopied(true)
              window.setTimeout(() => setCopied(false), 1200)
            })
          }}
          className="rounded p-1.5 transition-colors hover:bg-accent hover:text-foreground"
        >
          <Copy className="size-3.5" />
        </button>
      </div>
      <AnimatePresence>
        {copied && (
          <motion.p initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="text-primary">
            Copied
          </motion.p>
        )}
      </AnimatePresence>
      <p>Then restart the server. Tapo cameras use ONVIF port 2020 and the same camera account (SECURITYCAM_ONVIF_PORT).</p>
    </div>
  )
}

function PtzPresets({ ptz, status }: { ptz: PtzControls; status: PtzStatus }) {
  const [name, setName] = useState("")
  return (
    <section className="mt-4 border-t border-border pt-4" aria-label="Camera positions">
      <p className="mb-2 text-sm">Positions</p>
      {status.presets.length === 0 && <p className="mb-2 text-xs text-muted-foreground">No saved positions on the camera yet.</p>}
      <ul className="-mx-2 space-y-0.5">
        <AnimatePresence initial={false}>
          {status.presets.map((p) => (
            <motion.li
              key={p.token}
              layout
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2, ease }}
              className="group flex items-center gap-2 overflow-hidden rounded-md px-2 py-1 hover:bg-accent/70"
            >
              <span className="min-w-0 flex-1 truncate text-sm">{p.name}</span>
              <button
                type="button"
                onClick={() => void ptz.deletePreset(p.token)}
                aria-label={`Delete position ${p.name}`}
                className="rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
              >
                <Trash2 className="size-3.5" />
              </button>
              <button
                type="button"
                onClick={() => void ptz.gotoPreset(p.token)}
                aria-label={`Go to ${p.name}`}
                className="flex items-center gap-1 rounded-full border border-primary/50 px-2.5 py-0.5 text-xs text-primary transition-colors hover:bg-primary/12"
              >
                <Navigation className="size-3" /> Go
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          const v = name.trim()
          if (!v) return
          void ptz.savePreset(v.slice(0, 32))
          setName("")
        }}
      >
        <input
          value={name}
          maxLength={32}
          placeholder="Position name"
          aria-label="New position name"
          onChange={(e) => setName(e.target.value)}
          className="h-8 min-w-0 flex-1 rounded-md border border-input bg-input/30 px-2.5 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring"
        />
        <button
          type="submit"
          disabled={!name.trim()}
          className="flex h-8 items-center gap-1 rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground disabled:opacity-40"
        >
          <Plus className="size-3.5" /> Save Position
        </button>
      </form>
    </section>
  )
}

function CalibrationSlider({
  def,
  value,
  onCommit,
}: {
  def: (typeof CALIBRATION)[number]
  value: number
  onCommit: (v: number) => void
}) {
  const [local, setLocal] = useState(value)
  useEffect(() => setLocal(value), [value])
  return (
    <div className="space-y-2.5">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{def.label}</span>
        <span className="tabular-nums">{def.format(local)}</span>
      </div>
      <Slider
        min={def.min}
        max={def.max}
        step={def.step}
        value={[local]}
        aria-label={def.label}
        onValueChange={([v]) => setLocal(v)}
        onValueCommit={([v]) => onCommit(v)}
      />
    </div>
  )
}
