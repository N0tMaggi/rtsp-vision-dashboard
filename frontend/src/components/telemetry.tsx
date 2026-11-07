import { AnimatePresence, motion } from "motion/react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { CameraStatus } from "@/lib/types"
import { RollingNumber } from "./rolling-number"

/**
 * Live readout, top right of the stage. While the live connection to the
 * server is down (the dashboard then falls back to polling) the dot turns grey.
 */
export function LiveReadout({
  status,
  live,
  reconnecting = false,
}: {
  status: CameraStatus | null
  live: boolean
  reconnecting?: boolean
}) {
  const on = live && !reconnecting
  const pill = (
    <div
      className="flex items-center gap-3 rounded-full border border-border/70 bg-background/70 py-1.5 pl-3 pr-4 text-[13px] backdrop-blur-md"
      aria-label={reconnecting ? "Connection to the server lost. Reconnecting…" : undefined}
    >
      <span className="relative flex size-2">
        {on && <span className="absolute inset-0 animate-ping rounded-full bg-primary opacity-60" />}
        <span className={on ? "relative size-2 rounded-full bg-primary" : "relative size-2 rounded-full bg-muted-foreground/50"} />
      </span>
      <span className="text-muted-foreground">{live ? "Live" : "Offline"}</span>
      <span className="tabular-nums">
        <RollingNumber value={status?.fps ?? 0} format={(v) => v.toFixed(1)} />
        <span className="ml-1 text-muted-foreground">fps</span>
      </span>
    </div>
  )
  if (!reconnecting) return pill
  return (
    <Tooltip>
      <TooltipTrigger asChild>{pill}</TooltipTrigger>
      <TooltipContent>Reconnecting…</TooltipContent>
    </Tooltip>
  )
}

/** Thin telemetry line along the bottom of the stage. */
export function TelemetryBar({ status }: { status: CameraStatus | null }) {
  const s = status
  const gpu = s?.gpu
  const aiOn = s?.yolo_enabled ?? false
  const memPct = gpu && gpu.mem_total > 0 ? (gpu.mem_used / gpu.mem_total) * 100 : 0

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border/70 bg-background/80 px-5 py-2.5 text-[13px] backdrop-blur-md">
      <Metric label="Targets" value={s?.num_detections ?? 0} highlight={aiOn && (s?.num_detections ?? 0) > 0} />
      <Metric label="Faces" value={s?.faces ?? 0} />
      <Metric label="Inference" value={s?.last_infer_ms ?? 0} unit="ms" dim={!aiOn} />
      <Metric label="AI fps" value={s?.inference_fps ?? 0} digits={1} dim={!aiOn} />

      <div className="flex items-center gap-2">
        <span className="text-muted-foreground">{gpu?.name ?? "GPU"}</span>
        <Meter pct={gpu?.util ?? 0} />
        <span className="w-9 tabular-nums">
          <RollingNumber value={gpu?.util ?? 0} />%
        </span>
      </div>
      <div className="hidden items-center gap-2 md:flex">
        <span className="text-muted-foreground">VRAM</span>
        <Meter pct={memPct} />
      </div>

      <ul className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-1.5" aria-label="Detected classes">
        <AnimatePresence initial={false}>
          {(s?.classes ?? []).map((c) => (
            <motion.li
              key={c}
              layout
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ type: "spring", stiffness: 400, damping: 28 }}
              className="rounded-full border border-primary/40 px-2.5 py-0.5 text-xs text-primary"
            >
              {c}
            </motion.li>
          ))}
        </AnimatePresence>
        <span className="hidden text-muted-foreground lg:inline">
          {s ? `${s.resolution} on ${s.device.toUpperCase()}` : ""}
        </span>
      </ul>
    </div>
  )
}

function Metric({
  label,
  value,
  unit,
  digits = 0,
  dim,
  highlight,
}: {
  label: string
  value: number
  unit?: string
  digits?: number
  dim?: boolean
  highlight?: boolean
}) {
  return (
    <div className={dim ? "flex items-baseline gap-2 opacity-40 transition-opacity" : "flex items-baseline gap-2 transition-opacity"}>
      <span className="text-muted-foreground">{label}</span>
      <span className={highlight ? "tabular-nums text-primary transition-colors" : "tabular-nums transition-colors"}>
        <RollingNumber value={value} format={(v) => v.toFixed(digits)} />
        {unit && <span className="ml-0.5 text-muted-foreground">{unit}</span>}
      </span>
    </div>
  )
}

function Meter({ pct }: { pct: number }) {
  return (
    <span className="h-1 w-16 overflow-hidden rounded-full bg-muted">
      <motion.span
        className="block h-full rounded-full bg-primary"
        animate={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
        transition={{ type: "spring", stiffness: 120, damping: 20 }}
      />
    </span>
  )
}
