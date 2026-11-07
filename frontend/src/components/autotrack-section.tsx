import { useCallback, useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Crosshair, ScanEye } from "lucide-react"
import { BodyPointPicker } from "@/components/body-point-picker"
import { PanelRange, PanelSegments, PanelToggle } from "@/components/panel-controls"
import {
  AIM_LABEL,
  consumeFocus,
  isMoving,
  OPEN_AUTOTRACK_EVENT,
  patchAutoTrack,
  requestOpenCrosshair,
  setDeadzonePreview,
  STATE_META,
  useAutoTrack,
} from "@/lib/autotrack"
import type { AutoTrackConfig, PtzStatus } from "@/lib/ops-api"
import { cn } from "@/lib/utils"
import { targetText } from "./autotrack-chip"

const MODES = [
  { value: "locked", label: "Locked Target" },
  { value: "auto", label: "Auto Pick" },
] as const

const pct = (v: number) => `${Math.round(v * 100)}%`
const secs = (v: number) => `${v % 1 ? v.toFixed(1) : v} s`

/** PTZ panel section: the real camera follows a target so its aim point stays in the framing window. */
export function AutoTrackSection({ status }: { status: PtzStatus }) {
  const { autotrack, ready, zoom } = useAutoTrack()
  const [error, setError] = useState<string | null>(null)
  const [flash, setFlash] = useState(false)
  const ref = useRef<HTMLElement>(null)
  const previewTimer = useRef<number | undefined>(undefined)

  // "open auto-track" from the stage chip: scroll here and highlight briefly
  useEffect(() => {
    const focus = () => {
      if (!consumeFocus("autotrack")) return
      window.requestAnimationFrame(() => {
        ref.current?.scrollIntoView({ behavior: "smooth", block: "start" })
        setFlash(true)
        window.setTimeout(() => setFlash(false), 1400)
      })
    }
    focus()
    window.addEventListener(OPEN_AUTOTRACK_EVENT, focus)
    return () => window.removeEventListener(OPEN_AUTOTRACK_EVENT, focus)
  }, [])

  useEffect(
    () => () => {
      window.clearTimeout(previewTimer.current)
      setDeadzonePreview(null)
    },
    [],
  )

  const patch = useCallback(async (p: Partial<AutoTrackConfig>) => {
    setError(await patchAutoTrack(p))
  }, [])

  if (!autotrack) return null
  const { config: c, status: st } = autotrack
  const meta = STATE_META[st.state]
  const who = targetText(st)
  const canZoom = zoom || Boolean(status.caps.zoom)

  return (
    <section
      ref={ref}
      aria-label="Auto-Track"
      className={cn(
        "-mx-2 mt-4 scroll-mt-4 rounded-lg border-t border-border px-2 pt-4 transition-[background-color,box-shadow] duration-700",
        flash && "bg-primary/8 shadow-[0_0_0_1px_rgba(255,210,31,0.45)]",
      )}
    >
      <PanelToggle
        id="ptz-autotrack"
        label="Auto-Track"
        icon={<ScanEye className={cn("size-4", c.enabled ? "text-primary" : "text-muted-foreground")} />}
        on={c.enabled}
        onChange={(v) => void patch({ enabled: v })}
      />

      <div className="mt-1 flex min-h-5 items-center gap-2 text-xs" role="status">
        <span className="relative flex size-2 shrink-0">
          {meta.live && isMoving(autotrack) && <span className="absolute inset-0 animate-ping rounded-full opacity-70" style={{ background: meta.color }} />}
          <span className="relative size-2 rounded-full" style={{ background: meta.color }} />
        </span>
        <span className="font-medium tracking-wide" style={{ color: meta.color }}>
          {meta.label}
        </span>
        {who && <span className="truncate text-foreground">{who}</span>}
        {st.reason && <span className="ml-auto min-w-0 truncate text-muted-foreground" title={st.reason}>{st.reason}</span>}
      </div>

      {!ready && (
        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
          Needs the PTZ camera connected. Settings are kept; tracking starts once the camera is ready.
        </p>
      )}

      <div className="mt-3 space-y-3">
        <PanelSegments label="Mode" id="autotrack-mode" options={MODES} value={c.mode} onChange={(v) => void patch({ mode: v })} />
        <p className="text-xs leading-relaxed text-muted-foreground">
          {c.mode === "locked"
            ? "Follows the target locked on the stage. Right-click a target and choose Track with PTZ, or press Shift+T."
            : "Picks the most prominent person by itself and stays with them until they are lost."}
        </p>
      </div>

      <div className="mt-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Aim Point</span>
          <span className="text-xs text-foreground">{AIM_LABEL[c.aim]}</span>
        </div>
        <BodyPointPicker value={c.aim} onChange={(k) => void patch({ aim: k })} />
        <button
          type="button"
          onClick={requestOpenCrosshair}
          className="mt-1 flex w-full items-center justify-center gap-1.5 rounded-lg border border-border py-2 text-xs text-foreground transition-colors hover:border-primary/60 hover:bg-accent"
        >
          <Crosshair className="size-3.5" /> Crosshair Settings
        </button>
      </div>

      <div className="mt-4 space-y-1 border-t border-border pt-3">
        <PanelRange
          label="Framing Window"
          min={0.03}
          max={0.45}
          step={0.01}
          value={c.deadzone}
          format={(v) => `${Math.round(v * 200)}%`}
          onChange={(v) => void patch({ deadzone: v })}
          onDrag={(v) => {
            window.clearTimeout(previewTimer.current)
            if (v !== null) setDeadzonePreview(v)
            else previewTimer.current = window.setTimeout(() => setDeadzonePreview(null), 900)
          }}
        />
        <PanelRange label="Gain" min={0.2} max={3} step={0.1} value={c.gain} format={(v) => `${v.toFixed(1)}×`} onChange={(v) => void patch({ gain: v })} />
        <PanelRange label="Max Speed" min={0.05} max={1} step={0.05} value={c.max_speed} format={pct} onChange={(v) => void patch({ max_speed: v })} />
        <p className="pb-1 text-xs leading-relaxed text-muted-foreground">
          The camera moves once the aim point leaves the framing window. Raise the gain if it lags behind, lower it if it overshoots.
        </p>
        <PanelToggle id="autotrack-invert-pan" label="Invert Pan" on={c.invert_pan} onChange={(v) => void patch({ invert_pan: v })} />
        <PanelToggle id="autotrack-invert-tilt" label="Invert Tilt" on={c.invert_tilt} onChange={(v) => void patch({ invert_tilt: v })} />
        <PanelToggle
          id="autotrack-zoom"
          label={canZoom ? "Auto Zoom" : "Auto Zoom (no zoom on this camera)"}
          on={c.auto_zoom && canZoom}
          disabled={!canZoom}
          onChange={(v) => void patch({ auto_zoom: v })}
        />
        <AnimatePresence initial={false}>
          {c.auto_zoom && canZoom && (
            <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
              <PanelRange
                label="Target Height"
                min={0.2}
                max={0.9}
                step={0.05}
                value={c.zoom_target}
                format={pct}
                onChange={(v) => void patch({ zoom_target: v })}
              />
            </motion.div>
          )}
        </AnimatePresence>
        <PanelRange label="Lost Timeout" min={1} max={30} step={1} value={c.lost_timeout_s} format={secs} onChange={(v) => void patch({ lost_timeout_s: v })} />
        <PanelRange
          label="Pause After Manual Move"
          min={0}
          max={30}
          step={1}
          value={c.manual_pause_s}
          format={(v) => (v === 0 ? "Off" : secs(v))}
          onChange={(v) => void patch({ manual_pause_s: v })}
        />
        <p className="text-xs leading-relaxed text-muted-foreground">
          After the target is lost the camera holds for the lost timeout, then searches again. Moving the camera by hand pauses auto-track.
        </p>
      </div>

      <AnimatePresence>
        {error && (
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} role="alert" className="mt-2 text-xs text-destructive">
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </section>
  )
}
