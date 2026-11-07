import { AnimatePresence, motion } from "motion/react"
import { ScanLine } from "lucide-react"
import { Slider } from "@/components/ui/slider"
import { CHAMS_MODES, SLIDERS } from "@/lib/esp-schema"
import { isSegModel, modelLabel, parseModel, segModelFor } from "@/lib/models"
import type { EspSettings } from "@/lib/types"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const ease = [0.22, 1, 0.36, 1] as const
const spring = { type: "spring", stiffness: 450, damping: 34 } as const
const OUTLINE_MODES = new Set(["outline", "silhouette_outline", "xray"])

interface Props {
  settings: EspSettings
  onUpdate: Update
  models: string[]
  /** Active YOLO model of this camera. */
  model: string | null
  cuda: boolean
  onModelChange?: (model: string) => void
  busy?: boolean
}

/**
 * Chams style: classic box or real body silhouettes from a Segment model
 * (filled, outlined with glow, or X-ray). Without a Segment model the backend
 * falls back to box chams; this panel offers a one-click switch.
 */
export function ChamsPanel({ settings, onUpdate, models, model, cuda, onModelChange, busy }: Props) {
  const mode = String(settings.chams_mode ?? "box")
  const silhouette = mode !== "box"
  const hasMasks = isSegModel(model)
  const target = segModelFor(models, model, cuda)
  const fromPose = model ? parseModel(model).task === "pose" : false
  const glow = SLIDERS.find((s) => s.key === "outline_glow")
  const glowValue = Number(settings.outline_glow ?? 8)

  return (
    <div className="mt-4 border-t border-border pt-4">
      <p className="mb-2 flex items-center gap-1.5 text-sm text-muted-foreground">
        <ScanLine className="size-3.5" /> Chams Style
      </p>
      <div
        role="radiogroup"
        aria-label="Chams style"
        className="grid gap-1 rounded-lg bg-background/60 p-1"
        style={{ gridTemplateColumns: `repeat(${CHAMS_MODES.length}, minmax(0, 1fr))` }}
      >
        {CHAMS_MODES.map((m) => {
          const active = m.value === mode
          return (
            <button
              key={m.value}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => {
                onUpdate("chams_mode", m.value)
                if (!settings.chams) onUpdate("chams", true)
              }}
              className={cn(
                "relative rounded-md py-1.5 text-xs transition-colors duration-200",
                active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {active && <motion.span layoutId="chams-mode-indicator" className="absolute inset-0 rounded-md bg-primary" transition={spring} />}
              <span className="relative">{m.label}</span>
            </button>
          )
        })}
      </div>

      <AnimatePresence initial={false}>
        {silhouette && !hasMasks && (
          <motion.div
            key="needs-seg"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.24, ease }}
            className="overflow-hidden"
          >
            <div className="mt-3 rounded-lg border border-primary/40 bg-primary/[0.06] p-3">
              <p className="text-sm text-foreground">Needs a Segment model</p>
              <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                {modelLabel(model)} has no body masks, so box chams are drawn instead.
                {fromPose && " Skeleton and head need a Pose model and stop while a Segment model runs."}
              </p>
              {target && onModelChange && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onModelChange(target)}
                  className="mt-2.5 w-full rounded-md bg-primary py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-40"
                >
                  Switch to {modelLabel(target)}
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence initial={false}>
        {glow && OUTLINE_MODES.has(mode) && (
          <motion.div
            key="glow"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            <div className="space-y-2.5 pt-4">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">{glow.label}</span>
                <span className="tabular-nums">{glow.format(glowValue)}</span>
              </div>
              <Slider
                min={glow.min}
                max={glow.max}
                step={glow.step}
                value={[glowValue]}
                aria-label={glow.label}
                onValueChange={([v]) => onUpdate("outline_glow", v, false)}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
