import { AnimatePresence, motion, useReducedMotion } from "motion/react"
import { Hand, ThumbsUp } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import { BAR_TOGGLES, BODY_TOGGLES, GESTURE_TOGGLES, IFF_COLORS, type HintToggleDef } from "@/lib/esp-schema"
import type { EspSettings } from "@/lib/types"
import { cn } from "@/lib/utils"
import { ColorRow, ToggleList } from "./esp-menu"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const ease = [0.22, 1, 0.36, 1] as const

/** "Body" category of the ESP menu: the pose ESP pack (needs a pose model for keypoints). */
export function BodyPanel({ settings, onUpdate }: { settings: EspSettings; onUpdate: Update }) {
  const on = (k: string) => Boolean(settings[k])
  return (
    <>
      <p className="mb-3 text-xs leading-relaxed text-muted-foreground">
        Reads the pose keypoints of every person. Use a pose model; without keypoints these overlays stay empty.
      </p>
      <ul className="-mx-2">
        {BODY_TOGGLES.map((t) => (
          <li key={t.key}>
            <HintRow def={t} settings={settings} onUpdate={onUpdate} />
            <Reveal show={on(t.key)}>
              {t.key === "iff_colors" && (
                <div className="mx-2 mb-2 grid grid-cols-3 gap-1 rounded-lg bg-background/50 p-1">
                  {IFF_COLORS.map((c) => (
                    <ColorRow key={c.key} def={c} value={String(settings[c.key] ?? "#ffffff")} onUpdate={onUpdate} />
                  ))}
                </div>
              )}
              {t.key === "posture" && (
                <div className="mx-2 mb-2 rounded-lg bg-background/50 px-1">
                  <HintRow
                    def={{ key: "fall_alarm", label: "Fall Alarm", hint: "Upright to lying within about a second raises a Fall event. Notifications: Alarms." }}
                    settings={settings}
                    onUpdate={onUpdate}
                    compact
                  />
                </div>
              )}
              {t.key === "status_bars" && (
                <div className="mx-2 mb-2 rounded-lg bg-background/50 px-1 pb-1">
                  <ToggleList items={BAR_TOGGLES} settings={settings} onUpdate={onUpdate} />
                  <p className="px-2 pb-1 text-xs leading-relaxed text-muted-foreground">
                    Dwell fills up to the loitering time and turns red near it.
                  </p>
                </div>
              )}
            </Reveal>
          </li>
        ))}
      </ul>

      <section className="mt-4 border-t border-border pt-4">
        <div className="-mx-2">
          <HintRow
            def={{ key: "gestures", label: "Gesture Control", hint: "People in view can control the camera with their hands." }}
            settings={settings}
            onUpdate={onUpdate}
          />
        </div>
        <Reveal show={on("gestures")}>
          <ul className="-mx-2 mt-1 rounded-lg bg-background/50 px-1">
            {GESTURE_TOGGLES.map((t) => (
              <li key={t.key}>
                <HintRow def={t} settings={settings} onUpdate={onUpdate} compact />
              </li>
            ))}
          </ul>
          <p className="mt-2 flex gap-2 text-xs leading-relaxed text-muted-foreground">
            <span className="mt-0.5 flex shrink-0 gap-0.5 text-primary">
              <Hand className="size-3.5" />
              <ThumbsUp className="size-3.5" />
            </span>
            Each gesture fires once per hold, then waits a few seconds. Palm and thumbs up use the hand landmark
            detector while this is on; wave needs a PTZ camera. Every gesture is logged in Events.
          </p>
        </Reveal>
      </section>
    </>
  )
}

function HintRow({ def, settings, onUpdate, compact }: { def: HintToggleDef; settings: EspSettings; onUpdate: Update; compact?: boolean }) {
  const id = `esp-${def.key}`
  const on = Boolean(settings[def.key])
  return (
    <label
      htmlFor={id}
      className={cn(
        "flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 text-sm transition-colors duration-150 hover:bg-accent/70",
        compact ? "py-1.5" : "py-2",
      )}
    >
      <span className="min-w-0">
        <span className={cn("block transition-colors duration-200", on ? "text-foreground" : "text-muted-foreground")}>{def.label}</span>
        <span className="block text-xs leading-snug text-muted-foreground/80">{def.hint}</span>
      </span>
      <Switch id={id} checked={on} onCheckedChange={(v) => onUpdate(def.key, v)} />
    </label>
  )
}

function Reveal({ show, children }: { show: boolean; children: React.ReactNode }) {
  const reduce = useReducedMotion()
  return (
    <AnimatePresence initial={false}>
      {show && (
        <motion.div
          initial={reduce ? { opacity: 0 } : { opacity: 0, height: 0 }}
          animate={reduce ? { opacity: 1 } : { opacity: 1, height: "auto" }}
          exit={reduce ? { opacity: 0 } : { opacity: 0, height: 0 }}
          transition={{ duration: 0.24, ease }}
          className="overflow-hidden"
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
