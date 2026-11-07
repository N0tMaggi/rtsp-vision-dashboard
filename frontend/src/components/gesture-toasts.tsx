import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion, useReducedMotion, useTransform } from "motion/react"
import { Hand, ThumbsUp, Waves } from "lucide-react"
import type { GestureEvent, GestureKind, NBox, Target } from "@/lib/types"
import { cn } from "@/lib/utils"
import type { ViewLike } from "./stage-visuals"

const ICON: Record<GestureKind, typeof Hand> = { wave: Waves, palm: Hand, thumbs_up: ThumbsUp }
const SHOW_MS = 3200

interface Shown extends GestureEvent {
  box: NBox | null
}

/**
 * Pose ESP pack: confirms fired gestures on the stage (a ring bursting on the person
 * plus a pill under the top edge) and marks people who are holding a gesture.
 */
export function GestureToasts({
  view,
  width: W,
  height: H,
  targets,
  gestures,
}: {
  view: ViewLike
  width: number
  height: number
  targets: Target[]
  gestures?: GestureEvent[]
}) {
  const [shown, setShown] = useState<Shown[]>([])
  const seen = useRef(new Set<string>())
  const timers = useRef<number[]>([])
  const targetsRef = useRef(targets)
  useEffect(() => {
    targetsRef.current = targets
  })

  useEffect(() => {
    if (!gestures?.length) return
    const fresh: Shown[] = []
    for (const g of gestures) {
      const key = `${g.seq}:${Math.round(g.ts * 1000)}`
      if (seen.current.has(key) || g.age > 2.5) continue
      seen.current.add(key)
      fresh.push({ ...g, box: targetsRef.current.find((t) => t.id === g.id)?.box ?? null })
    }
    if (!fresh.length) return
    setShown((s) => [...fresh, ...s].slice(0, 3))
    for (const f of fresh) {
      timers.current.push(window.setTimeout(() => setShown((s) => s.filter((x) => x.seq !== f.seq || x.ts !== f.ts)), SHOW_MS))
    }
    if (timers.current.length > 16) timers.current = timers.current.slice(-8)
    if (seen.current.size > 64) seen.current = new Set([...seen.current].slice(-32))
  }, [gestures])
  useEffect(() => {
    const list = timers
    return () => list.current.forEach((t) => window.clearTimeout(t))
  }, [])

  const holding = targets.filter((t) => t.gesture?.length)
  if (W <= 0 || H <= 0) return null
  return (
    <div className="pointer-events-none absolute inset-0 z-[2] overflow-hidden" aria-hidden={shown.length === 0}>
      <AnimatePresence>
        {holding.map((t) => (
          <HoldMark key={`hold-${t.id}`} target={t} view={view} width={W} height={H} />
        ))}
        {shown.filter((g) => g.box).map((g) => (
          <Burst key={`burst-${g.seq}-${g.ts}`} box={g.box!} ok={g.ok} view={view} width={W} height={H} />
        ))}
      </AnimatePresence>
      <div className="absolute inset-x-0 top-3 flex flex-col items-center gap-1.5 px-3" role="status" aria-live="polite">
        <AnimatePresence initial={false}>
          {shown.map((g) => {
            const Icon = ICON[g.gesture] ?? Hand
            return (
              <motion.div
                key={`pill-${g.seq}-${g.ts}`}
                layout
                initial={{ opacity: 0, y: -10, scale: 0.92 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -6, scale: 0.96 }}
                transition={{ type: "spring", stiffness: 420, damping: 30 }}
                className="flex items-center gap-2 rounded-full border border-white/10 bg-background/75 py-1 pl-1 pr-3 text-xs shadow-[0_8px_30px_-12px_rgba(0,0,0,0.7)] backdrop-blur-md"
              >
                <span className={cn("grid size-6 place-items-center rounded-full", g.ok ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>
                  <Icon className="size-3.5" />
                </span>
                <span className="font-medium text-foreground">{g.label}</span>
                <span className="text-muted-foreground">{g.message}</span>
              </motion.div>
            )
          })}
        </AnimatePresence>
      </div>
    </div>
  )
}

function useBoxPoint(view: ViewLike, W: number, H: number, u: number, v: number) {
  const x = useTransform(() => view.x.get() + u * W * view.s.get())
  const y = useTransform(() => view.y.get() + v * H * view.s.get())
  return { x, y }
}

function Burst({ box, ok, view, width: W, height: H }: { box: NBox; ok: boolean; view: ViewLike; width: number; height: number }) {
  const reduce = useReducedMotion()
  const { x, y } = useBoxPoint(view, W, H, (box[0] + box[2]) / 2, box[1] + (box[3] - box[1]) * 0.25)
  const size = Math.max(48, Math.min(220, (box[2] - box[0]) * W * view.s.get() * 1.1))
  const color = ok ? "#ffd21f" : "#f4f4f2"
  return (
    <motion.div className="absolute left-0 top-0" style={{ x, y }}>
      {[0, 0.18].map((delay) => (
        <motion.span
          key={delay}
          className="absolute rounded-full"
          style={{ width: size, height: size, left: -size / 2, top: -size / 2, border: `2px solid ${color}` }}
          initial={{ opacity: 0.9, scale: reduce ? 1 : 0.3 }}
          animate={{ opacity: 0, scale: reduce ? 1 : 1.25 }}
          exit={{ opacity: 0 }}
          transition={{ duration: reduce ? 0.6 : 0.9, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      ))}
    </motion.div>
  )
}

function HoldMark({ target, view, width: W, height: H }: { target: Target; view: ViewLike; width: number; height: number }) {
  const reduce = useReducedMotion()
  const b = target.box
  const { x, y } = useBoxPoint(view, W, H, (b[0] + b[2]) / 2, b[1])
  const g = target.gesture?.[0] ?? "palm"
  const Icon = ICON[g] ?? Hand
  return (
    <motion.div className="absolute left-0 top-0" style={{ x, y }}>
      <motion.div
        initial={{ opacity: 0, scale: 0.6, y: 0 }}
        animate={{ opacity: 1, scale: 1, y: -30 }}
        exit={{ opacity: 0, scale: 0.8 }}
        transition={{ type: "spring", stiffness: 380, damping: 26 }}
        className="relative -translate-x-1/2"
      >
        <span className="grid size-7 place-items-center rounded-full bg-background/75 text-primary ring-1 ring-primary/50 backdrop-blur-md">
          <Icon className="size-3.5" />
        </span>
        {!reduce && (
          <motion.span
            className="absolute inset-0 rounded-full ring-2 ring-primary/60"
            animate={{ scale: [1, 1.5], opacity: [0.7, 0] }}
            transition={{ duration: 1.1, repeat: Infinity, ease: "easeOut" }}
          />
        )}
      </motion.div>
    </motion.div>
  )
}
