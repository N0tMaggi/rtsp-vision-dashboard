import { useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion, useSpring, useTransform, type MotionValue } from "motion/react"
import type { CameraPose, NBox, Target } from "@/lib/types"
import { useVisualPrefs } from "@/lib/visual-prefs"
import { cn } from "@/lib/utils"
import { TagChips } from "./stage-context"
import { SoundRadarPings } from "./sound-hud" // Sound ESP

/** The parts of useViewTransform() the HUD needs. */
export interface ViewLike {
  s: MotionValue<number>
  x: MotionValue<number>
  y: MotionValue<number>
  zoom: number
}

interface Props {
  view: ViewLike
  /** Size of the visible video rect in px (the HUD's coordinate space). */
  width: number
  height: number
  targets: Target[]
  lockedId: number | null
  hoveredId: number | null
  /** Pan so that `box` is centered (keeping the zoom). */
  onFocus: (box: NBox) => void
  /** PTZ camera pose while overlays are world-anchored: the radar shows world bearings. */
  pose?: CameraPose
}

const spring = { type: "spring", stiffness: 420, damping: 32 } as const

/** Radar minimap, off-screen arrows and target info cards, layered over the video. */
export function StageVisuals(p: Props) {
  const prefs = useVisualPrefs()
  if (p.width <= 0 || p.height <= 0) return null
  const cards = [p.lockedId, p.hoveredId !== p.lockedId ? p.hoveredId : null]
    .map((id) => p.targets.find((t) => t.id === id))
    .filter((t): t is Target => Boolean(t))
  return (
    <div className="pointer-events-none absolute inset-0 z-[1]">
      {prefs.arrows && p.view.zoom > 1.01 && <OffscreenArrows {...p} />}
      {prefs.infoCard && (
        <AnimatePresence>
          {cards.map((t) => (
            <TargetCard key={t.id} target={t} locked={t.id === p.lockedId} view={p.view} width={p.width} height={p.height} />
          ))}
        </AnimatePresence>
      )}
      {prefs.radar && (
        <Radar targets={p.targets} lockedId={p.lockedId} hoveredId={p.hoveredId} compact={p.width < 520} pose={p.pose} />
      )}
    </div>
  )
}

// ------------------------------------------------------------------ formatting
export const fmtDuration = (sec?: number) => {
  if (sec === undefined || !Number.isFinite(sec)) return "–"
  const t = Math.max(0, Math.floor(sec))
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = String(t % 60).padStart(2, "0")
  return h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`
}
const fmtMeters = (m?: number) => (m === undefined ? "–" : m < 10 ? `${m.toFixed(1)} m` : `${Math.round(m)} m`)
const fmtTime = (epoch?: number) =>
  epoch === undefined
    ? "–"
    : new Date(epoch * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })

// ------------------------------------------------------------------ view snapshot
/** Current (animated) view transform as React state, batched per animation frame. */
function useViewSnapshot(view: ViewLike) {
  const [v, setV] = useState(() => ({ s: view.s.get(), x: view.x.get(), y: view.y.get() }))
  useEffect(() => {
    let raf = 0
    const push = () => {
      raf = 0
      setV({ s: view.s.get(), x: view.x.get(), y: view.y.get() })
    }
    const on = () => {
      if (!raf) raf = requestAnimationFrame(push)
    }
    const offs = [view.s.on("change", on), view.x.on("change", on), view.y.on("change", on)]
    return () => {
      offs.forEach((off) => off())
      cancelAnimationFrame(raf)
    }
  }, [view.s, view.x, view.y])
  return v
}

// ------------------------------------------------------------------ off-screen arrows
const EDGE = 26

function OffscreenArrows({ view, width: W, height: H, targets, lockedId, onFocus }: Props) {
  const v = useViewSnapshot(view)
  const arrows = useMemo(() => {
    const cx = W / 2
    const cy = H / 2
    const out: { t: Target; x: number; y: number; angle: number; locked: boolean }[] = []
    for (const t of targets) {
      const tx = v.x + ((t.box[0] + t.box[2]) / 2) * W * v.s
      const ty = v.y + ((t.box[1] + t.box[3]) / 2) * H * v.s
      // fully visible box center -> no arrow
      if (tx >= 0 && tx <= W && ty >= 0 && ty <= H) continue
      const dx = tx - cx
      const dy = ty - cy
      const k = Math.min(
        dx !== 0 ? (W / 2 - EDGE) / Math.abs(dx) : Infinity,
        dy !== 0 ? (H / 2 - EDGE) / Math.abs(dy) : Infinity,
      )
      out.push({ t, x: cx + dx * k, y: cy + dy * k, angle: (Math.atan2(dy, dx) * 180) / Math.PI, locked: t.id === lockedId })
    }
    // locked arrow last so it paints on top
    return out.sort((a, b) => Number(a.locked) - Number(b.locked))
  }, [targets, v, W, H, lockedId])

  return (
    <AnimatePresence>
      {arrows.map(({ t, x, y, angle, locked }) => {
        // label sits on the inner side of the arrow
        const side = x <= EDGE + 1 ? "left" : x >= W - EDGE - 1 ? "right" : y <= EDGE + 1 ? "top" : "bottom"
        const labelPos = {
          left: "left-full ml-1.5 top-1/2 -translate-y-1/2",
          right: "right-full mr-1.5 top-1/2 -translate-y-1/2",
          top: "top-full mt-1.5 left-1/2 -translate-x-1/2",
          bottom: "bottom-full mb-1.5 left-1/2 -translate-x-1/2",
        }[side]
        return (
          <motion.button
            key={t.id}
            type="button"
            aria-label={`Pan to ${t.label} #${t.id}`}
            onPointerDown={(e) => e.stopPropagation()}
            onDoubleClick={(e) => e.stopPropagation()}
            onClick={(e) => {
              e.stopPropagation()
              onFocus(t.box)
            }}
            initial={{ opacity: 0, scale: 0.6, x, y }}
            animate={{ opacity: locked ? 1 : 0.8, scale: 1, x, y }}
            exit={{ opacity: 0, scale: 0.6 }}
            transition={{ x: { type: "spring", stiffness: 500, damping: 40 }, y: { type: "spring", stiffness: 500, damping: 40 }, default: spring }}
            className="group pointer-events-auto absolute left-0 top-0 -ml-4 -mt-4 grid size-8 place-items-center"
          >
            <span
              className={cn(
                "grid size-8 place-items-center rounded-full border backdrop-blur-md transition-transform duration-200 group-hover:scale-110",
                locked ? "border-primary bg-primary/90 text-primary-foreground" : "border-white/25 bg-background/75 text-foreground",
              )}
              style={{ transform: `rotate(${angle}deg)` }}
            >
              <svg viewBox="0 0 16 16" className="size-3.5" aria-hidden>
                <path d="M3 3 L13 8 L3 13 L6 8 Z" fill="currentColor" />
              </svg>
            </span>
            {locked && (
              <span className="absolute inset-0 animate-ping rounded-full border border-primary/70" aria-hidden />
            )}
            <span
              className={cn(
                "absolute whitespace-nowrap rounded-full px-2 py-0.5 text-[11px] tabular-nums backdrop-blur-md",
                labelPos,
                locked ? "bg-primary text-primary-foreground" : "bg-background/80 text-foreground",
              )}
            >
              {t.label} #{t.id}
              {t.distance !== undefined && <span className="opacity-70"> · {fmtMeters(t.distance)}</span>}
            </span>
          </motion.button>
        )
      })}
    </AnimatePresence>
  )
}

// ------------------------------------------------------------------ info card
const CARD_W = 212
const GAP = 12
const MARGIN = 8

function TargetCard({
  target,
  locked,
  view,
  width: W,
  height: H,
}: {
  target: Target
  locked: boolean
  view: ViewLike
  width: number
  height: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const boxSpring = { stiffness: 300, damping: 36, mass: 0.6 }
  const [b0, b1, b2, b3] = target.box
  const u1 = useSpring(b0, boxSpring)
  const v1 = useSpring(b1, boxSpring)
  const u2 = useSpring(b2, boxSpring)
  const v2 = useSpring(b3, boxSpring)
  useEffect(() => {
    u1.set(b0)
    v1.set(b1)
    u2.set(b2)
    v2.set(b3)
  }, [b0, b1, b2, b3, u1, v1, u2, v2])

  // Screen-space anchor from the (smoothed) normalized box and the live view transform.
  const left = useTransform(() => {
    const s = view.s.get()
    const x = view.x.get()
    const right = x + u2.get() * W * s + GAP
    const leftSide = x + u1.get() * W * s - GAP - CARD_W
    const px = right + CARD_W <= W - MARGIN || leftSide < MARGIN ? right : leftSide
    return Math.min(W - CARD_W - MARGIN, Math.max(MARGIN, px))
  })
  const top = useTransform(() => {
    const s = view.s.get()
    const y = view.y.get()
    const ch = ref.current?.offsetHeight ?? 150
    return Math.min(H - ch - MARGIN, Math.max(MARGIN, y + v1.get() * H * s))
  })

  // Off-screen target: fade the card out (the edge arrow takes over).
  const onScreen = useTransform((): number => {
    const s = view.s.get()
    const cx = view.x.get() + ((u1.get() + u2.get()) / 2) * W * s
    const cy = view.y.get() + ((v1.get() + v2.get()) / 2) * H * s
    return cx >= 0 && cx <= W && cy >= 0 && cy <= H ? 1 : 0
  })
  const opacity = useSpring(onScreen, { stiffness: 260, damping: 30 })

  const rows: [string, string][] = [
    ["Distance", fmtMeters(target.distance)],
    ["Dwell", fmtDuration(target.dwell)],
    ["Path", fmtMeters(target.path_m)],
    ["Speed", target.speed === undefined ? "–" : `${target.speed.toFixed(1)} m/s`],
    ["First Seen", fmtTime(target.first_seen)],
  ]

  return (
    <motion.div ref={ref} style={{ left, top, width: CARD_W, opacity }} className="absolute">
    <motion.div
      initial={{ opacity: 0, scale: 0.92, filter: "blur(4px)" }}
      animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
      exit={{ opacity: 0, scale: 0.94, filter: "blur(4px)" }}
      transition={spring}
      className={cn(
        "origin-top-left overflow-hidden rounded-lg border bg-background/82 text-xs shadow-[0_18px_50px_-20px_rgba(0,0,0,0.7)] backdrop-blur-xl",
        locked ? "border-primary/60" : "border-white/12",
      )}
    >
      <div className="flex items-center gap-2 border-b border-white/8 px-3 py-2">
        <span className={cn("h-3.5 w-1 rounded-full", locked ? "bg-primary" : "bg-foreground/60")} />
        <span className="truncate font-semibold uppercase tracking-wide">{target.label}</span>
        <span className="text-muted-foreground">#{target.id}</span>
        <span className="ml-auto tabular-nums text-muted-foreground">{Math.round(target.conf * 100)}%</span>
      </div>
      {(target.identity?.name || target.identity?.face_id !== undefined || target.tags?.length) && (
        <div className="space-y-1 border-b border-white/8 px-3 py-1.5">
          {target.identity?.name ? (
            <div className="truncate text-[11px] font-semibold text-[#7ee0a1]">{target.identity.name}</div>
          ) : target.identity?.face_id !== undefined ? (
            <div className="text-[11px] font-semibold text-[#ff8a80]">Face #{target.identity.face_id}</div>
          ) : null}
          <TagChips tags={target.tags} />
        </div>
      )}
      {locked && (
        <div className="flex items-center gap-1.5 bg-primary/12 px-3 py-1 text-[10px] font-semibold uppercase tracking-widest text-primary">
          <span className="size-1.5 animate-pulse rounded-full bg-primary" /> Locked
        </div>
      )}
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 px-3 py-2">
        {rows.map(([k, val]) => (
          <div key={k} className="contents">
            <dt className="text-muted-foreground">{k}</dt>
            <dd className="text-right tabular-nums">{val}</dd>
          </div>
        ))}
      </dl>
    </motion.div>
    </motion.div>
  )
}

// ------------------------------------------------------------------ radar
const FOV_DEG = 100
const RANGES = [5, 10, 20, 40, 80, 160]

type Shape = "person" | "vehicle" | "animal" | "other"
const VEHICLES = new Set(["car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane"])
const ANIMALS = new Set(["dog", "cat", "bird", "horse", "sheep", "cow", "bear"])
const shapeOf = (label: string): Shape => {
  const l = label.toLowerCase()
  if (l === "person") return "person"
  if (VEHICLES.has(l)) return "vehicle"
  if (ANIMALS.has(l)) return "animal"
  return "other"
}

function Blip({ shape, r }: { shape: Shape; r: number }) {
  switch (shape) {
    case "person":
      return <circle r={r} />
    case "vehicle":
      return <rect x={-r * 1.1} y={-r * 0.8} width={r * 2.2} height={r * 1.6} rx={r * 0.4} />
    case "animal":
      return <path d={`M0 ${-r * 1.2} L${r * 1.1} ${r * 0.8} L${-r * 1.1} ${r * 0.8} Z`} />
    default:
      return <path d={`M0 ${-r * 1.2} L${r * 1.1} 0 L0 ${r * 1.2} L${-r * 1.1} 0 Z`} />
  }
}

/** Bearing of a normalized image column in degrees: pinhole model when the pose is known. */
function bearing(u: number, pose?: CameraPose) {
  if (!pose) return (u - 0.5) * FOV_DEG
  const fx = 0.5 / Math.tan(((pose.fov_h / 2) * Math.PI) / 180)
  return pose.pan + (Math.atan((u - 0.5) / fx) * 180) / Math.PI
}

function Radar({
  targets,
  lockedId,
  hoveredId,
  compact,
  pose,
}: {
  targets: Target[]
  lockedId: number | null
  hoveredId: number | null
  compact: boolean
  pose?: CameraPose
}) {
  // Anchored PTZ: full circle in world bearings (0° = the camera's home direction, up),
  // the view wedge turns with the camera. Otherwise a fan relative to the picture.
  const world = Boolean(pose)
  const W = compact ? 132 : 176
  const R = W / 2 - 8
  const H = world ? W + 12 : R + 22
  const cx = W / 2
  const cy = world ? W / 2 : R + 10
  const half = (world ? pose!.fov_h : FOV_DEG) / 2
  const rad = (half * Math.PI) / 180
  const heading = world ? pose!.pan : 0

  const withDist = targets.filter((t) => t.distance !== undefined)
  const maxD = withDist.reduce((m, t) => Math.max(m, t.distance ?? 0), 0)
  const range = RANGES.find((r) => r >= maxD * 1.1) ?? RANGES[RANGES.length - 1]

  const pos = (t: Target) => {
    const u = (t.box[0] + t.box[2]) / 2
    const a = (bearing(u, pose) * Math.PI) / 180
    // sqrt scale: spreads the near field, where most targets are
    const r = Math.sqrt(Math.min(1, (t.distance ?? range) / range)) * R
    return { x: cx + r * Math.sin(a), y: cy - r * Math.cos(a) }
  }
  const fan = `M${cx} ${cy} L${cx - R * Math.sin(rad)} ${cy - R * Math.cos(rad)} A${R} ${R} 0 0 1 ${cx + R * Math.sin(rad)} ${cy - R * Math.cos(rad)} Z`

  return (
    <motion.div
      initial={{ opacity: 0, y: -6, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={spring}
      className="absolute right-3 top-3 rounded-xl border border-white/10 bg-background/70 p-1.5 backdrop-blur-md"
      role="img"
      aria-label={`Radar: ${targets.length} targets, range ${range} m`}
    >
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block overflow-visible">
        <defs>
          <linearGradient id="radar-sweep" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0" stopColor="#ffd21f" stopOpacity="0" />
            <stop offset="1" stopColor="#ffd21f" stopOpacity="0.35" />
          </linearGradient>
          <clipPath id="radar-fan">
            <path d={fan} />
          </clipPath>
        </defs>
        {world && (
          <>
            <circle cx={cx} cy={cy} r={R} fill="rgba(244,244,242,0.02)" stroke="rgba(244,244,242,0.14)" />
            {[1 / 3, 2 / 3].map((f) => (
              <circle key={f} cx={cx} cy={cy} r={R * f} fill="none" stroke="rgba(244,244,242,0.1)" strokeDasharray="2 3" />
            ))}
            <line x1={cx} y1={cy - R - 3} x2={cx} y2={cy - R + 4} stroke="rgba(244,244,242,0.5)" />
            <text x={cx} y={cy - R - 5} textAnchor="middle" className="fill-muted-foreground" fontSize={8}>
              0°
            </text>
          </>
        )}
        <g transform={`rotate(${heading} ${cx} ${cy})`} style={{ transition: "transform 120ms linear" }}>
        <path d={fan} fill="rgba(255,210,31,0.04)" stroke="rgba(244,244,242,0.18)" />
        {!world &&
          [1 / 3, 2 / 3].map((f) => (
            <path
              key={f}
              d={`M${cx - R * f * Math.sin(rad)} ${cy - R * f * Math.cos(rad)} A${R * f} ${R * f} 0 0 1 ${cx + R * f * Math.sin(rad)} ${cy - R * f * Math.cos(rad)}`}
              fill="none"
              stroke="rgba(244,244,242,0.12)"
              strokeDasharray="2 3"
            />
          ))}
        <line x1={cx} y1={cy} x2={cx} y2={cy - R} stroke="rgba(244,244,242,0.08)" />
        <g clipPath="url(#radar-fan)">
          <g>
            <animateTransform
              attributeName="transform"
              type="rotate"
              values={`${-half} ${cx} ${cy}; ${half} ${cx} ${cy}; ${-half} ${cx} ${cy}`}
              keyTimes="0; 0.5; 1"
              calcMode="spline"
              keySplines="0.45 0 0.55 1; 0.45 0 0.55 1"
              dur="5.2s"
              repeatCount="indefinite"
            />
            <path d={`M${cx} ${cy} L${cx - R * 0.16} ${cy - R - 4} L${cx + 1} ${cy - R - 4} Z`} fill="url(#radar-sweep)" />
            <line x1={cx} y1={cy} x2={cx} y2={cy - R - 4} stroke="#ffd21f" strokeOpacity="0.55" strokeWidth={1} />
          </g>
        </g>
        </g>
        <AnimatePresence>
          {withDist.map((t) => {
            const { x, y } = pos(t)
            const locked = t.id === lockedId
            const hovered = t.id === hoveredId
            return (
              <motion.g
                key={t.id}
                initial={{ x, y, opacity: 0, scale: 0.4 }}
                animate={{ x, y, opacity: 1, scale: hovered ? 1.35 : 1 }}
                exit={{ opacity: 0, scale: 0.4 }}
                transition={{ x: { type: "spring", stiffness: 120, damping: 20 }, y: { type: "spring", stiffness: 120, damping: 20 }, default: spring }}
                fill={locked ? "#ffd21f" : "#f4f4f2"}
                fillOpacity={locked ? 1 : 0.85}
              >
                {locked && (
                  <motion.circle
                    r={7}
                    fill="none"
                    stroke="#ffd21f"
                    animate={{ r: [5, 10], opacity: [0.9, 0] }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: "easeOut" }}
                  />
                )}
                <Blip shape={shapeOf(t.label)} r={compact ? 2.6 : 3.2} />
              </motion.g>
            )
          })}
        </AnimatePresence>
        <SoundRadarPings cx={cx} cy={cy} r={R} />
        <circle cx={cx} cy={cy} r={2.5} fill="#ffd21f" />
        <text x={cx + R} y={H - 2} textAnchor="end" className="fill-muted-foreground" fontSize={9}>
          {world ? `${Math.round((((heading % 360) + 540) % 360) - 180)}° · ` : ""}
          {range} m
        </text>
        <text x={cx - R} y={H - 2} className="fill-muted-foreground" fontSize={9}>
          {withDist.length ? `${withDist.length} TGT` : "No Targets"}
        </text>
      </svg>
    </motion.div>
  )
}
