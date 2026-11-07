import { useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion, useSpring, useTransform } from "motion/react"
import { Maximize2, Minimize2 } from "lucide-react"
import type { CameraPose, DepthStatus, Target } from "@/lib/types"
import { useVisualPrefs } from "@/lib/visual-prefs"
import { cn } from "@/lib/utils"

/**
 * Top-down room map (depth package): the camera's view frustum seen from above,
 * targets as dots at their floor positions (meters), trails of the last seconds,
 * distance rings, optional depth "floor scan". PTZ cameras with world anchoring
 * use world bearings: dots stay put while the camera pans and the FOV wedge turns.
 */
interface Props {
  targets: Target[]
  lockedId: number | null
  hoveredId: number | null
  pose?: CameraPose
  depth?: DepthStatus
  compact?: boolean
  onLock: (id: number | null) => void
}

const YELLOW = "#ffd21f"
const WHITE = "#f4f4f2"
const ANTHRACITE = "#222427"
const TRAIL_S = 8
const RANGES = [3, 5, 8, 12, 20, 30, 50, 80, 120]
const RING_STEPS = [0.5, 1, 2, 2.5, 5, 10, 20, 25, 50]
const spring = { type: "spring", stiffness: 420, damping: 32 } as const
const dotSpring = { type: "spring", stiffness: 140, damping: 22 } as const

type XZ = [number, number]

/** Optional IFF field added by another feature: a state string or { state, color }. */
type IffLike = string | { state?: string; color?: string } | undefined
const IFF_COLORS: Record<string, string> = {
  friend: "#7ee0a1",
  friendly: "#7ee0a1",
  foe: "#ff5a4e",
  hostile: "#ff5a4e",
  neutral: WHITE,
  unknown: YELLOW,
}
function iffColor(t: Target): string | null {
  const iff = (t as Target & { iff?: IffLike }).iff
  if (!iff) return null
  if (typeof iff === "string") return IFF_COLORS[iff.toLowerCase()] ?? null
  return iff.color ?? (iff.state ? IFF_COLORS[iff.state.toLowerCase()] ?? null : null)
}

/** Rotate a camera-frame floor point by the pan (degrees, + = right) into world bearings. */
function rotate([x, z]: XZ, panDeg: number): XZ {
  const p = (panDeg * Math.PI) / 180
  return [x * Math.cos(p) + z * Math.sin(p), -x * Math.sin(p) + z * Math.cos(p)]
}

export function RoomMap(p: Props) {
  const prefs = useVisualPrefs()
  if (!prefs.roomMap) return null
  return <RoomMapInner {...p} />
}

function RoomMapInner({ targets, lockedId, hoveredId, pose, depth, compact, onLock }: Props) {
  const [large, setLarge] = useState(false)
  const world = Boolean(pose)
  const pan = pose?.pan ?? 0
  const fovH = pose?.fov_h ?? depth?.fov_h ?? 90
  const half = Math.min(85, fovH / 2)

  // ---- floor positions (meters, map frame) -------------------------------------
  const placed = useMemo(() => {
    const fx = 0.5 / Math.tan((fovH / 2) * (Math.PI / 180))
    const out: { t: Target; pos: XZ }[] = []
    for (const t of targets) {
      let pos: XZ | null = null
      if (world && t.ground_world) pos = t.ground_world
      else if (t.ground) pos = world ? rotate(t.ground, pan) : t.ground
      else if (t.distance !== undefined) {
        const u = (t.box[0] + t.box[2]) / 2
        const b = Math.atan((u - 0.5) / fx)
        pos = world ? rotate([t.distance * Math.sin(b), t.distance * Math.cos(b)], pan) : [t.distance * Math.sin(b), t.distance * Math.cos(b)]
      }
      if (pos && Number.isFinite(pos[0]) && Number.isFinite(pos[1])) out.push({ t, pos })
    }
    return out
  }, [targets, world, pan, fovH])

  const scan = useMemo(() => {
    const s = depth?.scan
    if (!s?.pts.length) return null
    const pts = s.world === world ? s.pts : s.pts.map(([x, z, w]) => [...(world ? rotate([x, z], pan) : rotate([x, z], -pan)), w] as [number, number, number])
    return { cell: s.cell, pts }
  }, [depth?.scan, world, pan])

  // ---- trails (meters, kept across renders) -------------------------------------
  const trails = useRef(new Map<number, { x: number; z: number; t: number }[]>())
  const [, bump] = useState(0)
  useEffect(() => {
    const now = performance.now() / 1000
    const map = trails.current
    for (const { t, pos } of placed) {
      const tr = map.get(t.id) ?? []
      const last = tr[tr.length - 1]
      if (!last || Math.hypot(last.x - pos[0], last.z - pos[1]) > 0.08 || now - last.t > 0.5) {
        tr.push({ x: pos[0], z: pos[1], t: now })
      }
      map.set(t.id, tr)
    }
    for (const [id, tr] of map) {
      while (tr.length && now - tr[0].t > TRAIL_S) tr.shift()
      if (!tr.length) map.delete(id)
    }
    bump((n) => (n + 1) % 1024)
  }, [placed])

  // ---- range with hysteresis (grows at once, shrinks after a while) -------------
  const wanted = useMemo(() => {
    let m = 0
    for (const { pos } of placed) m = Math.max(m, Math.hypot(pos[0], pos[1]))
    const fromTargets = m > 0
    if (!m && scan) {
      // empty scene: frame the bulk of the floor scan (not its far outliers)
      const ds = scan.pts.map(([x, z]) => Math.hypot(x, z)).sort((a, b) => a - b)
      m = ds[Math.floor(ds.length * 0.6)] ?? 0
    }
    const need = (m || 6) * 1.18
    return { r: RANGES.find((r) => r >= need) ?? RANGES[RANGES.length - 1], fromTargets }
  }, [placed, scan])
  const [range, setRange] = useState(wanted.r)
  useEffect(() => {
    if (wanted.r === range) return
    // grow at once for targets; otherwise wait, so flickering detections don't make the map pump
    const delay = wanted.r > range && wanted.fromTargets ? 0 : wanted.fromTargets ? 2500 : 6000
    const id = setTimeout(() => setRange(wanted.r), delay)
    return () => clearTimeout(id)
  }, [wanted.r, wanted.fromTargets, range])

  // ---- layout -------------------------------------------------------------------
  const W = large ? 340 : compact ? 176 : 236
  const sinH = Math.sin((half * Math.PI) / 180)
  const H = world ? W : Math.round(W * (half > 60 ? 0.62 : 0.78))
  const cx = W / 2
  const cy = world ? W / 2 : H - 12
  const R = world ? W / 2 - 12 : Math.min(H - 24, (W / 2 - 10) / Math.max(0.35, sinH))
  const k = R / range
  const px = ([x, z]: XZ) => ({ x: cx + x * k, y: cy - z * k })
  const ringStep = RING_STEPS.find((s) => range / s <= 4) ?? 50
  const rings: number[] = []
  for (let r = ringStep; r < range + 1e-6; r += ringStep) rings.push(Math.round(r * 100) / 100)

  // FOV wedge turns smoothly with the camera (unwrapped heading)
  const unwrapped = useRef(pan)
  const target = useMemo(() => {
    const prev = unwrapped.current
    const d = ((((pan - prev) % 360) + 540) % 360) - 180
    unwrapped.current = prev + d
    return unwrapped.current
  }, [pan])
  const heading = useSpring(world ? target : 0, { stiffness: 160, damping: 24 })
  useEffect(() => heading.set(world ? target : 0), [heading, target, world])
  const wedge = useTransform(heading, (h) => {
    const a0 = ((h - half) * Math.PI) / 180
    const a1 = ((h + half) * Math.PI) / 180
    const arc = half * 2 > 180 ? 1 : 0
    return `M${cx} ${cy} L${cx + R * Math.sin(a0)} ${cy - R * Math.cos(a0)} A${R} ${R} 0 ${arc} 1 ${cx + R * Math.sin(a1)} ${cy - R * Math.cos(a1)} Z`
  })
  const clipId = useMemo(() => `rm-clip-${Math.random().toString(36).slice(2, 8)}`, [])
  const outline = world
    ? null
    : `M${cx} ${cy} L${cx - R * sinH} ${cy - R * Math.cos((half * Math.PI) / 180)} A${R} ${R} 0 0 1 ${cx + R * sinH} ${cy - R * Math.cos((half * Math.PI) / 180)} Z`

  const st = depth?.state
  const pill = !depth
    ? "Box estimate"
    : st === "running"
      ? `${depth.device_label ?? depth.device ?? ""}${depth.latency_ms != null ? ` · ${Math.round(depth.latency_ms)} ms` : ""}`
      : depth.model.state === "downloading"
        ? "Downloading"
        : st === "loading"
          ? "Preparing"
          : st === "failed"
            ? "Failed"
            : "Starting"
  const calibrated = Boolean(depth?.calibration.ready)

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8, scale: 0.96 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.96 }}
      transition={spring}
      className="pointer-events-none absolute bottom-3 right-3 z-[2] rounded-xl border border-white/10 bg-background/72 p-2 shadow-[0_18px_50px_-24px_rgba(0,0,0,0.8)] backdrop-blur-md"
      role="group"
      aria-label={`Room map: ${placed.length} targets, range ${range} m`}
    >
      <div className="mb-1 flex items-center gap-2 px-0.5 text-[10px] uppercase tracking-widest text-muted-foreground">
        <span className="font-semibold text-foreground/90">Room Map</span>
        <span
          className={cn("size-1.5 rounded-full transition-colors duration-500", calibrated ? "bg-primary" : "border border-muted-foreground")}
          title={calibrated ? "Depth calibrated" : "Not calibrated yet: distances from box heights"}
        />
        <span className="ml-auto max-w-[9rem] truncate normal-case tracking-normal tabular-nums">{pill}</span>
        <button
          type="button"
          aria-label={large ? "Smaller room map" : "Larger room map"}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation()
            setLarge((v) => !v)
          }}
          className="pointer-events-auto -mr-0.5 grid size-5 place-items-center rounded text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          {large ? <Minimize2 className="size-3" /> : <Maximize2 className="size-3" />}
        </button>
      </div>
      <motion.svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        animate={{ width: W, height: H }}
        transition={spring}
        className="block overflow-visible"
      >
        <defs>
          <clipPath id={clipId}>{world ? <circle cx={cx} cy={cy} r={R} /> : <path d={outline!} />}</clipPath>
          <radialGradient id={`${clipId}-g`} cx="50%" cy="50%" r="50%">
            <stop offset="0" stopColor={YELLOW} stopOpacity="0.16" />
            <stop offset="1" stopColor={YELLOW} stopOpacity="0.02" />
          </radialGradient>
        </defs>

        {/* frame: circle (world) or fan (fixed camera) */}
        {world ? (
          <circle cx={cx} cy={cy} r={R} fill="rgba(244,244,242,0.02)" stroke="rgba(244,244,242,0.16)" />
        ) : (
          <path d={outline!} fill="rgba(244,244,242,0.02)" stroke="rgba(244,244,242,0.16)" />
        )}

        <g clipPath={`url(#${clipId})`}>
          {/* depth floor scan */}
          {scan &&
            scan.pts.map(([x, z, w], i) => {
              const q = px([x, z])
              const s = Math.max(1.4, scan.cell * k * 0.9)
              return <rect key={i} x={q.x - s / 2} y={q.y - s / 2} width={s} height={s} rx={s * 0.25} fill={WHITE} opacity={0.08 + Math.sqrt(w) * 0.3} />
            })}
          {/* FOV wedge */}
          <motion.path d={wedge} fill={`url(#${clipId}-g)`} stroke="rgba(255,210,31,0.35)" strokeWidth={1} />
          {/* distance rings */}
          {rings.map((r) => (
            <motion.circle
              key={r}
              cx={cx}
              cy={cy}
              initial={false}
              animate={{ r: r * k }}
              transition={spring}
              fill="none"
              stroke="rgba(244,244,242,0.11)"
              strokeDasharray="2 3"
            />
          ))}
          {/* trails */}
          {[...trails.current.entries()].map(([id, tr]) =>
            tr.length > 1 ? (
              <polyline
                key={`trail-${id}`}
                points={tr.map((p) => `${cx + p.x * k},${cy - p.z * k}`).join(" ")}
                fill="none"
                stroke={id === lockedId ? YELLOW : WHITE}
                strokeOpacity={id === lockedId ? 0.55 : 0.28}
                strokeWidth={1.4}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ) : null,
          )}
        </g>

        {/* ring labels along the right edge of the view */}
        {rings.map((r) => (
          <text
            key={`l-${r}`}
            x={world ? cx + 3 : cx + r * k * Math.sin(Math.min(half, 60) * (Math.PI / 180)) + 3}
            y={world ? cy - r * k - 2 : cy - r * k * Math.cos(Math.min(half, 60) * (Math.PI / 180)) - 2}
            className="fill-muted-foreground"
            fontSize={8}
          >
            {r} m
          </text>
        ))}
        {world && (
          <text x={cx} y={cy - R - 3} textAnchor="middle" className="fill-muted-foreground" fontSize={8}>
            0°
          </text>
        )}

        {/* targets */}
        <AnimatePresence>
          {placed.map(({ t, pos }) => {
            const { x, y } = px(pos)
            const locked = t.id === lockedId
            const hovered = t.id === hoveredId
            const color = iffColor(t) ?? (locked ? YELLOW : WHITE)
            const name = t.identity?.name ?? (t.identity?.face_id !== undefined ? `Face #${t.identity.face_id}` : null)
            const label = compact && !large ? `#${t.id}` : name ? `${name}` : `${t.label} #${t.id}`
            return (
              <motion.g
                key={t.id}
                initial={{ x, y, opacity: 0, scale: 0.3 }}
                animate={{ x, y, opacity: 1, scale: hovered ? 1.3 : 1 }}
                exit={{ opacity: 0, scale: 0.3 }}
                transition={{ x: dotSpring, y: dotSpring, default: spring }}
                className="pointer-events-auto cursor-pointer"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation()
                  onLock(locked ? null : t.id)
                }}
              >
                <title>{`${t.label} #${t.id}${t.distance !== undefined ? ` · ${t.distance.toFixed(1)} m` : ""}${locked ? " · locked" : ""}`}</title>
                <circle r={9} fill="transparent" />
                {locked && (
                  <motion.circle
                    r={6}
                    fill="none"
                    stroke={YELLOW}
                    animate={{ r: [5, 12], opacity: [0.9, 0] }}
                    transition={{ duration: 1.3, repeat: Infinity, ease: "easeOut" }}
                  />
                )}
                <circle r={locked ? 4.2 : 3.4} fill={color} stroke={ANTHRACITE} strokeWidth={1.2} />
                {t.ground_src === "depth" && <circle r={1.1} fill={ANTHRACITE} />}
                <text
                  x={7}
                  y={3}
                  fontSize={9}
                  fill={locked ? YELLOW : WHITE}
                  fillOpacity={locked || hovered ? 1 : 0.8}
                  stroke={ANTHRACITE}
                  strokeWidth={2.5}
                  paintOrder="stroke"
                  className="tabular-nums"
                >
                  {label}
                </text>
              </motion.g>
            )
          })}
        </AnimatePresence>

        {/* camera */}
        <g transform={`translate(${cx} ${cy})`}>
          <circle r={3} fill={YELLOW} />
          <circle r={6} fill="none" stroke={YELLOW} strokeOpacity={0.35} />
        </g>
      </motion.svg>
      <div className="mt-1 flex items-center justify-between gap-3 px-0.5 text-[10px] text-muted-foreground">
        <span className="tabular-nums">{placed.length ? `${placed.length} TGT` : "No Targets"}</span>
        <span className="truncate" title={depth?.note ?? "Distances are estimated from a single camera."}>
          Single-camera estimate
        </span>
        <span className="tabular-nums">
          {world ? `${Math.round((((pan % 360) + 540) % 360) - 180)}° · ` : ""}
          {range} m
        </span>
      </div>
    </motion.div>
  )
}
