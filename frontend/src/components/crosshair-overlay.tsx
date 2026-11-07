import { useEffect, useMemo } from "react"
import { AnimatePresence, motion, useSpring, useTransform } from "motion/react"
import { AIM_LABEL, isMoving, STATE_META, useAutoTrack, useDeadzonePreview } from "@/lib/autotrack"
import { aimPoint, useCrosshair, type CrosshairSettings } from "@/lib/crosshair"
import type { AimKey, Target } from "@/lib/types"
import { CrosshairGlyph } from "./crosshair-glyph"
import type { ViewLike } from "./stage-visuals"

interface Props {
  view: ViewLike
  width: number
  height: number
  /** Visible targets (view filter applied). */
  targets: Target[]
  lockedId: number | null
}

const PTZ_LIVE = new Set(["searching", "tracking", "holding", "lost"])
const fmtMeters = (m?: number) => (m === undefined ? null : m < 10 ? `${m.toFixed(1)} m` : `${Math.round(m)} m`)

/**
 * Crosshairs on the locked and the PTZ-tracked target (optionally on every person),
 * plus the auto-track framing window and error vector. Screen space, following the
 * stage zoom / pan like the target info card.
 */
export function CrosshairOverlay({ view, width: W, height: H, targets, lockedId }: Props) {
  const { settings } = useCrosshair()
  const at = useAutoTrack()
  const preview = useDeadzonePreview()
  const cfg = at.autotrack?.config
  const st = at.autotrack?.status
  const ptzOn = Boolean(cfg?.enabled && st && PTZ_LIVE.has(st.state))
  const ptzId = ptzOn ? (st?.target_id ?? null) : null
  const aim: AimKey = settings.aim === "autotrack" ? (cfg?.aim ?? "chest") : settings.aim

  const marks = useMemo(() => {
    if (!settings.enabled) return []
    const out: { t: Target; primary: boolean; ptz: boolean }[] = []
    for (const t of targets) {
      const isLocked = t.id === lockedId
      const isPtz = t.id === ptzId
      if (isLocked || isPtz) out.push({ t, primary: true, ptz: isPtz })
      else if (settings.allPersons && t.label.toLowerCase() === "person") out.push({ t, primary: false, ptz: false })
    }
    return out
  }, [targets, lockedId, ptzId, settings.enabled, settings.allPersons])

  if (W <= 0 || H <= 0) return null
  const showWindow = preview !== null || (ptzOn && cfg)
  return (
    <div className="pointer-events-none absolute inset-0 z-[1]" aria-hidden>
      <AnimatePresence>
        {showWindow && cfg && (
          <FramingLayer
            key="framing"
            view={view}
            width={W}
            height={H}
            deadzone={preview ?? cfg.deadzone}
            previewing={preview !== null}
            moving={isMoving(at.autotrack)}
            aimPoint={ptzOn ? (st?.aim_point ?? null) : null}
            color={st ? STATE_META[st.state].color : "#ffd21f"}
          />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {marks.map(({ t, primary, ptz }) => (
          <Marker key={t.id} target={t} aim={aim} primary={primary} ptz={ptz} settings={settings} view={view} width={W} height={H} />
        ))}
      </AnimatePresence>
    </div>
  )
}

const pointSpring = { stiffness: 320, damping: 34, mass: 0.55 }

function Marker({
  target,
  aim,
  primary,
  ptz,
  settings,
  view,
  width: W,
  height: H,
}: {
  target: Target
  aim: AimKey
  primary: boolean
  ptz: boolean
  settings: CrosshairSettings
  view: ViewLike
  width: number
  height: number
}) {
  const [pu, pv] = aimPoint(target, aim)
  const u = useSpring(pu, pointSpring)
  const v = useSpring(pv, pointSpring)
  useEffect(() => {
    u.set(pu)
    v.set(pv)
  }, [pu, pv, u, v])

  const x = useTransform(() => view.x.get() + u.get() * W * view.s.get())
  const y = useTransform(() => view.y.get() + v.get() * H * view.s.get())
  const onScreen = useTransform((): number => {
    const px = x.get()
    const py = y.get()
    return px >= -4 && px <= W + 4 && py >= -4 && py <= H + 4 ? 1 : 0
  })
  const visible = useSpring(onScreen, { stiffness: 260, damping: 30 })

  const look = primary ? settings : { ...settings, opacity: settings.opacity * 0.5, spin: false, pulse: false }
  const half = settings.size / 2 + settings.thickness + 3
  const isPerson = target.label.toLowerCase() === "person"
  const point = isPerson ? AIM_LABEL[aim] : AIM_LABEL.center
  const name = target.identity?.name
  const dist = fmtMeters(target.distance)
  const label = [name, point.toUpperCase(), dist].filter(Boolean).join(" · ")

  return (
    <motion.div className="absolute left-0 top-0" style={{ x, y, opacity: visible }}>
      <motion.div
        initial={{ opacity: 0, scale: 1.6 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.6 }}
        transition={{ type: "spring", stiffness: 380, damping: 26 }}
        className="relative"
      >
        <div style={{ position: "absolute", left: -half, top: -half }}>
          <CrosshairGlyph look={look} />
        </div>
        {settings.label && primary && (
          <div
            className="absolute flex items-center gap-1 whitespace-nowrap rounded-full bg-background/75 px-2 py-0.5 text-[10px] font-medium tracking-wider tabular-nums backdrop-blur-md"
            style={{ left: 0, top: half + 2, transform: "translateX(-50%)", color: settings.color }}
          >
            {ptz && <span className="rounded-sm bg-current px-1 text-[9px] leading-tight"><span className="text-background">PTZ</span></span>}
            {label}
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

function FramingLayer({
  view,
  width: W,
  height: H,
  deadzone,
  previewing,
  moving,
  aimPoint: ap,
  color,
}: {
  view: ViewLike
  width: number
  height: number
  deadzone: number
  previewing: boolean
  moving: boolean
  aimPoint: [number, number] | null
  color: string
}) {
  const d = Math.min(0.45, Math.max(0.03, deadzone))
  const strength = previewing ? 0.85 : moving ? 0.5 : 0.2
  const stroke = previewing ? "#ffd21f" : "#f4f4f2"
  const corner = Math.min(0.04, d * 0.6)
  const corners = [
    [0.5 - d, 0.5 - d, 1, 1],
    [0.5 + d, 0.5 - d, -1, 1],
    [0.5 - d, 0.5 + d, 1, -1],
    [0.5 + d, 0.5 + d, -1, -1],
  ]
    .map(([cx, cy, sx, sy]) => `M${cx} ${cy + sy * corner}V${cy}H${cx + sx * corner}`)
    .join("")
  const lineT = { type: "spring", stiffness: 220, damping: 30 } as const
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      className="absolute left-0 top-0 origin-top-left"
      style={{ width: W, height: H, x: view.x, y: view.y, scale: view.s }}
    >
      <svg className="absolute inset-0 size-full overflow-visible" viewBox="0 0 1 1" preserveAspectRatio="none">
        <motion.rect
          initial={false}
          animate={{ x: 0.5 - d, y: 0.5 - d, width: 2 * d, height: 2 * d, strokeOpacity: strength * 0.6 }}
          transition={{ type: "spring", stiffness: 300, damping: 34 }}
          fill={previewing ? "rgba(255,210,31,0.05)" : "none"}
          stroke={stroke}
          strokeWidth={1}
          strokeDasharray="4 6"
          vectorEffect="non-scaling-stroke"
        />
        <motion.path
          d={corners}
          initial={false}
          animate={{ strokeOpacity: Math.min(1, strength + 0.15) }}
          fill="none"
          stroke={stroke}
          strokeWidth={1.5}
          strokeLinecap="round"
          vectorEffect="non-scaling-stroke"
        />
        <path d="M0.49 0.5H0.51M0.5 0.49V0.51" stroke="#f4f4f2" strokeOpacity={0.35} strokeWidth={1} vectorEffect="non-scaling-stroke" />
        {ap && (
          <motion.line
              x1={0.5}
              y1={0.5}
              initial={false}
              animate={{ x2: ap[0], y2: ap[1], strokeOpacity: moving ? 0.6 : 0.3 }}
              transition={lineT}
              stroke={color}
              strokeWidth={1}
              strokeDasharray="2 4"
              vectorEffect="non-scaling-stroke"
            />
        )}
      </svg>
    </motion.div>
  )
}
