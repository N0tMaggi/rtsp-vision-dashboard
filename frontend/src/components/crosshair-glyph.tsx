import { motion, useReducedMotion } from "motion/react"
import type { CrosshairLook } from "@/lib/crosshair"

type GlyphLook = Pick<CrosshairLook, "style" | "color" | "size" | "thickness" | "gap" | "outline" | "opacity" | "rotation" | "spin" | "pulse">

const OUTLINE = "rgba(17,18,20,0.78)"

/** Paths of one crosshair style, centered on (0, 0), in screen px. Chevron points down at the center. */
function shapes(l: GlyphLook): { strokes: string; fills: { cx: number; cy: number; r: number }[]; rings: number[] } {
  const r = l.size / 2
  const g = Math.min(l.gap, r - 1)
  const t = l.thickness
  const dot = Math.max(1.2, t * 0.9)
  const cross = (from: number, to: number, top = true) =>
    [`M${from} 0H${to}`, `M${-from} 0H${-to}`, `M0 ${from}V${to}`, top ? `M0 ${-from}V${-to}` : ""].join("")
  switch (l.style) {
    case "cross":
      return { strokes: cross(g, r), fills: [], rings: [] }
    case "dot":
      return { strokes: "", fills: [{ cx: 0, cy: 0, r: Math.max(1.5, r / 2) }], rings: [] }
    case "circle":
      return { strokes: "", fills: [], rings: [Math.max(2, r - t / 2)] }
    case "cross-dot":
      return { strokes: cross(g + dot, r), fills: [{ cx: 0, cy: 0, r: dot }], rings: [] }
    case "t":
      return { strokes: cross(g + dot, r, false), fills: [{ cx: 0, cy: 0, r: dot }], rings: [] }
    case "brackets": {
      const a = Math.max(3, r * 0.42)
      const c = [
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
      ]
        .map(([sx, sy]) => `M${sx * r} ${sy * (r - a)}V${sy * r}H${sx * (r - a)}`)
        .join("")
      return { strokes: c, fills: [{ cx: 0, cy: 0, r: dot * 0.8 }], rings: [] }
    }
    case "chevron": {
      // two stacked chevrons above the point, the lower tip at the gap
      const w = r * 0.62
      const h = r * 0.42
      const tip = -g - t / 2
      const v = (y: number) => `M${-w} ${y - h}L0 ${y}L${w} ${y - h}`
      return { strokes: v(tip) + v(tip - h - t * 1.6), fills: [], rings: [] }
    }
    case "sniper": {
      const ring = r * 0.82
      const ticks = [0.32, 0.56]
        .flatMap((f) => {
          const d = ring * f
          const k = Math.max(2, r * 0.07)
          return [`M${d} ${-k}V${k}`, `M${-d} ${-k}V${k}`, `M${-k} ${d}H${k}`, `M${-k} ${-d}H${k}`]
        })
        .join("")
      return { strokes: cross(g, r) + ticks, fills: [{ cx: 0, cy: 0, r: Math.max(1, t * 0.7) }], rings: [ring] }
    }
  }
}

/**
 * One crosshair as crisp SVG (1 unit = 1 screen px, so it stays sharp at any stage zoom).
 * Spin and pulse respect prefers-reduced-motion.
 */
export function CrosshairGlyph({ look, className }: { look: GlyphLook; className?: string }) {
  const reduce = useReducedMotion()
  const { strokes, fills, rings } = shapes(look)
  const pad = look.thickness + 3
  const half = look.size / 2 + pad
  const box = half * 2
  const t = look.thickness
  const draw = (stroke: string, width: number, fill: string) => (
    <>
      {strokes && <path d={strokes} fill="none" stroke={stroke} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round" />}
      {rings.map((rr) => (
        <circle key={rr} r={rr} fill="none" stroke={stroke} strokeWidth={width} />
      ))}
      {fills.map((f) => (
        <circle key={`${f.cx}:${f.cy}`} cx={f.cx} cy={f.cy} r={f.r + (width - t) / 2} fill={fill} />
      ))}
    </>
  )
  const spin = look.spin && !reduce
  const pulse = look.pulse && !reduce
  return (
    <motion.div
      className={className}
      style={{ width: box, height: box, opacity: look.opacity }}
      animate={pulse ? { scale: [1, 1.18, 1], opacity: [look.opacity, look.opacity * 0.55, look.opacity] } : { scale: 1, opacity: look.opacity }}
      transition={pulse ? { duration: 1.3, repeat: Infinity, ease: "easeInOut" } : { duration: 0.2 }}
    >
      <motion.svg
        width={box}
        height={box}
        viewBox={`${-half} ${-half} ${box} ${box}`}
        className="block overflow-visible"
        animate={spin ? { rotate: 360 } : { rotate: 0 }}
        transition={spin ? { duration: 3.2, repeat: Infinity, ease: "linear" } : { duration: 0.3 }}
        aria-hidden
      >
        <g transform={`rotate(${look.rotation})`}>
          {look.outline && draw(OUTLINE, t + 2.5, OUTLINE)}
          {draw(look.color, t, look.color)}
        </g>
      </motion.svg>
    </motion.div>
  )
}
