import { useId, useState } from "react"
import { motion } from "motion/react"
import { AIM_KEYS, AIM_LABEL } from "@/lib/autotrack"
import type { AimKey } from "@/lib/types"
import { cn } from "@/lib/utils"

/** Point positions on the silhouette (viewBox 0 0 100 200). */
const POINTS: Record<AimKey, [number, number]> = {
  head: [50, 10],
  face: [50, 21],
  neck: [50, 33],
  chest: [50, 54],
  belly: [50, 80],
  center: [50, 96],
  groin: [50, 109],
  feet: [50, 189],
}

const BODY =
  "M44 29V34C36 35 29 37 27 43L20 95C19.5 99.5 23 101.5 25.5 98.5L31 58L33 102L35 187C35 193 45 193 45 187L48 112H52L55 187C55 193 65 193 65 187L67 102L69 58L74.5 98.5C77 101.5 80.5 99.5 80 95L73 43C71 37 64 35 56 34V29Z"

const spring = { type: "spring", stiffness: 520, damping: 32 } as const

interface Props {
  value: AimKey | null
  onChange: (k: AimKey) => void
  disabled?: boolean
  /** Accent for the selected point (defaults to the theme yellow). */
  color?: string
  className?: string
}

/** Clickable body silhouette for choosing an aim point; the list on the right does the same. */
export function BodyPointPicker({ value, onChange, disabled, color = "#ffd21f", className }: Props) {
  const [hover, setHover] = useState<AimKey | null>(null)
  const uid = useId()
  return (
    <div className={cn("flex items-stretch gap-3", disabled && "pointer-events-none opacity-50", className)}>
      <svg viewBox="0 0 100 200" className="h-[176px] w-[88px] shrink-0" role="group" aria-label="Body point">
        <rect x="14" y="2" width="72" height="194" rx="3" fill="none" stroke="rgba(244,244,242,0.14)" strokeDasharray="3 3" />
        <circle cx="50" cy="16" r="12" fill="rgba(244,244,242,0.07)" stroke="rgba(244,244,242,0.28)" strokeWidth="1" />
        <path d={BODY} fill="rgba(244,244,242,0.07)" stroke="rgba(244,244,242,0.28)" strokeWidth="1" strokeLinejoin="round" />
        {AIM_KEYS.map((k) => {
          const [x, y] = POINTS[k]
          const active = k === value
          const hot = k === hover
          return (
            <g
              key={k}
              role="radio"
              aria-checked={active}
              aria-label={AIM_LABEL[k]}
              tabIndex={-1}
              className="cursor-pointer"
              onPointerEnter={() => setHover(k)}
              onPointerLeave={() => setHover((h) => (h === k ? null : h))}
              onClick={() => onChange(k)}
            >
              <circle cx={x} cy={y} r="6.5" fill="transparent" />
              {active && (
                <motion.circle
                  cx={x}
                  cy={y}
                  initial={{ r: 3, opacity: 0.9 }}
                  animate={{ r: [4.5, 8], opacity: [0.8, 0] }}
                  transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
                  fill="none"
                  stroke={color}
                  strokeWidth="1"
                />
              )}
              {k === "center" ? (
                <motion.path
                  d={`M${x} ${y - 4}L${x + 4} ${y}L${x} ${y + 4}L${x - 4} ${y}Z`}
                  animate={{ scale: active || hot ? 1.2 : 1 }}
                  transition={spring}
                  style={{ transformOrigin: `${x}px ${y}px`, transformBox: "view-box" }}
                  fill={active ? color : "rgba(34,36,39,0.9)"}
                  stroke={active ? color : hot ? "#f4f4f2" : "rgba(244,244,242,0.6)"}
                  strokeWidth="1"
                />
              ) : (
                <motion.circle
                  cx={x}
                  cy={y}
                  animate={{ r: active ? 3.6 : hot ? 3.4 : 2.6 }}
                  transition={spring}
                  fill={active ? color : hot ? "#f4f4f2" : "rgba(244,244,242,0.55)"}
                />
              )}
            </g>
          )
        })}
      </svg>
      <div role="radiogroup" aria-label="Aim point" className="grid min-w-0 flex-1 grid-cols-1 content-between gap-0.5">
        {AIM_KEYS.map((k) => {
          const active = k === value
          return (
            <button
              key={k}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={disabled}
              onPointerEnter={() => setHover(k)}
              onPointerLeave={() => setHover((h) => (h === k ? null : h))}
              onFocus={() => setHover(k)}
              onBlur={() => setHover(null)}
              onClick={() => onChange(k)}
              className={cn(
                "relative flex items-center gap-2 rounded-full px-2.5 py-[3px] text-left text-xs transition-colors duration-200",
                active ? "text-primary-foreground" : hover === k ? "bg-accent text-foreground" : "text-muted-foreground",
              )}
            >
              {active && <motion.span layoutId={`aim-pill-${uid}`} className="absolute inset-0 rounded-full" style={{ background: color }} transition={spring} />}
              <span className="relative">{AIM_LABEL[k]}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}
