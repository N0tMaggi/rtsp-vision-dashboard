import { useEffect, useRef, useState, type ReactNode } from "react"
import { motion } from "motion/react"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

/** Small shared controls for the ESP menu panels (auto-track, crosshair). */

export function PanelToggle({
  id,
  label,
  on,
  onChange,
  disabled,
  icon,
}: {
  id: string
  label: ReactNode
  on: boolean
  onChange: (v: boolean) => void
  disabled?: boolean
  icon?: ReactNode
}) {
  return (
    <label
      htmlFor={id}
      className={cn(
        "-mx-2 flex items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors duration-150",
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer hover:bg-accent/70",
      )}
    >
      <span className={cn("flex items-center gap-2 transition-colors duration-200", on ? "text-foreground" : "text-muted-foreground")}>
        {icon}
        {label}
      </span>
      <Switch id={id} checked={on} onCheckedChange={onChange} disabled={disabled} />
    </label>
  )
}

export function PanelSegments<T extends string>({
  label,
  id,
  options,
  value,
  onChange,
  disabled,
}: {
  label: string
  id: string
  options: ReadonlyArray<{ value: T; label: string }>
  value: T
  onChange: (v: T) => void
  disabled?: boolean
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn("grid gap-1 rounded-lg bg-background/60 p-1", disabled && "pointer-events-none opacity-50")}
      style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
    >
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={cn(
              "relative rounded-md py-1.5 text-xs transition-colors duration-200",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId={`${id}-indicator`}
                className="absolute inset-0 rounded-md bg-primary"
                transition={{ type: "spring", stiffness: 450, damping: 34 }}
              />
            )}
            <span className="relative">{o.label}</span>
          </button>
        )
      })}
    </div>
  )
}

/**
 * Slider with a local value while dragging. `onChange` fires debounced while
 * dragging (debounceMs, 0 = every step) and once more on release.
 */
export function PanelRange({
  label,
  min,
  max,
  step,
  value,
  format,
  onChange,
  onDrag,
  debounceMs = 250,
  disabled,
}: {
  label: string
  min: number
  max: number
  step: number
  value: number
  format: (v: number) => string
  onChange: (v: number) => void
  /** Live value while the thumb moves; null after release. */
  onDrag?: (v: number | null) => void
  debounceMs?: number
  disabled?: boolean
}) {
  const [local, setLocal] = useState(value)
  const dragging = useRef(false)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => {
    if (!dragging.current) setLocal(value)
  }, [value])
  useEffect(() => () => window.clearTimeout(timer.current), [])
  return (
    <div className={cn("space-y-2.5 py-1.5", disabled && "opacity-50")}>
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="tabular-nums">{format(local)}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[local]}
        disabled={disabled}
        aria-label={label}
        onValueChange={([v]) => {
          dragging.current = true
          setLocal(v)
          onDrag?.(v)
          window.clearTimeout(timer.current)
          if (debounceMs <= 0) onChange(v)
          else timer.current = window.setTimeout(() => onChange(v), debounceMs)
        }}
        onValueCommit={([v]) => {
          dragging.current = false
          window.clearTimeout(timer.current)
          onChange(v)
          onDrag?.(null)
        }}
      />
    </div>
  )
}
