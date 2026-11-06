import { motion } from "motion/react"
import { LayoutGrid, Square } from "lucide-react"
import type { ViewMode } from "@/hooks/use-view-mode"
import { cn } from "@/lib/utils"

const OPTIONS = [
  { value: "single", label: "Single view", icon: Square },
  { value: "grid", label: "Grid view (G)", icon: LayoutGrid },
] as const

export function ViewToggle({ mode, onChange }: { mode: ViewMode; onChange: (m: ViewMode) => void }) {
  return (
    <div
      role="radiogroup"
      aria-label="View"
      className="flex shrink-0 items-center gap-0.5 rounded-full border border-border/70 bg-background/70 p-1 backdrop-blur-md"
    >
      {OPTIONS.map(({ value, label, icon: Icon }) => {
        const active = value === mode
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={label}
            title={label}
            onClick={() => onChange(value)}
            className={cn(
              "relative grid size-7 place-items-center rounded-full transition-colors duration-200",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="view-mode-indicator"
                className="absolute inset-0 rounded-full bg-primary"
                transition={{ type: "spring", stiffness: 420, damping: 32 }}
              />
            )}
            <Icon className="relative size-3.5" />
          </button>
        )
      })}
    </div>
  )
}
