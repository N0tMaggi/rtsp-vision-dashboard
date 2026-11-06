import { motion } from "motion/react"
import type { Camera } from "@/lib/types"
import { cn } from "@/lib/utils"

interface Props {
  cameras: Camera[]
  activeId: string | null
  onSelect: (id: string) => void
}

export function CameraSwitcher({ cameras, activeId, onSelect }: Props) {
  return (
    <nav
      aria-label="Kamera wählen"
      className="flex w-fit max-w-full items-center gap-0.5 overflow-x-auto rounded-full border border-border/70 bg-background/70 p-1 backdrop-blur-md"
    >
      {cameras.map((cam) => {
        const active = cam.id === activeId
        return (
          <button
            key={cam.id}
            type="button"
            onClick={() => onSelect(cam.id)}
            aria-current={active}
            className={cn(
              "relative shrink-0 rounded-full px-3.5 py-1.5 text-[13px] transition-colors duration-300 outline-none focus-visible:ring-2 focus-visible:ring-ring",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId="camera-indicator"
                className="absolute inset-0 rounded-full bg-primary"
                transition={{ type: "spring", stiffness: 420, damping: 34 }}
              />
            )}
            <span className="relative flex items-center gap-2">
              {!cam.has_stream && <span className="size-1.5 rounded-full bg-current opacity-50" aria-label="nicht konfiguriert" />}
              {cam.label}
            </span>
          </button>
        )
      })}
    </nav>
  )
}
