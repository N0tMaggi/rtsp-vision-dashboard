import { motion } from "motion/react"
import { Film, Radio, X } from "lucide-react"
import type { Camera } from "@/lib/types"
import { cn } from "@/lib/utils"

interface Props {
  cameras: Camera[]
  activeId: string | null
  onSelect: (id: string) => void
  /** Close a video-analysis camera (file / URL source). */
  onClose?: (id: string) => void
}

export function CameraSwitcher({ cameras, activeId, onSelect, onClose }: Props) {
  return (
    <nav
      aria-label="Select camera"
      className="flex w-fit max-w-full items-center gap-0.5 overflow-x-auto rounded-full border border-border/70 bg-background/70 p-1 backdrop-blur-md"
    >
      {cameras.map((cam) => {
        const active = cam.id === activeId
        const video = cam.source_kind === "file" || cam.source_kind === "url"
        const Icon = cam.source_kind === "url" ? Radio : Film
        return (
          <motion.div
            key={cam.id}
            layout="position"
            initial={video ? { opacity: 0, scale: 0.85 } : false}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: "spring", stiffness: 420, damping: 34 }}
            className="relative flex shrink-0 items-center rounded-full"
          >
            {active && (
              <motion.span
                layoutId="camera-indicator"
                className="absolute inset-0 rounded-full bg-primary"
                transition={{ type: "spring", stiffness: 420, damping: 34 }}
              />
            )}
            <button
              type="button"
              onClick={() => onSelect(cam.id)}
              aria-current={active}
              title={video ? `${cam.label} (${cam.source_kind === "url" ? "stream URL" : "video"})` : undefined}
              className={cn(
                "relative rounded-full py-1.5 text-[13px] transition-colors duration-300 outline-none focus-visible:ring-2 focus-visible:ring-ring",
                video && onClose ? "pl-3 pr-1.5" : "px-3.5",
                active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="relative flex items-center gap-2">
                {!cam.has_stream && <span className="size-1.5 rounded-full bg-current opacity-50" aria-label="not configured" />}
                {video && <Icon className="size-3.5 shrink-0" aria-hidden />}
                <span className={cn(video && "max-w-40 truncate")}>{cam.label}</span>
              </span>
            </button>
            {video && onClose && (
              <button
                type="button"
                onClick={() => onClose(cam.id)}
                aria-label={`Close ${cam.label}`}
                title="Close video"
                className={cn(
                  "relative mr-1 grid size-5 place-items-center rounded-full transition-colors duration-200",
                  active ? "text-primary-foreground/70 hover:bg-primary-foreground/15 hover:text-primary-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                <X className="size-3" />
              </button>
            )}
          </motion.div>
        )
      })}
    </nav>
  )
}
