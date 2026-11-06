import { AnimatePresence, motion } from "motion/react"
import { Crosshair, Minus, MousePointer2, Plus, RotateCcw, Slash, SquareDashed, X } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { Target } from "@/lib/types"
import { cn } from "@/lib/utils"

export type Tool = "select" | "zone" | "tripwire"

interface Props {
  tool: Tool
  onToolChange: (t: Tool) => void
  zoom: number
  onZoom: (factor: number) => void
  onReset: () => void
  aiOn: boolean
  locked: Target | null
  lockedId: number | null
  hovered: Target | null
  follow: boolean
  onFollowChange: (f: boolean) => void
  onUnlock: () => void
}

const spring = { type: "spring", stiffness: 420, damping: 32 } as const

export function StageToolbar(p: Props) {
  let hint: string
  if (p.tool === "tripwire") hint = "Drag a line to create a tripwire. Esc cancels."
  else if (p.tool === "zone") hint = "Ziehe ein Rechteck, um eine Zone anzulegen. Esc bricht ab."
  else if (!p.aiOn) hint = "Schalte die Erkennung ein, um Objekte anzuklicken."
  else if (p.hovered) hint = `${p.hovered.label} #${p.hovered.id} · Klick markiert, Doppelklick zoomt`
  else hint = "Mausrad zoomt, Ziehen verschiebt, Doppelklick setzt zurück"

  return (
    <div
      className="absolute bottom-[80px] left-9 z-10 flex flex-col items-start gap-2"
      onPointerDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <AnimatePresence>
        {p.lockedId !== null && (
          <motion.div
            initial={{ opacity: 0, y: 8, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 8, scale: 0.96 }}
            transition={spring}
            className="flex items-center gap-1 rounded-full border border-primary/50 bg-background/80 py-1 pl-3 pr-1 text-[13px] backdrop-blur-md"
          >
            <span className="relative mr-1 flex size-2">
              <span className="absolute inset-0 animate-ping rounded-full bg-primary opacity-60" />
              <span className="relative size-2 rounded-full bg-primary" />
            </span>
            <span>{p.locked ? `${p.locked.label} #${p.locked.id}` : `Ziel #${p.lockedId}`}</span>
            <button
              type="button"
              onClick={() => p.onFollowChange(!p.follow)}
              aria-pressed={p.follow}
              className={cn(
                "ml-2 flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-colors duration-200",
                p.follow ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              <Crosshair className="size-3.5" /> Folgen
            </button>
            <button
              type="button"
              onClick={p.onUnlock}
              aria-label="Markierung lösen"
              className="rounded-full p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <X className="size-3.5" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-0.5 rounded-full border border-border/70 bg-background/70 p-1 backdrop-blur-md">
          <ToolButton active={p.tool === "select"} label="Auswählen (V)" onClick={() => p.onToolChange("select")}>
            <MousePointer2 className="size-4" />
          </ToolButton>
          <ToolButton active={p.tool === "zone"} label="Zone zeichnen (Z)" onClick={() => p.onToolChange(p.tool === "zone" ? "select" : "zone")}>
            <SquareDashed className="size-4" />
          </ToolButton>
          <ToolButton active={p.tool === "tripwire"} label="Draw Tripwire (T)" onClick={() => p.onToolChange(p.tool === "tripwire" ? "select" : "tripwire")}>
            <Slash className="size-4" />
          </ToolButton>
        </div>

        <div className="flex items-center rounded-full border border-border/70 bg-background/70 p-1 backdrop-blur-md">
          <IconButton label="Herauszoomen (−)" onClick={() => p.onZoom(1 / 1.4)} disabled={p.zoom <= 1.01}>
            <Minus className="size-3.5" />
          </IconButton>
          <span className="w-12 text-center text-xs tabular-nums">{p.zoom.toFixed(1)}×</span>
          <IconButton label="Hineinzoomen (+)" onClick={() => p.onZoom(1.4)} disabled={p.zoom >= 7.99}>
            <Plus className="size-3.5" />
          </IconButton>
          <AnimatePresence initial={false}>
            {p.zoom > 1.01 && (
              <motion.span initial={{ width: 0, opacity: 0 }} animate={{ width: "auto", opacity: 1 }} exit={{ width: 0, opacity: 0 }} className="overflow-hidden">
                <IconButton label="Zoom zurücksetzen (0)" onClick={p.onReset}>
                  <RotateCcw className="size-3.5" />
                </IconButton>
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        <AnimatePresence mode="wait">
          <motion.p
            key={hint}
            initial={{ opacity: 0, x: -4 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="hidden rounded-full bg-background/60 px-3 py-1 text-xs text-muted-foreground backdrop-blur-md lg:block"
          >
            {hint}
          </motion.p>
        </AnimatePresence>
      </div>
    </div>
  )
}

function ToolButton({ active, label, onClick, children }: { active: boolean; label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={label}
          aria-pressed={active}
          className={cn(
            "relative grid size-8 place-items-center rounded-full transition-colors duration-200",
            active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {active && <motion.span layoutId="tool-indicator" className="absolute inset-0 rounded-full bg-primary" transition={spring} />}
          <span className="relative">{children}</span>
        </button>
      </TooltipTrigger>
      <TooltipContent side="top">{label}</TooltipContent>
    </Tooltip>
  )
}

function IconButton({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="grid size-7 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:pointer-events-none disabled:opacity-30"
    >
      {children}
    </button>
  )
}
