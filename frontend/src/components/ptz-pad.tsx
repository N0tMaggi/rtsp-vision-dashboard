import { useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, House, Minus, Plus } from "lucide-react"
import type { PtzControls, Vector } from "@/hooks/use-ptz"
import { cn } from "@/lib/utils"

const spring = { type: "spring", stiffness: 380, damping: 30 } as const

/** On-screen D-pad for continuous PTZ moves (press and hold). */
export function PtzPad({ ptz, rightInset = 24 }: { ptz: PtzControls; rightInset?: number }) {
  const [held, setHeld] = useState<string | null>(null)
  const caps = ptz.status?.caps ?? {}
  const zoom = Boolean(caps.zoom)

  const press = (id: string, v: Vector) => ({
    onPointerDown: (e: React.PointerEvent) => {
      e.currentTarget.setPointerCapture(e.pointerId)
      setHeld(id)
      ptz.hold(v)
    },
    onPointerUp: () => {
      setHeld(null)
      ptz.hold({ x: 0, y: 0, zoom: 0 })
    },
    onPointerCancel: () => {
      setHeld(null)
      ptz.hold({ x: 0, y: 0, zoom: 0 })
    },
  })

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 12 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.9, y: 12 }}
      transition={spring}
      className="absolute bottom-[80px] z-20 flex items-end gap-2 transition-[right] duration-500"
      style={{ right: rightInset + 12 }}
      onPointerDown={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      {zoom && (
        <div className="flex flex-col overflow-hidden rounded-full border border-border/70 bg-background/75 backdrop-blur-md">
          <PadButton id="zin" held={held} label="Zoom in (Page Up)" {...press("zin", { x: 0, y: 0, zoom: 1 })}>
            <Plus className="size-4" />
          </PadButton>
          <PadButton id="zout" held={held} label="Zoom out (Page Down)" {...press("zout", { x: 0, y: 0, zoom: -1 })}>
            <Minus className="size-4" />
          </PadButton>
        </div>
      )}
      <div
        className="relative grid size-[132px] grid-cols-3 grid-rows-3 place-items-center rounded-full border border-border/70 bg-background/75 shadow-[0_18px_50px_-20px_rgba(0,0,0,0.7)] backdrop-blur-md"
        aria-label="Camera movement"
        role="group"
      >
        <span className="col-start-2 row-start-1">
          <PadButton id="up" held={held} label="Tilt up (↑)" {...press("up", { x: 0, y: 1, zoom: 0 })}>
            <ChevronUp className="size-5" />
          </PadButton>
        </span>
        <span className="col-start-1 row-start-2">
          <PadButton id="left" held={held} label="Pan left (←)" {...press("left", { x: -1, y: 0, zoom: 0 })}>
            <ChevronLeft className="size-5" />
          </PadButton>
        </span>
        <span className="col-start-2 row-start-2">
          {caps.home ? (
            <button
              type="button"
              onClick={() => ptz.home()}
              aria-label="Home position"
              title="Home position"
              className="grid size-9 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <House className="size-4" />
            </button>
          ) : (
            <AnimatePresence>
              <motion.span
                key={ptz.status?.moving ? "moving" : "still"}
                className={cn("block size-2 rounded-full", ptz.status?.moving || held ? "bg-primary" : "bg-muted-foreground/50")}
                initial={{ scale: 0.6 }}
                animate={{ scale: 1 }}
              />
            </AnimatePresence>
          )}
        </span>
        <span className="col-start-3 row-start-2">
          <PadButton id="right" held={held} label="Pan right (→)" {...press("right", { x: 1, y: 0, zoom: 0 })}>
            <ChevronRight className="size-5" />
          </PadButton>
        </span>
        <span className="col-start-2 row-start-3">
          <PadButton id="down" held={held} label="Tilt down (↓)" {...press("down", { x: 0, y: -1, zoom: 0 })}>
            <ChevronDown className="size-5" />
          </PadButton>
        </span>
      </div>
    </motion.div>
  )
}

function PadButton({
  id,
  held,
  label,
  children,
  ...handlers
}: {
  id: string
  held: string | null
  label: string
  children: React.ReactNode
  onPointerDown: (e: React.PointerEvent) => void
  onPointerUp: () => void
  onPointerCancel: () => void
}) {
  const active = held === id
  const ref = useRef<HTMLButtonElement>(null)
  return (
    <button
      ref={ref}
      type="button"
      aria-label={label}
      title={label}
      {...handlers}
      onContextMenu={(e) => e.preventDefault()}
      className={cn(
        "relative grid size-10 touch-none place-items-center rounded-full transition-colors duration-150",
        active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
      )}
    >
      {active && <motion.span layoutId="ptz-pad-held" className="absolute inset-0 rounded-full bg-primary" transition={spring} />}
      <span className="relative">{children}</span>
    </button>
  )
}
