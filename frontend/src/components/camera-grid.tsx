import { useEffect, useRef, useState } from "react"
import { motion } from "motion/react"
import { opsApi, snapTileWidth, type CameraOverview } from "@/lib/ops-api"
import type { Camera } from "@/lib/types"
import { cn } from "@/lib/utils"

interface Props {
  cameras: Camera[]
  activeId: string | null
  overview: Map<string, CameraOverview>
  onOpen: (id: string) => void
  rightInset?: number
}

const spring = { type: "spring", stiffness: 300, damping: 34 } as const

/**
 * All cameras at once (2 columns up to 4 cameras, 3 above). Every tile pulls
 * a downscaled stream; clicking (or Enter) opens it, animating into the
 * single view through the shared `cam-<id>` layoutId.
 */
export function CameraGrid({ cameras, activeId, overview, onOpen, rightInset = 24 }: Props) {
  const cols = cameras.length <= 1 ? 1 : cameras.length <= 4 ? 2 : 3
  const rows = Math.ceil(cameras.length / cols)
  return (
    <div
      className="absolute inset-0 overflow-y-auto bg-[#1b1c1f] pb-16 pl-6 pt-20 transition-[padding] duration-500"
      style={{ paddingRight: rightInset }}
    >
      <ul
        className="grid h-full min-h-[320px] gap-3"
        style={{
          gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
          gridTemplateRows: rows <= 3 ? `repeat(${rows}, minmax(140px, 1fr))` : undefined,
        }}
        aria-label="Camera grid"
      >
        {cameras.map((cam, i) => (
          <li key={cam.id} className="min-h-0">
            <Tile
              camera={cam}
              index={i}
              info={overview.get(cam.id)}
              active={cam.id === activeId}
              onOpen={() => onOpen(cam.id)}
            />
          </li>
        ))}
      </ul>
    </div>
  )
}

function Tile({
  camera,
  index,
  info,
  active,
  onOpen,
}: {
  camera: Camera
  index: number
  info: CameraOverview | undefined
  active: boolean
  onOpen: () => void
}) {
  const ref = useRef<HTMLButtonElement>(null)
  const [width, setWidth] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [nonce] = useState(() => Date.now())

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => {
      // Only the bucket matters; a new bucket means a new (shared) stream.
      setWidth(snapTileWidth(e.contentRect.width * Math.min(2, window.devicePixelRatio || 1)))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const live = Boolean(info?.live) && loaded
  const error = info?.stream_error
  return (
    <motion.button
      ref={ref}
      type="button"
      layoutId={`cam-${camera.id}`}
      transition={spring}
      onClick={onOpen}
      aria-label={`Open ${camera.label}${index < 9 ? ` (${index + 1})` : ""}`}
      className={cn(
        "group relative block size-full overflow-hidden rounded-lg border bg-[#16171a] text-left outline-none",
        "transition-[border-color,box-shadow] duration-300 focus-visible:ring-2 focus-visible:ring-ring",
        active ? "border-primary/70 shadow-[0_0_0_1px_rgba(255,210,31,0.25)]" : "border-border/70 hover:border-primary/50",
      )}
    >
      {camera.has_stream && width > 0 && (
        <motion.img
          key={width}
          src={opsApi.tileStreamUrl(camera.id, width, nonce)}
          alt={`Live view ${camera.label}`}
          draggable={false}
          onLoad={() => setLoaded(true)}
          initial={{ opacity: 0 }}
          animate={{ opacity: loaded ? 1 : 0 }}
          transition={{ duration: 0.5 }}
          className="absolute inset-0 size-full object-contain transition-transform duration-500 ease-[var(--ease-hud)] group-hover:scale-[1.015]"
        />
      )}

      {!loaded && (
        <span className="absolute inset-0 grid place-items-center px-4 text-center text-xs text-muted-foreground">
          {!camera.has_stream ? "No RTSP stream configured" : error || "Connecting…"}
        </span>
      )}

      {/* minimal HUD */}
      <span className="pointer-events-none absolute inset-x-0 top-0 flex items-center gap-2 bg-gradient-to-b from-black/60 to-transparent px-3 pb-6 pt-2.5 text-[12px]">
        <span className="relative flex size-2">
          {live && <span className="absolute inset-0 animate-ping rounded-full bg-primary opacity-60" />}
          <span className={cn("relative size-2 rounded-full", live ? "bg-primary" : error ? "bg-destructive" : "bg-muted-foreground/50")} />
        </span>
        <span className="truncate font-medium">{camera.label}</span>
        {index < 9 && (
          <kbd className="ml-auto rounded border border-white/20 px-1.5 text-[10px] text-white/60">{index + 1}</kbd>
        )}
      </span>
      <span className="pointer-events-none absolute bottom-2 left-3 flex items-center gap-3 rounded-full bg-black/45 px-2.5 py-1 text-[11px] tabular-nums backdrop-blur-sm">
        <span>
          {(info?.fps ?? 0).toFixed(1)}
          <span className="ml-1 text-white/55">fps</span>
        </span>
        <span className={cn(info?.yolo_enabled && (info?.num_detections ?? 0) > 0 ? "text-primary" : "")}>
          {info?.yolo_enabled ? info.num_detections : "–"}
          <span className="ml-1 text-white/55">targets</span>
        </span>
      </span>
    </motion.button>
  )
}
