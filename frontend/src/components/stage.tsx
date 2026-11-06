import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { useViewTransform } from "@/hooks/use-view-transform"
import { api } from "@/lib/api"
import type { Camera, NBox, Target, TargetsResponse } from "@/lib/types"
import { cn } from "@/lib/utils"
import { StageToolbar, type Tool } from "./stage-toolbar"
import { StageVisuals } from "./stage-visuals"

type Phase = "idle" | "connecting" | "live" | "error"

interface Props {
  camera: Camera | undefined
  error: string | null
  aiOn: boolean
  targets: TargetsResponse | null
  lockedId: number | null
  tool: Tool
  onToolChange: (tool: Tool) => void
  follow: boolean
  onFollowChange: (follow: boolean) => void
  onLock: (id: number | null) => void
  onZoneDrawn: (box: NBox) => void
  /** Tripwire tool finished a line (normalized points). */
  onTripwireDrawn?: (a: [number, number], b: [number, number]) => void
  /** Space kept free on the right (e.g. for the open ESP menu), in px. */
  rightInset?: number
  /** Right-click on the picture (normalized frame coords); e.g. PTZ "center here". */
  onAimAt?: (u: number, v: number, clientX: number, clientY: number) => void
  /** Space kept free at the bottom (e.g. for the video playback bar), in px. */
  bottomInset?: number
}

// Free area between header (~72px) and telemetry bar (~44px); the video is fitted inside.
const INSET = { top: 80, bottom: 64, left: 24, right: 24 }

const ease = [0.22, 1, 0.36, 1] as const
const DRAG_THRESHOLD_PX = 4

function containRect(cw: number, ch: number, vw: number, vh: number, right: number, bottom = INSET.bottom) {
  const aw = Math.max(0, cw - INSET.left - right)
  const ah = Math.max(0, ch - INSET.top - bottom)
  if (!vw || !vh || !aw || !ah) return { left: INSET.left, top: INSET.top, width: aw, height: ah }
  const k = Math.min(aw / vw, ah / vh)
  const width = vw * k
  const height = vh * k
  return { left: INSET.left + (aw - width) / 2, top: INSET.top + (ah - height) / 2, width, height }
}

function hitTest(targets: Target[], u: number, v: number): Target | null {
  let best: Target | null = null
  let bestArea = Infinity
  for (const t of targets) {
    const [x1, y1, x2, y2] = t.box
    if (u < x1 || u > x2 || v < y1 || v > y2) continue
    const area = (x2 - x1) * (y2 - y1)
    if (area < bestArea) {
      best = t
      bestArea = area
    }
  }
  return best
}

type Gesture =
  | { kind: "none" }
  | { kind: "press"; startX: number; startY: number; lastX: number; lastY: number; panning: boolean }
  | { kind: "draw"; start: [number, number] }
  | { kind: "line"; start: [number, number] }

/**
 * Full-bleed camera view with digital zoom, click-to-lock and zone drawing.
 * The yellow corner brackets are the one loud element: wide while connecting,
 * snapping onto the frame once video arrives.
 */
export function Stage(props: Props) {
  const { camera, error, aiOn, targets, lockedId, tool, onToolChange, follow, onFollowChange, onLock, onZoneDrawn } = props
  const rightInset = Math.max(INSET.right, props.rightInset ?? INSET.right)
  const [loaded, setLoaded] = useState(false)
  const [nonce, setNonce] = useState(() => Date.now())
  const [shownId, setShownId] = useState(camera?.id)
  const [natural, setNatural] = useState({ w: 0, h: 0 })
  const [box, setBox] = useState({ w: 0, h: 0 })
  const [hoverId, setHoverId] = useState<number | null>(null)
  const [draft, setDraft] = useState<NBox | null>(null)
  const [lineDraft, setLineDraft] = useState<NBox | null>(null)
  const [panning, setPanning] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const gesture = useRef<Gesture>({ kind: "none" })

  // New camera -> new stream URL. Adjusted during render (not in an effect) so
  // the first mount opens the MJPEG stream once instead of twice.
  if (camera?.id !== shownId) {
    setShownId(camera?.id)
    setLoaded(false)
    setNonce(Date.now())
  }

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setBox({ w: entry.contentRect.width, h: entry.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const frameW = natural.w || targets?.frame.w || 0
  const frameH = natural.h || targets?.frame.h || 0
  const bottomInset = Math.max(INSET.bottom, props.bottomInset ?? INSET.bottom)
  const rect = useMemo(
    () => containRect(box.w, box.h, frameW, frameH, rightInset, bottomInset),
    [box.w, box.h, frameW, frameH, rightInset, bottomInset],
  )
  const view = useViewTransform(rect.width, rect.height)
  // First real layout appears instantly; later changes (menu open/close, resize) glide.
  const laidOut = useRef(false)
  const layoutTransition = laidOut.current ? { type: "spring" as const, stiffness: 260, damping: 32 } : { duration: 0 }
  useEffect(() => {
    if (rect.width > 0 && rect.height > 0) laidOut.current = true
  })
  const { reset, focus, zoomAt, panBy, toNormalized } = view

  // Layout or camera changed -> start unzoomed.
  useEffect(() => {
    reset()
  }, [rect.width, rect.height, camera?.id, reset])

  const list = useMemo(() => targets?.targets ?? [], [targets])
  const locked = list.find((t) => t.id === lockedId) ?? null

  // Follow the locked target like a digital PTZ.
  useEffect(() => {
    if (follow && locked) focus(locked.box)
  }, [follow, locked, focus])

  useEffect(() => {
    if (follow && lockedId === null) onFollowChange(false)
  }, [follow, lockedId, onFollowChange])

  const local = useCallback(
    (e: { clientX: number; clientY: number }) => {
      const b = containerRef.current!.getBoundingClientRect()
      return [e.clientX - b.left - rect.left, e.clientY - b.top - rect.top] as const
    },
    [rect.left, rect.top],
  )

  // Wheel zoom (non-passive so the page never scrolls).
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!loaded) return
      e.preventDefault()
      const [px, py] = local(e)
      onFollowChange(false)
      zoomAt(px, py, Math.exp(-e.deltaY * 0.0015))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [loaded, local, zoomAt, onFollowChange])

  const inside = (u: number, v: number) => u >= 0 && u <= 1 && v >= 0 && v <= 1

  const onPointerDown = (e: React.PointerEvent) => {
    if (!loaded || e.button !== 0) return
    const [px, py] = local(e)
    const [u, v] = toNormalized(px, py)
    if (!inside(u, v)) return
    e.currentTarget.setPointerCapture(e.pointerId)
    if (tool === "zone") {
      gesture.current = { kind: "draw", start: [u, v] }
      setDraft([u, v, u, v])
    } else if (tool === "tripwire") {
      gesture.current = { kind: "line", start: [u, v] }
      setLineDraft([u, v, u, v])
    } else {
      gesture.current = { kind: "press", startX: e.clientX, startY: e.clientY, lastX: e.clientX, lastY: e.clientY, panning: false }
    }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const g = gesture.current
    const [px, py] = local(e)
    const [u, v] = toNormalized(px, py)
    if (g.kind === "line") {
      setLineDraft([g.start[0], g.start[1], Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v))])
      return
    }
    if (g.kind === "draw") {
      const cu = Math.min(1, Math.max(0, u))
      const cv = Math.min(1, Math.max(0, v))
      setDraft([g.start[0], g.start[1], cu, cv])
      return
    }
    if (g.kind === "press") {
      const moved = Math.hypot(e.clientX - g.startX, e.clientY - g.startY) > DRAG_THRESHOLD_PX
      if (moved && view.view.current.s > 1.001) {
        if (!g.panning) {
          g.panning = true
          setPanning(true)
          onFollowChange(false)
        }
        panBy(e.clientX - g.lastX, e.clientY - g.lastY)
      }
      g.lastX = e.clientX
      g.lastY = e.clientY
      return
    }
    if (tool === "select") {
      const hit = inside(u, v) ? hitTest(list, u, v) : null
      setHoverId(hit?.id ?? null)
    }
  }

  const onPointerUp = (e: React.PointerEvent) => {
    const g = gesture.current
    gesture.current = { kind: "none" }
    setPanning(false)
    if (g.kind === "line") {
      setLineDraft(null)
      const [px, py] = local(e)
      const [u, v] = toNormalized(px, py)
      const b: [number, number] = [Math.min(1, Math.max(0, u)), Math.min(1, Math.max(0, v))]
      if (Math.hypot(b[0] - g.start[0], b[1] - g.start[1]) > 0.02) props.onTripwireDrawn?.(g.start, b)
      onToolChange("select")
      return
    }
    if (g.kind === "draw") {
      setDraft(null)
      const [px, py] = local(e)
      const [u, v] = toNormalized(px, py)
      const nb: NBox = [
        Math.min(g.start[0], u),
        Math.min(g.start[1], v),
        Math.max(g.start[0], u),
        Math.max(g.start[1], v),
      ].map((n) => Math.min(1, Math.max(0, n))) as NBox
      if (nb[2] - nb[0] > 0.01 && nb[3] - nb[1] > 0.01) onZoneDrawn(nb)
      onToolChange("select")
      return
    }
    if (g.kind === "press" && !g.panning) {
      const [px, py] = local(e)
      const [u, v] = toNormalized(px, py)
      const hit = hitTest(list, u, v)
      if (hit) onLock(hit.id === lockedId ? null : hit.id)
    }
  }

  const onDoubleClick = (e: React.MouseEvent) => {
    if (!loaded || tool !== "select") return
    const [px, py] = local(e)
    const [u, v] = toNormalized(px, py)
    const hit = hitTest(list, u, v)
    onFollowChange(false)
    if (hit && view.view.current.s < 1.2) focus(hit.box, 0.5, 4)
    else reset()
  }

  // Tool & zoom shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, [role=combobox], [role=listbox]")) return
      const k = e.key.toLowerCase()
      if (k === "v") onToolChange("select")
      else if (k === "z") onToolChange(tool === "zone" ? "select" : "zone")
      else if (k === "t") onToolChange(tool === "tripwire" ? "select" : "tripwire")
      else if (k === "f" && lockedId !== null) onFollowChange(!follow)
      else if (k === "0") {
        onFollowChange(false)
        reset()
      } else if (k === "+" || k === "=") zoomAt(rect.width / 2, rect.height / 2, 1.4)
      else if (k === "-") zoomAt(rect.width / 2, rect.height / 2, 1 / 1.4)
      else if (k === "escape") {
        if (gesture.current.kind === "draw" || gesture.current.kind === "line") {
          gesture.current = { kind: "none" }
          setDraft(null)
          setLineDraft(null)
        }
        if (tool === "zone" || tool === "tripwire") onToolChange("select")
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [tool, follow, lockedId, rect.width, rect.height, onToolChange, onFollowChange, reset, zoomAt])

  const phase: Phase = !camera?.has_stream ? "idle" : error ? "error" : loaded ? "live" : "connecting"
  let message: string | null = null
  if (!camera) message = "Loading camera"
  else if (!camera.has_stream) message = `No RTSP stream is configured for ${camera.label}. Add its IP and stream path to .env.`
  else if (phase === "error") message = error
  else if (phase === "connecting") message = `Connecting to ${camera.label}`

  const hovered = tool === "select" && hoverId !== lockedId ? (list.find((t) => t.id === hoverId) ?? null) : null
  const cursor =
    tool === "zone" || tool === "tripwire"
      ? "cursor-crosshair"
      : panning
        ? "cursor-grabbing"
        : hovered
          ? "cursor-pointer"
          : view.zoom > 1.01
            ? "cursor-grab"
            : "cursor-default"

  return (
    <div
      ref={containerRef}
      className={cn("absolute inset-0 touch-none select-none overflow-hidden bg-[#1b1c1f]", cursor)}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={() => gesture.current.kind === "none" && setHoverId(null)}
      onDoubleClick={onDoubleClick}
      onContextMenu={(e) => {
        if (!props.onAimAt || !loaded) return
        const [u, v] = toNormalized(...local(e))
        if (!inside(u, v)) return
        e.preventDefault()
        props.onAimAt(u, v, e.clientX, e.clientY)
      }}
    >
      <motion.div
        className="absolute overflow-hidden rounded-md"
        initial={false}
        animate={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
        transition={layoutTransition}
      >
      <motion.div
        className="absolute inset-0 origin-top-left will-change-transform"
        style={{ x: view.x, y: view.y, scale: view.s }}
      >
        <AnimatePresence mode="popLayout">
          {camera?.has_stream && (
            <motion.img
              key={`${camera.id}-${nonce}`}
              src={api.streamUrl(camera.id, nonce)}
              alt={`Live view of ${camera.label}`}
              draggable={false}
              onLoad={(e) => {
                const img = e.currentTarget
                setNatural({ w: img.naturalWidth, h: img.naturalHeight })
                setLoaded(true)
              }}
              initial={{ opacity: 0, scale: 1.015, filter: "blur(8px)" }}
              animate={loaded ? { opacity: 1, scale: 1, filter: "blur(0px)" } : { opacity: 0 }}
              exit={{ opacity: 0, filter: "blur(8px)" }}
              transition={{ duration: 0.6, ease }}
              className="absolute inset-0 size-full object-contain"
            />
          )}
        </AnimatePresence>

        <svg className="pointer-events-none absolute inset-0 size-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
          <AnimatePresence>
            {hovered && (
              <motion.rect
                key={hovered.id}
                initial={{ opacity: 0, x: hovered.box[0], y: hovered.box[1], width: hovered.box[2] - hovered.box[0], height: hovered.box[3] - hovered.box[1] }}
                animate={{ opacity: 1, x: hovered.box[0], y: hovered.box[1], width: hovered.box[2] - hovered.box[0], height: hovered.box[3] - hovered.box[1] }}
                exit={{ opacity: 0 }}
                transition={{ opacity: { duration: 0.15 }, default: { duration: 0.12, ease: "linear" } }}
                fill="rgba(255,210,31,0.08)"
                stroke="#f4f4f2"
                strokeWidth={1.5}
                strokeDasharray="6 4"
                vectorEffect="non-scaling-stroke"
                rx={0.004}
              />
            )}
          </AnimatePresence>
          {draft && (
            <rect
              x={Math.min(draft[0], draft[2])}
              y={Math.min(draft[1], draft[3])}
              width={Math.abs(draft[2] - draft[0])}
              height={Math.abs(draft[3] - draft[1])}
              fill="rgba(255,210,31,0.12)"
              stroke="#ffd21f"
              strokeWidth={1.5}
              strokeDasharray="8 5"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {lineDraft && <TripwireDraft line={lineDraft} aspect={frameW && frameH ? frameW / frameH : 16 / 9} />}
        </svg>
      </motion.div>
      {loaded && aiOn && (
        <StageVisuals
          view={view}
          width={rect.width}
          height={rect.height}
          targets={list}
          lockedId={lockedId}
          hoveredId={hovered?.id ?? null}
          pose={targets?.pose}
          onFocus={(b) => {
            onFollowChange(false)
            focus(b, 10, view.view.current.s)
          }}
        />
      )}
      </motion.div>

      <Brackets phase={phase} rect={rect} area={{ w: box.w, h: box.h, right: rightInset }} instant={!laidOut.current} />

      <AnimatePresence mode="wait">
        {message && (
          <motion.p
            key={message}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.35, ease }}
            role="status"
            className={cn(
              "pointer-events-none absolute inset-x-6 top-1/2 mx-auto max-w-md -translate-y-1/2 text-center text-sm leading-relaxed",
              phase === "error" ? "text-destructive" : "text-muted-foreground",
            )}
          >
            {message}
            {phase === "connecting" && <Ellipsis />}
          </motion.p>
        )}
      </AnimatePresence>

      {loaded && (
        <StageToolbar
          tool={tool}
          onToolChange={onToolChange}
          zoom={view.zoom}
          onZoom={(f) => {
            onFollowChange(false)
            zoomAt(rect.width / 2, rect.height / 2, f)
          }}
          onReset={() => {
            onFollowChange(false)
            reset()
          }}
          aiOn={aiOn}
          locked={locked}
          lockedId={lockedId}
          hovered={hovered}
          follow={follow}
          onFollowChange={onFollowChange}
          onUnlock={() => onLock(null)}
        />
      )}
    </div>
  )
}

function Brackets({
  phase,
  rect,
  area,
  instant,
}: {
  instant: boolean
  phase: Phase
  rect: { left: number; top: number; width: number; height: number }
  area: { w: number; h: number; right: number }
}) {
  const live = phase === "live"
  const color = phase === "error" ? "border-destructive" : phase === "idle" ? "border-muted-foreground/40" : "border-primary"
  // Live: hug the actual picture. Otherwise: hover wide inside the free area.
  const pad = 8
  const spread = 36
  const box = live
    ? { left: rect.left - pad, top: rect.top - pad, width: rect.width + 2 * pad, height: rect.height + 2 * pad }
    : {
        left: INSET.left + spread,
        top: INSET.top + spread,
        width: Math.max(0, area.w - INSET.left - area.right - 2 * spread),
        height: Math.max(0, area.h - INSET.top - INSET.bottom - 2 * spread),
      }
  const corners = [
    "left-0 top-0 border-l-2 border-t-2 rounded-tl-md",
    "right-0 top-0 border-r-2 border-t-2 rounded-tr-md",
    "left-0 bottom-0 border-l-2 border-b-2 rounded-bl-md",
    "right-0 bottom-0 border-r-2 border-b-2 rounded-br-md",
  ]
  return (
    <motion.div
      aria-hidden
      className="pointer-events-none absolute"
      initial={false}
      animate={{ ...box, opacity: live ? 0.9 : 0.55 }}
      transition={instant ? { duration: 0 } : { type: "spring", stiffness: 160, damping: 22 }}
    >
      {corners.map((c) => (
        <span key={c} className={cn("absolute size-7 transition-colors duration-500", color, c)} />
      ))}
      {phase === "connecting" && (
        <motion.span
          className="absolute inset-x-0 h-px bg-primary/60"
          initial={{ top: "0%" }}
          animate={{ top: ["0%", "100%"] }}
          transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut", repeatType: "reverse" }}
        />
      )}
    </motion.div>
  )
}

function Ellipsis() {
  return (
    <span className="inline-flex w-5">
      {[0, 1, 2].map((i) => (
        <motion.span key={i} animate={{ opacity: [0.2, 1, 0.2] }} transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}>
          .
        </motion.span>
      ))}
    </span>
  )
}

/** Line being drawn with the tripwire tool, plus the "in" arrow (a→b rotated 90° clockwise on screen). */
function TripwireDraft({ line, aspect }: { line: NBox; aspect: number }) {
  const [ax, ay, bx, by] = line
  // work in an aspect-correct space (x scaled by aspect) so the arrow is perpendicular on screen
  const dx = (bx - ax) * aspect
  const dy = by - ay
  const len = Math.hypot(dx, dy)
  const mx = (ax + bx) / 2
  const my = (ay + by) / 2
  let arrow: string | null = null
  if (len > 0.02) {
    const nx = -dy / len
    const ny = dx / len
    const L = 0.05
    const tip = [mx + (nx * L) / aspect, my + ny * L]
    const head = 0.018
    const left = [tip[0] - ((nx * head - ny * head * 0.7) / aspect), tip[1] - (ny * head + nx * head * 0.7)]
    const right = [tip[0] - ((nx * head + ny * head * 0.7) / aspect), tip[1] - (ny * head - nx * head * 0.7)]
    arrow = `M${mx},${my} L${tip[0]},${tip[1]} M${left[0]},${left[1]} L${tip[0]},${tip[1]} L${right[0]},${right[1]}`
  }
  return (
    <g>
      <line x1={ax} y1={ay} x2={bx} y2={by} stroke="#222427" strokeWidth={5} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      <line x1={ax} y1={ay} x2={bx} y2={by} stroke="#ffd21f" strokeWidth={2.5} strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      {arrow && <path d={arrow} fill="none" stroke="#ffd21f" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />}
    </g>
  )
}
