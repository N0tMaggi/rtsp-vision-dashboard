import { useCallback, useRef, useState } from "react"
import { useSpring } from "motion/react"
import type { NBox } from "@/lib/types"

export const MIN_ZOOM = 1
export const MAX_ZOOM = 8

interface View {
  s: number
  x: number
  y: number
}

const SPRING = { stiffness: 280, damping: 34, mass: 0.7 }

/**
 * Digital zoom/pan for the video. The transformed element is exactly the size of
 * the visible video (w × h) with transform-origin at its top-left, so a content
 * point u lands on screen at x + u·s. Pan is clamped so the video always covers
 * its own frame - no empty borders when zoomed.
 */
export function useViewTransform(w: number, h: number) {
  const s = useSpring(1, SPRING)
  const x = useSpring(0, SPRING)
  const y = useSpring(0, SPRING)
  const view = useRef<View>({ s: 1, x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)

  const apply = useCallback(
    (next: View, immediate = false) => {
      const ns = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next.s))
      const nx = Math.min(0, Math.max(w * (1 - ns), next.x))
      const ny = Math.min(0, Math.max(h * (1 - ns), next.y))
      view.current = { s: ns, x: nx, y: ny }
      if (immediate) {
        s.jump(ns)
        x.jump(nx)
        y.jump(ny)
      } else {
        s.set(ns)
        x.set(nx)
        y.set(ny)
      }
      setZoom((z) => (Math.abs(z - ns) > 0.01 ? ns : z))
    },
    [w, h, s, x, y],
  )

  /** Zoom by `factor` keeping the point (px, py) - in element-local px - fixed. */
  const zoomAt = useCallback(
    (px: number, py: number, factor: number) => {
      const v = view.current
      const ns = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.s * factor))
      const ux = (px - v.x) / v.s
      const uy = (py - v.y) / v.s
      apply({ s: ns, x: px - ux * ns, y: py - uy * ns })
    },
    [apply],
  )

  const panBy = useCallback(
    (dx: number, dy: number) => {
      const v = view.current
      apply({ ...v, x: v.x + dx, y: v.y + dy }, true)
    },
    [apply],
  )

  /** Center a normalized box, filling ~`fill` of the view. */
  const focus = useCallback(
    (box: NBox, fill = 0.45, maxZoom = 5) => {
      const bw = Math.max(0.01, box[2] - box[0])
      const bh = Math.max(0.01, box[3] - box[1])
      const ns = Math.min(maxZoom, Math.max(1.4, Math.min(fill / bw, fill / bh)))
      const cx = ((box[0] + box[2]) / 2) * w
      const cy = ((box[1] + box[3]) / 2) * h
      apply({ s: ns, x: w / 2 - cx * ns, y: h / 2 - cy * ns })
    },
    [apply, w, h],
  )

  const reset = useCallback(() => apply({ s: 1, x: 0, y: 0 }), [apply])

  /** Element-local px -> normalized frame coordinates. */
  const toNormalized = useCallback(
    (px: number, py: number) => {
      const v = view.current
      return [(px - v.x) / v.s / w, (py - v.y) / v.s / h] as const
    },
    [w, h],
  )

  return { s, x, y, zoom, view, zoomAt, panBy, focus, reset, toNormalized }
}
