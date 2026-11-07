import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { createPortal } from "react-dom"
import { AnimatePresence, motion } from "motion/react"
import { ChevronLeft, ChevronRight, type LucideIcon } from "lucide-react"
import { cn } from "@/lib/utils"

export type MenuEntry =
  | {
      kind: "item"
      id: string
      label: string
      icon?: LucideIcon
      /** Keyboard hint shown on the right (e.g. "F", "Alt+Right-click"). */
      shortcut?: string
      /** Tooltip / secondary explanation. */
      hint?: string
      disabled?: boolean
      danger?: boolean
      active?: boolean
      /** Runs and closes the menu. */
      onSelect?: () => void
      /** Opens an inline panel instead (Back / Esc / ArrowLeft returns to the list). */
      panel?: (api: PanelApi) => ReactNode
    }
  | { kind: "separator"; id: string }
  | { kind: "header"; id: string; content: ReactNode }

export interface PanelApi {
  close: () => void
  back: () => void
}

interface Props {
  open: boolean
  x: number
  y: number
  entries: MenuEntry[]
  onClose: () => void
  label: string
  /** Open straight into this item's panel (e.g. "Add Tag…" from a shortcut). */
  initialPanel?: string | null
}

const MARGIN = 8
const stop = (e: React.SyntheticEvent) => e.stopPropagation()
const spring = { type: "spring", stiffness: 520, damping: 36, mass: 0.7 } as const

/**
 * Animated context menu at the cursor, kept on-screen. Keyboard: arrows / Home / End
 * move, Enter / Space select, typing jumps to an item, ArrowRight opens a panel,
 * Esc / ArrowLeft go back, Esc closes. Click outside, scroll or resize close it.
 */
export function ContextMenu({ open, x, y, entries, onClose, label, initialPanel = null }: Props) {
  return createPortal(
    <AnimatePresence>
      {open && <MenuBody key={`${x}:${y}`} x={x} y={y} entries={entries} onClose={onClose} label={label} initialPanel={initialPanel} />}
    </AnimatePresence>,
    document.body,
  )
}

function MenuBody({ x, y, entries, onClose, label, initialPanel }: Omit<Props, "open">) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ left: x, top: y, originX: 0, originY: 0 })
  const [panel, setPanel] = useState<string | null>(initialPanel ?? null)
  const items = useMemo(() => entries.filter((e) => e.kind === "item"), [entries])
  const enabled = useMemo(() => items.filter((e) => !e.disabled).map((e) => e.id), [items])
  const [active, setActive] = useState<string | null>(null)
  const restore = useRef<HTMLElement | null>(null)

  useEffect(() => {
    restore.current = document.activeElement as HTMLElement | null
    return () => restore.current?.focus?.({ preventScroll: true })
  }, [])

  // Keep the menu on-screen (re-measured when a panel changes the size).
  const place = useCallback(() => {
    const el = ref.current
    if (!el) return
    const w = el.offsetWidth
    const h = el.offsetHeight
    const vw = window.innerWidth
    const vh = window.innerHeight
    let left = x
    let top = y
    let originX = 0
    let originY = 0
    if (left + w > vw - MARGIN) {
      left = Math.max(MARGIN, x - w)
      originX = 1
    }
    if (top + h > vh - MARGIN) {
      top = Math.max(MARGIN, Math.min(y - h, vh - MARGIN - h))
      originY = 1
    }
    setPos((p) => (p.left === left && p.top === top ? p : { left, top, originX, originY }))
  }, [x, y])
  useLayoutEffect(place, [place, panel])
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(place)
    ro.observe(el)
    return () => ro.disconnect()
  }, [place])

  // Focus the menu (not an item) so keys work right away; panels focus their own first field.
  useEffect(() => {
    if (!panel) ref.current?.focus({ preventScroll: true })
  }, [panel])

  // Click outside / resize / scroll / blur close.
  useEffect(() => {
    const down = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose()
    }
    const wheel = (e: WheelEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose()
    }
    const blur = () => onClose()
    window.addEventListener("pointerdown", down, true)
    window.addEventListener("wheel", wheel, true)
    window.addEventListener("resize", blur)
    window.addEventListener("blur", blur)
    return () => {
      window.removeEventListener("pointerdown", down, true)
      window.removeEventListener("wheel", wheel, true)
      window.removeEventListener("resize", blur)
      window.removeEventListener("blur", blur)
    }
  }, [onClose])

  const select = (id: string) => {
    const it = items.find((e) => e.id === id)
    if (!it || it.kind !== "item" || it.disabled) return
    if (it.panel) {
      setPanel(id)
      return
    }
    onClose()
    it.onSelect?.()
  }

  const move = (delta: number) => {
    if (!enabled.length) return
    const i = active ? enabled.indexOf(active) : -1
    const next = i < 0 ? (delta > 0 ? 0 : enabled.length - 1) : (i + delta + enabled.length) % enabled.length
    setActive(enabled[next])
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    // keep app / stage shortcuts (S, P, Space, …) from firing while the menu is open
    e.stopPropagation()
    const typing = (e.target as HTMLElement).closest("input, textarea, select")
    if (e.key === "Escape") {
      e.preventDefault()
      if (panel) setPanel(null)
      else onClose()
      return
    }
    if (panel) {
      if (e.key === "ArrowLeft" && !typing) {
        e.preventDefault()
        setPanel(null)
      }
      return
    }
    if (e.key === "ArrowDown" || (e.key === "Tab" && !e.shiftKey)) {
      e.preventDefault()
      move(1)
    } else if (e.key === "ArrowUp" || (e.key === "Tab" && e.shiftKey)) {
      e.preventDefault()
      move(-1)
    } else if (e.key === "Home") {
      e.preventDefault()
      setActive(enabled[0] ?? null)
    } else if (e.key === "End") {
      e.preventDefault()
      setActive(enabled[enabled.length - 1] ?? null)
    } else if (e.key === "Enter" || e.key === " " || e.key === "ArrowRight") {
      if (!active) return
      const it = items.find((i) => i.id === active)
      if (e.key === "ArrowRight" && !(it && it.kind === "item" && it.panel)) return
      e.preventDefault()
      select(active)
    } else if (e.key.length === 1 && /\S/.test(e.key) && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const k = e.key.toLowerCase()
      const start = active ? enabled.indexOf(active) + 1 : 0
      const order = [...enabled.slice(start), ...enabled.slice(0, start)]
      const hit = order.find((id) => {
        const it = items.find((i) => i.id === id)
        return it?.kind === "item" && it.label.toLowerCase().startsWith(k)
      })
      if (hit) setActive(hit)
    }
  }

  const current = panel ? items.find((e) => e.id === panel) : null
  const api: PanelApi = { close: onClose, back: () => setPanel(null) }

  return (
    <motion.div
      ref={ref}
      role="menu"
      aria-label={label}
      aria-activedescendant={!panel && active ? `ctx-${active}` : undefined}
      tabIndex={-1}
      onKeyDown={onKeyDown}
      // React events bubble through portals: keep the stage under the menu from seeing them
      onPointerDown={stop}
      onPointerUp={stop}
      onPointerMove={stop}
      onClick={stop}
      onDoubleClick={stop}
      onWheel={stop}
      onContextMenu={(e) => {
        e.preventDefault()
        e.stopPropagation()
      }}
      initial={{ opacity: 0, scale: 0.94, filter: "blur(3px)" }}
      animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
      exit={{ opacity: 0, scale: 0.96, filter: "blur(2px)", transition: { duration: 0.12 } }}
      transition={spring}
      style={{ left: pos.left, top: pos.top, transformOrigin: `${pos.originX * 100}% ${pos.originY * 100}%` }}
      className="fixed z-[60] min-w-56 max-w-[min(20rem,calc(100vw-16px))] overflow-hidden rounded-xl border border-white/10 bg-[#26282c]/95 p-1 text-[13px] text-foreground shadow-[0_24px_60px_-18px_rgba(0,0,0,0.75)] outline-none backdrop-blur-xl"
    >
      <AnimatePresence mode="popLayout" initial={false}>
        {current && current.kind === "item" && current.panel ? (
          <motion.div
            key={`panel-${current.id}`}
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 24 }}
            transition={spring}
            className="w-72"
          >
            <div className="mb-1 flex items-center gap-1 border-b border-white/8 px-1 pb-1">
              <button
                type="button"
                onClick={() => setPanel(null)}
                aria-label="Back"
                className="grid size-7 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-white/8 hover:text-foreground"
              >
                <ChevronLeft className="size-4" />
              </button>
              <span className="truncate font-medium">{current.label.replace(/…$/, "")}</span>
            </div>
            <div className="px-1.5 pb-1.5 pt-0.5">{current.panel(api)}</div>
          </motion.div>
        ) : (
          <motion.ul
            key="list"
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -24 }}
            transition={spring}
            className="flex flex-col"
            onPointerLeave={() => setActive(null)}
          >
            {entries.map((e) => {
              if (e.kind === "separator") return <li key={e.id} role="separator" className="mx-2 my-1 h-px bg-white/8" />
              if (e.kind === "header")
                return (
                  <li key={e.id} role="presentation" className="px-2.5 pb-1.5 pt-1">
                    {e.content}
                  </li>
                )
              const Icon = e.icon
              const isActive = active === e.id
              return (
                <li
                  key={e.id}
                  id={`ctx-${e.id}`}
                  role="menuitem"
                  aria-disabled={e.disabled || undefined}
                  aria-haspopup={e.panel ? "dialog" : undefined}
                  title={e.hint}
                  onPointerMove={() => !e.disabled && setActive(e.id)}
                  onClick={() => select(e.id)}
                  className={cn(
                    "relative flex cursor-default select-none items-center gap-2.5 rounded-md px-2.5 py-1.5 transition-colors duration-100",
                    e.disabled ? "opacity-40" : e.danger ? "text-[#ff8a80]" : "",
                  )}
                >
                  {isActive && (
                    <motion.span
                      layoutId="ctx-highlight"
                      className={cn("absolute inset-0 rounded-md", e.danger ? "bg-[#ff5a4e]/15" : "bg-white/8")}
                      transition={{ type: "spring", stiffness: 700, damping: 45 }}
                    />
                  )}
                  {Icon ? (
                    <Icon className={cn("relative size-4 shrink-0", e.active ? "text-primary" : "text-muted-foreground")} />
                  ) : (
                    <span className="relative size-4 shrink-0" />
                  )}
                  <span className={cn("relative flex-1 truncate", e.active && "text-primary")}>{e.label}</span>
                  {e.shortcut && <kbd className="relative font-sans text-[11px] text-muted-foreground">{e.shortcut}</kbd>}
                  {e.panel && <ChevronRight className="relative size-3.5 text-muted-foreground" />}
                </li>
              )
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
