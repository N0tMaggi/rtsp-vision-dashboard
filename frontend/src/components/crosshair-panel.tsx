import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Check, ChevronDown, ClipboardPaste, Copy, Pencil, Plus, Save, Trash2, X } from "lucide-react"
import { BodyPointPicker } from "@/components/body-point-picker"
import { CrosshairGlyph } from "@/components/crosshair-glyph"
import { PanelRange, PanelToggle } from "@/components/panel-controls"
import { AIM_LABEL, consumeFocus, OPEN_CROSSHAIR_EVENT, useAutoTrack } from "@/lib/autotrack"
import {
  allTemplates,
  applyTemplate,
  CROSSHAIR_STYLES,
  deleteTemplate,
  exportJson,
  importJson,
  LIMITS,
  normalizeHex,
  overwriteTemplate,
  PALETTE,
  renameTemplate,
  saveTemplate,
  setCrosshair,
  STYLE_LABEL,
  useCrosshair,
  type CrosshairSettings,
} from "@/lib/crosshair"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const
const spring = { type: "spring", stiffness: 450, damping: 34 } as const
const inputCls =
  "h-8 min-w-0 rounded-md border border-input bg-input/30 px-2.5 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring"

/** Crosshair look, aim point and templates (Overlays category; this browser only). */
export function CrosshairPanel() {
  const store = useCrosshair()
  const s = store.settings
  const { autotrack } = useAutoTrack()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // "Crosshair Settings" from the auto-track section: expand and scroll here
  useEffect(() => {
    const focus = () => {
      if (!consumeFocus("crosshair")) return
      setOpen(true)
      window.setTimeout(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60)
    }
    focus()
    window.addEventListener(OPEN_CROSSHAIR_EVENT, focus)
    return () => window.removeEventListener(OPEN_CROSSHAIR_EVENT, focus)
  }, [])

  const atAim = autotrack?.config.aim ?? null
  const shownAim = s.aim === "autotrack" ? (atAim ?? "chest") : s.aim
  const set = (p: Partial<CrosshairSettings>) => setCrosshair(p)

  return (
    <div ref={ref} className="scroll-mt-4 space-y-2">
      <PanelToggle id="crosshair-enabled" label="Crosshair" on={s.enabled} onChange={(v) => set({ enabled: v })} />
      <Preview settings={s} aimLabel={AIM_LABEL[shownAim]} />
      <Templates activeId={store.activeId} />

      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="-mx-2 flex w-[calc(100%+1rem)] items-center justify-between rounded-md px-2 py-2 text-sm text-muted-foreground transition-colors hover:bg-accent/70 hover:text-foreground"
      >
        Customize
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={spring}>
          <ChevronDown className="size-4" />
        </motion.span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease }}
            className="overflow-hidden"
          >
            <div className="space-y-4 pb-1">
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm text-muted-foreground">Aim Point</span>
                  {autotrack && (
                    <button
                      type="button"
                      aria-pressed={s.aim === "autotrack"}
                      onClick={() => set({ aim: s.aim === "autotrack" ? shownAim : "autotrack" })}
                      className={cn(
                        "rounded-full border px-2.5 py-0.5 text-xs transition-colors",
                        s.aim === "autotrack" ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground hover:text-foreground",
                      )}
                    >
                      Same as Auto-Track
                    </button>
                  )}
                </div>
                <BodyPointPicker value={shownAim} color={s.color} onChange={(k) => set({ aim: k })} />
              </div>

              <StylePicker settings={s} />
              <ColorPicker value={s.color} onChange={(color) => set({ color })} />

              <div className="space-y-0.5">
                <PanelRange label="Size" min={LIMITS.size[0]} max={LIMITS.size[1]} step={1} value={s.size} debounceMs={0} format={(v) => `${v} px`} onChange={(v) => set({ size: v })} />
                <PanelRange label="Thickness" min={LIMITS.thickness[0]} max={LIMITS.thickness[1]} step={0.5} value={s.thickness} debounceMs={0} format={(v) => `${v} px`} onChange={(v) => set({ thickness: v })} />
                <PanelRange label="Gap" min={LIMITS.gap[0]} max={LIMITS.gap[1]} step={1} value={s.gap} debounceMs={0} format={(v) => `${v} px`} onChange={(v) => set({ gap: v })} />
                <PanelRange label="Opacity" min={LIMITS.opacity[0]} max={LIMITS.opacity[1]} step={0.05} value={s.opacity} debounceMs={0} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => set({ opacity: v })} />
                <PanelRange label="Rotation" min={LIMITS.rotation[0]} max={LIMITS.rotation[1]} step={1} value={s.rotation} debounceMs={0} format={(v) => `${v}°`} onChange={(v) => set({ rotation: v })} />
              </div>

              <div>
                <PanelToggle id="crosshair-outline" label="Outline" on={s.outline} onChange={(v) => set({ outline: v })} />
                <PanelToggle id="crosshair-spin" label="Spin" on={s.spin} onChange={(v) => set({ spin: v })} />
                <PanelToggle id="crosshair-pulse" label="Pulse" on={s.pulse} onChange={(v) => set({ pulse: v })} />
                <PanelToggle id="crosshair-label" label="Show Label" on={s.label} onChange={(v) => set({ label: v })} />
                <PanelToggle id="crosshair-all" label="Show on All Persons" on={s.allPersons} onChange={(v) => set({ allPersons: v })} />
              </div>

              <ImportExport />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <p className="text-xs leading-relaxed text-muted-foreground">
        Drawn on the locked target and the one the PTZ camera follows. Saved in this browser only.
      </p>
    </div>
  )
}

function Preview({ settings, aimLabel }: { settings: CrosshairSettings; aimLabel: string }) {
  return (
    <div
      className={cn(
        "relative grid h-32 place-items-center overflow-hidden rounded-lg border border-border bg-[#1b1c1f] transition-opacity duration-300",
        !settings.enabled && "opacity-40",
      )}
      style={{
        backgroundImage:
          "linear-gradient(rgba(244,244,242,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(244,244,242,0.04) 1px, transparent 1px)",
        backgroundSize: "16px 16px",
        backgroundPosition: "center",
      }}
      aria-label="Crosshair preview"
      role="img"
    >
      <div className="relative">
        <CrosshairGlyph look={settings} />
        {settings.label && (
          <span
            className="absolute left-1/2 top-full mt-0.5 -translate-x-1/2 whitespace-nowrap rounded-full bg-background/75 px-2 py-0.5 text-[10px] font-medium tracking-wider tabular-nums"
            style={{ color: settings.color }}
          >
            {aimLabel.toUpperCase()} · 4.2 m
          </span>
        )}
      </div>
    </div>
  )
}

function StylePicker({ settings }: { settings: CrosshairSettings }) {
  return (
    <div className="space-y-2">
      <span className="text-sm text-muted-foreground">Style</span>
      <div role="radiogroup" aria-label="Crosshair style" className="grid grid-cols-4 gap-1.5">
        {CROSSHAIR_STYLES.map((st) => {
          const active = st === settings.style
          return (
            <button
              key={st}
              type="button"
              role="radio"
              aria-checked={active}
              title={STYLE_LABEL[st]}
              onClick={() => setCrosshair({ style: st })}
              className={cn(
                "group relative flex flex-col items-center gap-1 rounded-lg border py-1.5 transition-colors duration-200",
                active ? "border-primary/70 bg-primary/10" : "border-border hover:border-foreground/30 hover:bg-accent/60",
              )}
            >
              <span className="grid size-9 place-items-center">
                <CrosshairGlyph
                  look={{
                    ...settings,
                    style: st,
                    size: 26,
                    gap: Math.min(settings.gap, 5),
                    thickness: Math.min(settings.thickness, 2.5),
                    opacity: 1,
                    spin: false,
                    pulse: false,
                    color: active ? settings.color : "#f4f4f2",
                  }}
                />
              </span>
              <span className={cn("text-[10px]", active ? "text-foreground" : "text-muted-foreground")}>{STYLE_LABEL[st]}</span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function ColorPicker({ value, onChange }: { value: string; onChange: (c: string) => void }) {
  const [hex, setHex] = useState(value)
  useEffect(() => setHex(value), [value])
  const valid = normalizeHex(hex)
  return (
    <div className="space-y-2">
      <span className="text-sm text-muted-foreground">Color</span>
      <div className="flex flex-wrap items-center gap-1.5">
        {PALETTE.map((c) => {
          const active = c === value
          return (
            <button
              key={c}
              type="button"
              aria-label={`Color ${c}`}
              aria-pressed={active}
              onClick={() => onChange(c)}
              className="relative grid size-7 place-items-center rounded-full"
            >
              {active && <motion.span layoutId="crosshair-color-ring" className="absolute inset-0 rounded-full border-2 border-foreground/80" transition={spring} />}
              <span className="size-5 rounded-full border border-black/30" style={{ background: c }} />
            </button>
          )
        })}
        <label className="relative grid size-7 cursor-pointer place-items-center rounded-full" title="Custom color">
          <span
            className="size-5 rounded-full border border-black/30"
            style={{ background: "conic-gradient(#ff5a4e, #ffd21f, #7ee0a1, #4dd2ff, #b48cff, #ff6bd6, #ff5a4e)" }}
          />
          <input type="color" value={value} onChange={(e) => onChange(e.target.value)} className="absolute inset-0 cursor-pointer opacity-0" aria-label="Custom color" />
        </label>
        <input
          value={hex}
          maxLength={7}
          spellCheck={false}
          aria-label="Hex color"
          aria-invalid={!valid}
          onChange={(e) => {
            setHex(e.target.value)
            const n = normalizeHex(e.target.value)
            if (n && n.length === 7 && e.target.value.replace("#", "").length === 6) onChange(n)
          }}
          onBlur={() => (valid ? onChange(valid) : setHex(value))}
          className={cn(inputCls, "ml-auto h-7 w-[5.5rem] font-mono text-xs uppercase", !valid && "border-destructive")}
        />
      </div>
    </div>
  )
}

function Templates({ activeId }: { activeId: string | null }) {
  const store = useCrosshair()
  const list = allTemplates(store)
  const active = list.find((t) => t.id === activeId) ?? null
  const [mode, setMode] = useState<"none" | "save" | "rename">("none")
  const [name, setName] = useState("")
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => {
    setConfirmDelete(false)
    setMode((m) => (m === "rename" ? "none" : m))
  }, [activeId])

  const submit = () => {
    const v = name.trim()
    if (!v) return
    if (mode === "save") saveTemplate(v)
    else if (mode === "rename" && active && !active.builtin) renameTemplate(active.id, v)
    setMode("none")
    setName("")
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">Templates</span>
        <span className="text-xs text-muted-foreground">{active ? active.name : "Custom"}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {list.map((t) => {
          const on = t.id === activeId
          return (
            <button
              key={t.id}
              type="button"
              aria-pressed={on}
              onClick={() => applyTemplate(t.id)}
              className={cn(
                "relative flex items-center gap-1.5 rounded-full border py-1 pl-1.5 pr-2.5 text-xs transition-colors duration-200",
                on ? "border-primary text-primary-foreground" : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
              )}
            >
              {on && <motion.span layoutId="crosshair-template" className="absolute inset-0 rounded-full bg-primary" transition={spring} />}
              <span className="relative grid size-4 place-items-center">
                <CrosshairGlyph
                  look={{ ...t.look, size: 13, gap: Math.min(t.look.gap, 2), thickness: 1.5, outline: false, opacity: 1, spin: false, pulse: false, color: on ? "#222427" : t.look.color }}
                />
              </span>
              <span className="relative">{t.name}</span>
            </button>
          )
        })}
      </div>

      <AnimatePresence mode="wait" initial={false}>
        {mode !== "none" ? (
          <motion.form
            key="form"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.18 }}
            className="flex gap-1.5"
            onSubmit={(e) => {
              e.preventDefault()
              submit()
            }}
          >
            <input
              autoFocus
              value={name}
              maxLength={32}
              placeholder={mode === "save" ? "Template name" : "New name"}
              aria-label={mode === "save" ? "New template name" : "Rename template"}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Escape" && (e.stopPropagation(), setMode("none"))}
              className={cn(inputCls, "flex-1")}
            />
            <button type="submit" disabled={!name.trim()} aria-label="Confirm" className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground disabled:opacity-40">
              <Check className="size-3.5" />
            </button>
            <button type="button" aria-label="Cancel" onClick={() => setMode("none")} className="grid size-8 place-items-center rounded-md border border-border text-muted-foreground hover:text-foreground">
              <X className="size-3.5" />
            </button>
          </motion.form>
        ) : (
          <motion.div
            key="actions"
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.18 }}
            className="flex flex-wrap gap-1.5"
          >
            <ActionButton
              onClick={() => {
                setName(active && !active.builtin ? `${active.name} copy` : "")
                setMode("save")
              }}
            >
              <Plus className="size-3.5" /> Save as New
            </ActionButton>
            {store.templates.length > 0 && activeId === null && <OverwriteMenu />}
            {active && !active.builtin && (
              <>
                <ActionButton
                  onClick={() => {
                    setName(active.name)
                    setMode("rename")
                  }}
                >
                  <Pencil className="size-3.5" /> Rename
                </ActionButton>
                <ActionButton
                  danger
                  onClick={() => {
                    if (!confirmDelete) return setConfirmDelete(true)
                    deleteTemplate(active.id)
                    setConfirmDelete(false)
                  }}
                >
                  <Trash2 className="size-3.5" /> {confirmDelete ? "Click to Delete" : "Delete"}
                </ActionButton>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/** Shown while the settings differ from every template: save them into an existing user template. */
function OverwriteMenu() {
  const store = useCrosshair()
  const [open, setOpen] = useState(false)
  return (
    <div className="relative">
      <ActionButton onClick={() => setOpen((o) => !o)}>
        <Save className="size-3.5" /> Overwrite…
      </ActionButton>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ opacity: 0, y: -4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.97 }}
            transition={spring}
            className="absolute left-0 top-full z-10 mt-1 max-h-48 min-w-40 overflow-auto rounded-lg border border-border bg-popover p-1 shadow-xl"
          >
            {store.templates.map((t) => (
              <li key={t.id}>
                <button
                  type="button"
                  onClick={() => {
                    overwriteTemplate(t.id)
                    setOpen(false)
                  }}
                  className="w-full truncate rounded-md px-2 py-1.5 text-left text-xs hover:bg-accent"
                >
                  {t.name}
                </button>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  )
}

function ActionButton({ onClick, danger, children }: { onClick: () => void; danger?: boolean; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
        danger ? "border-border text-muted-foreground hover:border-destructive/60 hover:text-destructive" : "border-border text-foreground hover:border-primary/60 hover:bg-accent",
      )}
    >
      {children}
    </button>
  )
}

function ImportExport() {
  const [importing, setImporting] = useState(false)
  const [text, setText] = useState("")
  const [msg, setMsg] = useState<{ text: string; tone: "ok" | "error" } | null>(null)
  const [fallback, setFallback] = useState<string | null>(null)

  const copy = () => {
    const json = exportJson()
    setFallback(null)
    if (!navigator.clipboard) {
      setFallback(json)
      return
    }
    navigator.clipboard
      .writeText(json)
      .then(() => setMsg({ text: "Copied to the clipboard.", tone: "ok" }))
      .catch(() => setFallback(json))
  }

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <span className="text-sm text-muted-foreground">Share</span>
      <div className="flex gap-1.5">
        <ActionButton onClick={copy}>
          <Copy className="size-3.5" /> Copy JSON
        </ActionButton>
        <ActionButton
          onClick={() => {
            setImporting((v) => !v)
            setMsg(null)
          }}
        >
          <ClipboardPaste className="size-3.5" /> Import…
        </ActionButton>
      </div>
      {fallback && (
        <textarea
          readOnly
          value={fallback}
          aria-label="Exported crosshair JSON"
          onFocus={(e) => e.currentTarget.select()}
          className="h-28 w-full resize-none rounded-md border border-input bg-input/30 p-2 font-mono text-[11px] outline-none"
        />
      )}
      <AnimatePresence initial={false}>
        {importing && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="space-y-1.5 overflow-hidden">
            <textarea
              value={text}
              placeholder="Paste crosshair JSON here"
              aria-label="Crosshair JSON to import"
              spellCheck={false}
              onChange={(e) => setText(e.target.value)}
              className="h-28 w-full resize-none rounded-md border border-input bg-input/30 p-2 font-mono text-[11px] outline-none placeholder:text-muted-foreground focus-visible:border-ring"
            />
            <button
              type="button"
              disabled={!text.trim()}
              onClick={() => {
                try {
                  setMsg({ text: importJson(text), tone: "ok" })
                  setText("")
                  setImporting(false)
                } catch (e) {
                  setMsg({ text: (e as Error).message, tone: "error" })
                }
              }}
              className="flex h-8 w-full items-center justify-center rounded-md bg-primary text-xs font-medium text-primary-foreground disabled:opacity-40"
            >
              Import
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {msg && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            role={msg.tone === "error" ? "alert" : "status"}
            className={cn("text-xs", msg.tone === "error" ? "text-destructive" : "text-primary")}
          >
            {msg.text}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}
