import { useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Box, Palette, Radar, Route, ScanFace, SquareDashed, Trash2, X } from "lucide-react"
import { OverlaysPanel } from "@/components/overlays-panel"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  BOX_STYLES,
  COLORS,
  FACE_BLUR_MODES,
  FACE_TOGGLES,
  SLIDERS,
  STYLE_TOGGLES,
  TRACKING_TOGGLES,
  VISUAL_TOGGLES,
  type SliderDef,
  type ToggleDef,
} from "@/lib/esp-schema"
import type { CameraStatus, EspSettings, Zone } from "@/lib/types"
import { cn } from "@/lib/utils"

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const CATEGORIES = [
  { id: "visual", label: "Darstellung", icon: Box },
  { id: "face", label: "Gesichter & Hände", icon: ScanFace },
  { id: "track", label: "Tracking", icon: Route },
  { id: "zones", label: "Zonen", icon: SquareDashed },
  { id: "overlays", label: "Overlays", icon: Radar },
  { id: "style", label: "Farben", icon: Palette },
] as const
type CategoryId = (typeof CATEGORIES)[number]["id"]

const ease = [0.22, 1, 0.36, 1] as const

interface Props {
  open: boolean
  onClose: () => void
  status: CameraStatus | null
  models: string[]
  settings: EspSettings
  onUpdate: Update
  onToggleAi: () => void
  onModelChange: (model: string) => void
  busy: boolean
  zones: ZoneActions
}

export interface ZoneActions {
  list: Zone[]
  onDraw: () => void
  onRename: (id: string, label: string) => void
  onRecolor: (id: string, color: string) => void
  onDelete: (id: string) => void
}

export function EspMenu({ open, onClose, status, models, settings, onUpdate, onToggleAi, onModelChange, busy, zones }: Props) {
  const [category, setCategory] = useState<CategoryId>("visual")
  const offline = !status
  const aiOn = status?.yolo_enabled ?? false
  const aiAvailable = status?.yolo_available ?? false

  return (
    <AnimatePresence>
      {open && (
        <motion.aside
          key="esp-menu"
          aria-label="ESP-Menü"
          initial={{ opacity: 0, x: 24, scale: 0.98 }}
          animate={{ opacity: 1, x: 0, scale: 1 }}
          exit={{ opacity: 0, x: 24, scale: 0.98 }}
          transition={{ type: "spring", stiffness: 340, damping: 32 }}
          className={cn(
            "absolute z-30 flex flex-col overflow-hidden border border-border bg-card/92 shadow-[0_24px_80px_-24px_rgba(0,0,0,0.6)] backdrop-blur-xl",
            // phone: bottom sheet; desktop: floating panel on the right
            "inset-x-2 bottom-2 max-h-[72dvh] rounded-xl",
            "md:inset-x-auto md:bottom-20 md:right-5 md:top-20 md:max-h-none md:w-[380px]",
          )}
        >
          {/* Header: detection on/off and model */}
          <div className="flex items-center gap-3 border-b border-border px-4 py-3">
            <div className="flex min-w-0 flex-1 items-center gap-3">
              <Switch
                id="ai-toggle"
                checked={aiOn}
                onCheckedChange={onToggleAi}
                disabled={!aiAvailable || busy}
                aria-label="Erkennung"
              />
              <label htmlFor="ai-toggle" className="text-sm">
                Erkennung
                <span className="block text-xs text-muted-foreground">
                  {!status ? "Keine Verbindung" : !aiAvailable ? "YOLO nicht installiert" : aiOn ? "Läuft" : "Aus"}
                </span>
              </label>
            </div>
            <Select value={status?.yolo_model ?? undefined} onValueChange={onModelChange} disabled={!aiAvailable || busy}>
              <SelectTrigger size="sm" className="w-[150px] text-xs" aria-label="Modell">
                <SelectValue placeholder="Modell" />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m} value={m} className="text-xs">
                    {m.replace(/\.pt$/, "")}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <button
              type="button"
              onClick={onClose}
              aria-label="Menü schließen"
              className="-mr-1 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="flex min-h-0 flex-1">
            {/* Category rail */}
            <nav className="flex flex-col gap-1 border-r border-border p-2" aria-label="Kategorien">
              {CATEGORIES.map(({ id, label, icon: Icon }) => {
                const active = id === category
                return (
                  <Tooltip key={id}>
                    <TooltipTrigger asChild>
                      <button
                        type="button"
                        onClick={() => setCategory(id)}
                        aria-label={label}
                        aria-current={active}
                        className={cn(
                          "relative grid size-10 place-items-center rounded-lg transition-colors duration-200",
                          active ? "text-primary" : "text-muted-foreground hover:bg-accent hover:text-foreground",
                        )}
                      >
                        {active && (
                          <motion.span
                            layoutId="category-indicator"
                            className="absolute inset-0 rounded-lg bg-primary/12 ring-1 ring-primary/40"
                            transition={{ type: "spring", stiffness: 450, damping: 34 }}
                          />
                        )}
                        <Icon className="relative size-[18px]" />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="left">{label}</TooltipContent>
                  </Tooltip>
                )
              })}
            </nav>

            {/* Options */}
            <div className={cn("relative min-w-0 flex-1 overflow-y-auto", offline && "pointer-events-none opacity-50")}>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={category}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.22, ease }}
                  className="p-4"
                >
                  <h2 className="mb-3 text-base font-medium">{CATEGORIES.find((c) => c.id === category)?.label}</h2>
                  <CategoryBody category={category} settings={settings} onUpdate={onUpdate} zones={zones} cameraId={status?.camera.id} />
                </motion.div>
              </AnimatePresence>
            </div>
          </div>

          <p className="hidden border-t border-border px-4 py-2 text-xs text-muted-foreground md:block">
            <kbd className="rounded border border-border px-1.5 py-0.5 text-[11px]">M</kbd> blendet das Menü ein und aus
          </p>
        </motion.aside>
      )}
    </AnimatePresence>
  )
}

function CategoryBody({
  category,
  settings,
  onUpdate,
  zones,
  cameraId,
}: {
  category: CategoryId
  settings: EspSettings
  onUpdate: Update
  zones: ZoneActions
  cameraId?: string
}) {
  const sliders = (keys: string[]) => SLIDERS.filter((s) => keys.includes(s.key))
  switch (category) {
    case "visual":
      return (
        <>
          <SegmentPicker
            label="Box-Stil"
            id="box-style"
            options={BOX_STYLES}
            value={String(settings.box_style ?? "corner")}
            onChange={(v) => onUpdate("box_style", v)}
          />
          <ToggleList items={VISUAL_TOGGLES} settings={settings} onUpdate={onUpdate} />
          <SliderList items={sliders(["confidence_threshold", "line_thickness", "chams_opacity"])} settings={settings} onUpdate={onUpdate} />
        </>
      )
    case "face":
      return (
        <>
          <ToggleList items={FACE_TOGGLES} settings={settings} onUpdate={onUpdate} />
          <div className="mt-4 border-t border-border pt-4">
            <p className="mb-2 text-sm text-muted-foreground">Gesichter verbergen als</p>
            <SegmentPicker
              label="Art der Unkenntlichmachung"
              id="face-blur-mode"
              options={FACE_BLUR_MODES}
              value={String(settings.face_blur_mode ?? "blur")}
              onChange={(v) => {
                onUpdate("face_blur_mode", v)
                if (!settings.face_blur) onUpdate("face_blur", true)
              }}
            />
          </div>
          <SliderList
            items={sliders(settings.face_blur_mode === "blackbox" ? ["face_zoom_scale"] : ["face_blur_strength", "face_zoom_scale"])}
            settings={settings}
            onUpdate={onUpdate}
          />
        </>
      )
    case "track":
      return (
        <>
          <ToggleList items={TRACKING_TOGGLES} settings={settings} onUpdate={onUpdate} />
          <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
            Spuren, Geschwindigkeit und Vorhersage brauchen eingeschaltetes Tracking.
          </p>
          <SliderList items={sliders(["smoothing"])} settings={settings} onUpdate={onUpdate} />
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            0 % zeigt die Boxen roh im YOLO-Takt, höher lässt sie zwischen den Erkennungen gleiten.
          </p>
        </>
      )
    case "overlays":
      return <OverlaysPanel cameraId={cameraId} settings={settings} onUpdate={onUpdate} />
    case "zones":
      return <ZonesPanel settings={settings} onUpdate={onUpdate} zones={zones} />
    case "style":
      return (
        <>
          <ToggleList items={STYLE_TOGGLES} settings={settings} onUpdate={onUpdate} />
          <SliderList items={sliders(["jpeg_quality"])} settings={settings} onUpdate={onUpdate} />
          <div className="mt-4 grid grid-cols-2 gap-1 border-t border-border pt-4">
            {COLORS.map((c) => (
              <ColorRow key={c.key} def={c} value={String(settings[c.key] ?? "#ffffff")} onUpdate={onUpdate} />
            ))}
          </div>
        </>
      )
  }
}

function SegmentPicker({
  label,
  id,
  options,
  value,
  onChange,
}: {
  label: string
  id: string
  options: ReadonlyArray<{ value: string; label: string }>
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div role="radiogroup" aria-label={label} className="mb-3 grid grid-cols-3 gap-1 rounded-lg bg-background/60 p-1">
      {options.map((b) => {
        const active = b.value === value
        return (
          <button
            key={b.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(b.value)}
            className={cn(
              "relative rounded-md py-1.5 text-xs transition-colors duration-200",
              active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId={`${id}-indicator`}
                className="absolute inset-0 rounded-md bg-primary"
                transition={{ type: "spring", stiffness: 450, damping: 34 }}
              />
            )}
            <span className="relative">{b.label}</span>
          </button>
        )
      })}
    </div>
  )
}

function ToggleList({ items, settings, onUpdate }: { items: ToggleDef[]; settings: EspSettings; onUpdate: Update }) {
  return (
    <ul className="-mx-2">
      {items.map((t) => {
        const id = `esp-${t.key}`
        const on = Boolean(settings[t.key])
        return (
          <li key={t.key}>
            <label
              htmlFor={id}
              className="flex cursor-pointer items-center justify-between gap-3 rounded-md px-2 py-2 text-sm transition-colors duration-150 hover:bg-accent/70"
            >
              <span className={cn("transition-colors duration-200", on ? "text-foreground" : "text-muted-foreground")}>{t.label}</span>
              <Switch id={id} checked={on} onCheckedChange={(v) => onUpdate(t.key, v)} />
            </label>
          </li>
        )
      })}
    </ul>
  )
}

function SliderList({ items, settings, onUpdate }: { items: SliderDef[]; settings: EspSettings; onUpdate: Update }) {
  return (
    <div className="mt-4 space-y-5 border-t border-border pt-4">
      {items.map((s) => {
        const value = Number(settings[s.key] ?? s.min)
        return (
          <div key={s.key} className="space-y-2.5">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">{s.label}</span>
              <span className="tabular-nums">{s.format(value)}</span>
            </div>
            <Slider
              min={s.min}
              max={s.max}
              step={s.step}
              value={[value]}
              aria-label={s.label}
              onValueChange={([v]) => onUpdate(s.key, v, false)}
            />
          </div>
        )
      })}
    </div>
  )
}

function ColorRow({ def, value, onUpdate }: { def: ToggleDef; value: string; onUpdate: Update }) {
  const [local, setLocal] = useState(value)
  const [prev, setPrev] = useState(value)
  if (value !== prev) {
    setPrev(value)
    setLocal(value)
  }
  const id = `esp-${def.key}`
  return (
    <label
      htmlFor={id}
      className="group flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-accent/70"
    >
      <span className="relative size-5 shrink-0 overflow-hidden rounded-full ring-1 ring-white/15 transition-transform duration-200 group-hover:scale-110">
        <span className="absolute inset-0" style={{ background: local }} />
        <input
          id={id}
          type="color"
          value={local}
          onChange={(e) => {
            setLocal(e.target.value)
            onUpdate(def.key, e.target.value, false)
          }}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
      </span>
      <span className="truncate text-muted-foreground group-hover:text-foreground" title={local.toUpperCase()}>
        {def.label}
      </span>
    </label>
  )
}

function ZonesPanel({ settings, onUpdate, zones }: { settings: EspSettings; onUpdate: Update; zones: ZoneActions }) {
  return (
    <>
      <ToggleList items={[{ key: "zones", label: "Zonen anzeigen" }]} settings={settings} onUpdate={onUpdate} />
      <button
        type="button"
        onClick={zones.onDraw}
        className="mt-3 flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-primary/50 py-2.5 text-sm text-primary transition-colors duration-200 hover:border-primary hover:bg-primary/10"
      >
        <SquareDashed className="size-4" /> Zone zeichnen
        <kbd className="rounded border border-primary/40 px-1.5 text-[11px]">Z</kbd>
      </button>
      {zones.list.length === 0 ? (
        <p className="mt-4 text-xs leading-relaxed text-muted-foreground">
          Noch keine Zonen. Zieh im Bild ein Rechteck auf, zum Beispiel um die Einfahrt. Die Zone zählt, wie viele Objekte gerade darin sind.
        </p>
      ) : (
        <ul className="mt-3 space-y-1">
          <AnimatePresence initial={false}>
            {zones.list.map((z) => (
              <motion.li
                key={z.id}
                layout
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease }}
                className="overflow-hidden"
              >
                <ZoneRow zone={z} zones={zones} />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      )}
    </>
  )
}

function ZoneRow({ zone, zones }: { zone: Zone; zones: ZoneActions }) {
  const [label, setLabel] = useState(zone.label)
  const [prev, setPrev] = useState(zone.label)
  if (zone.label !== prev) {
    setPrev(zone.label)
    setLabel(zone.label)
  }
  const commit = () => {
    const v = label.trim().slice(0, 32)
    if (v && v !== zone.label) zones.onRename(zone.id, v)
    else setLabel(zone.label)
  }
  return (
    <div className="group flex items-center gap-2 rounded-md px-2 py-1.5 transition-colors hover:bg-accent/70">
      <span className="relative size-4 shrink-0 overflow-hidden rounded-full ring-1 ring-white/15">
        <span className="absolute inset-0" style={{ background: zone.color }} />
        <input
          type="color"
          value={zone.color}
          aria-label={`Farbe von ${zone.label}`}
          onChange={(e) => zones.onRecolor(zone.id, e.target.value)}
          className="absolute inset-0 cursor-pointer opacity-0"
        />
      </span>
      <input
        value={label}
        maxLength={32}
        aria-label="Name der Zone"
        onChange={(e) => setLabel(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        className="min-w-0 flex-1 rounded bg-transparent px-1 py-0.5 text-sm outline-none focus:bg-background/60 focus:ring-1 focus:ring-ring"
      />
      <span
        className={cn(
          "min-w-7 rounded-full px-2 py-0.5 text-center text-xs tabular-nums transition-colors duration-300",
          zone.count > 0 ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
        )}
        title="Objekte in der Zone"
      >
        {zone.count}
      </span>
      <button
        type="button"
        onClick={() => zones.onDelete(zone.id)}
        aria-label={`${zone.label} löschen`}
        className="rounded-md p-1 text-muted-foreground opacity-0 transition-all hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
      >
        <Trash2 className="size-3.5" />
      </button>
    </div>
  )
}
