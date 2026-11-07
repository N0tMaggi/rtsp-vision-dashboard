import { useCallback, useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ImagePlus, Loader2, Trash2 } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import type { EspSettings } from "@/lib/types"
import { cn } from "@/lib/utils"

export interface FaceMask {
  id: string
  name: string
  url: string
  w: number
  h: number
  alpha: boolean
}

type Update = (key: string, value: boolean | number | string, immediate?: boolean) => void

const FITS = [
  { value: "cover", label: "Cover", hint: "Fills the face area, crops the image" },
  { value: "contain", label: "Fit", hint: "Whole image visible" },
  { value: "stretch", label: "Stretch", hint: "Distorts to the face area" },
] as const

const ERRORS: Record<string, string> = {
  invalid_image: "That file is not an image OpenCV can read (PNG, JPEG, WebP, BMP).",
  too_large: "Image too large (max 6 MB).",
  too_many_images: "Image limit reached - delete one first.",
}

async function listMasks(): Promise<FaceMask[]> {
  const r = await fetch("/api/face-masks", { cache: "no-store" })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return ((await r.json()) as { masks: FaceMask[] }).masks
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => resolve(String(fr.result))
    fr.onerror = () => reject(fr.error)
    fr.readAsDataURL(file)
  })
}

/** Library of custom face-cover images for the "Image" face blur mode. */
export function FaceImagePicker({ settings, onUpdate }: { settings: EspSettings; onUpdate: Update }) {
  const [masks, setMasks] = useState<FaceMask[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const selected = String(settings.face_blur_image ?? "")
  const fit = String(settings.face_blur_image_fit ?? "cover")
  const underlay = settings.face_blur_image_underlay !== false

  const refresh = useCallback(() => {
    listMasks()
      .then(setMasks)
      .catch(() => setMasks([]))
  }, [])
  useEffect(refresh, [refresh])

  const upload = async (files: FileList | File[] | null) => {
    const file = files?.[0]
    if (!file) return
    setError(null)
    if (!file.type.startsWith("image/")) {
      setError("Pick an image file.")
      return
    }
    if (file.size > 6 * 1024 * 1024) {
      setError(ERRORS.too_large)
      return
    }
    setBusy(true)
    try {
      const data = await readAsDataUrl(file)
      const r = await fetch("/api/face-masks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: file.name.replace(/\.[^.]+$/, ""), data }),
      })
      const body = (await r.json().catch(() => ({}))) as { mask?: FaceMask; error?: string }
      if (!r.ok || !body.mask) throw new Error(ERRORS[body.error ?? ""] ?? "Upload failed.")
      setMasks((m) => [...(m ?? []), body.mask!])
      onUpdate("face_blur_image", body.mask.id, true)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.")
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ""
    }
  }

  const remove = async (m: FaceMask) => {
    const r = await fetch(`/api/face-masks/${m.id}`, { method: "DELETE" })
    if (r.ok || r.status === 404) {
      setMasks((list) => (list ?? []).filter((x) => x.id !== m.id))
      if (selected === m.id) onUpdate("face_blur_image", "", true)
    }
  }

  const missing = masks !== null && selected !== "" && !masks.some((m) => m.id === selected)

  return (
    <div className="mb-3 space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          void upload(e.dataTransfer.files)
        }}
        className={cn(
          "grid grid-cols-4 gap-1.5 rounded-lg border border-dashed p-1.5 transition-colors duration-200",
          dragOver ? "border-primary bg-primary/10" : "border-border/70 bg-background/40",
        )}
      >
        <AnimatePresence initial={false}>
          {(masks ?? []).map((m) => {
            const active = m.id === selected
            return (
              <motion.div
                key={m.id}
                layout
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                transition={{ type: "spring", stiffness: 420, damping: 32 }}
                className="group relative aspect-square"
              >
                <button
                  type="button"
                  title={m.name}
                  aria-pressed={active}
                  onClick={() => onUpdate("face_blur_image", m.id, true)}
                  className={cn(
                    "size-full overflow-hidden rounded-md border-2 transition-[border-color,transform] duration-200 active:scale-95",
                    "bg-[conic-gradient(#2a2b2f_25%,#1f2023_0_50%,#2a2b2f_0_75%,#1f2023_0)] bg-[length:12px_12px]",
                    active ? "border-primary" : "border-transparent hover:border-border",
                  )}
                >
                  <img src={m.url} alt={m.name} draggable={false} className="size-full object-contain" />
                </button>
                <button
                  type="button"
                  aria-label={`Delete ${m.name}`}
                  onClick={() => void remove(m)}
                  className="absolute -right-1 -top-1 hidden rounded-full border border-border bg-background p-1 text-muted-foreground hover:text-destructive group-hover:block group-focus-within:block"
                >
                  <Trash2 className="size-3" />
                </button>
              </motion.div>
            )
          })}
        </AnimatePresence>
        <button
          type="button"
          onClick={() => input.current?.click()}
          disabled={busy}
          title="Upload an image (or drop it here). Transparent PNGs work best."
          className="flex aspect-square flex-col items-center justify-center gap-1 rounded-md border border-border/60 text-[10px] text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground disabled:opacity-60"
        >
          {busy ? <Loader2 className="size-4 animate-spin" /> : <ImagePlus className="size-4" />}
          {busy ? "Uploading" : "Upload"}
        </button>
        <input
          ref={input}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp,image/gif"
          className="hidden"
          onChange={(e) => void upload(e.target.files)}
        />
      </div>

      {error && <p className="text-xs text-destructive">{error}</p>}
      {masks !== null && masks.length === 0 && !error && (
        <p className="text-xs leading-relaxed text-muted-foreground">
          Upload an image to put over every face. Until one is picked, faces get a black box.
        </p>
      )}
      {missing && <p className="text-xs text-muted-foreground">Selected image was deleted - faces get a black box.</p>}

      <div>
        <p className="mb-1.5 text-xs text-muted-foreground">Image Fit</p>
        <div role="radiogroup" aria-label="Image Fit" className="grid grid-cols-3 gap-1 rounded-lg bg-background/60 p-1">
          {FITS.map((f) => {
            const active = f.value === fit
            return (
              <button
                key={f.value}
                type="button"
                role="radio"
                aria-checked={active}
                title={f.hint}
                onClick={() => onUpdate("face_blur_image_fit", f.value, true)}
                className={cn(
                  "relative rounded-md py-1.5 text-xs transition-colors duration-200",
                  active ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {active && (
                  <motion.span
                    layoutId="face-image-fit-indicator"
                    className="absolute inset-0 rounded-md bg-primary"
                    transition={{ type: "spring", stiffness: 450, damping: 34 }}
                  />
                )}
                <span className="relative">{f.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      <label className="flex items-center justify-between gap-3 text-sm">
        <span>
          Blur Underneath
          <span className="block text-xs text-muted-foreground">Transparent parts of the image never show the face.</span>
        </span>
        <Switch checked={underlay} onCheckedChange={(v) => onUpdate("face_blur_image_underlay", v, true)} />
      </label>
    </div>
  )
}
