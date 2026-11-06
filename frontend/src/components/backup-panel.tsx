import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { ArrowRight, Download, Upload } from "lucide-react"
import { opsApi, type ImportResult, type PersistState } from "@/lib/ops-api"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const
const MAX_SHOWN = 60

function show(v: unknown): string {
  if (v === null || v === undefined) return "–"
  if (typeof v === "string") return v
  const s = JSON.stringify(v)
  return s.length > 48 ? `${s.slice(0, 45)}…` : s
}

function timeOf(ts: number) {
  return ts ? new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : null
}

/** Export all saved configuration as JSON; import with a diff preview first. */
export function BackupPanel({ onImported }: { onImported: () => void }) {
  const [persist, setPersist] = useState<PersistState | null>(null)
  const [preview, setPreview] = useState<{ name: string; text: string; result: ImportResult } | null>(null)
  const [message, setMessage] = useState<{ text: string; tone: "ok" | "error" } | null>(null)
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let alive = true
    const load = () => opsApi.persistState().then((s) => alive && setPersist(s), () => undefined)
    void load()
    const t = window.setInterval(load, 5000)
    return () => {
      alive = false
      window.clearInterval(t)
    }
  }, [])

  const pick = async (file: File) => {
    setMessage(null)
    setBusy(true)
    try {
      const text = await file.text()
      const result = await opsApi.importSettings(text, true)
      if (!result.ok) {
        setPreview(null)
        setMessage({ text: "This file is not a valid SecurityCam settings export.", tone: "error" })
      } else setPreview({ name: file.name, text, result })
    } catch {
      setMessage({ text: "The file could not be read.", tone: "error" })
    } finally {
      setBusy(false)
      if (fileRef.current) fileRef.current.value = ""
    }
  }

  const apply = async () => {
    if (!preview) return
    setBusy(true)
    try {
      const r = await opsApi.importSettings(preview.text, false)
      if (!r.ok) throw new Error(r.error)
      setPreview(null)
      setMessage({ text: `Imported ${r.applied ?? 0} change${r.applied === 1 ? "" : "s"}.`, tone: "ok" })
      onImported()
    } catch {
      setMessage({ text: "Import failed. Nothing was changed.", tone: "error" })
    } finally {
      setBusy(false)
    }
  }

  const diff = preview?.result.diff ?? []
  const saved = timeOf(persist?.last_saved_at ?? 0)

  return (
    <>
      <p className="-mt-1 mb-4 text-xs leading-relaxed text-muted-foreground">
        ESP settings, detection, zones, presets, schedules and PTZ calibration are saved automatically
        {persist ? <> to <code className="text-foreground/80">data/state.json</code></> : null}
        {saved ? `, last at ${saved}` : ""}.
      </p>
      {persist?.last_error && <p className="mb-3 text-xs text-destructive">{persist.last_error}</p>}

      <div className="grid grid-cols-2 gap-2">
        <a
          href={opsApi.exportUrl}
          download
          className="flex items-center justify-center gap-2 rounded-lg border border-border py-2.5 text-sm transition-colors duration-200 hover:border-primary/60 hover:text-primary"
        >
          <Download className="size-4" /> Export
        </a>
        <button
          type="button"
          disabled={busy}
          onClick={() => fileRef.current?.click()}
          className="flex items-center justify-center gap-2 rounded-lg border border-dashed border-primary/50 py-2.5 text-sm text-primary transition-colors duration-200 hover:border-primary hover:bg-primary/10 disabled:opacity-50"
        >
          <Upload className="size-4" /> Import
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          aria-label="Settings file"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) void pick(f)
          }}
        />
      </div>

      <AnimatePresence mode="wait">
        {preview && (
          <motion.section
            key="preview"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease }}
            className="mt-4 rounded-lg border border-border bg-background/50 p-3"
            aria-label="Import preview"
          >
            <p className="truncate text-sm">{preview.name}</p>
            <p className="mb-2 text-xs text-muted-foreground">
              {diff.length === 0 ? "Nothing would change." : `${diff.length} change${diff.length === 1 ? "" : "s"} in ${preview.result.sections.join(", ")}`}
            </p>
            {diff.length > 0 && (
              <ul className="max-h-56 space-y-1 overflow-y-auto pr-1 text-[11px]">
                {diff.slice(0, MAX_SHOWN).map((d) => (
                  <li key={d.path} className="rounded bg-muted/40 px-2 py-1">
                    <span className="block truncate text-muted-foreground" title={d.path}>
                      {d.path}
                    </span>
                    <span className="flex items-center gap-1.5 tabular-nums">
                      <span className="truncate text-foreground/60 line-through decoration-foreground/30">{show(d.before)}</span>
                      <ArrowRight className="size-3 shrink-0 text-primary" />
                      <span className="truncate text-primary">{show(d.after)}</span>
                    </span>
                  </li>
                ))}
                {diff.length > MAX_SHOWN && <li className="px-2 text-muted-foreground">and {diff.length - MAX_SHOWN} more</li>}
              </ul>
            )}
            {preview.result.warnings.length > 0 && (
              <ul className="mt-2 space-y-0.5 text-[11px] text-muted-foreground">
                {preview.result.warnings.slice(0, 8).map((w) => (
                  <li key={w}>· {w}</li>
                ))}
              </ul>
            )}
            <div className="mt-3 flex gap-2">
              <button
                type="button"
                disabled={busy || diff.length === 0}
                onClick={() => void apply()}
                className="flex-1 rounded-md bg-primary py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-40"
              >
                Apply
              </button>
              <button
                type="button"
                onClick={() => setPreview(null)}
                className="flex-1 rounded-md border border-border py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
              >
                Discard
              </button>
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {message && (
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role={message.tone === "error" ? "alert" : "status"}
            className={cn("mt-3 text-xs", message.tone === "error" ? "text-destructive" : "text-primary")}
          >
            {message.text}
          </motion.p>
        )}
      </AnimatePresence>
    </>
  )
}
