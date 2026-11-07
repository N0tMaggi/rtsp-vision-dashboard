import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Check, Loader2, Pencil, ScanText, X } from "lucide-react"
import { contextApi, contextError, type PlateCandidate } from "@/lib/context-api"
import type { PlateLogEntry } from "@/lib/platelog"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const

/**
 * Plate viewer side panel: run the multi-variant OCR again on the stored crop,
 * or correct the text by hand. Nothing changes until "Apply".
 */
export function PlateReread({ entry }: { entry: PlateLogEntry }) {
  const [state, setState] = useState<"idle" | "reading" | "editing" | "saving">("idle")
  const [cands, setCands] = useState<PlateCandidate[] | null>(null)
  const [text, setText] = useState(entry.plate)
  const [err, setErr] = useState<string | null>(null)
  const [saved, setSaved] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  // another entry in the viewer -> start over
  useEffect(() => {
    setState("idle")
    setCands(null)
    setText(entry.plate)
    setErr(null)
    setSaved(null)
  }, [entry.id, entry.plate])

  useEffect(() => {
    if (state === "editing") input.current?.select()
  }, [state])

  const reread = async () => {
    setState("reading")
    setErr(null)
    try {
      const r = await contextApi.reocr(entry.id)
      setCands(r.candidates)
      if (r.text) setText(r.text)
      setState("editing")
    } catch (e) {
      setErr(contextError(e))
      setState("idle")
    }
  }

  const apply = async (plate: string) => {
    setState("saving")
    try {
      const r = await contextApi.setPlate(entry.id, plate)
      setSaved(r.entry.plate)
      setState("idle")
      setCands(null)
    } catch (e) {
      setErr(contextError(e))
      setState("editing")
    }
  }

  return (
    <div className="space-y-2" onKeyDown={(e) => state === "editing" && e.stopPropagation()}>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={!entry.plate_url || state === "reading" || state === "saving"}
          onClick={() => void reread()}
          className="flex flex-1 items-center justify-center gap-2 rounded-lg border border-border px-3 py-2 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground disabled:opacity-40"
        >
          {state === "reading" ? <Loader2 className="size-3.5 animate-spin" /> : <ScanText className="size-3.5" />} Re-Read Plate
        </button>
        <button
          type="button"
          aria-label="Correct plate text"
          title="Correct plate text"
          onClick={() => setState(state === "editing" ? "idle" : "editing")}
          className={cn(
            "grid size-[34px] place-items-center rounded-lg border transition-colors",
            state === "editing" ? "border-primary/60 text-primary" : "border-border text-muted-foreground hover:text-foreground",
          )}
        >
          <Pencil className="size-3.5" />
        </button>
      </div>
      <AnimatePresence initial={false}>
        {(state === "editing" || state === "saving") && (
          <motion.div
            key="edit"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease }}
            className="overflow-hidden"
          >
            <form
              className="flex gap-1.5"
              onSubmit={(e) => {
                e.preventDefault()
                if (text.trim()) void apply(text.trim())
              }}
            >
              <input
                ref={input}
                value={text}
                maxLength={20}
                aria-label="Plate text"
                onChange={(e) => setText(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === "Escape" && setState("idle")}
                className="min-w-0 flex-1 rounded-md border border-border bg-background/60 px-2 py-1.5 text-sm font-semibold tracking-widest outline-none focus:border-primary/60"
              />
              <button type="submit" aria-label="Apply" disabled={state === "saving"} className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground disabled:opacity-50">
                {state === "saving" ? <Loader2 className="size-3.5 animate-spin" /> : <Check className="size-3.5" />}
              </button>
              <button type="button" aria-label="Cancel" onClick={() => setState("idle")} className="grid size-8 place-items-center rounded-md text-muted-foreground hover:text-foreground">
                <X className="size-3.5" />
              </button>
            </form>
            {cands && cands.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {cands.map((c) => (
                  <button
                    key={c.text}
                    type="button"
                    onClick={() => setText(c.text)}
                    className={cn(
                      "rounded-full border px-2 py-0.5 text-[11px] tracking-wider transition-colors",
                      c.text === text ? "border-primary/70 text-primary" : "border-white/10 hover:border-white/30",
                    )}
                  >
                    {c.text} <span className="text-muted-foreground">{Math.round(c.conf * 100)}%</span>
                  </button>
                ))}
              </div>
            )}
            {cands && cands.length === 0 && <p className="mt-1.5 text-[11px] text-muted-foreground">The reader found nothing. Type the plate.</p>}
          </motion.div>
        )}
      </AnimatePresence>
      {saved && <p className="text-[11px] text-[#7ee0a1]">Saved as {saved}.</p>}
      {err && <p className="text-[11px] text-destructive">{err}</p>}
    </div>
  )
}
