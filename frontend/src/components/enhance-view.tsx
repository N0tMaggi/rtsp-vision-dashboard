import { useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react"
import { AnimatePresence, motion } from "motion/react"
import {
  Check,
  ClipboardCopy,
  Cpu,
  Download,
  ImageDown,
  Info,
  Loader2,
  Maximize2,
  RefreshCw,
  ScanLine,
  Sparkles,
  Trash2,
  X,
} from "lucide-react"
import { useViewTransform } from "@/hooks/use-view-transform"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import {
  ACTIVE,
  NOTICE,
  EnhanceError,
  closeEnhance,
  enhanceApi,
  errorText,
  loadPrefs,
  openEnhance,
  savePrefs,
  useEnhanceState,
  type EnhanceIndex,
  type EnhanceJob,
  type EnhanceOptions,
} from "@/lib/enhance"
import type { NBox } from "@/lib/types"
import { cn } from "@/lib/utils"

const spring = { type: "spring", stiffness: 380, damping: 34 } as const
const POLL_MS = 250
const HISTORY_MAX = 12

/**
 * Enhance view: AI-upscales the frame region that was on screen (paused / zoomed)
 * for exports. Mounted once in App; opened via `openEnhance()` (toolbar button,
 * playback bar, X key, or a context menu). Every result is labelled as AI
 * enhanced: badge, persistent notice, and the watermark band on exports.
 */
export function EnhanceView() {
  const st = useEnhanceState()

  // X opens the view for the camera on the stage (current zoom viewport).
  useEffect(() => {
    if (st.open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return
      if ((e.target as HTMLElement).closest("input, textarea, [contenteditable], [role=combobox], [role=listbox], [role=dialog]")) return
      if ((e.key === "x" || e.key === "X") && st.stageCameraId) {
        e.preventDefault()
        openEnhance()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [st.open, st.stageCameraId])

  return (
    <AnimatePresence>
      {st.open && st.cameraId && (
        <motion.div
          key="enhance-view"
          className="absolute inset-0 z-50 grid place-items-center bg-[#16171a]/85 p-2 backdrop-blur-md md:p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onPointerDown={(e) => e.target === e.currentTarget && closeEnhance()}
          role="dialog"
          aria-modal="true"
          aria-label="Enhance"
        >
          <EnhancePanel key={st.nonce} cameraId={st.cameraId} region={st.region} />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function statusText(job: EnhanceJob): string {
  switch (job.status) {
    case "queued":
      return job.queue_position && job.queue_position > 1 ? `Queued (#${job.queue_position})` : "Queued"
    case "downloading":
      return "Downloading model"
    case "loading":
      return "Loading model"
    case "running":
      return `Enhancing ${Math.round(job.progress * 100)}%`
    case "saving":
      return "Saving PNG"
    case "done":
      return "Done"
    case "canceled":
      return "Canceled"
    default:
      return "Failed"
  }
}

function secs(v?: number) {
  if (v === undefined) return "–"
  return v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`
}

function EnhancePanel({ cameraId, region }: { cameraId: string; region: NBox | null }) {
  const [index, setIndex] = useState<EnhanceIndex | null>(null)
  const [opts, setOpts] = useState<EnhanceOptions>(loadPrefs)
  const [job, setJob] = useState<EnhanceJob | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showBand, setShowBand] = useState(true)
  const [copy, setCopy] = useState<"idle" | "ok" | "err">("idle")
  const [copyMsg, setCopyMsg] = useState("")
  const compareRef = useRef<CompareHandle | null>(null)
  const started = useRef(false)

  const refreshIndex = useCallback(() => {
    enhanceApi
      .index()
      .then(setIndex)
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    refreshIndex()
  }, [refreshIndex])

  // keep chosen options valid once the model list is known
  const models = index?.models ?? []
  const model = models.find((m) => m.id === opts.model) ?? models.find((m) => m.id === index?.default_model) ?? models[0]

  const updateOpts = (patch: Partial<EnhanceOptions>) =>
    setOpts((o) => {
      const n = { ...o, ...patch }
      savePrefs(n)
      return n
    })

  const start = useCallback(
    async (fromJob: EnhanceJob | null) => {
      setError(null)
      try {
        const res = fromJob
          ? await enhanceApi.start(fromJob.camera_id, { ...opts, source_job: fromJob.source_job ?? fromJob.id })
          : await enhanceApi.start(cameraId, { ...opts, region })
        setJob(res.job)
      } catch (e) {
        setError(errorText(e instanceof EnhanceError ? e.code : "network"))
      }
    },
    [cameraId, region, opts],
  )

  // Capture right away: for live cameras the frame at the moment the view opened.
  useEffect(() => {
    if (started.current) return
    started.current = true
    void start(null)
  }, [start])

  // Poll the running job.
  const jobId = job?.id
  const active = job ? ACTIVE.includes(job.status) : false
  useEffect(() => {
    if (!jobId || !active) return
    let stopped = false
    let timer: number | undefined
    const ctrl = new AbortController()
    const tick = async () => {
      try {
        const j = await enhanceApi.job(jobId, ctrl.signal)
        if (stopped) return
        setJob(j)
        if (!ACTIVE.includes(j.status)) {
          refreshIndex()
          if (j.status === "failed") setError(errorText(j.error ?? "enhance_failed"))
          return
        }
      } catch {
        if (stopped) return
      }
      timer = window.setTimeout(tick, POLL_MS)
    }
    timer = window.setTimeout(tick, POLL_MS)
    return () => {
      stopped = true
      window.clearTimeout(timer)
      ctrl.abort()
    }
  }, [jobId, active, refreshIndex])

  const cancel = useCallback(() => {
    if (job && ACTIVE.includes(job.status)) void enhanceApi.cancel(job.id).catch(() => undefined)
  }, [job])

  const close = useCallback(() => closeEnhance(), [])

  const remove = useCallback(
    async (j: EnhanceJob) => {
      try {
        await enhanceApi.remove(j.id)
      } catch {
        // already gone
      }
      if (job?.id === j.id) setJob(null)
      refreshIndex()
    },
    [job, refreshIndex],
  )

  const done = job?.status === "done"
  const enhancedUrl = done ? job?.urls.enhanced : undefined

  const copyImage = useCallback(async () => {
    if (!enhancedUrl) return
    try {
      if (!window.isSecureContext || typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
        throw new Error("insecure")
      }
      const blob = await fetch(enhancedUrl).then((r) => r.blob())
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
      setCopy("ok")
      setCopyMsg("Copied with the AI ENHANCED band")
    } catch (e) {
      setCopy("err")
      setCopyMsg(
        e instanceof Error && e.message === "insecure"
          ? "Copying images needs HTTPS or localhost. Use Download PNG."
          : "The browser refused the clipboard.",
      )
    }
    window.setTimeout(() => setCopy("idle"), 2200)
  }, [enhancedUrl])

  // Keyboard: captured while open so stage shortcuts never fire underneath.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return
      if ((e.target as HTMLElement).closest("input, textarea, [role=slider]")) return
      e.stopImmediatePropagation()
      const k = e.key
      if (k === "Escape") close()
      else if (k === "0") compareRef.current?.reset()
      else if (k === "+" || k === "=") compareRef.current?.zoom(1.4)
      else if (k === "-") compareRef.current?.zoom(1 / 1.4)
      else if (k === "ArrowLeft") compareRef.current?.nudge(-0.05)
      else if (k === "ArrowRight") compareRef.current?.nudge(0.05)
      else if (k === "b" || k === "B") setShowBand((v) => !v)
      else if (k === "Enter" && job && !ACTIVE.includes(job.status)) void start(job)
    }
    window.addEventListener("keydown", onKey, true)
    return () => window.removeEventListener("keydown", onKey, true)
  }, [close, job, start])

  const history = (index?.jobs ?? []).filter((j) => j.status === "done").slice(0, HISTORY_MAX)
  const aiResult = done && job?.ai_enhanced
  const optionsChanged =
    job && (job.options.model !== (model?.id ?? opts.model) || job.options.scale !== opts.scale || (model?.denoise && job.options.denoise !== opts.denoise))

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 10, scale: 0.98 }}
      transition={spring}
      className="flex h-full max-h-[960px] w-full max-w-7xl flex-col overflow-hidden rounded-xl border border-border bg-card shadow-[0_32px_120px_-24px_rgba(0,0,0,0.8)]"
    >
      {/* header */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-4 py-3">
        <span className="grid size-8 place-items-center rounded-full bg-primary/15 text-primary">
          <Sparkles className="size-4" />
        </span>
        <div className="min-w-0 flex-1 text-xs leading-relaxed text-muted-foreground">
          <p className="truncate text-sm text-foreground">
            Enhance
            {job && <span className="text-muted-foreground"> · {job.camera_label}</span>}
          </p>
          {job ? (
            <p className="truncate tabular-nums">
              {job.source.timestamp_iso.replace("T", " ")}
              {job.source.position_s != null && ` · video ${job.source.position_s.toFixed(2)} s`} · {job.crop_size[0]}×{job.crop_size[1]} px region of{" "}
              {job.frame_size[0]}×{job.frame_size[1]}
            </p>
          ) : (
            <p>Capturing the frame on screen</p>
          )}
        </div>
        <AnimatePresence mode="popLayout">
          {done && (
            <motion.span
              key={aiResult ? "ai" : "noai"}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={spring}
              className={cn(
                "flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold tracking-wider",
                aiResult ? "bg-primary text-primary-foreground" : "border border-border text-muted-foreground",
              )}
            >
              {aiResult ? <Sparkles className="size-3" /> : <ScanLine className="size-3" />}
              {aiResult ? "AI ENHANCED" : "UPSCALED · NO AI"}
            </motion.span>
          )}
        </AnimatePresence>
        <button
          type="button"
          onClick={close}
          aria-label="Close (Esc)"
          title="Close (Esc)"
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <X className="size-4" />
        </button>
      </div>

      {/* body */}
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <div className="relative flex min-h-[300px] flex-1 flex-col bg-[#1b1c1f]">
          <div className="relative min-h-0 flex-1">
            {job?.urls.original ? (
              <Compare
                ref={compareRef}
                job={job}
                before={job.urls.original}
                after={enhancedUrl}
                showBand={showBand}
              />
            ) : (
              <div className="grid h-full place-items-center">
                {error ? (
                  <p role="alert" className="max-w-sm px-6 text-center text-sm text-destructive">
                    {error}
                  </p>
                ) : (
                  <p className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Loader2 className="size-4 animate-spin text-primary" /> {job ? statusText(job) : "Capturing frame"}
                  </p>
                )}
              </div>
            )}
            {job?.urls.original && (
              <div className="absolute bottom-3 left-3 flex gap-1">
                <PillButton label="Fit (0)" onClick={() => compareRef.current?.reset()}>
                  <Maximize2 className="size-4" />
                </PillButton>
              </div>
            )}
          </div>
          {/* persistent honesty notice */}
          <div className="flex items-start gap-2 border-t border-primary/30 bg-primary/[0.07] px-4 py-2 text-[12px] leading-relaxed text-foreground">
            <Info className="mt-0.5 size-3.5 shrink-0 text-primary" />
            <span>{index?.notice ?? NOTICE}</span>
          </div>
        </div>

        {/* side rail */}
        <aside className="flex shrink-0 flex-col gap-4 overflow-y-auto *:shrink-0 border-t border-border p-4 md:w-[300px] md:border-l md:border-t-0">
          <Section title="Model">
            <div className="flex flex-col gap-1">
              {models.map((m) => {
                const on = m.id === model?.id
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => updateOpts({ model: m.id })}
                    aria-pressed={on}
                    className={cn(
                      "relative rounded-lg px-3 py-2 text-left transition-colors duration-200",
                      on ? "text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
                    )}
                  >
                    {on && (
                      <motion.span
                        layoutId="enhance-model"
                        className="absolute inset-0 rounded-lg border border-primary/70 bg-primary/10"
                        transition={spring}
                      />
                    )}
                    <span className="relative flex items-center gap-2 text-[13px]">
                      {m.label}
                      {m.ai && <span className="rounded bg-primary/20 px-1 text-[10px] font-semibold text-primary">AI</span>}
                    </span>
                    <span className="relative mt-0.5 block text-[11px] leading-snug text-muted-foreground">
                      {m.note}
                      {m.ai && !m.ready && ` · ${m.downloading ? "downloading" : `${Math.round(m.size_mb)} MB download on first use`}`}
                    </span>
                  </button>
                )
              })}
              {!models.length && <p className="text-xs text-muted-foreground">Loading models</p>}
            </div>
          </Section>

          <Section title="Scale">
            <div className="flex rounded-full border border-border p-0.5 text-xs">
              {(index?.scales ?? [2, 4]).map((s) => {
                const on = opts.scale === s
                return (
                  <button
                    key={s}
                    type="button"
                    onClick={() => updateOpts({ scale: s })}
                    aria-pressed={on}
                    className={cn("relative flex-1 rounded-full py-1 tabular-nums transition-colors", on ? "text-primary-foreground" : "text-muted-foreground hover:text-foreground")}
                  >
                    {on && <motion.span layoutId="enhance-scale" className="absolute inset-0 rounded-full bg-primary" transition={spring} />}
                    <span className="relative">{s}×</span>
                  </button>
                )
              })}
            </div>
          </Section>

          <AnimatePresence initial={false}>
            {model?.denoise && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                <Section title="Denoise" aside={`${Math.round(opts.denoise * 100)}%`}>
                  <Slider
                    value={[opts.denoise]}
                    min={0}
                    max={1}
                    step={0.05}
                    onValueChange={(v) => updateOpts({ denoise: v[0] ?? 0.5 })}
                    aria-label="Denoise strength"
                  />
                  <p className="mt-1.5 text-[11px] text-muted-foreground">Low keeps grain and texture, high smooths noise.</p>
                </Section>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex flex-col gap-1.5">
            {active ? (
              <button
                type="button"
                onClick={cancel}
                className="flex items-center justify-center gap-2 rounded-lg border border-border px-3 py-2.5 text-sm transition-colors hover:border-destructive/60 hover:text-destructive"
              >
                <Loader2 className="size-4 animate-spin" /> {job ? statusText(job) : ""} · Cancel
              </button>
            ) : (
              <motion.button
                type="button"
                whileTap={{ scale: 0.98 }}
                onClick={() => void start(job)}
                disabled={!index}
                className={cn(
                  "flex items-center justify-center gap-2 rounded-lg px-3 py-2.5 text-sm font-semibold transition-colors disabled:opacity-50",
                  optionsChanged || !job ? "bg-primary text-primary-foreground" : "border border-border text-foreground hover:border-primary/60",
                )}
              >
                <Sparkles className="size-4" /> {job ? "Enhance Again" : "Enhance"}
                <kbd className="ml-1 rounded border border-current/30 px-1 text-[10px] font-normal opacity-70">Enter</kbd>
              </motion.button>
            )}
            <button
              type="button"
              onClick={() => void start(null)}
              disabled={active}
              className="flex items-center justify-center gap-1.5 text-[11px] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
              title="Captures the frame on screen now (live cameras move on)"
            >
              <RefreshCw className="size-3" /> New Capture
            </button>
            {job && <p className="text-[11px] text-muted-foreground">Enhance Again reuses the captured pixels.</p>}
            {error && job?.urls.original && <p role="alert" className="text-[11px] text-destructive">{error}</p>}
          </div>

          <AnimatePresence initial={false}>
            {done && job && (
              <motion.div
                key={job.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={spring}
                className="flex flex-col gap-4"
              >
                <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
                  <Stat label="Device" value={job.device_used ?? "–"} wide icon />
                  <Stat label="Inference" value={secs(job.timings.infer_s)} />
                  <Stat label="Total" value={secs(job.timings.total_s)} />
                  <Stat label="Output" value={`${job.output_size[0]}×${job.output_size[1]}`} />
                  <Stat label="Model" value={`${job.options.scale}× ${job.model.ai ? "AI" : "classic"}`} />
                  {job.fallback && (
                    <p className="col-span-2 text-[11px] text-destructive">
                      {job.fallback === "model_unavailable"
                        ? "The AI model could not be downloaded; Lanczos was used instead."
                        : "The chosen device failed; the CPU was used instead."}
                    </p>
                  )}
                </dl>

                <label className="flex items-center justify-between gap-3 text-xs">
                  <span>
                    Watermark band in preview
                    <span className="block text-[11px] text-muted-foreground">Exports always include it.</span>
                  </span>
                  <Switch checked={showBand} onCheckedChange={setShowBand} aria-label="Show watermark band in preview (B)" />
                </label>

                <div className="flex flex-col gap-1.5">
                  <a
                    href={`${job.urls.enhanced}?download=1`}
                    className="flex items-center justify-center gap-2 rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-primary-foreground transition-transform active:scale-[0.98]"
                  >
                    <Download className="size-4" /> Download PNG
                  </a>
                  <div className="grid grid-cols-2 gap-1.5">
                    <a
                      href={`${job.urls.original}?download=1`}
                      className="flex items-center justify-center gap-1.5 rounded-lg border border-border px-2 py-2 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
                    >
                      <ImageDown className="size-3.5" /> Original Crop
                    </a>
                    <button
                      type="button"
                      onClick={() => void copyImage()}
                      className="flex items-center justify-center gap-1.5 rounded-lg border border-border px-2 py-2 text-xs text-muted-foreground transition-colors hover:border-primary/60 hover:text-foreground"
                    >
                      {copy === "ok" ? <Check className="size-3.5 text-primary" /> : <ClipboardCopy className="size-3.5" />} Copy
                    </button>
                  </div>
                  <AnimatePresence>
                    {copy !== "idle" && (
                      <motion.p
                        initial={{ opacity: 0, y: -4 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className={cn("text-[11px]", copy === "ok" ? "text-muted-foreground" : "text-destructive")}
                      >
                        {copyMsg}
                      </motion.p>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <Section title="Recent" aside={history.length ? String(history.length) : undefined}>
            {history.length ? (
              <ul className="flex flex-col gap-1">
                <AnimatePresence initial={false}>
                  {history.map((h) => (
                    <motion.li
                      key={h.id}
                      layout
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={spring}
                      className={cn(
                        "group flex items-center gap-2 rounded-lg p-1 pr-1.5 transition-colors",
                        h.id === job?.id ? "bg-primary/10 ring-1 ring-primary/50" : "hover:bg-accent/60",
                      )}
                    >
                      <button type="button" onClick={() => setJob(h)} className="flex min-w-0 flex-1 items-center gap-2 text-left">
                        {h.urls.original && (
                          <img src={h.urls.original} alt="" className="h-9 w-14 shrink-0 rounded object-cover [image-rendering:pixelated]" loading="lazy" />
                        )}
                        <span className="min-w-0 text-[11px] leading-snug">
                          <span className="block truncate text-foreground">
                            {h.model.ai ? "AI" : "No AI"} · {h.options.scale}× · {h.camera_label}
                          </span>
                          <span className="block truncate text-muted-foreground">{relTime(h.created_at)}</span>
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={() => void remove(h)}
                        aria-label="Delete result"
                        title="Delete result"
                        className="rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100"
                      >
                        <Trash2 className="size-3.5" />
                      </button>
                    </motion.li>
                  ))}
                </AnimatePresence>
              </ul>
            ) : (
              <p className="text-[11px] text-muted-foreground">Finished enhancements appear here.</p>
            )}
          </Section>

          <p className="mt-auto hidden text-[10px] leading-relaxed text-muted-foreground md:block">
            Wheel zoom · drag to pan · drag the divider to compare · ← → divider · B band · Enter enhance · Esc close
          </p>
        </aside>
      </div>
    </motion.div>
  )
}

function relTime(ts: number) {
  const d = Date.now() / 1000 - ts
  if (d < 60) return "just now"
  if (d < 3600) return `${Math.round(d / 60)} min ago`
  if (d < 86400) return `${Math.round(d / 3600)} h ago`
  return new Date(ts * 1000).toLocaleDateString()
}

function Section({ title, aside, children }: { title: string; aside?: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-1.5 flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
        <span>{title}</span>
        {aside && <span className="tabular-nums normal-case tracking-normal">{aside}</span>}
      </div>
      {children}
    </section>
  )
}

function Stat({ label, value, wide, icon }: { label: string; value: string; wide?: boolean; icon?: boolean }) {
  return (
    <div className={wide ? "col-span-2" : undefined}>
      <dt className="text-[11px] text-muted-foreground">{label}</dt>
      <dd className="flex items-center gap-1.5 truncate tabular-nums text-foreground">
        {icon && <Cpu className="size-3.5 shrink-0 text-primary" />}
        <span className="truncate">{value}</span>
      </dd>
    </div>
  )
}

function PillButton({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="rounded-full border border-border bg-background/80 p-1.5 text-muted-foreground backdrop-blur-md transition-colors hover:text-primary"
    >
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------- before / after
interface CompareHandle {
  reset: () => void
  zoom: (factor: number) => void
  nudge: (d: number) => void
}

const HANDLE_HIT_PX = 14

/**
 * Before/after with a draggable divider. Both layers share one view transform
 * (zoom/pan stay in sync); the "after" layer is clipped in screen space so the
 * divider doesn't move while zooming. The enhanced PNG carries its watermark
 * band below the picture: it's cut off here and shown as a strip underneath.
 */
function Compare({
  job,
  before,
  after,
  showBand,
  ref,
}: {
  job: EnhanceJob
  before: string
  after?: string
  showBand: boolean
  ref: React.Ref<CompareHandle>
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  const [split, setSplit] = useState(0.5)
  const [mode, setMode] = useState<"none" | "split" | "pan">("none")
  const [afterLoaded, setAfterLoaded] = useState(false)
  const drag = useRef<{ x: number; y: number; id: number } | null>(null)

  useLayoutEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const [outW, outH] = job.output_size
  const band = after ? (job.band_px ?? 0) : 0
  const bandShown = showBand && band > 0
  // fit picture (+ band strip when shown) into the area
  const fit = useMemo(() => {
    const pad = 16
    const aw = Math.max(0, size.w - 2 * pad)
    const ah = Math.max(0, size.h - 2 * pad)
    const [cw, ch] = job.crop_size
    if (!aw || !ah || !cw || !ch) return { left: 0, top: 0, w: 0, h: 0, bandH: 0 }
    const bandRatio = bandShown ? band / outW : 0 // band height per picture width
    const totalH = ch / cw + bandRatio
    const k = Math.min(aw, ah / totalH)
    const w = k
    const h = (k * ch) / cw
    const bandH = k * bandRatio
    return { left: pad + (aw - w) / 2, top: pad + (ah - h - bandH) / 2, w, h, bandH }
  }, [size, job.crop_size, bandShown, band, outW])

  const view = useViewTransform(fit.w, fit.h)
  const { reset, zoomAt } = view

  useEffect(() => setAfterLoaded(false), [after])

  useImperativeHandle(
    ref,
    () => ({
      reset,
      zoom: (f: number) => zoomAt(fit.w / 2, fit.h / 2, f),
      nudge: (d: number) => setSplit((s) => Math.min(1, Math.max(0, s + d))),
    }),
    [reset, zoomAt, fit.w, fit.h],
  )

  const local = useCallback(
    (e: { clientX: number; clientY: number }) => {
      const b = containerRef.current!.getBoundingClientRect()
      return [e.clientX - b.left - fit.left, e.clientY - b.top - fit.top] as const
    },
    [fit.left, fit.top],
  )

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const [px, py] = local(e)
      zoomAt(px, py, Math.exp(-e.deltaY * 0.0015))
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [local, zoomAt])

  const showAfter = Boolean(after)
  const running = ACTIVE.includes(job.status)

  return (
    <div
      ref={containerRef}
      className={cn(
        "absolute inset-0 touch-none select-none overflow-hidden",
        mode === "split" ? "cursor-ew-resize" : mode === "pan" ? "cursor-grabbing" : "cursor-grab",
      )}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        const [px] = local(e)
        e.currentTarget.setPointerCapture(e.pointerId)
        drag.current = { x: e.clientX, y: e.clientY, id: e.pointerId }
        setMode(showAfter && Math.abs(px - split * fit.w) <= HANDLE_HIT_PX ? "split" : "pan")
      }}
      onPointerMove={(e) => {
        const d = drag.current
        if (!d || d.id !== e.pointerId) return
        if (mode === "split") {
          const [px] = local(e)
          setSplit(Math.min(1, Math.max(0, px / Math.max(1, fit.w))))
        } else {
          view.panBy(e.clientX - d.x, e.clientY - d.y)
        }
        d.x = e.clientX
        d.y = e.clientY
      }}
      onPointerUp={() => {
        drag.current = null
        setMode("none")
      }}
      onPointerCancel={() => {
        drag.current = null
        setMode("none")
      }}
      onDoubleClick={(e) => {
        const [px, py] = local(e)
        if (view.view.current.s > 1.3) reset()
        else zoomAt(px, py, 3)
      }}
    >
      <motion.div
        className="absolute overflow-hidden rounded-sm ring-1 ring-border"
        initial={false}
        animate={{ left: fit.left, top: fit.top, width: fit.w, height: fit.h }}
        transition={{ type: "spring", stiffness: 300, damping: 34 }}
      >
        {/* before: the real camera pixels, nearest-neighbour so nothing is smoothed in */}
        <motion.div className="absolute inset-0 origin-top-left will-change-transform" style={{ x: view.x, y: view.y, scale: view.s }}>
          <img src={before} alt="Original crop" draggable={false} className="block size-full [image-rendering:pixelated]" />
        </motion.div>

        {/* after */}
        {showAfter && (
          <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }}>
            <motion.div className="absolute inset-0 origin-top-left overflow-hidden will-change-transform" style={{ x: view.x, y: view.y, scale: view.s }}>
              <motion.img
                src={after}
                alt="AI enhanced result"
                draggable={false}
                onLoad={() => setAfterLoaded(true)}
                initial={{ opacity: 0, filter: "blur(10px)" }}
                animate={afterLoaded ? { opacity: 1, filter: "blur(0px)" } : { opacity: 0 }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                className="absolute left-0 top-0 block w-full max-w-none"
                style={{ height: `${((outH + band) / outH) * 100}%` }}
              />
            </motion.div>
          </div>
        )}

        {/* labels */}
        <span className="pointer-events-none absolute left-2 top-2 rounded-full bg-background/80 px-2 py-0.5 text-[11px] text-muted-foreground backdrop-blur-md">
          Before · camera pixels
        </span>
        <AnimatePresence>
          {showAfter && (
            <motion.span
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className={cn(
                "pointer-events-none absolute right-2 top-2 rounded-full px-2 py-0.5 text-[11px] font-semibold",
                job.ai_enhanced ? "bg-primary text-primary-foreground" : "bg-background/80 text-foreground",
              )}
            >
              After · {job.ai_enhanced ? "AI" : "no AI"} {job.options.scale}×
            </motion.span>
          )}
        </AnimatePresence>

        {/* divider */}
        {showAfter && (
          <div className="pointer-events-none absolute inset-y-0" style={{ left: `${split * 100}%` }}>
            <div className="absolute inset-y-0 -left-px w-0.5 bg-primary shadow-[0_0_12px_rgba(255,210,31,0.6)]" />
            <motion.div
              className="absolute top-1/2 grid size-8 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border-2 border-primary bg-[#222427] text-primary shadow-lg"
              animate={{ scale: mode === "split" ? 1.15 : 1 }}
              transition={spring}
            >
              <svg viewBox="0 0 16 16" className="size-3.5" fill="currentColor" aria-hidden>
                <path d="M6 3 1 8l5 5V3Zm4 0v10l5-5-5-5Z" />
              </svg>
            </motion.div>
          </div>
        )}

        {/* progress sweep while running */}
        <AnimatePresence>
          {running && (
            <motion.div className="pointer-events-none absolute inset-0" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="absolute inset-0 bg-[#16171a]/35" />
              <motion.div
                className="absolute inset-y-0 w-0.5 bg-primary shadow-[0_0_18px_4px_rgba(255,210,31,0.45)]"
                animate={{ left: `${(job.status === "running" || job.status === "saving" ? job.progress : 0) * 100}%` }}
                transition={{ type: "spring", stiffness: 120, damping: 24 }}
              />
              <span className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full border border-border bg-background/85 px-3 py-1 text-xs tabular-nums backdrop-blur-md">
                <Loader2 className="size-3.5 animate-spin text-primary" />
                {statusText(job)}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* the watermark band of the export, shown under the picture */}
      <AnimatePresence>
        {bandShown && after && fit.bandH > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, left: fit.left, top: fit.top + fit.h, width: fit.w, height: fit.bandH }}
            exit={{ opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 34 }}
            className="pointer-events-none absolute"
            style={{ backgroundImage: `url(${after})`, backgroundSize: "100% auto", backgroundPosition: "bottom", backgroundRepeat: "no-repeat" }}
            aria-label="AI ENHANCED watermark band (always part of exports)"
            role="img"
          />
        )}
      </AnimatePresence>
    </div>
  )
}
