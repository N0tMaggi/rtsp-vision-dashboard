import { useEffect, useRef, useState, type DragEvent, type ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Check, Film, Link2, LoaderCircle, Pencil, Play, Radio, Trash2, UploadCloud, X } from "lucide-react"
import { Switch } from "@/components/ui/switch"
import type { UploadItem, VideoLibraryState } from "@/hooks/use-video-library"
import { formatBytes, formatTime, type Video } from "@/lib/videos"
import { cn } from "@/lib/utils"

const ease = [0.22, 1, 0.36, 1] as const
const spring = { type: "spring", stiffness: 380, damping: 34 } as const

interface Props {
  lib: VideoLibraryState
  onClose: () => void
  onOpen: (v: Video) => void
  onOpenUrl: (url: string) => Promise<void>
  /** Video camera id currently on stage (highlighted). */
  activeCameraId: string | null
  copySettings: boolean
  onCopySettingsChange: (on: boolean) => void
  copyFromLabel: string | null
  opening: string | null
}

/**
 * Video Analysis library: drop zone, uploads with live progress, and the clip
 * list. Opening a clip turns it into a virtual camera on the normal stage.
 */
export function VideoLibrary({
  lib,
  onClose,
  onOpen,
  onOpenUrl,
  activeCameraId,
  copySettings,
  onCopySettingsChange,
  copyFromLabel,
  opening,
}: Props) {
  const [dragDepth, setDragDepth] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const dragging = dragDepth > 0
  const accept = (lib.data?.extensions ?? ["mp4", "mkv", "avi", "mov", "webm"]).map((e) => `.${e}`).join(",")
  const videos = lib.data?.videos ?? []
  const empty = lib.data !== null && videos.length === 0 && lib.uploads.length === 0

  const hasFiles = (e: DragEvent) => Array.from(e.dataTransfer.types).includes("Files")

  return (
    <motion.section
      aria-label="Video Analysis"
      initial={{ opacity: 0, y: -14, scale: 0.985 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.985 }}
      transition={{ type: "spring", stiffness: 300, damping: 32 }}
      className="absolute inset-x-4 top-[76px] z-30 mx-auto flex max-h-[calc(100dvh-150px)] max-w-[880px] origin-top flex-col overflow-hidden rounded-xl border border-border/70 bg-background/90 shadow-[0_24px_80px_-24px_rgba(0,0,0,0.75)] backdrop-blur-xl"
      onDragEnter={(e) => {
        if (!hasFiles(e)) return
        e.preventDefault()
        setDragDepth((d) => d + 1)
      }}
      onDragOver={(e) => hasFiles(e) && e.preventDefault()}
      onDragLeave={(e) => hasFiles(e) && setDragDepth((d) => Math.max(0, d - 1))}
      onDrop={(e) => {
        e.preventDefault()
        setDragDepth(0)
        lib.upload(Array.from(e.dataTransfer.files))
      }}
    >
      <header className="flex items-start gap-3 border-b border-border/60 px-5 pb-4 pt-4">
        <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-md bg-primary text-primary-foreground">
          <Film className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="text-sm font-semibold">
            Video Analysis<span className="text-primary">_</span>
          </h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Clips play back in real time through the full pipeline: detection, tracking, ESP, zones, alarms, recognition.
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close Video Analysis"
          className="grid size-8 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <X className="size-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-5 pt-4">
        {/* drop zone */}
        <motion.button
          type="button"
          onClick={() => inputRef.current?.click()}
          animate={{ scale: dragging ? 1.01 : 1 }}
          transition={spring}
          className={cn(
            "group relative flex w-full items-center gap-4 overflow-hidden rounded-lg border border-dashed px-5 py-5 text-left transition-colors duration-300",
            dragging ? "border-primary bg-primary/10" : "border-border hover:border-primary/60 hover:bg-accent/40",
          )}
        >
          <motion.span
            animate={dragging ? { y: [0, -4, 0] } : { y: 0 }}
            transition={dragging ? { duration: 0.9, repeat: Infinity, ease: "easeInOut" } : spring}
            className={cn(
              "grid size-11 shrink-0 place-items-center rounded-full border transition-colors duration-300",
              dragging ? "border-primary bg-primary text-primary-foreground" : "border-border text-primary group-hover:border-primary/60",
            )}
          >
            <UploadCloud className="size-5" />
          </motion.span>
          <span className="min-w-0">
            <span className="block text-sm">{dragging ? "Drop to upload" : "Drop clips here or browse"}</span>
            <span className="mt-0.5 block text-xs text-muted-foreground">
              {(lib.data?.extensions ?? []).map((e) => e.toUpperCase()).join(" · ") || "MP4 · MKV · AVI · MOV · WEBM"}
              {lib.data ? ` — up to ${formatBytes(lib.data.max_upload_bytes)} each` : ""}
            </span>
          </span>
        </motion.button>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          hidden
          onChange={(e) => {
            lib.upload(Array.from(e.target.files ?? []))
            e.target.value = ""
          }}
        />

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
          <label className="flex cursor-pointer items-center gap-2 text-muted-foreground">
            <Switch checked={copySettings} onCheckedChange={onCopySettingsChange} disabled={!copyFromLabel} aria-label="Copy settings from current camera" />
            <span>
              Copy settings from current camera
              {copyFromLabel && <span className="text-foreground"> ({copyFromLabel})</span>}
            </span>
          </label>
          {lib.data?.allow_urls ? (
            <UrlOpener onOpen={onOpenUrl} />
          ) : (
            lib.data && (
              <span className="ml-auto flex items-center gap-1.5 text-muted-foreground/80">
                <Link2 className="size-3.5" /> Stream URLs are off — set SECURITYCAM_VIDEO_URLS=true in .env
              </span>
            )
          )}
        </div>

        {/* uploads in flight / failed */}
        <ul className="mt-4 flex flex-col gap-1.5" aria-label="Uploads">
          <AnimatePresence initial={false}>
            {lib.uploads.map((u) => (
              <UploadRow key={u.key} item={u} onCancel={() => lib.cancel(u.key)} onDismiss={() => lib.dismiss(u.key)} />
            ))}
          </AnimatePresence>
        </ul>

        {lib.error && <p className="mt-3 text-xs text-destructive">{lib.error}</p>}

        {empty ? (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease }}
            className="mt-8 flex flex-col items-center gap-2 pb-6 text-center"
          >
            <FilmStrip />
            <p className="mt-2 text-sm">Upload a surveillance clip to test detection — cars, people, plates.</p>
            <p className="max-w-md text-xs leading-relaxed text-muted-foreground">
              Public CCTV footage works great. The clip plays in real time like a live camera; turn on tracking, zones or
              plate reading in the ESP menu and scrub through the action.
            </p>
          </motion.div>
        ) : (
          <>
            {videos.length > 0 && (
              <div className="mt-5 flex items-baseline justify-between text-xs text-muted-foreground">
                <span>
                  Library <span className="tabular-nums text-foreground">{videos.length}</span>
                </span>
                {lib.data && <UsageMeter used={lib.data.total_bytes} max={lib.data.max_total_bytes} />}
              </div>
            )}
            <ul className="mt-2 flex flex-col" aria-label="Video library">
              <AnimatePresence initial={false}>
                {videos.map((v, i) => (
                  <VideoRow
                    key={v.id}
                    video={v}
                    index={i}
                    active={v.camera_id !== null && v.camera_id === activeCameraId}
                    opening={opening === v.id}
                    onOpen={() => onOpen(v)}
                    onRename={(name) => lib.rename(v.id, name)}
                    onDelete={() => lib.remove(v.id)}
                  />
                ))}
              </AnimatePresence>
            </ul>
            {lib.data === null && !lib.error && (
              <p className="mt-6 flex items-center justify-center gap-2 text-xs text-muted-foreground">
                <LoaderCircle className="size-3.5 animate-spin" /> Loading library
              </p>
            )}
          </>
        )}
      </div>
    </motion.section>
  )
}

function UsageMeter({ used, max }: { used: number; max: number }) {
  const pct = max > 0 ? Math.min(100, (used / max) * 100) : 0
  return (
    <span className="flex items-center gap-2">
      <span className="tabular-nums">
        {formatBytes(used)} / {formatBytes(max)}
      </span>
      <span className="h-1 w-20 overflow-hidden rounded-full bg-muted">
        <motion.span
          className={cn("block h-full rounded-full", pct > 90 ? "bg-destructive" : "bg-primary")}
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 120, damping: 20 }}
        />
      </span>
    </span>
  )
}

function UploadRow({ item, onCancel, onDismiss }: { item: UploadItem; onCancel: () => void; onDismiss: () => void }) {
  const pct = item.total > 0 ? (item.loaded / item.total) * 100 : 0
  const processing = item.state === "uploading" && pct >= 100
  return (
    <motion.li
      layout
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3, ease }}
      className="overflow-hidden"
    >
      <div
        className={cn(
          "relative flex items-center gap-3 overflow-hidden rounded-md border px-3 py-2 text-xs",
          item.state === "error" ? "border-destructive/50 text-destructive" : "border-border/70",
        )}
      >
        {item.state === "uploading" && (
          <motion.span
            aria-hidden
            className="absolute inset-y-0 left-0 bg-primary/10"
            initial={false}
            animate={{ width: `${pct}%` }}
            transition={{ type: "spring", stiffness: 140, damping: 24 }}
          />
        )}
        <span className="relative min-w-0 flex-1 truncate">{item.name}</span>
        <span className="relative shrink-0 tabular-nums text-muted-foreground">
          {item.state === "error"
            ? item.error
            : processing
              ? "Checking video…"
              : `${pct.toFixed(0)}% · ${formatBytes(item.loaded)} / ${formatBytes(item.total)}`}
        </span>
        <button
          type="button"
          onClick={item.state === "error" ? onDismiss : onCancel}
          aria-label={item.state === "error" ? "Dismiss" : `Cancel upload of ${item.name}`}
          title={item.state === "error" ? "Dismiss" : "Cancel upload"}
          className="relative grid size-6 shrink-0 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
        >
          <X className="size-3.5" />
        </button>
      </div>
    </motion.li>
  )
}

function VideoRow({
  video,
  index,
  active,
  opening,
  onOpen,
  onRename,
  onDelete,
}: {
  video: Video
  index: number
  active: boolean
  opening: boolean
  onOpen: () => void
  onRename: (name: string) => Promise<void>
  onDelete: () => Promise<void>
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(video.name)
  const [confirm, setConfirm] = useState(false)
  const open = video.camera_id !== null

  useEffect(() => {
    if (!confirm) return
    const t = window.setTimeout(() => setConfirm(false), 3000)
    return () => window.clearTimeout(t)
  }, [confirm])

  const commit = () => {
    setEditing(false)
    const name = draft.trim()
    if (name && name !== video.name) void onRename(name)
    else setDraft(video.name)
  }

  return (
    <motion.li
      layout
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0, transition: { delay: Math.min(index, 8) * 0.03, duration: 0.35, ease } }}
      exit={{ opacity: 0, x: 12, transition: { duration: 0.2 } }}
      className="group relative border-b border-border/50 last:border-b-0"
    >
      {active && (
        <motion.span layoutId="video-row-active" className="absolute inset-y-2 left-0 w-0.5 rounded-full bg-primary" transition={spring} />
      )}
      <div className="flex items-center gap-4 py-2.5 pl-3 pr-1">
        <button
          type="button"
          onClick={onOpen}
          aria-label={`Open ${video.name}`}
          className="relative aspect-video w-28 shrink-0 overflow-hidden rounded-md border border-border/60 bg-[#1b1c1f] outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {video.thumb_url ? (
            <img
              src={video.thumb_url}
              alt=""
              loading="lazy"
              className="size-full object-cover transition-transform duration-500 ease-[var(--ease-hud)] group-hover:scale-105"
            />
          ) : (
            <Film className="m-auto size-5 text-muted-foreground" />
          )}
          <span className="absolute inset-0 grid place-items-center bg-[#222427]/50 opacity-0 transition-opacity duration-300 group-hover:opacity-100">
            {opening ? <LoaderCircle className="size-5 animate-spin text-primary" /> : <Play className="size-5 fill-primary text-primary" />}
          </span>
          <span className="absolute bottom-1 right-1 rounded bg-[#222427]/85 px-1 text-[10px] tabular-nums text-foreground">
            {formatTime(video.duration)}
          </span>
        </button>

        <div className="min-w-0 flex-1">
          {editing ? (
            <input
              autoFocus
              value={draft}
              maxLength={120}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => {
                if (e.key === "Enter") commit()
                if (e.key === "Escape") {
                  e.stopPropagation()
                  setDraft(video.name)
                  setEditing(false)
                }
              }}
              aria-label="Clip name"
              className="w-full rounded border border-primary/60 bg-transparent px-1.5 py-0.5 text-sm outline-none"
            />
          ) : (
            <button type="button" onClick={onOpen} className="block max-w-full truncate text-left text-sm hover:text-primary">
              {video.name}
            </button>
          )}
          <p className="mt-0.5 flex flex-wrap gap-x-2 text-xs tabular-nums text-muted-foreground">
            <span>
              {video.width}×{video.height}
            </span>
            <span>· {Math.round(video.fps * 100) / 100} fps</span>
            <span>· {formatBytes(video.size)}</span>
            <span className="hidden sm:inline">· {new Date(video.created * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
          </p>
        </div>

        {open && (
          <span className="hidden shrink-0 items-center gap-1.5 text-xs text-primary sm:flex">
            <span className="size-1.5 rounded-full bg-primary" /> Open
          </span>
        )}
        <div className="flex shrink-0 items-center gap-0.5 opacity-70 transition-opacity group-hover:opacity-100">
          <RowButton label="Rename" onClick={() => setEditing(true)}>
            <Pencil className="size-3.5" />
          </RowButton>
          <AnimatePresence mode="popLayout" initial={false}>
            {confirm ? (
              <motion.button
                key="confirm"
                type="button"
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                transition={spring}
                onClick={() => void onDelete()}
                className="flex h-7 items-center gap-1 overflow-hidden whitespace-nowrap rounded-full bg-destructive px-2.5 text-xs text-[#222427]"
              >
                <Trash2 className="size-3.5" /> Delete
              </motion.button>
            ) : (
              <motion.span key="trash" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <RowButton label="Delete" onClick={() => setConfirm(true)}>
                  <Trash2 className="size-3.5" />
                </RowButton>
              </motion.span>
            )}
          </AnimatePresence>
          <button
            type="button"
            onClick={onOpen}
            disabled={opening}
            className={cn(
              "ml-1.5 flex h-7 items-center gap-1.5 rounded-full border px-3 text-xs transition-colors duration-200 disabled:opacity-60",
              active ? "border-primary bg-primary text-primary-foreground" : "border-border/70 hover:border-primary/60 hover:text-primary",
            )}
          >
            {active ? <Check className="size-3.5" /> : <Play className="size-3 fill-current" />}
            {active ? "On stage" : "Analyze"}
          </button>
        </div>
      </div>
    </motion.li>
  )
}

function RowButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="grid size-7 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
    >
      {children}
    </button>
  )
}

function UrlOpener({ onOpen }: { onOpen: (url: string) => Promise<void> }) {
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    if (!url.trim()) return
    setBusy(true)
    setError(null)
    try {
      await onOpen(url.trim())
      setUrl("")
      setOpen(false)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex min-w-0 flex-1 flex-wrap items-center justify-end gap-2">
      <AnimatePresence mode="popLayout" initial={false}>
        {open ? (
          <motion.form
            key="form"
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "100%" }}
            exit={{ opacity: 0, width: 0 }}
            transition={spring}
            className="flex min-w-0 max-w-md items-center gap-1.5 overflow-hidden"
            onSubmit={(e) => {
              e.preventDefault()
              void submit()
            }}
          >
            <Radio className="size-3.5 shrink-0 text-muted-foreground" />
            <input
              autoFocus
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  e.stopPropagation()
                  setOpen(false)
                }
              }}
              placeholder="https://… / rtsp://… (MP4, MJPEG, HLS)"
              aria-label="Stream URL"
              className="h-7 min-w-0 flex-1 rounded-full border border-border/70 bg-transparent px-3 outline-none focus:border-primary/60"
            />
            <button
              type="submit"
              disabled={busy || !url.trim()}
              className="flex h-7 shrink-0 items-center gap-1 rounded-full bg-primary px-3 text-primary-foreground disabled:opacity-50"
            >
              {busy ? <LoaderCircle className="size-3.5 animate-spin" /> : "Open"}
            </button>
          </motion.form>
        ) : (
          <motion.button
            key="btn"
            type="button"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setOpen(true)}
            className="flex items-center gap-1.5 text-muted-foreground transition-colors hover:text-primary"
          >
            <Link2 className="size-3.5" /> Open stream URL
          </motion.button>
        )}
      </AnimatePresence>
      {error && <span className="w-full text-right text-destructive">{error}</span>}
    </div>
  )
}

/** Small animated film strip for the empty state. */
function FilmStrip() {
  return (
    <div className="relative h-14 w-44 overflow-hidden rounded-md border border-border/70 bg-[#1b1c1f]" aria-hidden>
      <motion.div
        className="absolute inset-y-0 left-0 flex gap-1.5 px-1.5 py-2"
        animate={{ x: [0, -46] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
      >
        {Array.from({ length: 6 }, (_, i) => (
          <span key={i} className={cn("h-full w-10 shrink-0 rounded-sm border", i === 2 ? "border-primary bg-primary/15" : "border-border/70")} />
        ))}
      </motion.div>
      <span className="absolute inset-x-0 top-0.5 flex justify-around px-1">
        {Array.from({ length: 9 }, (_, i) => (
          <span key={i} className="size-1 rounded-[1px] bg-muted-foreground/40" />
        ))}
      </span>
      <span className="absolute inset-x-0 bottom-0.5 flex justify-around px-1">
        {Array.from({ length: 9 }, (_, i) => (
          <span key={i} className="size-1 rounded-[1px] bg-muted-foreground/40" />
        ))}
      </span>
    </div>
  )
}
