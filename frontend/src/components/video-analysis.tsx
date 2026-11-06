import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { AnimatePresence, motion } from "motion/react"
import { Film } from "lucide-react"
import { PlaybackBar, SPEEDS } from "@/components/playback-bar"
import { VideoLibrary } from "@/components/video-library"
import { usePlayback } from "@/hooks/use-playback"
import { useVideoLibrary } from "@/hooks/use-video-library"
import type { Camera } from "@/lib/types"
import { isVideoCamera, videosApi, type Video } from "@/lib/videos"
import { cn } from "@/lib/utils"

interface Options {
  cameras: Camera[]
  activeId: string | null
  open: (id: string) => void
  /** Re-fetch /api/cameras (virtual cameras appear / disappear). */
  reloadCameras: () => Promise<unknown>
  rightInset: number
  /** Stage is visible (single view). */
  single: boolean
}

const COPY_KEY = "securitycam.video.copy-settings.v1"
const isTyping = (e: KeyboardEvent) =>
  Boolean((e.target as HTMLElement).closest("input, textarea, select, [role=combobox], [role=listbox], [contenteditable=true]"))

function loadCopy() {
  try {
    return localStorage.getItem(COPY_KEY) === "1"
  } catch {
    return false
  }
}

/**
 * Video Analysis wired into the app shell: header entry, library overlay,
 * playback bar and shortcuts for video-source cameras. App only places the pieces.
 */
export function useVideoAnalysis({ cameras, activeId, open, reloadCameras, rightInset, single }: Options) {
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [copySettings, setCopySettings] = useState(loadCopy)
  const [opening, setOpening] = useState<string | null>(null)
  const pendingOpen = useRef<string | null>(null)
  const lastCamera = useRef<string | null>(null)

  const active = cameras.find((c) => c.id === activeId)
  const isVideo = isVideoCamera(active)
  const isFile = active?.source_kind === "file"
  const pb = usePlayback(isFile && single ? activeId : null)

  // Remember the last real camera: closing a video returns there.
  useEffect(() => {
    if (active && !isVideoCamera(active)) lastCamera.current = active.id
  }, [active])

  // Open a freshly created virtual camera once the camera list contains it
  // (after the workspace reconciled the new list, so the selection sticks).
  useEffect(() => {
    const id = pendingOpen.current
    if (id && cameras.some((c) => c.id === id)) {
      pendingOpen.current = null
      open(id)
    }
  }, [cameras, open])

  const setCopy = useCallback((on: boolean) => {
    setCopySettings(on)
    try {
      localStorage.setItem(COPY_KEY, on ? "1" : "0")
    } catch {
      /* not remembered */
    }
  }, [])

  const leave = useCallback(
    (cameraId: string | null) => {
      if (!cameraId || cameraId !== activeId) return
      const fallback = lastCamera.current ?? cameras.find((c) => !isVideoCamera(c))?.id
      if (fallback) open(fallback)
    },
    [activeId, cameras, open],
  )

  const lib = useVideoLibrary(libraryOpen, (closed) => {
    leave(closed)
    void reloadCameras()
  })

  const copyFrom = copySettings && activeId ? activeId : null

  const openVideo = useCallback(
    async (v: Video) => {
      if (v.camera_id && cameras.some((c) => c.id === v.camera_id)) {
        open(v.camera_id)
        setLibraryOpen(false)
        return
      }
      setOpening(v.id)
      try {
        const r = await videosApi.open(v.id, copyFrom)
        pendingOpen.current = r.camera_id
        await reloadCameras()
        setLibraryOpen(false)
      } catch {
        void lib.refresh()
      } finally {
        setOpening(null)
      }
    },
    [cameras, copyFrom, lib, open, reloadCameras],
  )

  const openUrl = useCallback(
    async (url: string) => {
      const r = await videosApi.openUrl(url, undefined, copyFrom)
      pendingOpen.current = r.camera_id
      await reloadCameras()
      setLibraryOpen(false)
    },
    [copyFrom, reloadCameras],
  )

  const closeCamera = useCallback(
    async (cameraId: string) => {
      leave(cameraId)
      try {
        await videosApi.close(cameraId)
      } finally {
        await reloadCameras()
      }
    },
    [leave, reloadCameras],
  )

  // Playback shortcuts (video on stage, library closed, not typing).
  const { send, playback, position } = pb
  const speedRef = useRef(1)
  speedRef.current = playback?.speed ?? 1
  useEffect(() => {
    if (!isFile || !single || libraryOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return
      const k = e.key
      if (k === " " || k === "k" || k === "K") {
        e.preventDefault() // no page scroll, no click on a focused button
        void send({ action: "toggle" })
      } else if (k === "j" || k === "J") void send({ action: "seek", seconds: Math.max(0, position() - 10) })
      else if (k === "l" || k === "L") void send({ action: "seek", seconds: position() + 10 })
      else if (k === ",") void send({ action: "step", frames: -1 })
      else if (k === ".") void send({ action: "step", frames: 1 })
      else if (k === "[" || k === "]") {
        const cur = speedRef.current
        const next =
          k === "]"
            ? (SPEEDS.find((s) => s > cur) ?? SPEEDS[SPEEDS.length - 1])
            : ([...SPEEDS].reverse().find((s) => s < cur) ?? SPEEDS[0])
        speedRef.current = next // repeated presses step on before the next render
        if (next !== cur) void send({ speed: next })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [isFile, single, libraryOpen, send, position])

  // Escape closes the library.
  useEffect(() => {
    if (!libraryOpen) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setLibraryOpen(false)
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [libraryOpen])

  const entry: ReactNode = (
    <button
      type="button"
      onClick={() => setLibraryOpen((o) => !o)}
      aria-expanded={libraryOpen}
      aria-label="Video Analysis"
      title="Video Analysis"
      className={cn(
        "relative flex h-9 shrink-0 items-center gap-2 rounded-full border px-3 text-[13px] backdrop-blur-md transition-colors duration-300 active:scale-95",
        libraryOpen
          ? "border-primary bg-primary text-primary-foreground"
          : isVideo
            ? "border-primary/60 bg-background/70 text-primary"
            : "border-border/70 bg-background/70 text-foreground hover:border-primary/60",
      )}
    >
      <Film className="size-4" />
      <span className="hidden lg:inline">Video Analysis</span>
    </button>
  )

  const library: ReactNode = (
    <AnimatePresence>
      {libraryOpen && (
        <>
          <motion.div
            key="video-library-scrim"
            aria-hidden
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 z-[25] bg-[#1b1c1f]/55 backdrop-blur-[2px]"
            onClick={() => setLibraryOpen(false)}
          />
          <VideoLibrary
            key="video-library"
            lib={lib}
            onClose={() => setLibraryOpen(false)}
            onOpen={(v) => void openVideo(v)}
            onOpenUrl={openUrl}
            activeCameraId={isVideo ? activeId : null}
            copySettings={copySettings}
            onCopySettingsChange={setCopy}
            copyFromLabel={active?.label ?? null}
            opening={opening}
          />
        </>
      )}
    </AnimatePresence>
  )

  const showBar = single && isVideo
  const playbackBar: ReactNode = (
    <AnimatePresence>
      {showBar && (
        <PlaybackBar
          key={activeId ?? "bar"}
          playback={playback}
          position={position}
          onCommand={(cmd) => void send(cmd)}
          live={active?.source_kind === "url" ? { label: active.label } : null}
          rightInset={rightInset}
        />
      )}
    </AnimatePresence>
  )

  return {
    entry,
    library,
    playbackBar,
    /** Extra space the stage keeps free at the bottom for the playback bar. */
    bottomInset: showBar ? 116 : undefined,
    closeCamera,
  }
}
