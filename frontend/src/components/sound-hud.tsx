import { useEffect, useRef, useState, useSyncExternalStore } from "react"
import { AnimatePresence, motion, useSpring, useTransform } from "motion/react"
import { AudioLines, Lock } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import {
  SOUND_PRIVACY,
  decodeColumns,
  soundIcon,
  soundKindIcon,
  soundPings,
  useSound,
  type SoundPayload,
} from "@/lib/sound"
import { cn } from "@/lib/utils"

const spring = { type: "spring", stiffness: 420, damping: 32 } as const
const METER_MIN_DB = -80
const SPEC_W = 168
const SPEC_H = 36
const COL_W = 2

/** dBFS -> 0..1 for the level bar. */
const meterPos = (db: number | null | undefined) =>
  db === null || db === undefined ? 0 : Math.max(0, Math.min(1, (db - METER_MIN_DB) / -METER_MIN_DB))

/** anthracite -> dim amber -> signal yellow -> white (rgb strings, precomputed) */
const LUT: string[] = (() => {
  const stops: [number, [number, number, number]][] = [
    [0, [27, 28, 31]],
    [0.35, [44, 42, 36]],
    [0.6, [120, 96, 20]],
    [0.82, [255, 210, 31]],
    [1, [244, 244, 242]],
  ]
  const out: string[] = []
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let j = 1
    while (j < stops.length - 1 && t > stops[j][0]) j++
    const [t0, c0] = stops[j - 1]
    const [t1, c1] = stops[j]
    const k = Math.max(0, Math.min(1, (t - t0) / (t1 - t0)))
    out.push(`rgb(${c0.map((v, n) => Math.round(v + (c1[n] - v) * k)).join(",")})`)
  }
  return out
})()

/** On-stage audio HUD: level meter, mini spectrogram, live sound labels. */
export function SoundHud({ cameraId, on }: { cameraId?: string; on: boolean }) {
  const data = useSound(cameraId, on)
  const primed = useRef(false)

  useEffect(() => {
    if (!data) return
    soundPings.ingest(data.events, primed.current)
    primed.current = true
  }, [data])
  useEffect(() => {
    primed.current = false
    return () => soundPings.clear()
  }, [cameraId, on])

  return (
    <AnimatePresence>
      {on && cameraId && (
        <motion.div
          key="sound-hud"
          initial={{ opacity: 0, y: -6, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -6, scale: 0.96 }}
          transition={spring}
          className="pointer-events-auto absolute left-3 top-3 z-[2] w-[188px] rounded-xl border border-white/10 bg-background/70 p-2 text-foreground backdrop-blur-md"
          role="group"
          aria-label="Sound"
        >
          <HudBody data={data} />
        </motion.div>
      )}
    </AnimatePresence>
  )
}

function HudBody({ data }: { data: SoundPayload | null }) {
  const live = data?.state === "live"
  const stateText = !data ? "Connecting…" : live ? "" : data.message
  return (
    <>
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
        <AudioLines className={cn("size-3.5", live ? "text-primary" : "text-muted-foreground")} />
        <span>Sound</span>
        {data?.loud && (
          <motion.span
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="rounded-full bg-primary px-1.5 py-px text-[9px] font-semibold tracking-normal text-primary-foreground"
          >
            LOUD
          </motion.span>
        )}
        <span className="ml-auto tabular-nums tracking-normal text-foreground/80">
          {live ? `${Math.round(data!.meter)} dB` : "–"}
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="grid size-4 place-items-center text-muted-foreground" aria-label="Audio stays local">
              <Lock className="size-3" />
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-56">
            {SOUND_PRIVACY}
          </TooltipContent>
        </Tooltip>
      </div>

      <LevelMeter data={data} />
      <Spectrogram data={data} />

      <AnimatePresence mode="popLayout" initial={false}>
        {stateText ? (
          <motion.p
            key={stateText}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="mt-1.5 text-[11px] leading-snug text-muted-foreground"
          >
            {stateText}
          </motion.p>
        ) : null}
      </AnimatePresence>

      <LabelChips data={data} />
      {live && data?.classifier.status !== "ready" && data?.classifier.reason && (
        <p className="mt-1 text-[10px] leading-snug text-muted-foreground">{data.classifier.reason}</p>
      )}
    </>
  )
}

function LevelMeter({ data }: { data: SoundPayload | null }) {
  const level = useSpring(0, { stiffness: 380, damping: 30 })
  const peak = useSpring(0, { stiffness: 120, damping: 24 })
  const width = useTransform(level, (v) => `${v * 100}%`)
  const peakLeft = useTransform(peak, (v) => `calc(${v * 100}% - 1px)`)
  const live = data?.state === "live"
  useEffect(() => {
    level.set(live ? meterPos(data!.meter) : 0)
    peak.set(live ? meterPos(data!.peak) : 0)
  }, [data, live, level, peak])
  const floor = live ? meterPos(data!.floor) : null
  return (
    <div className="relative mt-2 h-1.5 overflow-hidden rounded-full bg-white/8" aria-hidden>
      <motion.div
        className={cn("absolute inset-y-0 left-0 rounded-full", data?.loud ? "bg-primary" : "bg-foreground/80")}
        style={{ width }}
      />
      <motion.div className="absolute inset-y-0 w-0.5 bg-primary" style={{ left: peakLeft }} />
      {floor !== null && (
        <motion.div
          className="absolute -inset-y-px w-px bg-white/40"
          animate={{ left: `${floor * 100}%` }}
          transition={{ type: "spring", stiffness: 80, damping: 20 }}
          title="Background level"
        />
      )}
    </div>
  )
}

/** Scrolling mel spectrogram on a canvas: columns are queued and drawn at the column rate. */
function Spectrogram({ data }: { data: SoundPayload | null }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const queue = useRef<Uint8Array[]>([])
  const lastSeq = useRef(0)
  const timer = useRef<number | undefined>(undefined)
  const rate = data?.rate || 10

  useEffect(() => {
    if (!data || data.state !== "live") return
    const cols = decodeColumns(data)
    if (data.seq < lastSeq.current) lastSeq.current = 0         // engine restarted
    const fresh = Math.min(cols.length, Math.max(0, data.seq - lastSeq.current))
    lastSeq.current = data.seq
    if (fresh) queue.current.push(...cols.slice(cols.length - fresh))
    if (queue.current.length > 12) queue.current.splice(0, queue.current.length - 12)
  }, [data])

  useEffect(() => {
    const draw = () => {
      const cv = canvas.current
      const ctx = cv?.getContext("2d")
      if (!cv || !ctx || document.hidden) return
      // catch up when behind, otherwise one column per tick
      const n = queue.current.length > 4 ? 2 : queue.current.length ? 1 : 0
      if (!n) return
      const dpr = cv.width / SPEC_W
      const shift = n * COL_W * dpr
      ctx.drawImage(cv, -shift, 0)
      ctx.clearRect(cv.width - shift, 0, shift, cv.height)
      for (let i = 0; i < n; i++) {
        const col = queue.current.shift()!
        const x = cv.width - shift + i * COL_W * dpr
        const bh = cv.height / col.length
        for (let b = 0; b < col.length; b++) {
          ctx.fillStyle = LUT[col[b]]
          ctx.fillRect(x, cv.height - (b + 1) * bh, COL_W * dpr, Math.ceil(bh))
        }
      }
    }
    timer.current = window.setInterval(draw, 1000 / rate)
    return () => window.clearInterval(timer.current)
  }, [rate])

  useEffect(() => {
    const cv = canvas.current
    if (!cv) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    cv.width = SPEC_W * dpr
    cv.height = SPEC_H * dpr
  }, [])

  return (
    <canvas
      ref={canvas}
      className="mt-2 block rounded-md bg-[#1b1c1f]"
      style={{ width: SPEC_W, height: SPEC_H }}
      aria-label="Spectrogram"
      role="img"
    />
  )
}

function LabelChips({ data }: { data: SoundPayload | null }) {
  const labels = data?.state === "live" ? data.labels : []
  return (
    <div className="mt-1.5 flex min-h-0 flex-wrap gap-1">
      <AnimatePresence mode="popLayout" initial={false}>
        {labels.map((l) => {
          const Icon = soundIcon(l.icon)
          return (
            <motion.span
              key={l.id}
              layout
              initial={{ opacity: 0, scale: 0.6, y: 4 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.6 }}
              transition={spring}
              className="flex items-center gap-1 rounded-full border border-white/10 bg-white/6 py-0.5 pl-1.5 pr-2 text-[11px]"
            >
              <Icon className="size-3 text-primary" />
              {l.label}
              <span className="tabular-nums text-muted-foreground">{Math.round(l.score * 100)}%</span>
            </motion.span>
          )
        })}
      </AnimatePresence>
      {data?.state === "live" && !labels.length && data.heard && data.classifier.status === "ready" && (
        <span className="truncate text-[10px] text-muted-foreground">Heard: {data.heard.label}</span>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ radar pings
/** Re-renders every animation-relevant tick while pings are visible. */
function useNow(active: boolean, ms = 250) {
  const [, setT] = useState(0)
  useEffect(() => {
    if (!active) return
    const t = window.setInterval(() => setT((v) => v + 1), ms)
    return () => window.clearInterval(t)
  }, [active, ms])
}

/**
 * Pulse rings around the camera on the radar for recent sound events. A mono
 * microphone has no direction, so they are full rings, never a bearing.
 */
export function SoundRadarPings({ cx, cy, r }: { cx: number; cy: number; r: number }) {
  const list = useSyncExternalStore(soundPings.subscribe, soundPings.get, soundPings.get)
  const now = performance.now()
  const visible = list.filter((p) => now - p.at < soundPings.showMs)
  useNow(visible.length > 0)
  const latest = visible[visible.length - 1]
  const LatestIcon = latest ? soundKindIcon(latest.kind) : null
  return (
    <g aria-hidden>
      {visible.map((p) => {
        const color = p.alarm ? "#ff5a4e" : p.kind === "loud" ? "#f4f4f2" : "#ffd21f"
        return (
          <g key={p.key}>
            {[0, 0.45].map((delay) => (
              <motion.circle
                key={delay}
                cx={cx}
                cy={cy}
                fill="none"
                stroke={color}
                strokeWidth={1.5}
                initial={{ r: 3, opacity: 0.9 }}
                animate={{ r, opacity: 0 }}
                transition={{ duration: 1.8, delay, ease: [0.16, 1, 0.3, 1] }}
              />
            ))}
          </g>
        )
      })}
      <AnimatePresence>
        {latest && LatestIcon && (
          <motion.g
            key={latest.key}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={spring}
          >
            <foreignObject x={cx - 60} y={cy - 22} width={120} height={16}>
              <div className="flex items-center justify-center gap-1 text-[9px] text-foreground/90">
                <LatestIcon className={cn("size-2.5", latest.alarm ? "text-destructive" : "text-primary")} />
                <span className="truncate">{latest.label}</span>
              </div>
            </foreignObject>
          </motion.g>
        )}
      </AnimatePresence>
    </g>
  )
}
