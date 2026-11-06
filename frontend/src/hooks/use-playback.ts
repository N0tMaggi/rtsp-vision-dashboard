import { useCallback, useEffect, useRef, useState } from "react"
import { videosApi, type Playback, type PlaybackCommand } from "@/lib/videos"

const POLL_MS = 700

/**
 * Playback state of a video source: polled from the server and extrapolated
 * locally between polls, so the scrubber moves smoothly. Commands update the
 * state optimistically; polls that started before a command are ignored.
 */
export function usePlayback(cameraId: string | null) {
  const [playback, setPlayback] = useState<Playback | null>(null)
  const receivedAt = useRef(0)
  const generation = useRef(0)

  const apply = useCallback((p: Playback) => {
    receivedAt.current = performance.now()
    setPlayback(p)
  }, [])

  useEffect(() => {
    setPlayback(null)
    if (!cameraId) return
    let stopped = false
    let ctrl: AbortController | null = null
    const poll = async () => {
      if (document.hidden) return
      ctrl?.abort()
      ctrl = new AbortController()
      const gen = generation.current
      try {
        const p = await videosApi.playback(cameraId, ctrl.signal)
        if (!stopped && gen === generation.current) apply(p)
      } catch {
        /* source closed or server busy: the next poll retries */
      }
    }
    void poll()
    const t = window.setInterval(() => void poll(), POLL_MS)
    return () => {
      stopped = true
      window.clearInterval(t)
      ctrl?.abort()
    }
  }, [cameraId, apply])

  /** Current position in seconds, extrapolated from the last server state. */
  const position = useCallback(() => {
    if (!playback) return 0
    if (!playback.playing) return playback.position
    const dt = (performance.now() - receivedAt.current) / 1000
    const p = playback.position + dt * playback.speed
    return playback.duration > 0 ? (playback.loop ? p % playback.duration : Math.min(p, playback.duration)) : p
  }, [playback])

  const send = useCallback(
    async (cmd: PlaybackCommand) => {
      if (!cameraId) return
      generation.current += 1
      const gen = generation.current
      // optimistic: the bar reacts immediately
      setPlayback((p) => {
        if (!p) return p
        const now = position()
        const next = { ...p, position: now }
        if ("speed" in cmd) next.speed = cmd.speed
        if ("loop" in cmd) next.loop = cmd.loop
        if ("action" in cmd) {
          if (cmd.action === "play") next.playing = true
          else if (cmd.action === "pause") next.playing = false
          else if (cmd.action === "toggle") next.playing = !p.playing
          else if (cmd.action === "restart") Object.assign(next, { playing: true, ended: false, position: 0 })
          else if (cmd.action === "step") next.playing = false
          else if (cmd.action === "seek") {
            const to = "seconds" in cmd ? cmd.seconds : "fraction" in cmd ? cmd.fraction * p.duration : now + cmd.delta
            next.position = Math.max(0, Math.min(p.duration || to, to))
            next.ended = false
          }
        }
        receivedAt.current = performance.now()
        return next
      })
      try {
        const r = await videosApi.control(cameraId, cmd)
        if (gen === generation.current) apply(r.playback)
      } catch {
        /* the next poll restores the real state */
      }
    },
    [cameraId, apply, position],
  )

  return { playback, position, send }
}
