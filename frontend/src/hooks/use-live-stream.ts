import { useEffect, useLayoutEffect, useRef, useSyncExternalStore } from "react"
import { liveStream, type LiveState, type LiveTopic, type LiveTopics } from "@/lib/live-stream"

const subscribeState = (fn: () => void) => liveStream.onState(fn)
const getState = () => liveStream.state

/** Connection state of the tab's live stream. */
export function useLiveState(): LiveState {
  return useSyncExternalStore(subscribeState, getState, getState)
}

/**
 * Opens the tab's single SSE connection for ``cameraId`` (switching camera
 * reconnects) and returns its state. Call once, in the app shell.
 */
export function useLiveStream(cameraId: string | null): { state: LiveState; healthy: boolean } {
  useEffect(() => {
    liveStream.setCamera(cameraId)
  }, [cameraId])
  useEffect(() => () => liveStream.setCamera(null), [])
  const state = useLiveState()
  return { state, healthy: state === "open" }
}

/**
 * True while hooks should use their polling fallback: the stream is down or
 * unsupported. During the first connect attempt (a few seconds at most) and
 * while the tab is hidden, neither the stream nor polling runs.
 */
export function useShouldPoll(): boolean {
  const s = useLiveState()
  return s === "reconnecting" || s === "idle"
}

/** Calls ``fn`` with every pushed payload of ``topic`` (and the newest one on subscribe). */
export function useLiveTopic<T extends LiveTopic>(topic: T, fn: (data: LiveTopics[T]) => void, enabled = true) {
  const ref = useRef(fn)
  useLayoutEffect(() => {
    ref.current = fn
  })
  useEffect(() => {
    if (!enabled) return
    return liveStream.subscribe(topic, (d) => ref.current(d as LiveTopics[T]))
  }, [topic, enabled])
}

/** Ask the stream for the grid overview topic while ``on``. */
export function useLiveOverview(on: boolean) {
  useEffect(() => {
    if (!on) return
    liveStream.wantOverview(true)
    return () => liveStream.wantOverview(false)
  }, [on])
}
