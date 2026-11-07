import { useCallback, useEffect, useRef, useState } from "react"
import { useLiveTopic, useShouldPoll } from "@/hooks/use-live-stream"
import { alertsApi, type ActiveAlarm, type AlarmEvent, type EventsResponse } from "@/lib/alerts"

const INTERVAL_MS = 2000
const SEEN_KEY = "securitycam.events.seen"

function readSeen(): number {
  try {
    return Number(localStorage.getItem(SEEN_KEY)) || 0
  } catch {
    return 0
  }
}

/**
 * Event list (all cameras) and the currently active alarms, pushed by the
 * live stream (polled while it is down). ``onNew`` fires for events that
 * appeared after the first received list.
 */
export function useEvents(onNew?: (events: AlarmEvent[]) => void) {
  const [events, setEvents] = useState<AlarmEvent[]>([])
  const [active, setActive] = useState<ActiveAlarm[]>([])
  const [seen, setSeen] = useState(readSeen)
  const known = useRef<Set<string> | null>(null)
  const onNewRef = useRef(onNew)
  useEffect(() => {
    onNewRef.current = onNew
  })

  const poll = useShouldPoll()

  const apply = useCallback((r: EventsResponse) => {
    setEvents(r.events)
    setActive(r.active)
    const keys = new Set(r.events.map((e) => `${e.camera}/${e.id}`))
    if (known.current) {
      const prev = known.current
      const fresh = r.events.filter((e) => !prev.has(`${e.camera}/${e.id}`) && e.ts > Date.now() / 1000 - 120)
      if (fresh.length) onNewRef.current?.(fresh)
    }
    known.current = keys
  }, [])

  const refresh = useCallback(
    async (signal?: AbortSignal) => {
      try {
        apply(await alertsApi.events(signal))
      } catch {
        /* server offline or older backend: keep the last state */
      }
    },
    [apply],
  )

  useLiveTopic("events", apply)

  useEffect(() => {
    if (!poll) return
    let stopped = false
    let timer: number | undefined
    let ctrl: AbortController | null = null
    const tick = async () => {
      if (stopped) return
      if (!document.hidden) {
        ctrl = new AbortController()
        await refresh(ctrl.signal)
      }
      if (!stopped) timer = window.setTimeout(tick, INTERVAL_MS)
    }
    void tick()
    return () => {
      stopped = true
      window.clearTimeout(timer)
      ctrl?.abort()
    }
  }, [refresh, poll])

  const markSeen = useCallback(() => {
    const newest = events.reduce((m, e) => Math.max(m, e.ts), 0)
    if (newest <= seen) return
    setSeen(newest)
    try {
      localStorage.setItem(SEEN_KEY, String(newest))
    } catch {
      /* private mode */
    }
  }, [events, seen])

  const remove = useCallback(
    async (e: AlarmEvent) => {
      setEvents((list) => list.filter((x) => !(x.id === e.id && x.camera === e.camera)))
      try {
        await alertsApi.deleteEvent(e)
      } finally {
        void refresh()
      }
    },
    [refresh],
  )

  const unread = events.filter((e) => e.ts > seen).length
  return { events, active, unread, seen, markSeen, remove, refresh }
}
