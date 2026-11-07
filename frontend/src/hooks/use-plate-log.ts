import { useCallback, useEffect, useRef, useState } from "react"
import {
  plateLogApi,
  type PlateGroup,
  type PlateLogEntry,
  type PlateLogFilters,
  type WatchItem,
} from "@/lib/platelog"
import { useLiveTopic, useShouldPoll } from "@/hooks/use-live-stream"
import type { PlatesSummary } from "@/lib/live-stream"

const SUMMARY_MS = 3000
const LIST_MS = 2000
const SEEN_KEY = "securitycam.platelog.seen"

function readSeen(): number {
  try {
    return Number(localStorage.getItem(SEEN_KEY)) || Date.now() / 1000
  } catch {
    return Date.now() / 1000
  }
}

/**
 * Plate log data. A cheap summary (badge for new reads) is pushed by the live
 * stream, or polled while it is down; the list / grouped plates are loaded
 * only while the drawer is open and only refetched when the server's version
 * or the filters change.
 */
export function usePlateLog(open: boolean) {
  const [filters, setFilters] = useState<PlateLogFilters>({})
  // search text is debounced; camera / date filters apply at once
  const [debounced, setDebounced] = useState<PlateLogFilters>({})
  useEffect(() => {
    const t = window.setTimeout(() => setDebounced(filters), filters.q !== debounced.q ? 220 : 0)
    return () => window.clearTimeout(t)
  }, [filters, debounced.q])
  const [items, setItems] = useState<PlateLogEntry[]>([])
  const [total, setTotal] = useState(0)
  const [groups, setGroups] = useState<PlateGroup[]>([])
  const [watch, setWatch] = useState<WatchItem[]>([])
  const [unread, setUnread] = useState(0)
  const [seen, setSeen] = useState(readSeen)
  const [loaded, setLoaded] = useState(false)
  const [available, setAvailable] = useState(true)
  const version = useRef(-1)
  const fetched = useRef("")
  const key = JSON.stringify(debounced)
  const filtersRef = useRef(debounced)
  useEffect(() => {
    filtersRef.current = debounced
  }, [debounced])

  const loadList = useCallback(async (signal?: AbortSignal) => {
    const f = filtersRef.current
    const [list, g] = await Promise.all([
      plateLogApi.list(f, 300, signal),
      plateLogApi.groups(f, signal),
    ])
    setItems(list.items)
    setTotal(list.total)
    setGroups(g.plates)
    setLoaded(true)
  }, [])

  const loadWatch = useCallback(async () => {
    try {
      setWatch((await plateLogApi.watchlist()).plates)
    } catch {
      /* older backend */
    }
  }, [])

  const poll = useShouldPoll()
  const [pushed, setPushed] = useState<PlatesSummary | null>(null)
  useLiveTopic("plates", setPushed)

  // badge + change detection from the stream
  useEffect(() => {
    if (poll || !pushed) return
    setAvailable(true)
    const recent = pushed.recent
    setUnread(recent ? recent.filter((t) => t > seen).length : pushed.new)
    if (open && (pushed.version !== version.current || fetched.current !== key)) {
      version.current = pushed.version
      fetched.current = key
      loadList().catch(() => {
        version.current = -1     // retry on the next push
      })
    }
  }, [poll, pushed, open, key, seen, loadList])

  // badge + change detection by polling (stream down)
  useEffect(() => {
    if (!poll) return
    let stopped = false
    let timer: number | undefined
    let ctrl: AbortController | null = null
    const tick = async () => {
      if (stopped) return
      if (!document.hidden) {
        ctrl = new AbortController()
        try {
          const s = await plateLogApi.summary(seen, ctrl.signal)
          setAvailable(true)
          setUnread(s.new)
          if (open && (s.version !== version.current || fetched.current !== key)) {
            version.current = s.version
            fetched.current = key
            await loadList(ctrl.signal)
          }
        } catch (e) {
          if ((e as { status?: number }).status === 404) setAvailable(false)
        }
      }
      if (!stopped) timer = window.setTimeout(tick, open ? LIST_MS : SUMMARY_MS)
    }
    void tick()
    return () => {
      stopped = true
      window.clearTimeout(timer)
      ctrl?.abort()
    }
  }, [open, key, seen, loadList, poll])

  useEffect(() => {
    if (open) void loadWatch()
  }, [open, loadWatch])

  const markSeen = useCallback(() => {
    const now = Date.now() / 1000
    setSeen(now)
    setUnread(0)
    try {
      localStorage.setItem(SEEN_KEY, String(now))
    } catch {
      /* private mode */
    }
  }, [])

  const refresh = useCallback(async () => {
    version.current = -1
    try {
      await loadList()
    } catch {
      /* keep the last state */
    }
  }, [loadList])

  const remove = useCallback(
    async (e: PlateLogEntry) => {
      setItems((list) => list.filter((x) => x.id !== e.id))
      try {
        await plateLogApi.remove(e.id)
      } finally {
        void refresh()
      }
    },
    [refresh],
  )

  const clear = useCallback(
    async (camera?: string) => {
      setItems((list) => (camera ? list.filter((x) => x.camera !== camera) : []))
      try {
        await plateLogApi.clear(camera)
      } finally {
        void refresh()
      }
    },
    [refresh],
  )

  const toggleWatch = useCallback(
    async (plate: string, note = "") => {
      const hit = watch.find((w) => w.plate === plate)
      try {
        if (hit) await plateLogApi.unwatch(hit.id)
        else await plateLogApi.watch(plate, note)
      } finally {
        void loadWatch()
      }
    },
    [watch, loadWatch],
  )

  const removeWatch = useCallback(
    async (id: string) => {
      setWatch((w) => w.filter((x) => x.id !== id))
      try {
        await plateLogApi.unwatch(id)
      } finally {
        void loadWatch()
      }
    },
    [loadWatch],
  )

  const addWatch = useCallback(
    async (plate: string, note: string) => {
      await plateLogApi.watch(plate, note)
      await loadWatch()
    },
    [loadWatch],
  )

  return {
    filters,
    setFilters,
    searching: filters.q !== debounced.q,
    items,
    total,
    groups,
    watch,
    unread,
    seen,
    loaded,
    available,
    markSeen,
    refresh,
    remove,
    clear,
    toggleWatch,
    addWatch,
    removeWatch,
  }
}
