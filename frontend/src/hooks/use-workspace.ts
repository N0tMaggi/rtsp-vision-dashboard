import { useCallback, useEffect, useState } from "react"
import type { Camera } from "@/lib/types"

const STORAGE_KEY = "securitycam.workspace.v2"

interface Workspace {
  openIds: string[]
  activeId: string | null
}

function load(): Partial<Workspace> {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}") as Partial<Workspace>
  } catch {
    return {}
  }
}

function save(ws: Workspace) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ws))
  } catch {
    /* storage unavailable — workspace is just not remembered */
  }
}

/** Open camera tabs + active tab, remembered per browser. */
export function useWorkspace(cameras: Camera[], defaultId: string | null) {
  const [ws, setWs] = useState<Workspace>({ openIds: [], activeId: null })

  // Reconcile stored workspace with the cameras the server knows about.
  useEffect(() => {
    if (cameras.length === 0) return
    const valid = new Set(cameras.map((c) => c.id))
    const stored = load()
    let openIds = (stored.openIds ?? []).filter((id) => valid.has(id))
    if (openIds.length === 0) openIds = cameras.map((c) => c.id)
    let activeId = stored.activeId && openIds.includes(stored.activeId) ? stored.activeId : null
    activeId ??= defaultId && openIds.includes(defaultId) ? defaultId : openIds[0]
    setWs({ openIds, activeId })
  }, [cameras, defaultId])

  useEffect(() => {
    if (ws.activeId) save(ws)
  }, [ws])

  const open = useCallback((id: string) => {
    setWs((prev) => ({
      openIds: prev.openIds.includes(id) ? prev.openIds : [...prev.openIds, id],
      activeId: id,
    }))
  }, [])

  const close = useCallback((id: string) => {
    setWs((prev) => {
      const openIds = prev.openIds.filter((x) => x !== id)
      if (openIds.length === 0) return prev // always keep one tab open
      const idx = prev.openIds.indexOf(id)
      const activeId = prev.activeId === id ? openIds[Math.max(0, idx - 1)] : prev.activeId
      return { openIds, activeId }
    })
  }, [])

  return { openIds: ws.openIds, activeId: ws.activeId, open, close }
}
