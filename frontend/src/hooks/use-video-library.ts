import { useCallback, useEffect, useRef, useState } from "react"
import { VideoApiError, videosApi, type UploadHandle, type VideoLibrary } from "@/lib/videos"

export interface UploadItem {
  key: string
  name: string
  loaded: number
  total: number
  state: "uploading" | "error"
  error?: string
}

export interface VideoLibraryState {
  data: VideoLibrary | null
  error: string | null
  uploads: UploadItem[]
  refresh: () => Promise<void>
  upload: (files: File[]) => void
  cancel: (key: string) => void
  dismiss: (key: string) => void
  rename: (id: string, name: string) => Promise<void>
  remove: (id: string) => Promise<void>
}

const message = (e: unknown) => (e instanceof Error ? e.message : "Request failed.")

/** Library listing + uploads (XHR progress, cancel). Lists while ``active``. */
export function useVideoLibrary(active: boolean, onRemoved?: (cameraId: string | null) => void): VideoLibraryState {
  const [data, setData] = useState<VideoLibrary | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploads, setUploads] = useState<UploadItem[]>([])
  const handles = useRef(new Map<string, UploadHandle>())
  const seq = useRef(0)

  const refresh = useCallback(async () => {
    try {
      setData(await videosApi.list())
      setError(null)
    } catch (e) {
      setError(message(e))
    }
  }, [])

  useEffect(() => {
    if (active) void refresh()
  }, [active, refresh])

  const patch = (key: string, p: Partial<UploadItem>) => setUploads((list) => list.map((u) => (u.key === key ? { ...u, ...p } : u)))
  const drop = (key: string) => setUploads((list) => list.filter((u) => u.key !== key))

  const upload = useCallback(
    (files: File[]) => {
      for (const file of files) {
        const key = `u${++seq.current}`
        const max = data?.max_upload_bytes
        const ext = file.name.includes(".") ? file.name.split(".").pop()!.toLowerCase() : ""
        const item: UploadItem = { key, name: file.name, loaded: 0, total: file.size, state: "uploading" }
        // Cheap checks before sending gigabytes; the server validates again.
        if (data && !data.extensions.includes(ext)) {
          setUploads((l) => [...l, { ...item, state: "error", error: new VideoApiError("unsupported_type", 415).message }])
          continue
        }
        if (max && file.size > max) {
          setUploads((l) => [...l, { ...item, state: "error", error: new VideoApiError("too_large", 413).message }])
          continue
        }
        setUploads((l) => [...l, item])
        const h = videosApi.upload(file, (loaded, total) => patch(key, { loaded, total }))
        handles.current.set(key, h)
        h.promise
          .then(() => {
            drop(key)
            void refresh()
          })
          .catch((e: unknown) => {
            if (e instanceof DOMException && e.name === "AbortError") drop(key)
            else patch(key, { state: "error", error: message(e) })
          })
          .finally(() => handles.current.delete(key))
      }
    },
    [data, refresh],
  )

  useEffect(() => {
    const live = handles.current
    return () => live.forEach((h) => h.abort())
  }, [])

  const cancel = useCallback((key: string) => {
    handles.current.get(key)?.abort()
  }, [])

  const rename = useCallback(
    async (id: string, name: string) => {
      try {
        await videosApi.rename(id, name)
      } catch (e) {
        setError(message(e))
      }
      await refresh()
    },
    [refresh],
  )

  const remove = useCallback(
    async (id: string) => {
      setData((d) => (d ? { ...d, videos: d.videos.filter((v) => v.id !== id) } : d))
      try {
        const r = await videosApi.remove(id)
        onRemoved?.(r.closed)
      } catch (e) {
        setError(message(e))
      }
      await refresh()
    },
    [refresh, onRemoved],
  )

  return { data, error, uploads, refresh, upload, cancel, dismiss: drop, rename, remove }
}
