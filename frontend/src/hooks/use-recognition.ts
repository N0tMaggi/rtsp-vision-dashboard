import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import {
  recognitionApi,
  type ModelInfo,
  type Person,
  type PlateEntry,
  type RecognitionStatus,
  type TensorRtCaps,
} from "@/lib/recognition"

function usePoll(fn: () => Promise<void>, ms: number, enabled = true) {
  const ref = useRef(fn)
  useLayoutEffect(() => {
    ref.current = fn
  })
  useEffect(() => {
    if (!enabled) return
    let alive = true
    let timer: number | undefined
    const tick = async () => {
      if (!document.hidden) {
        try {
          await ref.current()
        } catch {
          /* keep the last good data */
        }
      }
      if (alive) timer = window.setTimeout(tick, ms)
    }
    void tick()
    return () => {
      alive = false
      window.clearTimeout(timer)
    }
  }, [ms, enabled])
}

/** Live data for the Recognition panel; only polls while the panel is mounted. */
export function useRecognition(cameraId: string | null, platesOn: boolean) {
  const [status, setStatus] = useState<RecognitionStatus | null>(null)
  const [people, setPeople] = useState<Person[] | null>(null)
  const [plates, setPlates] = useState<PlateEntry[]>([])
  const [models, setModels] = useState<ModelInfo[]>([])
  const [trt, setTrt] = useState<TensorRtCaps | null>(null)

  const loadPeople = useCallback(async () => {
    const r = await recognitionApi.people()
    setPeople(r.people)
  }, [])
  const loadModels = useCallback(async () => {
    const r = await recognitionApi.models()
    setModels(r.models)
    setTrt(r.tensorrt)
  }, [])

  usePoll(async () => setStatus(await recognitionApi.status(cameraId ?? undefined)), 3000)
  usePoll(loadPeople, 15000)
  usePoll(
    async () => {
      if (cameraId) setPlates((await recognitionApi.plates(cameraId)).plates)
    },
    2000,
    Boolean(cameraId) && platesOn,
  )
  const exporting = models.some((m) => m.job?.status === "running")
  usePoll(loadModels, exporting ? 1500 : 6000)

  useEffect(() => setPlates([]), [cameraId])

  return { status, people, setPeople, reloadPeople: loadPeople, plates, setPlates, models, trt, reloadModels: loadModels }
}
