/** Compute devices API (/api/compute) + one shared, low-frequency store for every component. */
import { useEffect, useSyncExternalStore } from "react"

export type DeviceId = "auto" | "cuda" | "npu" | "igpu" | "cpu"
export type WorkloadId = "yolo" | "faces" | "plates" | "depth"
export type NoticeLevel = "info" | "warn" | "error"

export interface HardwareDevice {
  id: "cuda" | "npu" | "igpu" | "cpu" | "cpu_torch"
  label: string
  name: string | null
  available: boolean
  reason: string | null
  backend: string
  selectable: boolean
  ov_device?: string | null
}

export interface DeviceOption {
  id: DeviceId
  label: string
  short?: string
  available: boolean
  reason: string | null
  recommended: boolean
}

export interface Effective {
  requested: DeviceId
  source: "global" | "camera"
  device: DeviceId
  fallback_reason: string | null
  error: string | null
}

export interface Notice {
  level: NoticeLevel
  text: string
}

export interface BenchRow {
  device: DeviceId | "cpu_torch"
  label: string
  ms: number | null
  error: string | null
  auto: boolean
  status?: "pending" | "running" | "done" | "failed" | "skipped"
  stage?: string
}

export interface BenchResult {
  workload: WorkloadId
  model: string | null
  finished: number
  rows: BenchRow[]
}

export interface BenchJob extends BenchResult {
  status: "running" | "done" | "failed"
  stage: string
  started: number
  error: string | null
  elapsed_s?: number
}

export interface PrepareJob {
  camera: string
  model: string
  device: DeviceId
  status: "queued" | "running" | "ready" | "failed" | "cancelled"
  stage: string
  error: string | null
  elapsed_s?: number
}

export type RunState = "auto" | "active" | "preparing" | "switching" | "pending" | "idle"

export interface Workload {
  id: WorkloadId
  label: string
  description: string
  per_camera: boolean
  assignment: DeviceId
  effective: Effective
  auto: { device: DeviceId; label: string }
  options: DeviceOption[]
  model: string | null
  warnings: Notice[]
  benchmark: BenchResult | null
  state?: RunState
}

export interface ComputeCamera {
  id: string
  label: string
  source_kind: string
  model: string
  yolo_enabled: boolean
  override: DeviceId | null
  effective: Effective
  badge: string | null
  warnings: Notice[]
  state: RunState
  backend: string | null
  job: PrepareJob | null
}

export interface ComputeState {
  ok: boolean
  hardware: {
    ready: boolean
    devices: HardwareDevice[]
    openvino: { installed: boolean; version: string | null; error: string | null; install_cmd: string | null }
  }
  workloads: Workload[]
  cameras: ComputeCamera[]
  non_default: boolean
  benchmark_job: BenchJob | null
  reference_machine: string
}

export class ComputeError extends Error {
  code: string
  detail: string | null
  constructor(code: string, detail: string | null) {
    super(code)
    this.code = code
    this.detail = detail
  }
}

async function call<T>(url: string, method = "GET", body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  let data: unknown = null
  try {
    data = await res.json()
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) {
    const d = data as { error?: string; detail?: string } | null
    throw new ComputeError(d?.error ?? `http_${res.status}`, d?.detail ?? null)
  }
  return data as T
}

export const computeApi = {
  get: () => call<ComputeState>("/api/compute"),
  patch: (body: { workloads?: Partial<Record<WorkloadId, DeviceId>>; cameras?: Record<string, DeviceId | null> }) =>
    call<ComputeState>("/api/compute", "PATCH", body),
  reset: () => call<ComputeState>("/api/compute/reset", "POST", {}),
  benchmark: (workload: WorkloadId, camera?: string | null) =>
    call<{ ok: true; job: BenchJob }>("/api/compute/benchmark", "POST", { workload, ...(camera ? { camera } : {}) }),
}

const ERRORS: Record<string, string> = {
  device_unavailable: "That device isn't available here.",
  invalid_device: "Unknown device.",
  unknown_camera: "That camera no longer exists.",
  busy: "A speed test is already running.",
  yolo_unavailable: "YOLO isn't installed, nothing to test.",
  invalid_model: "Unknown model.",
}

export function computeErrorText(e: unknown): string {
  if (e instanceof ComputeError) return e.detail ?? ERRORS[e.code] ?? `Request failed (${e.code}).`
  return "Request failed. Is the server running?"
}

// ---------------------------------------------------------------------------- shared store
// One poller for all subscribers: every 20 s for the small "non-default" chip, every
// 1.5 s while the Compute tab is open or something is preparing / being measured.
let snapshot: ComputeState | null = null
const listeners = new Set<() => void>()
let fastSubscribers = 0
let timer: number | undefined
let inflight = false

function busy(s: ComputeState | null) {
  if (!s) return false
  return (
    s.benchmark_job?.status === "running" ||
    s.cameras.some((c) => c.state === "preparing" || c.state === "switching") ||
    s.workloads.some((w) => w.state === "preparing") ||
    !s.hardware.ready
  )
}

export function setComputeState(s: ComputeState) {
  snapshot = s
  listeners.forEach((l) => l())
  schedule()
}

export async function refreshCompute() {
  if (inflight) return
  inflight = true
  try {
    setComputeState(await computeApi.get())
  } catch {
    /* keep the last good data */
  } finally {
    inflight = false
  }
}

function schedule() {
  window.clearTimeout(timer)
  if (listeners.size === 0) return
  const ms = fastSubscribers > 0 || busy(snapshot) ? 1500 : 20000
  timer = window.setTimeout(() => {
    if (!document.hidden) void refreshCompute()
    else schedule()
  }, ms)
}

function subscribe(l: () => void) {
  listeners.add(l)
  if (listeners.size === 1) void refreshCompute()
  else schedule()
  return () => {
    listeners.delete(l)
    if (listeners.size === 0) window.clearTimeout(timer)
  }
}

/** Shared compute state. ``live`` polls fast while the caller is mounted (the Compute tab). */
export function useCompute(live = false): ComputeState | null {
  useEffect(() => {
    if (!live) return
    fastSubscribers += 1
    void refreshCompute()
    return () => {
      fastSubscribers -= 1
    }
  }, [live])
  return useSyncExternalStore(subscribe, () => snapshot, () => null)
}

export const OPEN_COMPUTE_EVENT = "securitycam:open-compute"

/** Short label of the YOLO device of a camera when it is not on Auto, e.g. "NPU". */
export function cameraBadge(s: ComputeState | null, cameraId: string | null | undefined) {
  if (!s || !cameraId) return null
  const cam = s.cameras.find((c) => c.id === cameraId)
  if (!cam || cam.effective.requested === "auto") return null
  return { label: cam.badge ?? cam.effective.requested.toUpperCase(), fallback: cam.effective.device !== cam.effective.requested, state: cam.state, cam }
}

export const DEVICE_SHORT: Record<DeviceId, string> = { auto: "Auto", cuda: "GPU", npu: "NPU", igpu: "iGPU", cpu: "CPU" }
