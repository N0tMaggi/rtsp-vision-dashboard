/** Model catalog helpers for the model picker (family / task / size). */

export type ModelFamily = "YOLO26" | "YOLO11" | "YOLOv8" | "RT-DETR" | "Other"
export type ModelTask = "detect" | "pose"

export interface ParsedModel {
  name: string
  family: ModelFamily
  task: ModelTask
  /** N, S, M, L, X (RT-DETR: L) */
  size: string
}

export interface FamilyMeta {
  id: ModelFamily
  label: string
  blurb: string
  recommended?: boolean
}

export const FAMILIES: FamilyMeta[] = [
  { id: "YOLO26", label: "YOLO26", blurb: "Newest, best on small and far-away objects", recommended: true },
  { id: "YOLO11", label: "YOLO11", blurb: "Previous generation" },
  { id: "YOLOv8", label: "YOLOv8", blurb: "Classic, widely used" },
  { id: "RT-DETR", label: "RT-DETR", blurb: "Transformer detector, slower" },
]

export const SIZE_ORDER = ["N", "S", "M", "L", "X"] as const

export const SIZE_META: Record<string, { label: string; hint: string }> = {
  N: { label: "Nano", hint: "Fastest" },
  S: { label: "Small", hint: "Fast" },
  M: { label: "Medium", hint: "Balanced" },
  L: { label: "Large", hint: "Accurate" },
  X: { label: "X-Large", hint: "Most accurate" },
}

export const TASK_META: Record<ModelTask, { label: string; hint: string }> = {
  detect: { label: "Detect", hint: "All objects: people, cars, plates, pets" },
  pose: { label: "Pose", hint: "People only, with skeleton keypoints" },
}

export function parseModel(name: string): ParsedModel {
  const stem = name.replace(/\.(pt|engine)$/, "")
  const task: ModelTask = stem.endsWith("-pose") ? "pose" : "detect"
  const base = stem.replace(/-pose$/, "")
  if (base.startsWith("rtdetr")) return { name, family: "RT-DETR", task, size: (base.split("-")[1] ?? "l").toUpperCase() }
  const family: ModelFamily = base.startsWith("yolo26")
    ? "YOLO26"
    : base.startsWith("yolo11")
      ? "YOLO11"
      : base.startsWith("yolov8")
        ? "YOLOv8"
        : "Other"
  return { name, family, task, size: base.slice(-1).toUpperCase() }
}

/** Short human label, e.g. "YOLO26 M" / "YOLOv8 L Pose". */
export function modelLabel(name: string | null | undefined): string {
  if (!name) return "Model"
  const m = parseModel(name)
  return `${m.family === "Other" ? name.replace(/\.(pt|engine)$/, "") : m.family} ${m.size}${m.task === "pose" ? " Pose" : ""}`
}

/** Best default for this machine: medium on a CUDA GPU, nano on CPU. */
export function recommendedModel(models: string[], cuda: boolean, task: ModelTask = "detect"): string | null {
  const want = `yolo26${cuda ? "m" : "n"}${task === "pose" ? "-pose" : ""}.pt`
  return models.includes(want) ? want : null
}

/** Find the model for a (family, task, size) combo, or null if it isn't offered. */
export function findModel(models: string[], family: ModelFamily, task: ModelTask, size: string): string | null {
  return models.find((n) => {
    const m = parseModel(n)
    return m.family === family && m.task === task && m.size === size
  }) ?? null
}

export function sizesFor(models: string[], family: ModelFamily, task: ModelTask): string[] {
  const set = new Set(models.map(parseModel).filter((m) => m.family === family && m.task === task).map((m) => m.size))
  return SIZE_ORDER.filter((s) => set.has(s))
}

export function tasksFor(models: string[], family: ModelFamily): ModelTask[] {
  const set = new Set(models.map(parseModel).filter((m) => m.family === family).map((m) => m.task))
  return (["detect", "pose"] as ModelTask[]).filter((t) => set.has(t))
}
