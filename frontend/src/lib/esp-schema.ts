export interface ToggleDef {
  key: string
  label: string
}

export interface SliderDef {
  key: string
  label: string
  min: number
  max: number
  step: number
  format: (v: number) => string
}

export const VISUAL_TOGGLES: ToggleDef[] = [
  { key: "boxes", label: "Boxes" },
  { key: "names", label: "Names" },
  { key: "conf", label: "Confidence" },
  { key: "skeleton", label: "Skeleton" },
  { key: "head", label: "Head" },
  { key: "snaplines", label: "Snaplines" },
  { key: "center_dot", label: "Crosshair" },
  { key: "distance", label: "Distance" },
  { key: "fill_box", label: "Box Fill" },
  { key: "chams", label: "Chams" },
  { key: "class_colors", label: "Class Colors" },
  { key: "person_only", label: "Person Only" },
  { key: "highlight_center", label: "Center Highlight" },
  { key: "thermal", label: "Thermal" },
]

export const FACE_TOGGLES: ToggleDef[] = [
  { key: "face_boxes", label: "Face Boxes" },
  { key: "face_zoom", label: "Face Zoom" },
  { key: "face_blur", label: "Face Blur" },
  { key: "hand_skeleton", label: "Hand Skeleton" },
]

export const TRACKING_TOGGLES: ToggleDef[] = [
  { key: "tracking", label: "Tracking" },
  { key: "tracers", label: "Tracers" },
  { key: "velocity", label: "Velocity" },
  { key: "prediction", label: "Prediction" },
]

export const STYLE_TOGGLES: ToggleDef[] = [
  { key: "glow", label: "Glow" },
  { key: "label_background", label: "Label Background" },
  { key: "rgb_mode", label: "RGB Mode" },
]

const pct = (v: number) => `${Math.round(v * 100)}%`

export const SLIDERS: SliderDef[] = [
  { key: "confidence_threshold", label: "Confidence", min: 0.05, max: 1, step: 0.05, format: pct },
  { key: "line_thickness", label: "Line", min: 1, max: 6, step: 1, format: (v) => `${Math.round(v)}px` },
  { key: "chams_opacity", label: "Chams Opacity", min: 0, max: 1, step: 0.05, format: pct },
  { key: "face_zoom_scale", label: "Face Zoom", min: 1, max: 4, step: 0.25, format: (v) => `${v.toFixed(2)}x` },
  { key: "face_blur_strength", label: "Blur Strength", min: 1, max: 100, step: 1, format: (v) => `${Math.round(v)}%` },
  { key: "smoothing", label: "Smoothing", min: 0, max: 1, step: 0.05, format: pct },
  { key: "jpeg_quality", label: "JPEG Quality", min: 30, max: 95, step: 5, format: (v) => `${Math.round(v)}` },
]

export const COLORS: ToggleDef[] = [
  { key: "color_primary", label: "Primary" },
  { key: "color_secondary", label: "Secondary" },
  { key: "color_boxes", label: "Boxes" },
  { key: "color_skeleton", label: "Skeleton" },
  { key: "color_head", label: "Head" },
  { key: "color_tracers", label: "Tracers" },
  { key: "color_velocity", label: "Velocity" },
  { key: "color_prediction", label: "Prediction" },
  { key: "color_face", label: "Face" },
  { key: "color_hand", label: "Hand" },
  { key: "color_text", label: "Text" },
  { key: "chams_color", label: "Chams" },
]

export const FACE_BLUR_MODES = [
  { value: "blur", label: "Blur" },
  { value: "pixelate", label: "Pixel" },
  { value: "blackbox", label: "Black Box" },
] as const

export const BOX_STYLES = [
  { value: "full", label: "Full" },
  { value: "corner", label: "Corner" },
  { value: "3d", label: "3D" },
] as const
