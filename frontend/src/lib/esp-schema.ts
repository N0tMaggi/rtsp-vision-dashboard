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

// --- pose ESP pack ("Body" category, components/body-panel.tsx) ---
export interface HintToggleDef extends ToggleDef {
  hint: string
}

export const BODY_TOGGLES: HintToggleDef[] = [
  { key: "iff_colors", label: "Friend / Foe Colors", hint: "Known people green, unknown faces red, unidentified yellow. Overrides box colors." },
  { key: "gaze", label: "View Cone", hint: "Head direction from the face keypoints. An eye marks people looking into the camera." },
  { key: "posture", label: "Posture & Actions", hint: "Standing, sitting, lying, walking, waving, hands up… shown under each person." },
  { key: "status_bars", label: "Status Bars", hint: "Game-style bars beside each box. Burned into recordings and snapshots." },
]

export const BAR_TOGGLES: ToggleDef[] = [
  { key: "bar_dwell", label: "Dwell" },
  { key: "bar_conf", label: "Confidence" },
  { key: "bar_speed", label: "Speed" },
]

export const GESTURE_TOGGLES: HintToggleDef[] = [
  { key: "gesture_wave", label: "Wave", hint: "Wave for 2 s: the PTZ camera starts following you." },
  { key: "gesture_palm", label: "Open Palm", hint: "Hold an open palm still for 1.5 s: auto-track stops." },
  { key: "gesture_thumbs", label: "Thumbs Up", hint: "Hold a thumbs up for 1 s: a snapshot is saved as an event." },
]

export const IFF_COLORS: ToggleDef[] = [
  { key: "iff_color_friend", label: "Friend" },
  { key: "iff_color_foe", label: "Foe" },
  { key: "iff_color_unknown", label: "Unidentified" },
]

export const FACE_BLUR_MODES = [
  { value: "blur", label: "Blur" },
  { value: "pixelate", label: "Pixel" },
  { value: "blackbox", label: "Black Box" },
  { value: "image", label: "Image" },
] as const

export const BOX_STYLES = [
  { value: "full", label: "Full" },
  { value: "corner", label: "Corner" },
  { value: "3d", label: "3D" },
] as const
