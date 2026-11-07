import { Eye } from "lucide-react"
import { formatPlate } from "@/lib/platelog"
import { cn } from "@/lib/utils"

/** License-plate glyph (lucide has none). */
export function PlateIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" className={className} aria-hidden>
      <rect x="2" y="6.5" width="20" height="11" rx="2" />
      <path d="M6 6.5v11" />
      <path d="M9.5 12h2.5M14.5 12h4" />
    </svg>
  )
}

/** Plate text rendered like a plate: white field, anthracite type, yellow band. */
export function PlateBadge({ plate, size = "md", className }: { plate: string; size?: "sm" | "md" | "lg"; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-stretch overflow-hidden rounded-[5px] bg-[#f4f4f2] font-mono font-bold text-[#222427] ring-1 ring-[#222427] [box-shadow:inset_0_0_0_1.5px_#f4f4f2,inset_0_0_0_2.5px_#222427]",
        size === "lg" ? "h-9 text-xl" : size === "md" ? "h-7 text-base" : "h-5 text-xs",
        className,
      )}
    >
      <span className={cn("bg-[#ffd21f]", size === "lg" ? "w-2.5" : size === "md" ? "w-2" : "w-1.5")} />
      <span className={cn("flex items-center whitespace-pre tracking-[0.12em]", size === "lg" ? "px-3" : size === "md" ? "px-2" : "px-1.5")}>
        {formatPlate(plate)}
      </span>
    </span>
  )
}

/** Plate-log entry the user marked by hand (stage "Mark Plate" / "Read Plate"). */
export function ManualBadge({ edited }: { edited?: boolean }) {
  return (
    <span
      title={edited ? "Marked by hand, text corrected" : "Marked by hand"}
      className="inline-flex items-center rounded-full border border-primary/50 px-1.5 py-px text-[10px] font-semibold uppercase tracking-wider text-primary"
    >
      Manual
    </span>
  )
}

export function WatchFlag({ note }: { note?: string }) {
  return (
    <span
      title={note ? `Watchlist: ${note}` : "On the watchlist"}
      className="inline-flex items-center gap-1 rounded-full bg-destructive/15 px-2 py-0.5 text-[11px] text-destructive"
    >
      <Eye className="size-3" />
      {note ? <span className="max-w-28 truncate">{note}</span> : "Watch"}
    </span>
  )
}
