import { useEffect } from "react"
import { motion, useSpring, useTransform } from "motion/react"

interface Props {
  value: number
  format?: (v: number) => string
  className?: string
}

/** A number that glides to its new value instead of jumping. */
export function RollingNumber({ value, format = (v) => v.toFixed(0), className }: Props) {
  const spring = useSpring(value, { stiffness: 140, damping: 22, mass: 0.6 })
  const text = useTransform(spring, (v) => format(v))
  useEffect(() => {
    spring.set(value)
  }, [spring, value])
  return <motion.span className={className}>{text}</motion.span>
}
