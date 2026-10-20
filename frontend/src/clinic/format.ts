import type { Measure, Patient } from "./types"

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

export function formatDate(iso: string): string {
  const [year, month, day] = iso.split("-").map(Number)
  return `${day} ${MONTHS[month - 1]} ${year}`
}

export function formatDay(iso: string): string {
  const [, month, day] = iso.split("-").map(Number)
  return `${day} ${MONTHS[month - 1]}`
}

export function formatMeasure(value: number, measure: Measure): string {
  const text = value.toFixed(measure.decimals)
  if (measure.unit === "%") return `${text}%`
  if (measure.unit === "mmHg") return `${text} mmHg`
  return text
}

export function formatGoal(measure: Measure): string {
  const text = measure.goal.toFixed(measure.decimals)
  if (measure.unit === "%") return `${text}%`
  if (measure.unit === "mmHg") return `${text} mmHg`
  return `${text} points`
}

export function scaleNote(measure: Measure): string {
  if (measure.unit === "%" || measure.unit === "mmHg") return "Higher is worse."
  return `Out of ${measure.high}. Higher is worse.`
}

export function fullName(patient: Patient): string {
  return `${patient.given} ${patient.family}`
}
