import type { Condition, Course, Measure, Patient, Reading, Treatment } from "./types"

export const INPUTS = [
  {
    label: "Severity",
    detail: "Latest result, scaled from the low end of this measure (0) to the high end (1). Higher is worse.",
  },
  {
    label: "Adherence",
    detail: "Share of the plan being followed, after this option raises or lowers it. 0 is none and 1 is the whole plan.",
  },
  {
    label: "Comorbidity",
    detail: "Count of other problems divided by 4, capped at 1.",
  },
  {
    label: "Age",
    detail: "Age scaled from 18 years (0) to 90 years (1).",
  },
  {
    label: "Trajectory",
    detail: "Recent direction. 0.5 is flat. Above 0.5 the number has been rising. Below 0.5 it has been falling.",
  },
  {
    label: "Intensity",
    detail: "A fixed strength for this option. It is not measured from the patient.",
  },
] as const

export const HIDDEN = [
  { label: "Burden", detail: "Rises with severity, other problems, age, and a rising course." },
  { label: "Engagement", detail: "Rises with adherence." },
  { label: "Treatment effect", detail: "Rises with the option's intensity and with adherence." },
  { label: "Residual risk", detail: "Rises with other problems, age, and a rising course. Falls as intensity rises." },
] as const

export const OUTPUTS = [
  { label: "Change in severity", detail: "Added to today's severity. Negative means the model expects the result to fall." },
  { label: "Response", detail: "Turned into a chance that the result improves by at least the stated goal within 12 weeks." },
] as const

export const W1 = [
  [1.1, -0.15, 0.55, 0.25, 0.6, -0.05],
  [-0.05, 1.25, -0.2, -0.05, -0.1, 0.1],
  [0.25, 0.4, -0.1, -0.05, 0.05, 1.2],
  [0.2, -0.25, 0.7, 0.35, 0.45, -0.4],
]

export const B1 = [-0.4, -0.3, -0.45, -0.15]

export const W2 = [
  [0.1, -0.12, -0.16, 0.09],
  [-0.9, 1.15, 1.55, -1.25],
]

export const B2 = [-0.03, -0.85]

export type Term = {
  label: string
  weight: number
  value: number
  product: number
}

export type UnitStep = {
  index: number
  label: string
  bias: number
  terms: Term[]
  z: number
  activation: number | null
}

export type Explanation = {
  inputs: { label: string; detail: string; value: number }[]
  hidden: UnitStep[]
  change: UnitStep
  response: UnitStep
  severity: number
  delta: number
  futureSeverity: number
  probability: number
  currentNative: number
  futureNative: number
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

export function sigmoid(value: number): number {
  const x = clamp(value, -20, 20)
  return 1 / (1 + Math.exp(-x))
}

export function trajectoryOf(readings: Reading[], measure: Measure): number {
  const window = readings.slice(-8)
  if (window.length < 2) return 0.5
  const delta = window[window.length - 1].value - window[0].value
  const span = measure.high - measure.low
  return clamp(0.5 + delta / span / 0.5, 0, 1)
}

export function recentDelta(readings: Reading[]): number {
  const window = readings.slice(-8)
  if (window.length < 2) return 0
  return window[window.length - 1].value - window[0].value
}

export function courseOf(delta: number, measure: Measure): Course {
  const band = (measure.high - measure.low) * 0.02
  if (delta < -band) return "Improving"
  if (delta > band) return "Worsening"
  return "Stable"
}

export function featureVector(patient: Patient, treatment: Treatment, condition: Condition): number[] {
  const latest = patient.readings[patient.readings.length - 1]?.value ?? condition.measure.low
  const span = condition.measure.high - condition.measure.low
  const severity = clamp((latest - condition.measure.low) / span, 0, 1)
  const adherence = clamp(patient.adherence + treatment.adherenceDelta, 0, 1)
  const comorbidity = clamp(patient.problems.length / 4, 0, 1)
  const age = clamp((patient.age - 18) / 72, 0, 1)
  const trajectory = trajectoryOf(patient.readings, condition.measure)
  return [severity, adherence, comorbidity, age, trajectory, treatment.intensity]
}

function combine(values: number[], weights: number[], bias: number, labels: readonly { label: string }[]): UnitStep {
  const terms = weights.map((weight, index) => ({
    label: labels[index].label,
    weight,
    value: values[index],
    product: weight * values[index],
  }))
  const z = bias + terms.reduce((sum, term) => sum + term.product, 0)
  return { index: 0, label: "", bias, terms, z, activation: null }
}

export function explain(features: number[], currentNative: number, measure: Measure): Explanation {
  const hidden = W1.map((weights, index) => {
    const step = combine(features, weights, B1[index], INPUTS)
    return {
      ...step,
      index,
      label: HIDDEN[index].label,
      activation: Math.max(0, step.z),
    }
  })
  const activations = hidden.map((step) => step.activation ?? 0)
  const outputs = W2.map((weights, index) => {
    const step = combine(activations, weights, B2[index], HIDDEN)
    return { ...step, index, label: OUTPUTS[index].label }
  })
  const severity = features[0]
  const delta = outputs[0].z
  const futureSeverity = clamp(severity + delta, 0, 1)
  const span = measure.high - measure.low
  return {
    inputs: INPUTS.map((input, index) => ({ ...input, value: features[index] })),
    hidden,
    change: outputs[0],
    response: outputs[1],
    severity,
    delta,
    futureSeverity,
    probability: sigmoid(outputs[1].z),
    currentNative,
    futureNative: measure.low + futureSeverity * span,
  }
}

export function forecastValues(lastDate: string, current: number, future: number, weeks = 12): { date: string; value: number }[] {
  const denom = 1 - Math.exp(-0.22 * weeks)
  return Array.from({ length: weeks }, (_, index) => {
    const week = index + 1
    const progress = (1 - Math.exp(-0.22 * week)) / denom
    return {
      date: addDays(lastDate, 7 * week),
      value: current + (future - current) * progress,
    }
  })
}

export function addDays(iso: string, days: number): string {
  const [year, month, day] = iso.split("-").map(Number)
  const date = new Date(Date.UTC(year, month - 1, day))
  date.setUTCDate(date.getUTCDate() + days)
  return date.toISOString().slice(0, 10)
}

export function allergyCaution(allergies: string, treatment: Treatment): string | null {
  const record = allergies.toLowerCase()
  const option = `${treatment.name} ${treatment.summary}`.toLowerCase()
  const rules: [string, string, string][] = [
    ["ace inhibitor", "ace", "The allergy list mentions an ACE inhibitor reaction. Choose a substitute before using this option."],
    ["penicillin", "penicillin", "The allergy list mentions penicillin. Do not use this option unless an alternative is chosen."],
    ["sulfa", "sulfa", "The allergy list mentions a sulfa reaction. Do not use this option unless an alternative is chosen."],
  ]
  for (const [drug, allergy, message] of rules) {
    if (option.includes(drug) && record.includes(allergy)) return message
  }
  return null
}

export type Projection = {
  treatment: Treatment
  features: number[]
  explanation: Explanation
  forecast: { date: string; value: number }[]
  caution: string | null
}

export function projectTreatment(patient: Patient, treatment: Treatment, condition: Condition): Projection {
  const current = patient.readings[patient.readings.length - 1]?.value ?? condition.measure.low
  const lastDate = patient.readings[patient.readings.length - 1]?.date ?? "2026-10-20"
  const features = featureVector(patient, treatment, condition)
  const explanation = explain(features, current, condition.measure)
  return {
    treatment,
    features,
    explanation,
    forecast: forecastValues(lastDate, current, explanation.futureNative),
    caution: allergyCaution(patient.allergies, treatment),
  }
}

export function projectAll(patient: Patient, condition: Condition): Projection[] {
  return condition.treatments
    .map((treatment) => projectTreatment(patient, treatment, condition))
    .sort(
      (a, b) =>
        b.explanation.probability - a.explanation.probability ||
        a.explanation.futureNative - b.explanation.futureNative,
    )
}

export function sumLine(bias: number, terms: Term[]): string {
  const body = terms.map((term) => signedTerm(term.weight, term.label)).join(" ")
  return `${bias.toFixed(2)} ${body}`
}

function signedTerm(weight: number, label: string): string {
  const mag = Math.abs(weight).toFixed(2)
  return weight < 0 ? `− ${mag}·${label}` : `+ ${mag}·${label}`
}
