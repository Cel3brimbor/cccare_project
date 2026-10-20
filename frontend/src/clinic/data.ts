import { addDays } from "./model"
import type { Condition, Course, NewPatientInput, Patient, Reading, Sex } from "./types"

export const PANEL_DATE = "2026-10-20"

export const STORAGE_KEY = "meridian.patients.v1"

export const EXTRA_PROBLEMS = [
  "Hypertension",
  "Chronic kidney disease",
  "Obesity",
  "Depression",
  "COPD",
  "Coronary artery disease",
  "Atrial fibrillation",
  "Osteoporosis",
  "Type 2 diabetes",
  "Hyperlipidemia",
]

const hba1c = {
  id: "hba1c",
  name: "HbA1c",
  unit: "%",
  low: 5,
  high: 12,
  goal: 0.5,
  decimals: 1,
}

const symptom = {
  id: "symptom",
  name: "Symptom score",
  unit: "score",
  low: 0,
  high: 10,
  goal: 1.5,
  decimals: 1,
}

const phq = {
  id: "phq9",
  name: "PHQ-9",
  unit: "points",
  low: 0,
  high: 27,
  goal: 5,
  decimals: 0,
}

const pain = {
  id: "pain",
  name: "Pain score",
  unit: "score",
  low: 0,
  high: 10,
  goal: 2,
  decimals: 1,
}

const copd = {
  id: "copd",
  name: "COPD symptom score",
  unit: "score",
  low: 0,
  high: 10,
  goal: 1.5,
  decimals: 1,
}

const sbp = {
  id: "sbp",
  name: "Systolic blood pressure",
  unit: "mmHg",
  low: 90,
  high: 180,
  goal: 10,
  decimals: 0,
}

export const CONDITIONS: Condition[] = [
  {
    id: "diabetes",
    name: "Type 2 diabetes",
    measure: hba1c,
    treatments: [
      {
        id: "dm-continue",
        name: "Continue the current regimen",
        summary: "Keep metformin and the current add-on medicine at the present dose.",
        intensity: 0.35,
        adherenceDelta: 0,
      },
      {
        id: "dm-glp1",
        name: "Add a GLP-1 receptor agonist",
        summary: "A weekly injection. Nausea is common at the start. It often lowers glucose and weight.",
        intensity: 0.84,
        adherenceDelta: -0.06,
      },
      {
        id: "dm-education",
        name: "Education and metformin titration",
        summary: "Coaching on meals, activity, and a higher metformin dose when the kidneys allow it.",
        intensity: 0.55,
        adherenceDelta: 0.08,
      },
      {
        id: "dm-insulin",
        name: "Add basal insulin",
        summary: "A daily injection with teaching on low glucose. The expected drop is larger, and so is the daily burden.",
        intensity: 0.92,
        adherenceDelta: -0.12,
      },
    ],
  },
  {
    id: "heart-failure",
    name: "Heart failure with reduced ejection fraction",
    measure: symptom,
    treatments: [
      {
        id: "hf-continue",
        name: "Continue the current regimen",
        summary: "Keep the present beta blocker, loop diuretic, and renin-angiotensin medicine.",
        intensity: 0.38,
        adherenceDelta: 0,
      },
      {
        id: "hf-sglt2",
        name: "Add an SGLT2 inhibitor",
        summary: "Dapagliflozin or empagliflozin if kidney function and fluid status allow it.",
        intensity: 0.8,
        adherenceDelta: 0.02,
      },
      {
        id: "hf-rehab",
        name: "Add cardiac rehabilitation",
        summary: "Supervised activity. Much of the gain depends on attending.",
        intensity: 0.58,
        adherenceDelta: 0.18,
      },
      {
        id: "hf-mra",
        name: "Add a mineralocorticoid receptor antagonist",
        summary: "Spironolactone or eplerenone, with potassium and kidney checks.",
        intensity: 0.74,
        adherenceDelta: 0,
      },
    ],
  },
  {
    id: "depression",
    name: "Major depressive disorder",
    measure: phq,
    treatments: [
      {
        id: "dep-continue",
        name: "Continue the current dose",
        summary: "Stay on the present antidepressant and review side effects.",
        intensity: 0.4,
        adherenceDelta: 0,
      },
      {
        id: "dep-dose",
        name: "Increase the antidepressant dose",
        summary: "Move one step up the usual range if the current dose is tolerated.",
        intensity: 0.72,
        adherenceDelta: 0,
      },
      {
        id: "dep-cbt",
        name: "Add cognitive behavioral therapy",
        summary: "Weekly therapy beside the current medicine. Attendance is part of the effect.",
        intensity: 0.68,
        adherenceDelta: 0.18,
      },
      {
        id: "dep-switch",
        name: "Switch antidepressant class",
        summary: "A planned cross-taper to another class if this one has stalled.",
        intensity: 0.7,
        adherenceDelta: -0.04,
      },
    ],
  },
  {
    id: "arthritis",
    name: "Rheumatoid arthritis",
    measure: pain,
    treatments: [
      {
        id: "ra-continue",
        name: "Continue methotrexate",
        summary: "Keep the weekly dose and folic acid, and recheck labs on the usual schedule.",
        intensity: 0.36,
        adherenceDelta: 0,
      },
      {
        id: "ra-hcq",
        name: "Add hydroxychloroquine",
        summary: "A second conventional medicine. Eye screening is part of the plan.",
        intensity: 0.62,
        adherenceDelta: 0,
      },
      {
        id: "ra-bio",
        name: "Advance to a biologic",
        summary: "A targeted injection or infusion when conventional therapy is not enough. Infection risk goes up.",
        intensity: 0.88,
        adherenceDelta: -0.07,
      },
      {
        id: "ra-pt",
        name: "Add structured physical therapy",
        summary: "Joint protection and strengthening. The gain depends on doing the sessions.",
        intensity: 0.5,
        adherenceDelta: 0.12,
      },
    ],
  },
  {
    id: "copd",
    name: "COPD",
    measure: copd,
    treatments: [
      {
        id: "copd-continue",
        name: "Continue the current inhalers",
        summary: "Keep the long-acting bronchodilator and the rescue inhaler.",
        intensity: 0.36,
        adherenceDelta: 0,
      },
      {
        id: "copd-ics",
        name: "Add an inhaled corticosteroid",
        summary: "For repeated flare-ups. Review pneumonia risk and inhaler technique.",
        intensity: 0.7,
        adherenceDelta: 0,
      },
      {
        id: "copd-rehab",
        name: "Pulmonary rehabilitation",
        summary: "A supervised program of activity and education. Attendance drives the result.",
        intensity: 0.6,
        adherenceDelta: 0.16,
      },
      {
        id: "copd-plan",
        name: "Action plan and inhaler review",
        summary: "A written flare-up plan, spacer check, and technique teaching.",
        intensity: 0.48,
        adherenceDelta: 0.1,
      },
    ],
  },
  {
    id: "hypertension",
    name: "Hypertension",
    measure: sbp,
    treatments: [
      {
        id: "htn-continue",
        name: "Continue the current medicine",
        summary: "Keep the present calcium-channel blocker and recheck the home readings.",
        intensity: 0.34,
        adherenceDelta: 0,
      },
      {
        id: "htn-ace",
        name: "Add an ACE inhibitor",
        summary: "If cough or angioedema is already on the record, use an angiotensin receptor blocker instead.",
        intensity: 0.7,
        adherenceDelta: 0,
      },
      {
        id: "htn-thiazide",
        name: "Add a thiazide-like diuretic",
        summary: "Chlortalidone or indapamide, with a sodium and kidney check.",
        intensity: 0.66,
        adherenceDelta: 0,
      },
      {
        id: "htn-home",
        name: "Home blood pressure program",
        summary: "Confirmed home readings, a cuff check, and support for taking the medicine.",
        intensity: 0.48,
        adherenceDelta: 0.15,
      },
    ],
  },
]

const BLOCKED_EXTRAS: Record<string, string[]> = {
  diabetes: ["Type 2 diabetes"],
  depression: ["Depression"],
  copd: ["COPD"],
  hypertension: ["Hypertension"],
}

export function extrasFor(conditionId: string): string[] {
  const blocked = new Set(BLOCKED_EXTRAS[conditionId] ?? [])
  return EXTRA_PROBLEMS.filter((problem) => !blocked.has(problem))
}

export function conditionById(id: string): Condition {
  return CONDITIONS.find((condition) => condition.id === id) ?? CONDITIONS[0]
}

function readings(values: number[]): Reading[] {
  return weeksEnding(PANEL_DATE, values.length).map((date, index) => ({ date, value: values[index] }))
}

export function weeksEnding(end: string, count: number): string[] {
  return Array.from({ length: count }, (_, index) => addDays(end, -7 * (count - 1 - index)))
}

export function buildReadings(latest: number, course: Course, low: number, high: number, decimals: number): Reading[] {
  const count = 16
  const dates = weeksEnding(PANEL_DATE, count)
  const span = high - low
  const drift = course === "Improving" ? span * 0.08 : course === "Worsening" ? -span * 0.08 : 0
  const start = clamp(latest + drift, low, high)
  const wobble = [0.02, -0.015, 0.01, -0.02, 0.015, -0.01, 0.02, 0, -0.015, 0.01, -0.02, 0.015, 0, -0.01, 0.01, 0]
  return dates.map((date, index) => {
    const t = index / (count - 1)
    const raw = start + (latest - start) * t + wobble[index] * span
    const value = index === count - 1 ? latest : clamp(raw, low, high)
    return { date, value: roundTo(value, decimals) }
  })
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value))
}

function roundTo(value: number, decimals: number): number {
  const factor = 10 ** decimals
  return Math.round(value * factor) / factor
}

export const MOCK_PATIENTS: Patient[] = [
  {
    id: "elena-voss",
    mrn: "MC-10482",
    given: "Elena",
    family: "Voss",
    age: 58,
    sex: "Female",
    conditionId: "diabetes",
    adherence: 0.82,
    problems: ["Hypertension", "Obesity"],
    medications: ["Metformin 1000 mg twice daily", "Empagliflozin 10 mg daily", "Lisinopril 10 mg daily"],
    allergies: "No known drug allergies",
    note: "Asking whether a GLP-1 receptor agonist adds more than coaching and a higher metformin dose.",
    readings: readings([8.6, 8.5, 8.7, 8.4, 8.5, 8.3, 8.4, 8.2, 8.3, 8.1, 8.2, 8.0, 8.1, 7.9, 8.0, 7.9]),
    source: "mock",
  },
  {
    id: "james-okonkwo",
    mrn: "MC-11007",
    given: "James",
    family: "Okonkwo",
    age: 71,
    sex: "Male",
    conditionId: "heart-failure",
    adherence: 0.7,
    problems: ["Chronic kidney disease", "Type 2 diabetes"],
    medications: ["Carvedilol 12.5 mg twice daily", "Furosemide 40 mg daily", "Losartan 50 mg daily"],
    allergies: "Sulfa drugs — rash",
    note: "Breathlessness on one flight of stairs. Weight is up 1.4 kg since the last visit.",
    readings: readings([4.8, 4.9, 5.0, 5.2, 5.1, 5.4, 5.3, 5.6, 5.5, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4]),
    source: "mock",
  },
  {
    id: "priya-shah",
    mrn: "MC-10841",
    given: "Priya",
    family: "Shah",
    age: 34,
    sex: "Female",
    conditionId: "depression",
    adherence: 0.62,
    problems: [],
    medications: ["Sertraline 50 mg daily"],
    allergies: "No known drug allergies",
    note: "Partial improvement. Sleep is still poor. Misses doses about two days a week.",
    readings: readings([18, 18, 17, 18, 17, 16, 17, 16, 16, 15, 16, 15, 15, 14, 15, 14]),
    source: "mock",
  },
  {
    id: "robert-chen",
    mrn: "MC-10119",
    given: "Robert",
    family: "Chen",
    age: 62,
    sex: "Male",
    conditionId: "arthritis",
    adherence: 0.85,
    problems: ["Hypertension"],
    medications: ["Methotrexate 15 mg weekly", "Folic acid 1 mg daily", "Amlodipine 5 mg daily"],
    allergies: "Penicillin — hives",
    note: "Morning stiffness lasts about 40 minutes. Methotrexate is tolerated.",
    readings: readings([6.8, 6.6, 6.7, 6.5, 6.4, 6.5, 6.2, 6.3, 6.1, 6.0, 6.1, 5.8, 5.9, 5.6, 5.7, 5.5]),
    source: "mock",
  },
  {
    id: "amira-hassan",
    mrn: "MC-11230",
    given: "Amira",
    family: "Hassan",
    age: 66,
    sex: "Female",
    conditionId: "copd",
    adherence: 0.78,
    problems: ["Osteoporosis"],
    medications: ["Tiotropium 2 inhalations daily", "Albuterol as needed"],
    allergies: "No known drug allergies",
    note: "Two short courses of prednisone this year. Inhaler technique was uneven at the last visit.",
    readings: readings([5.2, 5.0, 5.4, 5.3, 5.6, 5.5, 5.8, 5.4, 5.7, 5.9, 5.6, 6.0, 5.8, 6.2, 5.9, 6.1]),
    source: "mock",
  },
  {
    id: "thomas-adler",
    mrn: "MC-10755",
    given: "Thomas",
    family: "Adler",
    age: 54,
    sex: "Male",
    conditionId: "hypertension",
    adherence: 0.75,
    problems: ["Hyperlipidemia"],
    medications: ["Amlodipine 5 mg daily", "Atorvastatin 20 mg nightly"],
    allergies: "No known drug allergies",
    note: "Office readings remain above the home average. Cuff size was checked.",
    readings: readings([152, 150, 154, 148, 149, 147, 150, 146, 148, 145, 146, 144, 145, 143, 144, 142]),
    source: "mock",
  },
  {
    id: "helen-brooks",
    mrn: "MC-10064",
    given: "Helen",
    family: "Brooks",
    age: 79,
    sex: "Female",
    conditionId: "heart-failure",
    adherence: 0.58,
    problems: ["Atrial fibrillation", "Chronic kidney disease", "Osteoporosis"],
    medications: ["Metoprolol succinate 50 mg daily", "Apixaban 5 mg twice daily", "Furosemide 80 mg daily"],
    allergies: "ACE inhibitors — cough",
    note: "Worsening effort tolerance. Needs a same-week review of volume status before any new medicine.",
    readings: readings([5.5, 5.6, 5.8, 5.7, 6.0, 6.2, 6.1, 6.4, 6.6, 6.5, 6.9, 7.0, 7.2, 7.4, 7.6, 7.8]),
    source: "mock",
  },
  {
    id: "luis-ortega",
    mrn: "MC-11502",
    given: "Luis",
    family: "Ortega",
    age: 41,
    sex: "Male",
    conditionId: "diabetes",
    adherence: 0.95,
    problems: [],
    medications: ["Metformin 500 mg twice daily"],
    allergies: "No known drug allergies",
    note: "At goal on diet, walking, and metformin. Here to confirm that adding a medicine is unnecessary.",
    readings: readings([7.1, 7.0, 7.2, 6.9, 7.0, 6.8, 6.9, 6.8, 6.7, 6.8, 6.6, 6.7, 6.6, 6.5, 6.6, 6.6]),
    source: "mock",
  },
]

export function loadAdded(): Patient[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isPatient)
  } catch {
    return []
  }
}

export function saveAdded(patients: Patient[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(patients))
}

export function makePatient(input: NewPatientInput, existing: Patient[]): Patient {
  return {
    ...input,
    id: `local-${crypto.randomUUID()}`,
    mrn: nextMrn(existing.map((patient) => patient.mrn)),
    source: "added",
  }
}

function nextMrn(existing: string[]): string {
  let n = 12000 + existing.length
  let mrn = `MC-${n}`
  const taken = new Set(existing)
  while (taken.has(mrn)) {
    n += 1
    mrn = `MC-${n}`
  }
  return mrn
}

function isPatient(value: unknown): value is Patient {
  if (!value || typeof value !== "object") return false
  const patient = value as Partial<Patient>
  const sex: Sex[] = ["Female", "Male", "Another sex or gender", "Not recorded"]
  return Boolean(
    typeof patient.id === "string" &&
      typeof patient.mrn === "string" &&
      typeof patient.given === "string" &&
      typeof patient.family === "string" &&
      typeof patient.age === "number" &&
      typeof patient.sex === "string" &&
      sex.includes(patient.sex) &&
      typeof patient.conditionId === "string" &&
      CONDITIONS.some((condition) => condition.id === patient.conditionId) &&
      typeof patient.adherence === "number" &&
      Array.isArray(patient.problems) &&
      Array.isArray(patient.medications) &&
      typeof patient.allergies === "string" &&
      typeof patient.note === "string" &&
      Array.isArray(patient.readings) &&
      patient.readings.length > 1 &&
      patient.readings.every(
        (reading) =>
          Boolean(reading) &&
          typeof reading.date === "string" &&
          typeof reading.value === "number",
      ) &&
      (patient.source === "mock" || patient.source === "added"),
  )
}
