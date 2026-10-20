export type Sex = "Female" | "Male" | "Another sex or gender" | "Not recorded"

export type Course = "Improving" | "Stable" | "Worsening"

export type Source = "mock" | "added"

export type Measure = {
  id: string
  name: string
  unit: string
  low: number
  high: number
  goal: number
  decimals: number
}

export type Treatment = {
  id: string
  name: string
  summary: string
  intensity: number
  adherenceDelta: number
}

export type Condition = {
  id: string
  name: string
  measure: Measure
  treatments: Treatment[]
}

export type Reading = {
  date: string
  value: number
}

export type Patient = {
  id: string
  mrn: string
  given: string
  family: string
  age: number
  sex: Sex
  conditionId: string
  adherence: number
  problems: string[]
  medications: string[]
  allergies: string
  note: string
  readings: Reading[]
  source: Source
}

export type NewPatientInput = {
  given: string
  family: string
  age: number
  sex: Sex
  conditionId: string
  adherence: number
  problems: string[]
  medications: string[]
  allergies: string
  note: string
  readings: Reading[]
}
