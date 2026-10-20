import { useEffect, useId, useLayoutEffect, useRef, useState, type FormEvent, type ReactNode } from "react"
import { X } from "@phosphor-icons/react"
import { CONDITIONS, buildReadings, extrasFor } from "./data"
import type { Course, NewPatientInput, Sex } from "./types"

const SEXES: Sex[] = ["Female", "Male", "Another sex or gender", "Not recorded"]
const COURSES: Course[] = ["Improving", "Stable", "Worsening"]

type Errors = Partial<Record<"given" | "family" | "age" | "sex" | "latest" | "adherence", string>>

export function PatientForm({
  onClose,
  onCreate,
}: {
  onClose: () => void
  onCreate: (input: NewPatientInput) => void
}) {
  const titleId = useId()
  const dialogRef = useRef<HTMLDivElement>(null)
  const summaryRef = useRef<HTMLDivElement>(null)
  const givenRef = useRef<HTMLInputElement>(null)
  const [given, setGiven] = useState("")
  const [family, setFamily] = useState("")
  const [age, setAge] = useState("")
  const [sex, setSex] = useState("")
  const [conditionId, setConditionId] = useState(CONDITIONS[0].id)
  const [latest, setLatest] = useState("")
  const [course, setCourse] = useState<Course>("Stable")
  const [adherence, setAdherence] = useState("80")
  const [problems, setProblems] = useState<string[]>([])
  const [medications, setMedications] = useState("")
  const [allergies, setAllergies] = useState("")
  const [note, setNote] = useState("")
  const [touched, setTouched] = useState<Record<string, boolean>>({})
  const [submitted, setSubmitted] = useState(false)
  const [submitCount, setSubmitCount] = useState(0)
  const condition = CONDITIONS.find((item) => item.id === conditionId) ?? CONDITIONS[0]
  const errors = validate(given, family, age, sex, latest, adherence, condition.measure.low, condition.measure.high)
  const errorItems = (Object.keys(errors) as (keyof Errors)[]).filter((key) => errors[key])

  useEffect(() => {
    givenRef.current?.focus()
    const node = dialogRef.current
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        onClose()
        return
      }
      if (event.key !== "Tab" || !node) return
      const items = [...node.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href]")].filter(
        (item) => item.tabIndex !== -1,
      )
      const first = items[0]
      const last = items[items.length - 1]
      if (!first || !last) return
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener("keydown", onKey)
    return () => document.removeEventListener("keydown", onKey)
  }, [onClose])

  useLayoutEffect(() => {
    if (submitCount > 0 && errorItems.length > 0) summaryRef.current?.focus()
  }, [submitCount, errorItems.length])

  function show(field: keyof Errors): string | undefined {
    if (!submitted && !touched[field]) return undefined
    return errors[field]
  }

  function touch(field: string) {
    setTouched((current) => ({ ...current, [field]: true }))
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    setSubmitted(true)
    if (errorItems.length > 0) {
      setSubmitCount((count) => count + 1)
      return
    }
    const parsedAge = Number(age)
    const parsedLatest = Number(latest)
    const parsedAdherence = Number(adherence) / 100
    const meds = medications
      .split(/\n|,/)
      .map((item) => item.trim())
      .filter(Boolean)
    onCreate({
      given: given.trim(),
      family: family.trim(),
      age: parsedAge,
      sex: sex as Sex,
      conditionId,
      adherence: parsedAdherence,
      problems: problems.filter((problem) => extrasFor(conditionId).includes(problem)),
      medications: meds.length ? meds : ["None recorded"],
      allergies: allergies.trim() || "No known drug allergies",
      note: note.trim(),
      readings: buildReadings(parsedLatest, course, condition.measure.low, condition.measure.high, condition.measure.decimals),
    })
  }

  return (
    <div className="c-scrim">
      <div
        ref={dialogRef}
        className="c-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
      >
        <div className="c-dialog-head">
          <h2 id={titleId}>New patient</h2>
          <button type="button" className="c-icon-btn" aria-label="Close new patient form" onClick={onClose}>
            <X size={20} aria-hidden="true" />
          </button>
        </div>
        <p>The record is stored in this browser. A 16-week course is built from the latest result and the direction you choose.</p>
        <form onSubmit={submit} noValidate>
          {submitted && errorItems.length > 0 ? (
            <div ref={summaryRef} tabIndex={-1} role="alert" className="c-summary" aria-labelledby="patient-error-title">
              <h3 id="patient-error-title">There is a problem</h3>
              <ul>
                {errorItems.map((field) => (
                  <li key={field}>
                    <a href={`#patient-${field}`}>{errors[field]}</a>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <div className="c-form-grid">
            <Field id="patient-given" label="Given name" error={show("given")}>
              <input
                ref={givenRef}
                id="patient-given"
                autoComplete="given-name"
                value={given}
                aria-invalid={show("given") ? true : undefined}
                aria-describedby={show("given") ? "patient-given-error" : undefined}
                onBlur={() => touch("given")}
                onChange={(event) => setGiven(event.target.value)}
              />
            </Field>
            <Field id="patient-family" label="Family name" error={show("family")}>
              <input
                id="patient-family"
                autoComplete="family-name"
                value={family}
                aria-invalid={show("family") ? true : undefined}
                aria-describedby={show("family") ? "patient-family-error" : undefined}
                onBlur={() => touch("family")}
                onChange={(event) => setFamily(event.target.value)}
              />
            </Field>
            <Field id="patient-age" label="Age" error={show("age")} hint="Years, from 18 to 110.">
              <input
                id="patient-age"
                inputMode="numeric"
                autoComplete="off"
                value={age}
                aria-invalid={show("age") ? true : undefined}
                aria-describedby={describedBy("patient-age", Boolean(show("age")))}
                onBlur={() => touch("age")}
                onChange={(event) => setAge(event.target.value)}
              />
            </Field>
            <Field id="patient-sex" label="Recorded sex" error={show("sex")}>
              <select
                id="patient-sex"
                value={sex}
                aria-invalid={show("sex") ? true : undefined}
                aria-describedby={show("sex") ? "patient-sex-error" : undefined}
                onBlur={() => touch("sex")}
                onChange={(event) => setSex(event.target.value)}
              >
                <option value="">Choose</option>
                {SEXES.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </Field>
            <Field id="patient-condition" label="Primary condition">
              <select
                id="patient-condition"
                value={conditionId}
                onChange={(event) => {
                  const next = event.target.value
                  setConditionId(next)
                  setProblems((current) => current.filter((problem) => extrasFor(next).includes(problem)))
                }}
              >
                {CONDITIONS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </Field>
            <Field
              id="patient-latest"
              label={`Latest ${condition.measure.name}`}
              error={show("latest")}
              hint={`From ${condition.measure.low} to ${condition.measure.high} ${condition.measure.unit}. Higher is worse.`}
            >
              <input
                id="patient-latest"
                inputMode="decimal"
                value={latest}
                aria-invalid={show("latest") ? true : undefined}
                aria-describedby={describedBy("patient-latest", Boolean(show("latest")))}
                onBlur={() => touch("latest")}
                onChange={(event) => setLatest(event.target.value)}
              />
            </Field>
            <Field
              id="patient-adherence"
              label="Adherence"
              error={show("adherence")}
              hint="Percent of the current plan the patient is following."
            >
              <input
                id="patient-adherence"
                inputMode="decimal"
                value={adherence}
                aria-invalid={show("adherence") ? true : undefined}
                aria-describedby={describedBy("patient-adherence", Boolean(show("adherence")))}
                onBlur={() => touch("adherence")}
                onChange={(event) => setAdherence(event.target.value)}
              />
            </Field>
          </div>
          <fieldset className="c-set">
            <legend>Recent course</legend>
            <div className="c-courses">
              {COURSES.map((item) => (
                <label key={item} className="c-check">
                  <input type="radio" name="course" checked={course === item} onChange={() => setCourse(item)} />
                  {item}
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className="c-set">
            <legend>Other problems</legend>
            <div className="c-checks">
              {extrasFor(conditionId).map((problem) => (
                <label key={problem} className="c-check">
                  <input
                    type="checkbox"
                    checked={problems.includes(problem)}
                    onChange={(event) => {
                      setProblems((current) =>
                        event.target.checked ? [...current, problem] : current.filter((item) => item !== problem),
                      )
                    }}
                  />
                  {problem}
                </label>
              ))}
            </div>
          </fieldset>
          <Field id="patient-meds" label="Current medicines" hint="One medicine on each line.">
            <textarea id="patient-meds" rows={4} value={medications} onChange={(event) => setMedications(event.target.value)} />
          </Field>
          <Field id="patient-allergies" label="Allergies" hint="Leave blank only if you have asked and none are known.">
            <input id="patient-allergies" value={allergies} onChange={(event) => setAllergies(event.target.value)} />
          </Field>
          <Field id="patient-note" label="Note" hint="Optional context for the next person who opens the chart.">
            <textarea id="patient-note" rows={3} value={note} onChange={(event) => setNote(event.target.value)} />
          </Field>
          <div className="c-dialog-actions">
            <button type="button" className="c-btn" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="c-btn c-btn-primary">
              Add patient
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function Field({
  id,
  label,
  hint,
  error,
  children,
}: {
  id: string
  label: string
  hint?: string
  error?: string
  children: ReactNode
}) {
  return (
    <div className="c-field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint ? <span id={`${id}-hint`} className="c-hint">{hint}</span> : null}
      {error ? (
        <span id={`${id}-error`} className="c-error">
          {error}
        </span>
      ) : null}
    </div>
  )
}

function describedBy(id: string, invalid: boolean): string {
  return invalid ? `${id}-hint ${id}-error` : `${id}-hint`
}

function validate(
  given: string,
  family: string,
  age: string,
  sex: string,
  latest: string,
  adherence: string,
  low: number,
  high: number,
): Errors {
  const errors: Errors = {}
  if (!given.trim()) errors.given = "Enter a given name."
  if (!family.trim()) errors.family = "Enter a family name."
  const parsedAge = Number(age)
  if (!age.trim() || !Number.isInteger(parsedAge) || parsedAge < 18 || parsedAge > 110) {
    errors.age = "Enter an age from 18 to 110."
  }
  if (!sex) errors.sex = "Choose a recorded sex."
  const parsedLatest = Number(latest)
  if (!latest.trim() || Number.isNaN(parsedLatest) || parsedLatest < low || parsedLatest > high) {
    errors.latest = `Enter a value from ${low} to ${high}.`
  }
  const parsedAdherence = Number(adherence)
  if (!adherence.trim() || Number.isNaN(parsedAdherence) || parsedAdherence < 0 || parsedAdherence > 100) {
    errors.adherence = "Enter adherence from 0 to 100."
  }
  return errors
}
