import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { DownloadSimple, MagnifyingGlass, Pulse, UserPlus } from "@phosphor-icons/react"
import { NetworkPanel } from "./NetworkPanel"
import { PatientForm } from "./PatientForm"
import { TrendChart } from "./TrendChart"
import { MOCK_PATIENTS, conditionById, loadAdded, makePatient, saveAdded } from "./data"
import { formatDate, formatGoal, formatMeasure, fullName, scaleNote } from "./format"
import { courseOf, projectAll, recentDelta } from "./model"
import { buildReport, reportFilename } from "./report"
import type { Course, NewPatientInput, Patient } from "./types"
import "./clinic.css"

type Filter = "all" | "worsening"

function patientIdFromHash(): string | null {
  const match = window.location.hash.match(/^#\/patients\/([^/?#]+)/)
  return match ? decodeURIComponent(match[1]) : null
}

export function Clinic({ onOpenBench }: { onOpenBench: () => void }) {
  const [added, setAdded] = useState<Patient[]>(() => loadAdded())
  const [patientId, setPatientId] = useState<string | null>(() => patientIdFromHash())
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState<Filter>("all")
  const [formOpen, setFormOpen] = useState(false)
  const [treatmentId, setTreatmentId] = useState<string | null>(null)
  const [unitId, setUnitId] = useState("out-1")
  const [armed, setArmed] = useState(false)
  const [status, setStatus] = useState("")
  const newButton = useRef<HTMLButtonElement>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const patients = useMemo(() => [...MOCK_PATIENTS, ...added], [added])
  const visible = useMemo(() => orderPatients(patients, filter, query), [patients, filter, query])

  useEffect(() => {
    const sync = () => setPatientId(patientIdFromHash())
    window.addEventListener("hashchange", sync)
    return () => window.removeEventListener("hashchange", sync)
  }, [])

  useEffect(() => {
    if (patientIdFromHash()) return
    const first = orderPatients(patients, "all", "")[0]
    if (first) window.location.hash = `#/patients/${first.id}`
  }, [patients])

  const known = patientId ? patients.find((item) => item.id === patientId) : undefined
  const missing = Boolean(patientId && !known)
  const patient = known ?? (patientId ? undefined : orderPatients(patients, "all", "")[0])
  const condition = patient ? conditionById(patient.conditionId) : undefined
  const ranked = useMemo(
    () => (patient && condition ? projectAll(patient, condition) : []),
    [patient, condition],
  )
  const selected = ranked.find((item) => item.treatment.id === treatmentId) ?? ranked[0]
  const latest = patient?.readings[patient.readings.length - 1]
  const delta = patient ? recentDelta(patient.readings) : 0
  const course = patient && condition ? courseOf(delta, condition.measure) : "Stable"

  useEffect(() => {
    setArmed(false)
  }, [patient?.id])

  useEffect(() => {
    document.title = patient ? `${patient.family}, ${patient.given} · Meridian Clinical` : "Meridian Clinical"
  }, [patient])

  const closeForm = useCallback(() => {
    setFormOpen(false)
    requestAnimationFrame(() => newButton.current?.focus())
  }, [])

  function select(id: string) {
    setPatientId(id)
    const next = `#/patients/${encodeURIComponent(id)}`
    if (window.location.hash !== next) window.location.hash = next
    requestAnimationFrame(() => heading.current?.focus())
  }

  function create(input: NewPatientInput) {
    const created = makePatient(input, patients)
    const next = [...added, created]
    setAdded(next)
    saveAdded(next)
    setFormOpen(false)
    setStatus(`Added ${fullName(created)} to the panel.`)
    select(created.id)
  }

  function remove(target: Patient) {
    const next = added.filter((item) => item.id !== target.id)
    setAdded(next)
    saveAdded(next)
    setArmed(false)
    setStatus(`Removed ${fullName(target)} from this browser.`)
    const fallback = patients.find((item) => item.id !== target.id)
    if (fallback) select(fallback.id)
  }

  function download() {
    if (!patient || !selected) return
    const blob = new Blob([buildReport(patient, selected.treatment.id)], { type: "text/html;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = reportFilename(patient)
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    setStatus(`Downloaded a projection file for ${fullName(patient)}.`)
  }

  return (
    <div className="clinic">
      <a className="c-skip" href="#patient-chart">
        Skip to chart
      </a>
      <div className="sr" aria-live="polite">
        {status}
      </div>
      <div className="c-shell" inert={formOpen}>
        <header className="c-topbar">
          <div className="c-topbar-inner">
            <div className="c-brand">
              <Pulse size={20} aria-hidden="true" />
              <div>
                <p className="c-brand-name">Meridian Clinical</p>
                <p className="c-brand-sub">Outpatient panel · 20 Oct 2026</p>
              </div>
            </div>
            <div className="c-actions">
              <button ref={newButton} type="button" className="c-btn c-btn-primary" onClick={() => setFormOpen(true)}>
                <UserPlus size={20} aria-hidden="true" />
                New patient
              </button>
              <button type="button" className="c-btn" onClick={onOpenBench}>
                Training bench
              </button>
            </div>
          </div>
        </header>
        <div className="c-frame">
          <aside className="c-census" aria-label="Patients">
            <h2>Patients</h2>
            <label className="c-field" htmlFor="patient-search">
              <span>Search</span>
              <span className="c-search">
                <MagnifyingGlass size={20} aria-hidden="true" />
                <input
                  id="patient-search"
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </span>
            </label>
            <div className="c-filters" role="group" aria-label="Filter records">
              <FilterButton current={filter} id="all" onSelect={setFilter}>
                All
              </FilterButton>
              <FilterButton current={filter} id="worsening" onSelect={setFilter}>
                Worsening
              </FilterButton>
            </div>
            {visible.length === 0 ? (
              <p className="c-empty">No records match that search.</p>
            ) : (
              <ul className="c-patient-list">
                {visible.map((item) => {
                  const itemCondition = conditionById(item.conditionId)
                  const itemCourse = courseOf(recentDelta(item.readings), itemCondition.measure)
                  const itemLatest = item.readings[item.readings.length - 1]
                  return (
                    <li key={item.id}>
                      <button
                        type="button"
                        className="c-patient"
                        aria-current={item.id === patient?.id ? "true" : undefined}
                        onClick={() => select(item.id)}
                      >
                        <span className="c-patient-name">{fullName(item)}</span>
                        <span className="c-patient-meta">
                          {item.mrn} · {itemCondition.name}
                        </span>
                        <span className="c-patient-meta">
                          {formatMeasure(itemLatest.value, itemCondition.measure)} · <CourseText course={itemCourse} />
                        </span>
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
            {query || filter !== "all" ? (
              <button
                type="button"
                className="c-btn"
                onClick={() => {
                  setQuery("")
                  setFilter("all")
                }}
              >
                Clear search
              </button>
            ) : null}
          </aside>
          <main className="c-main">
            {missing ? (
              <h1 id="patient-chart" tabIndex={-1} ref={heading}>
                That record is not on this panel.
              </h1>
            ) : null}
            {patient && condition && latest && selected ? (
              <div className="c-grid">
                <div className="c-span-12 c-identity">
                  <div>
                    <p className="c-kicker">{course === "Worsening" ? "Needs review" : "On the panel"}</p>
                    <h1 id="patient-chart" tabIndex={-1} ref={heading}>
                      {fullName(patient)}
                    </h1>
                    <p className="c-meta">
                      {patient.mrn} · {patient.age} years · {patient.sex}
                    </p>
                    <p>{condition.name}</p>
                  </div>
                  <div className="c-actions">
                    <button type="button" className="c-btn" onClick={download}>
                      <DownloadSimple size={20} aria-hidden="true" />
                      Export for the patient
                    </button>
                    {patient.source === "added" ? (
                      armed ? (
                        <div className="c-confirm" role="group" aria-label="Confirm removal">
                          <p>Remove {fullName(patient)} from this browser?</p>
                          <button type="button" className="c-btn c-btn-danger" onClick={() => remove(patient)}>
                            Remove record
                          </button>
                          <button type="button" className="c-btn" onClick={() => setArmed(false)}>
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button type="button" className="c-btn" onClick={() => setArmed(true)}>
                          Remove record
                        </button>
                      )
                    ) : null}
                  </div>
                </div>
                <p className="c-banner c-span-12" role="note">
                  Synthetic record. The projection is a fixed six-input network for a conversation, not a device cleared
                  for diagnosis or treatment, and it was not trained on these charts.
                </p>
                <dl className="c-facts c-span-12">
                  <div>
                    <dt>Problems</dt>
                    <dd>{[condition.name, ...patient.problems].join(", ")}</dd>
                  </div>
                  <div>
                    <dt>Medicines</dt>
                    <dd>{patient.medications.join("; ")}</dd>
                  </div>
                  <div>
                    <dt>Allergies</dt>
                    <dd className={patient.allergies.toLowerCase().includes("no known") ? undefined : "c-bad"}>
                      {patient.allergies}
                    </dd>
                  </div>
                  <div>
                    <dt>Note</dt>
                    <dd>{patient.note || "No note on the record."}</dd>
                  </div>
                </dl>
                <article className="c-stat c-span-3">
                  <h2>Latest {condition.measure.name}</h2>
                  <p className="c-figure">{formatMeasure(latest.value, condition.measure)}</p>
                  <p className="c-meta">
                    {formatDate(latest.date)}. {scaleNote(condition.measure)}
                  </p>
                </article>
                <article className="c-stat c-span-3">
                  <h2>Recent course</h2>
                  <p className="c-figure">
                    <CourseText course={course} />
                  </p>
                  <p className="c-meta">{changeText(delta, condition.measure)} over the last 8 readings.</p>
                </article>
                <article className="c-stat c-span-3">
                  <h2>Adherence</h2>
                  <p className="c-figure">{Math.round(patient.adherence * 100)}%</p>
                  <p className="c-meta">Share of the current plan being followed.</p>
                </article>
                <article className="c-stat c-span-3">
                  <h2>12-week projection</h2>
                  <p className="c-figure">{formatMeasure(selected.explanation.futureNative, condition.measure)}</p>
                  <p className="c-meta">
                    {Math.round(selected.explanation.probability * 100)}% chance of a drop of at least{" "}
                    {formatGoal(condition.measure)}.
                  </p>
                </article>
                <section className="c-card c-span-7" aria-labelledby="course-heading">
                  <h2 id="course-heading">Course and projection</h2>
                  <p className="c-lead" aria-live="polite">
                    {lead(patient.given, condition.measure.name, selected, ranked, condition.measure)}
                  </p>
                  <TrendChart
                    key={`${patient.id}-${selected.treatment.id}`}
                    recorded={patient.readings}
                    forecast={selected.forecast}
                    measure={condition.measure}
                    treatmentName={selected.treatment.name}
                  />
                </section>
                <section className="c-card c-span-5" aria-labelledby="options-heading">
                  <h2 id="options-heading">Treatment options</h2>
                  <p>Sorted by the estimated chance of a meaningful improvement. The chart follows the selected option.</p>
                  <fieldset className="c-options">
                    <legend className="sr">Treatment options</legend>
                    {ranked.map((item) => {
                      const on = item.treatment.id === selected.treatment.id
                      return (
                        <label key={item.treatment.id} className={on ? "c-option is-on" : "c-option"}>
                          <input
                            type="radio"
                            name="treatment"
                            value={item.treatment.id}
                            checked={on}
                            onChange={() => setTreatmentId(item.treatment.id)}
                          />
                          <span className="c-option-copy">
                            <span className="c-option-name">{item.treatment.name}</span>
                            <span>{item.treatment.summary}</span>
                            <span className="c-option-stats">
                              <span>{formatMeasure(item.explanation.futureNative, condition.measure)} in 12 weeks</span>
                              <span>
                                {Math.round(item.explanation.probability * 100)}% chance of a drop of at least{" "}
                                {formatGoal(condition.measure)}
                              </span>
                              <span>
                                Adherence used {Math.round(item.features[1] * 100)}%. Intensity {item.features[5].toFixed(2)}.
                              </span>
                            </span>
                            {item.caution ? <span className="c-bad">{item.caution}</span> : null}
                          </span>
                          <span className="c-meter" aria-hidden="true">
                            <i style={{ width: `${Math.round(item.explanation.probability * 100)}%` }} />
                          </span>
                        </label>
                      )
                    })}
                  </fieldset>
                </section>
                <NetworkPanel
                  projection={selected}
                  measure={condition.measure}
                  unitId={unitId}
                  onUnit={setUnitId}
                />
              </div>
            ) : null}
          </main>
        </div>
      </div>
      {formOpen ? <PatientForm onClose={closeForm} onCreate={create} /> : null}
    </div>
  )
}

function FilterButton({
  current,
  id,
  onSelect,
  children,
}: {
  current: Filter
  id: Filter
  onSelect: (id: Filter) => void
  children: string
}) {
  return (
    <button type="button" className="c-btn" aria-pressed={current === id} onClick={() => onSelect(id)}>
      {children}
    </button>
  )
}

function CourseText({ course }: { course: Course }) {
  return <span className={course === "Worsening" ? "c-bad" : course === "Improving" ? "c-good" : undefined}>{course}</span>
}

function orderPatients(patients: Patient[], filter: Filter, query: string): Patient[] {
  const q = query.trim().toLowerCase()
  const rank: Record<Course, number> = { Worsening: 0, Stable: 1, Improving: 2 }
  return patients
    .filter((patient) => {
      const condition = conditionById(patient.conditionId)
      const course = courseOf(recentDelta(patient.readings), condition.measure)
      if (filter === "worsening" && course !== "Worsening") return false
      if (!q) return true
      return `${patient.given} ${patient.family} ${patient.mrn} ${condition.name}`.toLowerCase().includes(q)
    })
    .sort((a, b) => {
      const leftCourse = courseOf(recentDelta(a.readings), conditionById(a.conditionId).measure)
      const rightCourse = courseOf(recentDelta(b.readings), conditionById(b.conditionId).measure)
      if (rank[leftCourse] !== rank[rightCourse]) return rank[leftCourse] - rank[rightCourse]
      if (leftCourse === "Worsening" && rightCourse === "Worsening") {
        return recentDelta(b.readings) - recentDelta(a.readings)
      }
      return a.family.localeCompare(b.family) || a.given.localeCompare(b.given)
    })
}

function changeText(delta: number, measure: { decimals: number; unit: string; low: number; high: number; goal: number; id: string; name: string }): string {
  const band = (measure.high - measure.low) * 0.02
  if (Math.abs(delta) <= band) return "Little change"
  const word = delta < 0 ? "Down" : "Up"
  return `${word} ${formatMeasure(Math.abs(delta), measure)}`
}

function lead(
  given: string,
  measureName: string,
  selected: ReturnType<typeof projectAll>[number],
  ranked: ReturnType<typeof projectAll>,
  measure: Parameters<typeof formatMeasure>[1],
): string {
  const current = selected.explanation.currentNative
  const future = selected.explanation.futureNative
  const chance = Math.round(selected.explanation.probability * 100)
  const drop = current - future
  const band = (measure.high - measure.low) * 0.02
  const goalText = formatGoal(measure)
  const landing =
    drop > band
      ? `down to ${formatMeasure(future, measure)}`
      : drop < -band
        ? `up to ${formatMeasure(future, measure)}`
        : `near ${formatMeasure(future, measure)}, close to the latest result`
  const reaches = drop + 1e-6 >= measure.goal
  const outcome = reaches
    ? `${selected.treatment.name} projects ${landing} in 12 weeks, which meets a drop of ${goalText}. The chance of that drop is ${chance}%.`
    : `${selected.treatment.name} projects ${landing} in 12 weeks. That line does not reach a drop of ${goalText}. The chance of a drop that large is ${chance}%.`
  const best = ranked[0]
  const bestChance = best ? Math.round(best.explanation.probability * 100) : chance
  const tail =
    best && selected.treatment.id !== best.treatment.id
      ? `${best.treatment.name} has the highest chance, ${bestChance}%.`
      : `That is the highest chance of the ${ranked.length} options.`
  return `${given}'s recorded ${measureName} is ${formatMeasure(current, measure)}. ${outcome} ${tail}`
}
