import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react"
import { CheckCircle, DownloadSimple, MagnifyingGlass, Pulse, UserPlus, WarningOctagon } from "@phosphor-icons/react"
import { CommandPalette, type PaletteAction } from "./CommandPalette"
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

const SHORT_CONDITIONS: Record<string, string> = {
  "heart-failure": "HFrEF",
}

const MEASURE_TAGS: Record<string, string> = {
  hba1c: "HbA1c",
  symptom: "Symptoms",
  phq9: "PHQ-9",
  pain: "Pain",
  copd: "Symptoms",
  sbp: "SBP",
}

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
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [treatmentId, setTreatmentId] = useState<string | null>(null)
  const [hoverId, setHoverId] = useState<string | null>(null)
  const [unitId, setUnitId] = useState("out-1")
  const [armed, setArmed] = useState(false)
  const [status, setStatus] = useState("")
  const [toast, setToast] = useState(0)
  const toastTimer = useRef<number | undefined>(undefined)
  const newButton = useRef<HTMLButtonElement>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const main = useRef<HTMLElement>(null)
  const shell = useRef<HTMLDivElement>(null)
  const bar = useRef<HTMLElement>(null)
  const opener = useRef<HTMLElement | null>(null)
  const returnFocus = useRef<"opener" | "chart" | null>(null)
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
  const summary = selected && condition ? lead(selected, ranked, condition.measure) : { headline: "", detail: "" }

  useEffect(() => {
    setArmed(false)
    setTreatmentId(null)
    setHoverId(null)
    main.current?.scrollTo({ top: 0 })
  }, [patient?.id])

  useEffect(() => {
    document.title = patient ? `${patient.family}, ${patient.given} · Meridian Clinical` : "Meridian Clinical"
  }, [patient])

  const closeForm = useCallback(() => {
    setFormOpen(false)
    requestAnimationFrame(() => newButton.current?.focus())
  }, [])

  useEffect(() => {
    const onKey = (event: globalThis.KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const typing = target?.tagName === "INPUT" || target?.tagName === "TEXTAREA" || target?.tagName === "SELECT" || target?.isContentEditable
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault()
        if (!formOpen) {
          if (!paletteOpen) opener.current = document.activeElement as HTMLElement | null
          setPaletteOpen((open) => !open)
        }
      } else if (event.key === "/" && !typing && !formOpen) {
        event.preventDefault()
        opener.current = document.activeElement as HTMLElement | null
        setPaletteOpen(true)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [formOpen, paletteOpen])

  useEffect(() => {
    if (paletteOpen || !returnFocus.current) return
    const target = returnFocus.current === "chart" ? heading.current : opener.current
    returnFocus.current = null
    target?.focus()
  }, [paletteOpen, patient?.id])

  useLayoutEffect(() => {
    const node = bar.current
    const root = shell.current
    if (!node || !root) return
    const fit = () => root.style.setProperty("--bar", `${Math.ceil(node.offsetHeight + node.offsetTop)}px`)
    fit()
    const observer = new ResizeObserver(fit)
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  function select(id: string, focusChart = true) {
    setPatientId(id)
    const next = `#/patients/${encodeURIComponent(id)}`
    if (window.location.hash !== next) window.location.hash = next
    if (focusChart) requestAnimationFrame(() => heading.current?.focus())
  }

  function stepThrough(event: KeyboardEvent<HTMLElement>) {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return
    const target = event.target as HTMLElement
    if (!target.classList.contains("c-patient")) return
    const buttons = [...event.currentTarget.querySelectorAll<HTMLButtonElement>(".c-patient")]
    const index = buttons.indexOf(target as HTMLButtonElement)
    const next = buttons[index + (event.key === "ArrowDown" ? 1 : -1)]
    if (!next) return
    event.preventDefault()
    next.focus()
    if (next.dataset.id) select(next.dataset.id, false)
  }

  function announce(text: string) {
    setStatus(text)
    setToast((count) => count + 1)
    window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setStatus(""), 4500)
  }

  function create(input: NewPatientInput) {
    const created = makePatient(input, patients)
    const next = [...added, created]
    setAdded(next)
    const saved = saveAdded(next)
    setFormOpen(false)
    announce(
      saved
        ? `Added ${fullName(created)} to the panel.`
        : `Added ${fullName(created)} for this visit only. This browser is not allowing saved records.`,
    )
    select(created.id)
  }

  function remove(target: Patient) {
    const next = added.filter((item) => item.id !== target.id)
    setAdded(next)
    saveAdded(next)
    setArmed(false)
    announce(`Removed ${fullName(target)} from this browser.`)
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
    announce(`Downloaded a projection file for ${fullName(patient)}.`)
  }

  const groups = (["Worsening", "Stable", "Improving"] as Course[])
    .map((name) => ({
      name,
      items: visible.filter(
        (item) => courseOf(recentDelta(item.readings), conditionById(item.conditionId).measure) === name,
      ),
    }))
    .filter((group) => group.items.length > 0)

  const paletteActions: PaletteAction[] = [
    { id: "new", label: "New patient", hint: "Add a record to this browser", run: () => setFormOpen(true) },
    ...(patient
      ? [{ id: "export", label: "Export for the patient", hint: `Download ${fullName(patient)}'s projection`, run: download }]
      : []),
    {
      id: "worsening",
      label: "Show worsening patients",
      hint: "Filter the list",
      run: () => {
        setQuery("")
        setFilter("worsening")
      },
    },
    { id: "bench", label: "Open the training bench", hint: "Train and inspect the network", run: onOpenBench },
  ]

  return (
    <div className="clinic">
      <a className="c-skip" href="#patient-chart">
        Skip to chart
      </a>
      <div className="c-toast-region" role="status" aria-live="polite">
        {status ? (
          <p key={toast} className="c-toast">
            <CheckCircle size={18} weight="fill" aria-hidden="true" />
            {status}
          </p>
        ) : null}
      </div>
      <div className="c-shell" ref={shell} inert={formOpen || paletteOpen}>
        <header className="c-topbar" ref={bar}>
          <div className="c-brand">
            <Pulse size={22} weight="bold" aria-hidden="true" />
            <p className="c-brand-name">Meridian Clinical</p>
            <p className="c-brand-sub">Outpatient panel, 20 Oct 2026</p>
          </div>
          <button
            type="button"
            className="c-jump"
            onClick={(event) => {
              opener.current = event.currentTarget
              setPaletteOpen(true)
            }} aria-keyshortcuts="Meta+K Control+K /">
            <MagnifyingGlass size={16} aria-hidden="true" />
            <span>Go to a patient or action</span>
            <kbd>{/Mac|iPhone|iPad/.test(navigator.userAgent) ? "⌘K" : "Ctrl K"}</kbd>
          </button>
          <div className="c-actions">
            <button type="button" className="c-btn c-btn-quiet" onClick={onOpenBench}>
              Training bench
            </button>
            <button ref={newButton} type="button" className="c-btn c-btn-primary" onClick={() => setFormOpen(true)}>
              <UserPlus size={20} aria-hidden="true" />
              <span className="c-btn-label">New patient</span>
            </button>
          </div>
        </header>
        <div className="c-frame">
          <aside className="c-census" aria-label="Patients" onKeyDown={stepThrough}>
            <div className="c-census-tools">
              <label className="c-search" htmlFor="patient-search">
                <MagnifyingGlass size={18} aria-hidden="true" />
                <span className="sr">Search patients</span>
                <input
                  id="patient-search"
                  type="search"
                  placeholder="Name, MRN or condition"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                />
              </label>
              <div className="c-filters" role="group" aria-label="Filter records">
                <FilterButton current={filter} id="all" onSelect={setFilter}>
                  {`All ${patients.length}`}
                </FilterButton>
                <FilterButton current={filter} id="worsening" onSelect={setFilter}>
                  Worsening only
                </FilterButton>
              </div>
            </div>
            {visible.length === 0 ? (
              <div className="c-empty">
                <p>No patient matches that search.</p>
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
              </div>
            ) : (
              groups.map((group) => (
                <section key={group.name} className="c-group" aria-labelledby={`group-${group.name}`}>
                  <h2 id={`group-${group.name}`} className={`c-group-title is-${group.name.toLowerCase()}`}>
                    {group.name}
                    <span className="c-count">{group.items.length}</span>
                  </h2>
                  <ul className="c-patient-list">
                    {group.items.map((item) => {
                      const itemCondition = conditionById(item.conditionId)
                      const itemLatest = item.readings[item.readings.length - 1]
                      return (
                        <li key={item.id}>
                          <button
                            type="button"
                            className={`c-patient is-${group.name.toLowerCase()}`}
                            aria-current={item.id === patient?.id ? "true" : undefined}
                            data-id={item.id}
                            onClick={() => select(item.id)}
                          >
                            <span className="c-patient-name">{fullName(item)}</span>
                            <span className="c-patient-value">
                              <span className="c-patient-tag">{MEASURE_TAGS[itemCondition.measure.id] ?? itemCondition.measure.name}</span>
                              {itemCondition.measure.unit === "mmHg"
                                ? itemLatest.value.toFixed(itemCondition.measure.decimals)
                                : formatMeasure(itemLatest.value, itemCondition.measure)}
                            </span>
                            <span className="c-patient-meta" title={itemCondition.name}>
                              {SHORT_CONDITIONS[itemCondition.id] ?? itemCondition.name}
                            </span>
                            <Sparkline readings={item.readings} />
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </section>
              ))
            )}
            {visible.length > 0 && (query || filter !== "all") ? (
              <button
                type="button"
                className="c-btn c-btn-quiet"
                onClick={() => {
                  setQuery("")
                  setFilter("all")
                }}
              >
                Clear search
              </button>
            ) : null}
          </aside>
          <main className="c-main" ref={main}>
            {missing ? (
              <div className="c-missing">
                <h1 id="patient-chart" tabIndex={-1} ref={heading}>
                  That record is not on this panel.
                </h1>
                <p>It may have been removed from this browser. Choose a patient from the list.</p>
              </div>
            ) : null}
            {patient && condition && latest && selected ? (
              <div className="c-chart-page" key={patient.id}>
                <header className="c-identity">
                  <div className="c-identity-copy">
                    <h1 id="patient-chart" tabIndex={-1} ref={heading}>
                      <span className="c-family">{patient.family},</span> {patient.given}
                    </h1>
                    <p className="c-condition">{condition.name}</p>
                    <p className="c-ids">
                      <span>{patient.mrn}</span>
                      <span>{patient.age} years</span>
                      <span>{patient.sex}</span>
                    </p>
                    {patient.allergies.toLowerCase().includes("no known") ? (
                      <p className="c-allergy is-none">{patient.allergies}</p>
                    ) : (
                      <p className="c-allergy">
                        <WarningOctagon size={18} weight="fill" aria-hidden="true" />
                        Allergy: {patient.allergies}
                      </p>
                    )}
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
                        <button type="button" className="c-btn c-btn-quiet" onClick={() => setArmed(true)}>
                          Remove record
                        </button>
                      )
                    ) : null}
                  </div>
                </header>
                <dl className="c-readout">
                  <div>
                    <dt>Latest {condition.measure.name}</dt>
                    <dd className="c-figure">{formatMeasure(latest.value, condition.measure)}</dd>
                    <dd className="c-meta">
                      {formatDate(latest.date)}. {scaleNote(condition.measure)}
                    </dd>
                  </div>
                  <div>
                    <dt>Recent course</dt>
                    <dd className="c-figure">
                      <CourseText course={course} />
                    </dd>
                    <dd className="c-meta">{changeText(delta, condition.measure)} over the last 8 readings.</dd>
                  </div>
                  <div>
                    <dt>Adherence</dt>
                    <dd className="c-figure">{Math.round(patient.adherence * 100)}%</dd>
                    <dd className="c-meta">Of the current plan.</dd>
                  </div>
                  <div>
                    <dt>In 12 weeks</dt>
                    <dd className="c-figure is-projected">
                      {formatMeasure(selected.explanation.futureNative, condition.measure)}
                    </dd>
                    <dd className="c-meta">
                      On the selected option.
                    </dd>
                  </div>
                </dl>
                <div className="c-decide">
                  <section className="c-sheet c-course" aria-labelledby="course-heading">
                    <div className="c-sheet-head">
                      <h2 id="course-heading">Course and projection</h2>
                      <div className="c-lead" aria-live="polite">
                        <p className="c-verdict">{summary.headline}</p>
                        <p>{summary.detail}</p>
                      </div>
                    </div>
                    <TrendChart
                      key={patient.id}
                      recorded={patient.readings}
                      forecast={selected.forecast}
                      measure={condition.measure}
                      treatmentName={selected.treatment.name}
                      alternatives={ranked
                        .filter((item) => item.treatment.id !== selected.treatment.id)
                        .map((item) => ({
                          id: item.treatment.id,
                          name: item.treatment.name,
                          values: item.forecast.map((point) => point.value),
                        }))}
                      highlightId={hoverId}
                      onChoose={setTreatmentId}
                    />
                    <p className="c-banner" role="note">
                      Synthetic record. The projection is a fixed six-input network for a conversation, not a device
                      cleared for diagnosis or treatment, and it was not trained on these charts.
                    </p>
                  </section>
                  <section className="c-sheet c-options-sheet" aria-labelledby="options-heading">
                    <h2 id="options-heading">Treatment options</h2>
                    <p className="c-meta">
                      Sorted by the chance of a drop of at least {formatGoal(condition.measure)}. The chart follows the
                      option you choose.
                    </p>
                    <fieldset className="c-options">
                      <legend className="sr">Treatment options</legend>
                      {ranked.map((item) => {
                        const on = item.treatment.id === selected.treatment.id
                        const chance = Math.round(item.explanation.probability * 100)
                        return (
                          <label
                            key={item.treatment.id}
                            className={on ? "c-option is-on" : "c-option"}
                            onPointerEnter={() => setHoverId(item.treatment.id)}
                            onPointerLeave={() => setHoverId(null)}
                          >
                            <input
                              type="radio"
                              name="treatment"
                              value={item.treatment.id}
                              checked={on}
                              onChange={() => setTreatmentId(item.treatment.id)}
                            />
                            <span className="c-option-copy">
                              <span className="c-option-name">{item.treatment.name}</span>
                              <span className="c-option-summary">{item.treatment.summary}</span>
                              <span className="c-option-stats">
                                {formatMeasure(item.explanation.futureNative, condition.measure)} in 12 weeks. Adherence
                                used {Math.round(item.features[1] * 100)}%, intensity {item.features[5].toFixed(2)}.
                              </span>
                              {item.caution ? <span className="c-caution">{item.caution}</span> : null}
                            </span>
                            <span className="c-chance">
                              <span className="c-chance-figure">{chance}%</span>
                              <span className="c-chance-label">chance</span>
                              <span className="c-meter" aria-hidden="true">
                                <i style={{ width: `${chance}%` }} />
                              </span>
                            </span>
                          </label>
                        )
                      })}
                    </fieldset>
                  </section>
                </div>
                <section className="c-sheet c-record" aria-labelledby="record-heading">
                  <h2 id="record-heading">On the record</h2>
                  <dl className="c-facts">
                    <div>
                      <dt>Problems</dt>
                      <dd>
                        <ul>
                          {[condition.name, ...patient.problems].map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </dd>
                    </div>
                    <div>
                      <dt>Medicines</dt>
                      <dd>
                        <ul>
                          {patient.medications.map((item) => (
                            <li key={item}>{item}</li>
                          ))}
                        </ul>
                      </dd>
                    </div>
                    <div>
                      <dt>Note</dt>
                      <dd className="c-note">{patient.note || "No note on the record."}</dd>
                    </div>
                  </dl>
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
      {paletteOpen ? (
        <CommandPalette
          patients={orderPatients(patients, "all", "")}
          actions={paletteActions}
          onPick={(id) => {
            returnFocus.current = "chart"
            select(id, false)
          }}
          onClose={() => {
            returnFocus.current ??= "opener"
            setPaletteOpen(false)
          }}
        />
      ) : null}
    </div>
  )
}

function Sparkline({ readings }: { readings: { value: number }[] }) {
  const values = readings.map((reading) => reading.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const points = values
    .map((value, index) => {
      const x = values.length === 1 ? 36 : (index / (values.length - 1)) * 72
      const y = 2 + (1 - (value - min) / span) * 16
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(" ")
  const last = points.split(" ").pop()?.split(",") ?? ["72", "10"]
  return (
    <svg className="c-spark" viewBox="-2 0 78 20" aria-hidden="true">
      <polyline points={points} />
      <rect x={Number(last[0]) - 2} y={Number(last[1]) - 2} width={4} height={4} />
    </svg>
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
    <button type="button" className="c-chip" aria-pressed={current === id} onClick={() => onSelect(id)}>
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

function lowerFirst(text: string): string {
  return text.charAt(0).toLowerCase() + text.slice(1)
}

function lead(
  selected: ReturnType<typeof projectAll>[number],
  ranked: ReturnType<typeof projectAll>,
  measure: Parameters<typeof formatMeasure>[1],
): { headline: string; detail: string } {
  const current = selected.explanation.currentNative
  const future = selected.explanation.futureNative
  const chance = Math.round(selected.explanation.probability * 100)
  const drop = current - future
  const band = (measure.high - measure.low) * 0.02
  const goalText = formatGoal(measure)
  const landing =
    drop > band
      ? `down from ${formatMeasure(current, measure)} today`
      : drop < -band
        ? `up from ${formatMeasure(current, measure)} today`
        : `about the same as today's ${formatMeasure(current, measure)}`
  const best = ranked[0] ?? selected
  const bestChance = Math.round(best.explanation.probability * 100)
  const headline =
    bestChance >= 50
      ? `Best chance: ${lowerFirst(best.treatment.name)}, ${bestChance}%.`
      : `No option is likely to bring a drop of ${goalText}. Best chance: ${lowerFirst(best.treatment.name)}, ${bestChance}%.`
  const detail =
    selected.treatment.id === best.treatment.id
      ? `Projected ${formatMeasure(future, measure)} in 12 weeks, ${landing}.`
      : `Selected: ${lowerFirst(selected.treatment.name)}, ${chance}%. Projected ${formatMeasure(future, measure)} in 12 weeks, ${landing}.`
  return { headline, detail }
}
