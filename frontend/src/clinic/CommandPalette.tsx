import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from "react"
import { ArrowRight, MagnifyingGlass } from "@phosphor-icons/react"
import { conditionById } from "./data"
import { formatMeasure, fullName } from "./format"
import { courseOf, recentDelta } from "./model"
import type { Patient } from "./types"

export type PaletteAction = {
  id: string
  label: string
  hint: string
  run: () => void
}

type Row =
  | { kind: "patient"; id: string; patient: Patient; text: string }
  | { kind: "action"; id: string; action: PaletteAction; text: string }

export function CommandPalette({
  patients,
  actions,
  onPick,
  onClose,
}: {
  patients: Patient[]
  actions: PaletteAction[]
  onPick: (id: string) => void
  onClose: () => void
}) {
  const listId = useId()
  const input = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState("")
  const [cursor, setCursor] = useState(0)

  const rows = useMemo<Row[]>(() => {
    const q = query.trim().toLowerCase()
    const all: Row[] = [
      ...patients.map((patient) => {
        const condition = conditionById(patient.conditionId)
        return {
          kind: "patient" as const,
          id: `patient-${patient.id}`,
          patient,
          text: `${patient.given} ${patient.family} ${patient.mrn} ${condition.name}`.toLowerCase(),
        }
      }),
      ...actions.map((action) => ({
        kind: "action" as const,
        id: `action-${action.id}`,
        action,
        text: `${action.label} ${action.hint}`.toLowerCase(),
      })),
    ]
    if (!q) return all
    const words = q.split(/\s+/)
    return all.filter((row) => words.every((word) => row.text.includes(word)))
  }, [patients, actions, query])

  const safeCursor = Math.min(cursor, Math.max(0, rows.length - 1))

  useEffect(() => {
    input.current?.focus()
  }, [])

  useEffect(() => {
    document.getElementById(`${listId}-${safeCursor}`)?.scrollIntoView({ block: "nearest" })
  }, [safeCursor, listId])

  function run(row: Row | undefined) {
    if (!row) return
    if (row.kind === "patient") onPick(row.patient.id)
    onClose()
    if (row.kind === "action") row.action.run()
  }

  function onKey(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") {
      event.preventDefault()
      setCursor((safeCursor + 1) % Math.max(1, rows.length))
    } else if (event.key === "ArrowUp") {
      event.preventDefault()
      setCursor((safeCursor - 1 + rows.length) % Math.max(1, rows.length))
    } else if (event.key === "Enter") {
      event.preventDefault()
      run(rows[safeCursor])
    } else if (event.key === "Escape") {
      event.preventDefault()
      onClose()
    } else if (event.key === "Tab") {
      event.preventDefault()
    }
  }

  const firstAction = rows.findIndex((row) => row.kind === "action")

  return (
    <div className="c-scrim c-palette-scrim" onPointerDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="c-palette" role="dialog" aria-modal="true" aria-label="Go to a patient or action">
        <div className="c-palette-field">
          <MagnifyingGlass size={20} aria-hidden="true" />
          <input
            ref={input}
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-activedescendant={rows.length ? `${listId}-${safeCursor}` : undefined}
            aria-autocomplete="list"
            placeholder="Find a patient or action"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              setCursor(0)
            }}
            onKeyDown={onKey}
          />
          <kbd>Esc</kbd>
        </div>
        {rows.length === 0 ? (
          <p className="c-palette-empty">Nothing matches “{query}”. Try a family name, an MRN or a condition.</p>
        ) : (
          <ul id={listId} role="listbox" className="c-palette-list" aria-label="Results">
            {rows.map((row, index) => {
              const on = index === safeCursor
              const heading =
                index === 0 && row.kind === "patient" ? "Patients" : index === firstAction ? "Actions" : null
              return (
                <li key={row.id} role="presentation">
                  {heading ? (
                    <p className="c-palette-heading" role="presentation">
                      {heading}
                    </p>
                  ) : null}
                  <div
                    id={`${listId}-${index}`}
                    role="option"
                    aria-selected={on}
                    className={on ? "c-palette-row is-on" : "c-palette-row"}
                    onPointerMove={() => setCursor(index)}
                    onClick={() => run(row)}
                  >
                    {row.kind === "patient" ? <PatientRow patient={row.patient} /> : <ActionRow action={row.action} />}
                    {on ? <ArrowRight size={16} aria-hidden="true" className="c-palette-go" /> : null}
                  </div>
                </li>
              )
            })}
          </ul>
        )}
        <p className="c-palette-foot">
          <span>
            <kbd>↑</kbd> <kbd>↓</kbd> to move
          </span>
          <span>
            <kbd>Enter</kbd> to open
          </span>
        </p>
      </div>
    </div>
  )
}

function PatientRow({ patient }: { patient: Patient }) {
  const condition = conditionById(patient.conditionId)
  const course = courseOf(recentDelta(patient.readings), condition.measure)
  const latest = patient.readings[patient.readings.length - 1]
  return (
    <span className="c-palette-copy">
      <span className="c-palette-name">{fullName(patient)}</span>
      <span className="c-palette-meta">
        {patient.mrn}, {condition.name}
      </span>
      <span className={`c-palette-value is-${course.toLowerCase()}`}>
        {formatMeasure(latest.value, condition.measure)} {course.toLowerCase()}
      </span>
    </span>
  )
}

function ActionRow({ action }: { action: PaletteAction }) {
  return (
    <span className="c-palette-copy">
      <span className="c-palette-name">{action.label}</span>
      <span className="c-palette-meta">{action.hint}</span>
    </span>
  )
}
