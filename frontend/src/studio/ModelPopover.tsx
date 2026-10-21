import { useEffect, useId, useRef, useState } from "react"
import { useDialog } from "./dialog"
import { networkParameters, parameterCount } from "./format"
import { Segmented } from "./Segmented"
import { send, useStudio } from "./store"
import type { Config } from "./types"

const HIDDEN_WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

function widthChoices(current: number): number[] {
  if (HIDDEN_WIDTHS.includes(current)) return HIDDEN_WIDTHS
  return [...HIDDEN_WIDTHS, current].sort((left, right) => left - right)
}

export function ModelPopover({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { config, dataset, status, task } = useStudio()
  const titleId = useId()
  const panelRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef(onClose)
  closeRef.current = onClose
  useDialog(open, panelRef, onClose)

  useEffect(() => {
    if (!open) return
    const onPointer = (event: PointerEvent) => {
      const target = event.target as HTMLElement | null
      if (!target?.closest(".popover") && !target?.closest("[data-model-button]")) closeRef.current()
    }
    window.addEventListener("pointerdown", onPointer)
    return () => window.removeEventListener("pointerdown", onPointer)
  }, [open])

  if (!open) return null
  const vocab = dataset?.task === "reviews" ? dataset.vocab : 12000
  const side = dataset?.task === "digits" ? dataset.width : 28
  const gait = task === "gait"
  const commit = (patch: Partial<Config>) => send({ type: "configure", config: patch })
  const parameters =
    gait && config.parameters
      ? config.parameters
      : task === "digits"
        ? networkParameters(side * side, config.hidden, 10)
        : gait
          ? networkParameters(1280, config.hidden, 10)
          : parameterCount(config.embedDim, config.hidden, vocab)

  return (
    <div
      ref={panelRef}
      id="model-popover"
      className="popover"
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      tabIndex={-1}
    >
      <div className="popover-head">
        <div>
          <h2 id={titleId}>Model</h2>
          <p>{status.locked ? "Reset to change the shape. Training settings apply on the next step." : "Shape is editable until the first step."}</p>
        </div>
        <button type="button" className="btn" data-initial-focus onClick={onClose}>
          Close
        </button>
      </div>
      <fieldset disabled={status.locked}>
        <legend>Shape</legend>
        {task === "reviews" && (
          <NumberField
            label="Embedding dimensions"
            value={config.embedDim}
            min={8}
            max={96}
            step={1}
            live={false}
            onCommit={(embedDim) => commit({ embedDim })}
          />
        )}
        <div className="field">
          <span>{gait ? "Head widths" : "Hidden widths"}</span>
          <div className="inline-fields">
            {config.hidden.map((width, index) => (
              <span key={index} className="hidden-row">
                <label className="field">
                  <span>{gait ? (config.hidden.length === 1 ? "Head" : `Head ${index + 1}`) : `Hidden ${index + 1}`}</span>
                  <select
                    value={width}
                    onChange={(event) => {
                      const hidden = config.hidden.slice()
                      hidden[index] = Number(event.target.value)
                      commit({ hidden })
                    }}
                  >
                    {widthChoices(width).map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
                {config.hidden.length > 1 && (
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => commit({ hidden: config.hidden.filter((_, item) => item !== index) })}
                  >
                    Remove
                  </button>
                )}
              </span>
            ))}
            {config.hidden.length < (gait ? 2 : 3) && (
              <button type="button" className="text-button" onClick={() => commit({ hidden: [...config.hidden, 16] })}>
                Add layer
              </button>
            )}
          </div>
        </div>
        <NumberField
          label="Seed"
          value={config.seed}
          min={0}
          max={2147483647}
          step={1}
          live={false}
          onCommit={(seed) => commit({ seed })}
        />
      </fieldset>
      <fieldset>
        <legend>Training</legend>
        <NumberField
          label="Epochs"
          value={config.epochs}
          min={1}
          max={30}
          step={1}
          onCommit={(epochs) => commit({ epochs })}
        />
        <NumberField
          label="Learning rate"
          value={config.lr}
          min={0.00001}
          max={1}
          step={0.0001}
          onCommit={(lr) => commit({ lr })}
        />
        <NumberField
          label="Batch size"
          value={config.batchSize}
          min={8}
          max={gait ? 768 : 256}
          step={8}
          onCommit={(batchSize) => commit({ batchSize })}
        />
        <div className="field">
          <span>Optimizer</span>
          <Segmented
            label="Optimizer"
            value={config.optimizer}
            options={[
              { id: "adam", label: "Adam" },
              { id: "sgd", label: "SGD" },
            ]}
            onChange={(name) => commit({ optimizer: name as Config["optimizer"] })}
          />
        </div>
        {config.optimizer === "adam" && (
          <div className="inline-fields">
            <NumberField
              label="β1"
              value={config.beta1}
              min={0}
              max={0.9999}
              step={0.001}
              onCommit={(beta1) => commit({ beta1 })}
            />
            <NumberField
              label="β2"
              value={config.beta2}
              min={0}
              max={0.9999}
              step={0.0001}
              onCommit={(beta2) => commit({ beta2 })}
            />
            <NumberField
              label="ε"
              value={config.eps}
              min={0}
              max={0.01}
              step={0.00000001}
              onCommit={(eps) => commit({ eps })}
            />
          </div>
        )}
        <NumberField
          label="L2"
          value={config.weightDecay}
          min={0}
          max={0.1}
          step={0.0001}
          onCommit={(weightDecay) => commit({ weightDecay })}
        />
      </fieldset>
      <p className="fine">Training settings apply on the next step. Switching optimizer clears its moments.</p>
      <p className="fine">
        {parameters.toLocaleString()} parameters.{" "}
        {gait
          ? "Almost all of them sit in the convolutional trunk: stem 768, mid 1024, deep 1280, two residual blocks in each stage. That trunk is what fills the GPU. The head is the dense layer this panel can still multiply one column at a time."
          : task === "digits"
            ? `The input is a ${side}×${side} image, flattened. The last layer is 10 digits.`
            : `Most of them are the ${vocab.toLocaleString()} × ${config.embedDim} embedding table, a row lookup. The canvas draws the pool and the layers you can still read cell by cell.`}
      </p>
    </div>
  )
}

function formatField(value: number): string {
  if (!Number.isFinite(value)) return ""
  const text = value.toFixed(8).replace(/\.?0+$/, "")
  return text === "-0" ? "0" : text
}

function parseNumber(raw: string, strict: boolean): number | null {
  if (!raw || raw === "-" || raw === "." || raw === "-.") return null
  if (strict && /[.eE+-]$/.test(raw)) return null
  const next = Number(raw)
  return Number.isFinite(next) ? next : null
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  hideLabel,
  live = true,
  onCommit,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  hideLabel?: boolean
  live?: boolean
  onCommit: (value: number) => void
}) {
  const noteId = useId()
  const [text, setText] = useState(formatField(value))
  const focused = useRef(false)
  const parsed = parseNumber(text, true)
  const outOfRange = parsed !== null && (parsed < min || parsed > max)
  const pending = useRef<number | null>(null)
  useEffect(() => {
    if (!focused.current) setText(formatField(value))
  }, [value])
  useEffect(
    () => () => {
      if (pending.current !== null) window.clearTimeout(pending.current)
    },
    [],
  )

  const commitText = (raw: string, strict: boolean) => {
    const next = parseNumber(raw, strict)
    if (next === null) {
      if (!strict) setText(formatField(value))
      return
    }
    const clamped = Math.min(max, Math.max(min, next))
    if (!strict) setText(formatField(clamped))
    if (clamped !== value) onCommit(clamped)
  }

  return (
    <label className="field">
      {hideLabel ? <span className="sr">{label}</span> : <span>{label}</span>}
      <input
        type="number"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        value={text}
        aria-invalid={outOfRange || undefined}
        aria-describedby={outOfRange ? noteId : undefined}
        onFocus={() => {
          focused.current = true
        }}
        onBlur={(event) => {
          if (pending.current !== null) window.clearTimeout(pending.current)
          pending.current = null
          focused.current = false
          commitText(event.currentTarget.value, false)
        }}
        onChange={(event) => {
          const raw = event.target.value
          setText(raw)
          if (!live) return
          if (pending.current !== null) window.clearTimeout(pending.current)
          if (parseNumber(raw, true) === null) return
          pending.current = window.setTimeout(() => {
            pending.current = null
            commitText(raw, true)
          }, 280)
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") (event.target as HTMLInputElement).blur()
        }}
      />
      {outOfRange && (
        <span className="field-note" id={noteId}>
          Use {formatField(min)} to {formatField(max)}
        </span>
      )}
    </label>
  )
}
