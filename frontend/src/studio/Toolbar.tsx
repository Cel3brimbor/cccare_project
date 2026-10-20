import { useEffect, useRef, useState } from "react"
import { CaretLeft } from "@phosphor-icons/react"
import { formatBytes, formatPct } from "./format"
import { Segmented } from "./Segmented"
import { clearError, send, useStudio } from "./store"
import type { TaskName } from "./types"

export function Toolbar({
  modelOpen,
  probeOpen,
  onToggleModel,
  onOpenProbe,
  onReturn,
}: {
  modelOpen: boolean
  probeOpen: boolean
  onToggleModel: () => void
  onOpenProbe: () => void
  onReturn?: () => void
}) {
  const { device, system, status, metrics, connected, error, task } = useStudio()
  const [armed, setArmed] = useState(false)
  const [machine, setMachine] = useState(false)
  const confirmRef = useRef<HTMLButtonElement>(null)
  const running = status.phase === "running"
  const blocked = !connected || !status.metal || status.phase === "preparing" || status.phase === "evaluating"
  const reason = blockReason(connected, status.metal, status.phase)
  const busy = status.phase === "preparing" || status.phase === "evaluating"
  const showProgress = status.progress != null || busy

  useEffect(() => {
    if (!error) return
    const id = window.setTimeout(clearError, 6000)
    return () => window.clearTimeout(id)
  }, [error])

  useEffect(() => {
    if (!armed) return
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    let restore = true
    confirmRef.current?.focus({ preventScroll: true })
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setArmed(false)
    }
    const onPointer = (event: PointerEvent) => {
      const target = event.target as HTMLElement | null
      if (!target?.closest(".confirm-anchor")) {
        restore = false
        setArmed(false)
      }
    }
    window.addEventListener("keydown", onKey)
    window.addEventListener("pointerdown", onPointer)
    return () => {
      window.removeEventListener("keydown", onKey)
      window.removeEventListener("pointerdown", onPointer)
      if (restore && previous && document.contains(previous)) previous.focus({ preventScroll: true })
    }
  }, [armed])

  const memory = system ? `${formatBytes(system.memoryUsed)} / ${formatBytes(system.memoryTotal)}` : "—"
  const tooltip = [
    device?.name ?? "GPU",
    device?.architecture,
    device?.memory ? `${formatBytes(device.memory)} unified` : null,
    system ? `System CPU ${formatPct(system.cpu)}` : null,
    system ? `This process ${formatPct(system.processCpu)}` : null,
    system ? `Metal active ${formatBytes(system.metalActive)}, peak ${formatBytes(system.metalPeak)}` : null,
    system?.gpuUtil == null ? "GPU utilization unavailable" : `GPU ${system.gpuUtil}%`,
  ]
    .filter(Boolean)
    .join(" · ")

  const taskLocked = !connected || status.phase === "preparing" || status.phase === "evaluating"
  const resetLocked = !connected || status.phase === "preparing"
  const link = !connected ? "Reconnecting" : !device ? "Detecting" : !device.available ? "No GPU" : system ? "Live" : "Waiting for samples"

  return (
    <header className="toolbar" aria-busy={busy}>
      <div className="brand">
        {onReturn ? (
          <button type="button" className="back" onClick={onReturn}>
            <CaretLeft size={12} weight="bold" aria-hidden="true" />
            Meridian Clinical
          </button>
        ) : null}
        <h1>Training bench</h1>
        <span>{task === "digits" ? "Handwritten digits" : "IMDB reviews"}</span>
      </div>
      <Segmented
        className="segments task-switch"
        label="Dataset"
        value={task}
        disabled={taskLocked}
        hint={taskLocked ? reason : undefined}
        options={[
          { id: "reviews", label: "Reviews" },
          { id: "digits", label: "Digits" },
        ]}
        onChange={(id) => send({ type: "task", task: id as TaskName })}
      />
      <p className={status.phase === "error" ? "status-line is-error" : "status-line"} title={status.message}>
        {status.message}
      </p>
      <div className="actions">
        <button
          type="button"
          data-model-button
          className={modelOpen ? "btn on" : "btn"}
          aria-expanded={modelOpen}
          aria-controls="model-popover"
          aria-haspopup="dialog"
          onClick={onToggleModel}
        >
          Model
        </button>
        <span className="divider" aria-hidden="true" />
        {status.phase === "error" ? (
          <button type="button" className="btn primary" onClick={() => send({ type: "prepare" })}>
            Retry
          </button>
        ) : (
          <span className="tip" title={blocked && !running ? reason : undefined}>
            <button
              type="button"
              className="btn primary"
              disabled={blocked && !running}
              title={blocked && !running ? undefined : running ? "Pause. Space also pauses while the network has focus." : "Train. Space also trains while the network has focus."}
              onClick={() => send({ type: running ? "pause" : "start" })}
            >
              {running ? "Pause" : "Train"}
            </button>
          </span>
        )}
        <span className="tip" title={blocked || running ? (running ? "Pause before taking a step" : reason) : undefined}>
          <button
            type="button"
            className="btn"
            disabled={blocked || running}
            title={blocked || running ? undefined : "One step. Period also steps while the network has focus."}
            onClick={() => send({ type: "step" })}
          >
            Step
          </button>
        </span>
        <span className="tip confirm-anchor" title={resetLocked ? reason : undefined}>
          <button
            type="button"
            className="btn"
            disabled={resetLocked}
            aria-expanded={armed}
            aria-controls="reset-confirm"
            onClick={() => {
              if (status.trainedSteps > 0) {
                setArmed((value) => !value)
                return
              }
              setArmed(false)
              send({ type: "reset" })
            }}
          >
            Reset
          </button>
          {armed && (
            <div id="reset-confirm" className="confirm" role="group" aria-label="Confirm reset">
              <p>Clear the learned weights, the curves, and the checkpoint.</p>
              <div className="confirm-actions">
                <button type="button" className="btn" onClick={() => setArmed(false)}>
                  Cancel
                </button>
                <button
                  ref={confirmRef}
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    setArmed(false)
                    send({ type: "reset" })
                  }}
                >
                  Reset
                </button>
              </div>
            </div>
          )}
        </span>
        <span className="divider" aria-hidden="true" />
        <span className="tip" title={connected ? undefined : reason}>
          <button
            type="button"
            className="btn"
            onClick={onOpenProbe}
            disabled={!connected}
            aria-expanded={probeOpen}
            aria-controls="probe-sheet"
            aria-haspopup="dialog"
          >
            {task === "digits" ? "Draw a digit" : "Try a review"}
          </button>
        </span>
      </div>
      {error && (
        <div className="alert" role="alert">
          <p>{error}</p>
          <button type="button" className="text-button" onClick={clearError}>
            Dismiss
          </button>
        </div>
      )}
      <div className="meta">
        <div className="chip" title={tooltip}>
          <i className={connected && device?.available ? "dot on" : connected ? "dot" : "dot wait"} aria-hidden="true" />
          <span>
            <b>{device ? (device.available ? device.name ?? "Metal" : "Metal unavailable") : "Detecting GPU"}</b>
            <small>{link}</small>
          </span>
        </div>
        <button
          type="button"
          className="machine-toggle"
          aria-expanded={machine}
          aria-controls="machine-vitals"
          onClick={() => setMachine((open) => !open)}
        >
          {machine ? "Hide machine" : "Machine"}
        </button>
        <p id="machine-vitals" className={machine ? "vitals is-open" : "vitals"} title={tooltip}>
          <span>CPU {system ? formatPct(system.cpu) : "—"}</span>
          <span>Memory {memory}</span>
          {system?.gpuUtil != null && <span>GPU {system.gpuUtil}%</span>}
          <span>Metal {system ? formatBytes(system.metalActive) : "—"}</span>
        </p>
        <p className="epoch">
          {metrics ? `Epoch ${metrics.epoch} of ${metrics.epochs}` : "Epoch —"}
          {metrics?.examplesPerSec ? ` · ${Math.round(metrics.examplesPerSec).toLocaleString()} / s` : ""}
        </p>
      </div>
      {showProgress && (
        <div
          className={status.progress == null ? "progress busy" : "progress"}
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={status.progress == null ? undefined : Math.round(Math.max(0, Math.min(1, status.progress)) * 100)}
          aria-label={status.phase === "evaluating" ? "Evaluating" : status.phase === "preparing" ? "Preparing" : "Training progress"}
        >
          <span
            style={
              status.progress == null
                ? undefined
                : { transform: `scaleX(${Math.max(0, Math.min(1, status.progress))})` }
            }
          />
        </div>
      )}
    </header>
  )
}

function blockReason(connected: boolean, metal: boolean, phase: string) {
  if (!connected) return "Waiting for the training process"
  if (!metal) return "Metal is unavailable on this machine"
  if (phase === "preparing") return "The dataset is still being prepared"
  if (phase === "evaluating") return "Wait until evaluation finishes"
  return "Unavailable"
}
