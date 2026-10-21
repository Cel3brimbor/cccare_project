import { useEffect, useState } from "react"
import { formatNum } from "./format"
import { SensorFrame } from "./SensorFrame"
import { send, useStudio } from "./store"

export function GaitProbe({ ready, open }: { ready: boolean; open: boolean }) {
  const { trace, messages, dataset } = useStudio()
  const view = trace?.task === "gait" && trace.kind === "probe" ? trace : null
  const reply = messages.find((message) => message.task === "gait") ?? null
  const total = reply?.total ?? (dataset?.task === "gait" ? dataset.test : 0)
  const [index, setIndex] = useState(0)

  useEffect(() => {
    if (!ready || !open) return
    send({ type: "predict", id: `gait-${index}`, index })
  }, [ready, open, index])

  const names = view?.activityNames ?? (dataset?.task === "gait" ? dataset.activities.map((item) => item.name) : [])
  const label = view?.label
  const pred = view?.pred ?? reply?.pred ?? null

  return (
    <div className="gait-probe">
      {view ? (
        <SensorFrame
          values={view.frame}
          rows={view.height}
          cols={view.width}
          maxWidth={560}
          maxHeight={240}
          label="Held-out sensor window. Rows are accelerometers and the two quadriceps sensors. Columns are time."
        />
      ) : (
        <p className="lesson">{ready ? "Loading a held-out window." : "Preparing the leg recordings."}</p>
      )}
      <p className="lesson">
        {view?.person != null ? `Person ${view.person}. ` : ""}
        {total ? `Test window ${((view?.index ?? index) % Math.max(total, 1)) + 1} of ${total.toLocaleString()}. ` : ""}
        {label != null && names[label] ? `Recorded ${names[label]}. ` : ""}
        {pred != null && names[pred] ? `Predicted ${names[pred]} at ${Math.round((view?.prob ?? reply?.prob ?? 0) * 100)}%.` : ""}
      </p>
      {view?.untrained ? <p className="fine">Untrained weights. This is the random initialization.</p> : null}
      <ol className="activity-list">
        {names.map((name, activity) => {
          const probability = view?.probs[activity] ?? 0
          const recorded = label === activity
          const chosen = pred === activity
          return (
            <li key={name} className={chosen ? "is-pred" : recorded ? "is-label" : undefined}>
              <span>
                {name}
                {recorded ? " · recorded" : ""}
                {chosen ? " · predicted" : ""}
              </span>
              <i aria-hidden="true">
                <b style={{ width: `${Math.round(probability * 100)}%` }} />
              </i>
              <span>{formatNum(probability, 2)}</span>
            </li>
          )
        })}
      </ol>
      <div className="digit-actions">
        <button
          type="button"
          className="btn"
          disabled={!ready || total < 2}
          onClick={() => setIndex((value) => (value - 1 + total) % total)}
        >
          Previous window
        </button>
        <button
          type="button"
          className="btn"
          disabled={!ready || total < 2}
          onClick={() => setIndex((value) => (value + 1) % total)}
        >
          Next window
        </button>
      </div>
    </div>
  )
}
