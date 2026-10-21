import { useEffect, useRef, useState } from "react"
import { Formula } from "./Formula"
import { Heatmap } from "./Heatmap"
import { formatNum } from "./format"
import { SensorFrame } from "./SensorFrame"
import { Segmented } from "./Segmented"
import { useStudio } from "./store"
import type { AffineLayer, GaitTrace, Selection } from "./types"

export function sensorCell(trace: GaitTrace | null, unit: number) {
  const cols = trace?.width ?? 128
  const rows = trace?.height ?? 20
  const safe = Math.min(Math.max(0, unit), rows * cols - 1)
  const row = Math.floor(safe / cols)
  const col = safe % cols
  const name = trace?.channels[row] ?? `Sensor ${row + 1}`
  const value = trace?.frame[safe] ?? 0
  return { safe, row, col, name, value, cols, rows }
}

export function SensorsForward({
  trace,
  unit,
  onSelect,
}: {
  trace: GaitTrace | null
  unit: number
  onSelect: (selection: Selection) => void
}) {
  const cell = sensorCell(trace, unit)
  return (
    <section>
      <Formula tex={`x \\in \\mathbb{R}^{20 \\times 128}`} />
      <p className="lesson">
        Eighteen rows are accelerometers, three axes on each foot, shin, and thigh. The last two rows are the quadriceps
        muscle sensors. Values are in g or a centered muscle reading, then z-scored on the training people. {cell.name}, sample{" "}
        {cell.col + 1}, is {formatNum(cell.value, 3)}.
      </p>
      {trace && (
        <SensorFrame
          values={trace.frame}
          rows={trace.height}
          cols={trace.width}
          selected={cell.safe}
          onSelect={(index) => onSelect({ layer: 0, unit: index })}
          label={`${cell.name}, sample ${cell.col + 1}`}
        />
      )}
    </section>
  )
}

export function ConvForward({ affine, unit }: { affine: AffineLayer; unit: number }) {
  const local = affine.colIndex ? affine.colIndex.indexOf(unit) : unit
  const kernel = local >= 0 ? affine.kernel?.[local] : undefined
  const temporal = local >= 0 ? affine.temporal?.[local] : affine.temporal?.[0]
  return (
    <section>
      <Formula tex={`(y_t)_c = \\sum_{k,i} x_{t+k,i} W_{c,k,i} + b_c`} />
      <p className="lesson">{affine.detail}</p>
      {kernel ? (
        <figure className="matrix-block">
          <figcaption>Filter for channel {unit + 1}</figcaption>
          <Heatmap label={`Convolution filter, ${kernel.length} taps by ${kernel[0]?.length ?? 0} inputs`} matrix={kernel} selected={null} cursor={null} mode="diverging" />
        </figure>
      ) : temporal ? (
        <figure className="matrix-block">
          <figcaption>Time shape of channel {unit + 1}</figcaption>
          <Heatmap label="Filter magnitude across taps" matrix={[temporal]} selected={null} cursor={null} mode="magnitude" />
        </figure>
      ) : (
        <p className="fine">This channel is outside the drawn sample of the stage.</p>
      )}
      <p className="fine">
        Canvas value {formatNum(affine.output[unit], 3)}. Ochre and blue on the graph are how strongly each incoming channel
        drives this one, summed over the filter taps.
      </p>
    </section>
  )
}

export function ConvWeights({ affine, unit }: { affine: AffineLayer; unit: number }) {
  const localCol = affine.colIndex ? affine.colIndex.indexOf(unit) : unit
  return (
    <section>
      <h3>
        {affine.title} · kernel {affine.taps ?? "—"}
      </h3>
      <p className="lesson">
        A line on the canvas is the root-mean-square of one input channel through this filter, not a single multiply. Wider
        stages are drawn from an even sample of channels.
      </p>
      <Heatmap
        label={`${affine.title} channel coupling`}
        matrix={affine.weight}
        selected={localCol >= 0 ? localCol : null}
        cursor={null}
        mode="diverging"
      />
      <dl className="stats">
        {[
          ["Mean coupling", formatNum(affine.weightMean, 4)],
          ["Std", formatNum(affine.weightStd, 4)],
          ["Batch grad RMS", formatNum(affine.batchGradRms, 4)],
          ["Moment RMS", formatNum(affine.momentRms, 4)],
        ].map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

export function ConvBackward({ affine, trainedSteps }: { affine: AffineLayer; trainedSteps: number }) {
  return (
    <section>
      <Formula tex={`\\frac{\\partial L}{\\partial W} = \\sum_t \\frac{\\partial L}{\\partial y_t} \\, x_{t+k}`} />
      <p className="lesson">
        The loss reaches this filter through every time step it touched. That batch gradient stays on the GPU so the step
        can keep the machine busy. The head, to the right, is where one window’s derivative is written out term by term.
      </p>
      {affine.batchGradRms != null && (
        <dl className="stats">
          <div>
            <dt>Batch grad RMS</dt>
            <dd>{formatNum(affine.batchGradRms, 4)}</dd>
          </div>
        </dl>
      )}
      {trainedSteps === 0 && <p className="fine">Train or step once and this filter starts to move.</p>}
    </section>
  )
}

const PAGE = 8

export function GaitData({ onEval, evaluating }: { onEval: () => void; evaluating: boolean }) {
  const { dataset, trace, metrics } = useStudio()
  const gait = dataset?.task === "gait" ? dataset : null
  const view = trace?.task === "gait" ? trace : null
  const peak = Math.max(1, ...(gait?.activities.map((item) => item.count) ?? [1]))
  return (
    <section>
      <h3>HuGaDB</h3>
      {gait ? (
        <>
          <p className="lesson">
            {gait.train.toLocaleString()} training windows and {gait.test.toLocaleString()} test windows. Each window is{" "}
            {gait.seconds} seconds ({gait.window} samples at about {Math.round(gait.rate)} Hz) of leg acceleration and
            quadriceps activity. The label is the activity at the middle of the window.
          </p>
          <p className="lesson">
            Training people {gait.trainPeople.join(", ")}. Test people {gait.testPeople.join(", ")} are held out entirely, so
            the score is for someone the network has not seen.
          </p>
          <div className="class-bars gait" aria-hidden="true">
            {gait.activities.map((activity) => (
              <div key={activity.name}>
                <i style={{ height: `${Math.max(8, (activity.count / peak) * 100)}%` }} />
                <small>{activity.short}</small>
              </div>
            ))}
          </div>
          <p className="fine">Training windows. Bicycling and riding in a car are not in these continuous recordings.</p>
          <ul className="channel-key">
            {gait.channels.map((channel) => (
              <li key={channel.name}>
                <i className={channel.kind === "muscle" ? "is-muscle" : undefined} aria-hidden="true" />
                {channel.name}
              </li>
            ))}
          </ul>
          <WindowBrowser />
        </>
      ) : (
        <p className="lesson">The recordings are still being prepared.</p>
      )}
      {view && (
        <>
          <h3>{view.kind === "probe" ? "Held-out window" : "Window in view"}</h3>
          <SensorFrame values={view.frame} rows={view.height} cols={view.width} label="Sensor window in view" />
          <dl className="stats">
            {[
              ["Person", view.person == null ? "—" : String(view.person)],
              ["Recorded", view.label == null ? "—" : view.activityNames[view.label] ?? "—"],
              ["Predicted", view.activityNames[view.pred] ?? "—"],
              ["Probability", formatNum(view.prob, 4)],
              ["Example loss", formatNum(view.exampleLoss, 4)],
              ["Batch loss", formatNum(view.batchLoss, 4)],
            ].map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>{value}</dd>
              </div>
            ))}
          </dl>
        </>
      )}
      <div className="eval-row">
        <button type="button" className="btn" onClick={onEval} disabled={!gait || evaluating}>
          {evaluating ? "Scoring" : "Score the test set"}
        </button>
        <span>
          {metrics?.testAccuracy == null
            ? `Full ${gait ? gait.test.toLocaleString() : "test"} windows when you ask`
            : `Test accuracy ${Math.round(metrics.testAccuracy * 100)}%`}
        </span>
      </div>
    </section>
  )
}

function WindowBrowser() {
  const [split, setSplit] = useState<"train" | "test">("train")
  const [rows, setRows] = useState<{ index: number; label: number; activity: string; person: number; frame: number[]; width: number; height: number }[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const request = useRef(0)

  async function load(nextSplit: "train" | "test", offset: number, replace: boolean) {
    const id = request.current + 1
    request.current = id
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/rows?split=${nextSplit}&offset=${offset}&limit=${PAGE}`)
      if (!response.ok) throw new Error("Could not read that page of windows.")
      const page = (await response.json()) as {
        total: number
        rows: { index: number; label: number; activity: string; person: number; frame: number[]; width: number; height: number }[]
      }
      if (request.current !== id) return
      setTotal(page.total)
      setRows((current) => (replace ? page.rows : [...current, ...page.rows]))
    } catch (reason) {
      if (request.current !== id) return
      setError(reason instanceof Error ? reason.message : "Could not read that page of windows.")
    } finally {
      if (request.current === id) setLoading(false)
    }
  }

  useEffect(() => {
    setRows([])
    setTotal(0)
    void load(split, 0, true)
  }, [split])

  const remaining = Math.max(0, total - rows.length)
  const nextCount = Math.min(PAGE, remaining)
  return (
    <section aria-busy={loading}>
      <h3>Windows</h3>
      <Segmented
        label="Split"
        value={split}
        options={[
          { id: "train", label: "Training" },
          { id: "test", label: "Test" },
        ]}
        onChange={(id) => setSplit(id as "train" | "test")}
      />
      <p className="fine">
        {total === 0 && loading ? "Reading the first page." : `Showing ${rows.length.toLocaleString()} of ${total.toLocaleString()}.`}
      </p>
      <ol className="row-list">
        {rows.map((row) => (
          <li key={`${split}-${row.index}`} className="digit-row">
            <SensorFrame values={row.frame} rows={row.height} cols={row.width} maxWidth={220} maxHeight={96} label={`${row.activity}, person ${row.person}`} />
            <div>
              <div className="row-meta">
                <span>Person {row.person}</span>
                <span>{row.activity}</span>
              </div>
            </div>
          </li>
        ))}
      </ol>
      {error && (
        <p className="alert inline" role="alert">
          {error}
        </p>
      )}
      {remaining > 0 && (
        <div className="row-more">
          <button type="button" className="btn" disabled={loading} onClick={() => void load(split, rows.length, false)}>
            {loading ? "Loading" : `Load ${nextCount} more`}
          </button>
        </div>
      )}
    </section>
  )
}
