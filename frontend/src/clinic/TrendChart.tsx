import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react"
import { formatDate, formatDay, formatMeasure } from "./format"
import { layoutChart, pointsAttr, timeline } from "./figures"
import type { Measure, Reading } from "./types"

export function TrendChart({
  recorded,
  forecast,
  measure,
  treatmentName,
}: {
  recorded: Reading[]
  forecast: { date: string; value: number }[]
  measure: Measure
  treatmentName: string
}) {
  const points = useMemo(() => timeline(recorded, forecast), [recorded, forecast])
  const [index, setIndex] = useState(() => Math.max(0, recorded.length - 1))
  const safeIndex = Math.min(points.length - 1, Math.max(0, index))
  const latest = recorded[recorded.length - 1]?.value ?? measure.low
  const goal = latest - measure.goal
  const frame = useRef<HTMLDivElement>(null)
  const [frameWidth, setFrameWidth] = useState(960)
  const chart = layoutChart(points, goal, safeIndex, formatDay, (value) => value.toFixed(measure.decimals), frameWidth)
  const active = points[safeIndex]
  const plotBottom = chart.height - chart.plot.bottom
  const plotRight = chart.width - chart.plot.right
  const innerW = plotRight - chart.plot.left
  const today = chart.recorded[chart.recorded.length - 1]
  const end = chart.projected[chart.projected.length - 1]
  const endValue = forecast[forecast.length - 1]?.value ?? latest
  const [goalLabelY, endLabelY] = separate(chart.goal?.y ?? 0, end?.y ?? 0, chart.goal && end ? 20 : 0)

  useEffect(() => {
    const node = frame.current
    if (!node) return
    const observer = new ResizeObserver(([entry]) => {
      setFrameWidth(Math.round(Math.min(960, Math.max(320, entry.contentRect.width))))
    })
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  function track(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - box.left) / box.width) * chart.width
    if (points.length < 2) return
    const next = Math.round(((x - chart.plot.left) / innerW) * (points.length - 1))
    setIndex(Math.min(points.length - 1, Math.max(0, next)))
  }

  return (
    <div className="c-chart-block" ref={frame}>
      <svg
        className="c-chart-svg"
        viewBox={`0 0 ${chart.width} ${chart.height}`}
        role="img"
        aria-label={`Recorded ${measure.name} as a solid line, the projection for ${treatmentName} as a dashed line, and the goal as a dotted line. Higher is worse.`}
        onPointerMove={track}
      >
        <defs>
          <pattern id="c-ecg-fine" width={8} height={8} patternUnits="userSpaceOnUse" x={chart.plot.left} y={chart.plot.top}>
            <path d="M 8 0 V 8 M 0 8 H 8" className="c-grid-fine" />
          </pattern>
          <pattern id="c-ecg" width={40} height={40} patternUnits="userSpaceOnUse" x={chart.plot.left} y={chart.plot.top}>
            <rect width={40} height={40} fill="url(#c-ecg-fine)" />
            <path d="M 40 0 V 40 M 0 40 H 40" className="c-grid-major" />
          </pattern>
        </defs>
        <rect
          x={chart.plot.left}
          y={chart.plot.top}
          width={innerW}
          height={plotBottom - chart.plot.top}
          fill="url(#c-ecg)"
          className="c-paper"
        />
        {today ? (
          <rect
            x={today.x}
            y={chart.plot.top}
            width={plotRight - today.x}
            height={plotBottom - chart.plot.top}
            className="c-future"
          />
        ) : null}
        {chart.yTicks.map((tick) => (
          <text key={tick.label} x={chart.plot.left - 10} y={tick.y + 4} textAnchor="end" className="c-tick">
            {tick.label}
          </text>
        ))}
        {chart.xTicks.map((tick) => (
          <text key={tick.label} x={tick.x} y={chart.height - 14} textAnchor="middle" className="c-tick">
            {tick.label}
          </text>
        ))}
        {today ? (
          <g>
            <line x1={today.x} x2={today.x} y1={chart.plot.top - 12} y2={plotBottom} className="c-today" />
            <text x={today.x + 6} y={chart.plot.top - 14} className="c-direct">
              Today
            </text>
          </g>
        ) : null}
        {chart.goal ? (
          <g>
            <line x1={chart.plot.left} x2={plotRight} y1={chart.goal.y} y2={chart.goal.y} className="c-goal" />
            <text x={plotRight + 12} y={goalLabelY + 4} className="c-direct is-goal">
              Goal {formatMeasure(goal, measure)}
            </text>
          </g>
        ) : null}
        <polyline points={pointsAttr(chart.recorded)} className="c-series recorded" />
        <polyline points={pointsAttr(chart.projected)} className="c-series projected" />
        {end ? (
          <text x={plotRight + 12} y={endLabelY + 4} className="c-direct is-projected">
            Projected {formatMeasure(endValue, measure)}
          </text>
        ) : null}
        {chart.active ? (
          <g>
            <line x1={chart.active.x} x2={chart.active.x} y1={chart.plot.top} y2={plotBottom} className="c-cursor" />
            <circle
              cx={chart.active.x}
              cy={chart.active.y}
              r={6}
              className={chart.active.kind === "projected" ? "c-active-point is-projected" : "c-active-point"}
            />
          </g>
        ) : null}
      </svg>
      <div className="c-reader">
        <label className="c-slider">
          <span className="sr">Read a point on the chart</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, points.length - 1)}
            value={safeIndex}
            aria-valuetext={active ? `${formatDate(active.date)}, ${active.kind} ${formatMeasure(active.value, measure)}` : undefined}
            onChange={(event) => setIndex(Number(event.target.value))}
          />
        </label>
        {active ? (
          <p className="c-point" aria-live="polite">
            <span className="c-point-date">{formatDate(active.date)}</span>
            <span className={active.kind === "projected" ? "c-point-value is-projected" : "c-point-value"}>
              {formatMeasure(active.value, measure)}
            </span>
            <span className="c-point-kind">{active.kind === "recorded" ? "Recorded" : "Projected"}</span>
          </p>
        ) : null}
      </div>
      <details className="c-disclose">
        <summary>
          All readings: {recorded.length} recorded, {forecast.length} projected
        </summary>
        <div className="c-split-tables">
          <ReadingTable title="Recorded" caption={`Recorded ${measure.name}`} rows={recorded} measure={measure} />
          <ReadingTable
            title="Projected"
            caption={`Projected ${measure.name} for ${treatmentName}`}
            rows={forecast}
            measure={measure}
          />
        </div>
      </details>
    </div>
  )
}

function separate(a: number, b: number, gap: number): [number, number] {
  if (Math.abs(a - b) >= gap) return [a, b]
  const mid = (a + b) / 2
  return a <= b ? [mid - gap / 2, mid + gap / 2] : [mid + gap / 2, mid - gap / 2]
}

function ReadingTable({
  title,
  caption,
  rows,
  measure,
}: {
  title: string
  caption: string
  rows: { date: string; value: number }[]
  measure: Measure
}) {
  return (
    <div>
      <h3>{title}</h3>
      <div className="c-table-wrap">
        <table className="c-table">
          <caption className="sr">{caption}</caption>
          <thead>
            <tr>
              <th scope="col">Date</th>
              <th scope="col" className="c-num">
                Value
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((reading) => (
              <tr key={reading.date}>
                <th scope="row">{formatDate(reading.date)}</th>
                <td className="c-num">{formatMeasure(reading.value, measure)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
