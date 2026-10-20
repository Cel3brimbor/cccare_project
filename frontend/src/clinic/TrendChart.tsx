import { useMemo, useState } from "react"
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
  const chart = layoutChart(points, goal, safeIndex, formatDay, (value) => value.toFixed(measure.decimals))
  const active = points[safeIndex]
  const plotBottom = chart.height - chart.plot.bottom
  const axis = `M ${chart.plot.left} ${chart.plot.top} V ${plotBottom} H ${chart.width - chart.plot.right}`

  return (
    <div className="c-chart-block">
      <ul className="c-legend">
        <li>
          <i className="c-swatch solid" aria-hidden="true" />
          Recorded, latest {formatMeasure(latest, measure)}
        </li>
        <li>
          <i className="c-swatch dashed" aria-hidden="true" />
          Projected for {treatmentName}, {formatMeasure(forecast[forecast.length - 1]?.value ?? latest, measure)}
        </li>
        <li>
          <i className="c-swatch dotted" aria-hidden="true" />
          Goal {formatMeasure(goal, measure)}
        </li>
      </ul>
      <svg
        className="c-chart-svg"
        viewBox={`0 0 ${chart.width} ${chart.height}`}
        role="img"
        aria-label={`Recorded ${measure.name} as a solid line, the projection as a dashed line, and the goal as a dotted line. Higher is worse.`}
      >
        <path d={axis} className="c-axis" />
        {chart.yTicks.map((tick) => (
          <text key={tick.label} x={4} y={tick.y + 4} className="c-tick">
            {tick.label}
          </text>
        ))}
        {chart.xTicks.map((tick) => (
          <text key={tick.label} x={tick.x} y={chart.height - 8} textAnchor="middle" className="c-tick">
            {tick.label}
          </text>
        ))}
        {chart.goal ? (
          <line
            x1={chart.plot.left}
            x2={chart.width - chart.plot.right}
            y1={chart.goal.y}
            y2={chart.goal.y}
            className="c-goal"
          />
        ) : null}
        <polyline points={pointsAttr(chart.recorded)} className="c-series recorded" />
        <polyline points={pointsAttr(chart.projected)} className="c-series projected" />
        {chart.active ? <circle cx={chart.active.x} cy={chart.active.y} r={5} className="c-active-point" /> : null}
      </svg>
      <label className="c-slider">
        <span>Read a point on the chart</span>
        <input
          type="range"
          min={0}
          max={Math.max(0, points.length - 1)}
          value={safeIndex}
          onChange={(event) => setIndex(Number(event.target.value))}
        />
      </label>
      {active ? (
        <p className="c-point" aria-live="polite">
          {formatDate(active.date)}: {active.kind === "recorded" ? "Recorded" : "Projected"} {formatMeasure(active.value, measure)}
        </p>
      ) : null}
      <div className="c-split-tables">
        <div>
          <h3>Recorded</h3>
          <div className="c-table-wrap">
            <table className="c-table">
              <caption className="sr">Recorded {measure.name}</caption>
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col" className="c-num">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {recorded.map((reading) => (
                  <tr key={reading.date}>
                    <th scope="row">{formatDate(reading.date)}</th>
                    <td className="c-num">{formatMeasure(reading.value, measure)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h3>Projected</h3>
          <div className="c-table-wrap">
            <table className="c-table">
              <caption className="sr">Projected {measure.name} for {treatmentName}</caption>
              <thead>
                <tr>
                  <th scope="col">Date</th>
                  <th scope="col" className="c-num">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {forecast.map((point) => (
                  <tr key={point.date}>
                    <th scope="row">{formatDate(point.date)}</th>
                    <td className="c-num">{formatMeasure(point.value, measure)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
