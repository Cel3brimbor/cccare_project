import { useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react"
import { motion, useReducedMotion } from "motion/react"
import { formatDate, formatDay, formatMeasure } from "./format"
import { layoutChart, pointsAttr, timeline } from "./figures"
import type { Measure, Reading } from "./types"

export function TrendChart({
  recorded,
  forecast,
  measure,
  treatmentName,
  alternatives = [],
  highlightId = null,
  onChoose,
}: {
  recorded: Reading[]
  forecast: { date: string; value: number }[]
  measure: Measure
  treatmentName: string
  alternatives?: { id: string; name: string; values: number[] }[]
  highlightId?: string | null
  onChoose?: (id: string) => void
}) {
  const reduced = useReducedMotion()
  const spring = reduced ? { duration: 0 } : { type: "spring" as const, bounce: 0, duration: 0.55 }
  const points = useMemo(() => timeline(recorded, forecast), [recorded, forecast])
  const [index, setIndex] = useState(() => Math.max(0, recorded.length - 1))
  const safeIndex = Math.min(points.length - 1, Math.max(0, index))
  const latest = recorded[recorded.length - 1]?.value ?? measure.low
  const goal = latest - measure.goal
  const frame = useRef<HTMLDivElement>(null)
  const [frameWidth, setFrameWidth] = useState<number | null>(null)
  const chart = layoutChart(
    points,
    goal,
    safeIndex,
    formatDay,
    (value) => value.toFixed(measure.decimals),
    frameWidth ?? 960,
    alternatives.map((item) => item.values),
  )
  const active = points[safeIndex]
  const plotBottom = chart.height - chart.plot.bottom
  const plotRight = chart.width - chart.plot.right
  const innerW = plotRight - chart.plot.left
  const today = chart.recorded[chart.recorded.length - 1]
  const end = chart.projected[chart.projected.length - 1]
  const endValue = forecast[forecast.length - 1]?.value ?? latest
  const [goalLabelY, endLabelY] = separate(chart.goal?.y ?? 0, end?.y ?? 0, chart.goal && end ? 20 : 0)
  const lit = alternatives.find((item) => item.id === highlightId)
  const litEnd = lit ? chart.place(lit.values).at(-1) : undefined
  const litLabelY = litEnd ? avoid(litEnd.y, [endLabelY, chart.goal ? goalLabelY : -999], 18) : 0

  useLayoutEffect(() => {
    const node = frame.current
    if (!node) return
    const fit = (width: number) => setFrameWidth(Math.round(Math.min(960, Math.max(320, width))))
    fit(node.getBoundingClientRect().width)
    const observer = new ResizeObserver(([entry]) => fit(entry.contentRect.width))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  const describe = useId()
  const inset = chart.plot.right < 60
  const labelX = inset ? plotRight - 6 : plotRight + 12
  const labelAnchor = inset ? "end" : "start"
  const labelLift = inset ? -8 : 4

  function key(event: KeyboardEvent<HTMLDivElement>) {
    const last = points.length - 1
    const moves: Record<string, number> = {
      ArrowLeft: safeIndex - 1,
      ArrowDown: safeIndex - 1,
      ArrowRight: safeIndex + 1,
      ArrowUp: safeIndex + 1,
      PageDown: safeIndex - 4,
      PageUp: safeIndex + 4,
      Home: 0,
      End: last,
    }
    if (!(event.key in moves)) return
    event.preventDefault()
    setIndex(Math.min(last, Math.max(0, moves[event.key])))
  }

  function track(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - box.left) / box.width) * chart.width
    if (points.length < 2) return
    const next = Math.round(((x - chart.plot.left) / innerW) * (points.length - 1))
    setIndex(Math.min(points.length - 1, Math.max(0, next)))
  }

  return (
    <div className="c-chart-block" ref={frame}>
      <p id={describe} className="sr">
        Recorded {measure.name} as a solid line, the projection for {treatmentName} as a dashed line, other options as
        dotted lines, and the goal as a dotted line. Higher is worse.
      </p>
      <div
        className="c-plot"
        role="slider"
        tabIndex={0}
        aria-label="Read a point on the chart"
        aria-describedby={describe}
        aria-valuemin={0}
        aria-valuemax={Math.max(0, points.length - 1)}
        aria-valuenow={safeIndex}
        aria-valuetext={active ? `${formatDate(active.date)}, ${active.kind} ${formatMeasure(active.value, measure)}` : undefined}
        onKeyDown={key}
      >
        {frameWidth === null ? null : (
          <svg
            className="c-chart-svg"
            viewBox={`0 0 ${chart.width} ${chart.height}`}
            aria-hidden="true"
            onPointerMove={track}
          >
            <defs>
              <pattern id="c-ecg" width={40} height={40} patternUnits="userSpaceOnUse" x={chart.plot.left} y={chart.plot.top}>
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
              <text
                key={tick.label}
                x={tick.x}
                y={chart.height - 14}
                textAnchor={inset && tick.x > plotRight - 30 ? "end" : "middle"}
                className="c-tick"
              >
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
                <text x={labelX} y={goalLabelY + labelLift} textAnchor={labelAnchor} className="c-direct is-goal">
                  Goal {formatMeasure(goal, measure)}
                </text>
              </g>
            ) : null}
            {alternatives.map((item) => {
              const line = pathOf(chart.place(item.values))
              const lit = item.id === highlightId
              return (
                <g key={item.id} className={lit ? "c-alt is-lit" : "c-alt"}>
                  <path d={line} className="c-alt-line" />
                  {onChoose ? (
                    <path d={line} className="c-alt-hit" onClick={() => onChoose(item.id)}>
                      <title>{item.name}</title>
                    </path>
                  ) : null}
                </g>
              )
            })}
            <polyline points={pointsAttr(chart.recorded)} className="c-series recorded" />
            <motion.path
              className="c-series projected"
              initial={false}
              animate={{ d: pathOf(chart.projected) }}
              transition={spring}
            />
            {end ? (
              <motion.text
                x={labelX}
                textAnchor={labelAnchor}
                className="c-direct is-projected"
                initial={false}
                animate={{ y: endLabelY + labelLift }}
                transition={spring}
              >
                Selected {formatMeasure(endValue, measure)}
              </motion.text>
            ) : null}
            {lit && litEnd ? (
              <text x={labelX} y={litLabelY + labelLift} textAnchor={labelAnchor} className="c-direct is-alt">
                Preview {formatMeasure(lit.values[lit.values.length - 1], measure)}
              </text>
            ) : null}
            {chart.recorded.map((point) => (
              <rect key={point.date} x={point.x - 2} y={point.y - 2} width={4} height={4} className="c-mark" />
            ))}
            {chart.active ? (
              <g className="c-active">
                <line x1={chart.active.x} x2={chart.active.x} y1={chart.plot.top} y2={plotBottom} className="c-cursor" />
                <motion.g
                  className={chart.active.kind === "projected" ? "c-reticle is-projected" : "c-reticle"}
                  initial={false}
                  animate={{ x: chart.active.x, y: chart.active.y }}
                  transition={{ x: { duration: 0 }, y: spring }}
                >
                  <circle r={7} className="c-active-point" />
                  <path d="M -15 0 H -9 M 9 0 H 15 M 0 -15 V -9 M 0 9 V 15" />
                  <text
                    x={chart.active.x > plotRight - 60 ? -14 : 14}
                    y={-12}
                    textAnchor={chart.active.x > plotRight - 60 ? "end" : "start"}
                    className="c-active-label"
                  >
                    {formatMeasure(chart.active.value, measure)}
                  </text>
                </motion.g>
              </g>
            ) : null}
          </svg>
        )}
      </div>
      <div className="c-reader">
        <p className="c-hint">Move across the chart, or focus it and use the arrow keys.</p>
        {active ? (
          <p className="c-point">
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

function pathOf(points: { x: number; y: number }[]): string {
  return points.map((point, index) => `${index ? "L" : "M"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ")
}

function avoid(y: number, taken: number[], gap: number): number {
  let next = y
  for (let pass = 0; pass < 3; pass += 1) {
    const clash = taken.find((other) => Math.abs(other - next) < gap)
    if (clash === undefined) break
    next = next >= clash ? clash + gap : clash - gap
  }
  return next
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
