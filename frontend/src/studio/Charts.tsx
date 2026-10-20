import { useEffect, useRef } from "react"
import { usePrefers } from "./media"
import { cssColor } from "./color"
import { useStudio } from "./store"

export function Charts() {
  const dark = usePrefers("(prefers-color-scheme: dark)")
  const metrics = useStudio().metrics
  const lossRef = useRef<HTMLCanvasElement>(null)
  const accRef = useRef<HTMLCanvasElement>(null)
  const history = metrics?.history

  useEffect(() => {
    const loss = lossRef.current
    const accuracy = accRef.current
    const points = history ?? []
    if (!loss || !accuracy) return
    drawLine(loss, points.map((point) => point.loss), cssColor("--accent", "#0a84ff"))
    drawLine(
      accuracy,
      points.map((point) => point.accuracy),
      cssColor("--text", "#f5f5f7"),
      0,
      1,
    )
    drawMarks(
      accuracy,
      points.map((point) => point.val),
      cssColor("--pos", "#e3924e"),
      0,
      1,
    )
  }, [history, dark])

  const points = history?.length ?? 0
  const lossLabel =
    metrics?.loss == null ? "Loss chart. The curve appears after two steps." : `Loss chart. Latest loss ${metrics.loss.toFixed(3)}.`
  const accuracyLabel =
    metrics?.accuracy == null
      ? "Accuracy chart. The line is this batch and the dots are held-out accuracy. Both appear after two steps."
      : `Accuracy chart. Batch ${Math.round(metrics.accuracy * 100)} percent. The line is this batch and the dots are held-out accuracy.`

  return (
    <footer className="charts">
      <div className="figure-block">
        <span>Loss</span>
        <strong>{metrics?.loss == null ? "—" : metrics.loss.toFixed(3)}</strong>
      </div>
      <div className="spark">
        <canvas ref={lossRef} role="img" aria-label={lossLabel} />
        {points < 2 && <span className="spark-empty">After two steps</span>}
      </div>
      <div className="figure-block">
        <span>Batch accuracy</span>
        <strong>{metrics?.accuracy == null ? "—" : `${Math.round(metrics.accuracy * 100)}%`}</strong>
        <small className="chart-key">
          <i className="key-line" aria-hidden="true" />
          line
          <i className="key-dot" aria-hidden="true" />
          held-out
        </small>
      </div>
      <div className="spark">
        <canvas ref={accRef} role="img" aria-label={accuracyLabel} />
        {points < 2 && <span className="spark-empty">After two steps</span>}
      </div>
      <div className="figure-block quiet">
        <span>Held-out</span>
        <strong>{metrics?.valAccuracy == null ? "—" : `${Math.round(metrics.valAccuracy * 100)}%`}</strong>
        <small>{metrics?.testAccuracy == null ? "2,000 each epoch" : `Test ${Math.round(metrics.testAccuracy * 100)}%`}</small>
      </div>
    </footer>
  )
}

function drawLine(canvas: HTMLCanvasElement, values: number[], color: string, min?: number, max?: number) {
  const context = prepare(canvas)
  if (!context || values.length < 2) return
  const { width, height } = context
  const lo = min ?? Math.min(...values)
  const hi = max ?? Math.max(...values)
  const span = hi - lo || 1
  context.beginPath()
  context.strokeStyle = color
  context.lineWidth = 1.7
  context.lineJoin = "round"
  context.lineCap = "round"
  values.forEach((value, index) => {
    const x = 6 + (index / (values.length - 1)) * (width - 12)
    const y = height - 10 - ((value - lo) / span) * (height - 20)
    if (index === 0) context.moveTo(x, y)
    else context.lineTo(x, y)
  })
  context.stroke()
}

function drawMarks(canvas: HTMLCanvasElement, values: (number | null)[], color: string, min: number, max: number) {
  const context = canvas.getContext("2d")
  if (!context) return
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  const span = max - min || 1
  context.fillStyle = color
  values.forEach((value, index) => {
    if (value === null) return
    const x = 6 + (index / Math.max(1, values.length - 1)) * (width - 12)
    const y = height - 10 - ((value - min) / span) * (height - 20)
    context.beginPath()
    context.arc(x, y, 2.4, 0, Math.PI * 2)
    context.fill()
  })
}

function prepare(canvas: HTMLCanvasElement) {
  const dpr = Math.min(2, window.devicePixelRatio || 1)
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  if (width < 2 || height < 2) return null
  canvas.width = Math.floor(width * dpr)
  canvas.height = Math.floor(height * dpr)
  const context = canvas.getContext("2d")
  if (!context) return null
  context.setTransform(dpr, 0, 0, dpr, 0, 0)
  context.clearRect(0, 0, width, height)
  return Object.assign(context, { width, height })
}
