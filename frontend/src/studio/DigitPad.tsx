import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react"
import { formatNum } from "./format"
import { send, useStudio } from "./store"

const RADIUS = 1.35

export function DigitPad({ ready }: { ready: boolean }) {
  const { messages, dataset } = useStudio()
  const side = dataset?.task === "digits" ? dataset.width : 28
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const ink = useRef(new Float32Array(side * side))
  const last = useRef<{ x: number; y: number } | null>(null)
  const timer = useRef<number | null>(null)
  const [sampleLabel, setSampleLabel] = useState<number | null>(null)
  const guess = messages.find((message) => message.task === "digits") ?? null

  function redraw() {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const css = canvas.clientWidth || 336
    canvas.width = Math.floor(css * dpr)
    canvas.height = Math.floor(css * dpr)
    const context = canvas.getContext("2d")
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    const cell = css / side
    const gap = cell >= 8 ? 1 : 0
    context.fillStyle = "#0c0c0e"
    context.fillRect(0, 0, css, css)
    const pixels = ink.current
    for (let index = 0; index < pixels.length; index += 1) {
      const value = Math.max(0, Math.min(1, pixels[index] ?? 0))
      const row = Math.floor(index / side)
      const col = index % side
      const shade = Math.round(34 + value * (245 - 34))
      context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`
      context.fillRect(col * cell, row * cell, Math.max(1, cell - gap), Math.max(1, cell - gap))
    }
  }

  useEffect(() => {
    if (ink.current.length !== side * side) ink.current = new Float32Array(side * side)
    redraw()
    const observer = new ResizeObserver(() => redraw())
    if (canvasRef.current) observer.observe(canvasRef.current)
    return () => observer.disconnect()
  }, [side])

  function stamp(x: number, y: number) {
    const pixels = ink.current
    const reach = Math.ceil(RADIUS)
    for (let row = Math.floor(y) - reach; row <= Math.ceil(y) + reach; row += 1) {
      for (let col = Math.floor(x) - reach; col <= Math.ceil(x) + reach; col += 1) {
        if (row < 0 || col < 0 || row >= side || col >= side) continue
        const distance = Math.hypot(col - x, row - y)
        if (distance > RADIUS) continue
        const index = row * side + col
        pixels[index] = Math.max(pixels[index] ?? 0, 1 - distance / RADIUS)
      }
    }
  }

  function stroke(from: { x: number; y: number }, to: { x: number; y: number }) {
    const distance = Math.hypot(to.x - from.x, to.y - from.y)
    const steps = Math.max(1, Math.ceil(distance / 0.35))
    for (let step = 1; step <= steps; step += 1) {
      const t = step / steps
      stamp(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t)
    }
  }

  function point(event: ReactPointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) / rect.width) * side - 0.5,
      y: ((event.clientY - rect.top) / rect.height) * side - 0.5,
    }
  }

  function schedule() {
    if (!ready) return
    if (timer.current !== null) return
    timer.current = window.setTimeout(() => {
      timer.current = null
      send({ type: "predict", id: `${Date.now()}`, pixels: Array.from(ink.current) })
    }, 90)
  }

  function flush() {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
    if (!ready) return
    send({ type: "predict", id: `${Date.now()}`, pixels: Array.from(ink.current) })
  }

  function clear() {
    ink.current.fill(0)
    setSampleLabel(null)
    redraw()
    flush()
  }

  async function loadSample() {
    if (!dataset || dataset.task !== "digits") return
    const offset = Math.floor(Math.random() * dataset.test)
    const response = await fetch(`/api/rows?split=test&offset=${offset}&limit=1`)
    if (!response.ok) return
    const page = (await response.json()) as { rows: { label: number; pixels: number[] }[] }
    const row = page.rows[0]
    if (!row || row.pixels.length !== ink.current.length) return
    row.pixels.forEach((value, index) => {
      ink.current[index] = value / 255
    })
    setSampleLabel(row.label)
    redraw()
    flush()
  }

  const probs = guess?.probs ?? []

  return (
    <div className="digit-body">
      <canvas
        ref={canvasRef}
        className="digit-pad"
        aria-label="28 by 28 pixel grid. Draw a digit."
        onPointerDown={(event) => {
          try {
            event.currentTarget.setPointerCapture(event.pointerId)
          } catch {}
          const next = point(event)
          last.current = next
          setSampleLabel(null)
          stamp(next.x, next.y)
          redraw()
          schedule()
        }}
        onPointerMove={(event) => {
          if (!last.current) return
          const next = point(event)
          stroke(last.current, next)
          last.current = next
          redraw()
          schedule()
        }}
        onPointerUp={() => {
          last.current = null
          flush()
        }}
        onPointerCancel={() => {
          last.current = null
          flush()
        }}
      />
      <div className="digit-side">
        <div className="digit-readout">
          <strong>{guess ? guess.pred : "—"}</strong>
          <span>
            {guess ? `${Math.round(guess.prob * 100)}%` : ready ? "Draw a number" : "Preparing the digits"}
            {sampleLabel !== null ? ` · test label ${sampleLabel}` : ""}
          </span>
        </div>
        <div className="prob-list" aria-label="Digit probabilities">
          {Array.from({ length: 10 }, (_, digit) => {
            const probability = probs[digit] ?? 0
            const on = guess?.pred === digit
            return (
              <div key={digit} className={on ? "prob on" : "prob"}>
                <span>{digit}</span>
                <b>
                  <i style={{ transform: `scaleX(${Math.max(0, Math.min(1, probability))})` }} />
                </b>
                <span>{guess ? formatNum(probability, 2) : ""}</span>
              </div>
            )
          })}
        </div>
        <div className="digit-actions">
          <button type="button" className="btn" onClick={clear}>
            Clear
          </button>
          <button type="button" className="btn" disabled={!ready || !dataset} onClick={() => void loadSample()}>
            Load a test digit
          </button>
        </div>
        {guess?.untrained && <p className="fine">Untrained weights. This is the random initialization.</p>}
      </div>
      <p className="fine digit-note">Bright ink on black, the same way these digits were written. Each cell is one input pixel.</p>
    </div>
  )
}
