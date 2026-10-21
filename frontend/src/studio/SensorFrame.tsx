import { useEffect, useRef, type PointerEvent as ReactPointerEvent } from "react"

type RGB = [number, number, number]

function readColor(name: string, fallback: string): RGB {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
  return parseColor(value)
}

function parseColor(value: string): RGB {
  const text = value.trim()
  if (text.startsWith("#")) {
    const hex = text.slice(1)
    const full = hex.length === 3 ? hex.split("").map((part) => part + part).join("") : hex
    const parsed = Number.parseInt(full, 16)
    return [(parsed >> 16) & 255, (parsed >> 8) & 255, parsed & 255]
  }
  const match = text.match(/rgba?\(([^)]+)\)/)
  if (!match) return [140, 140, 140]
  const parts = match[1].split(",").map((part) => Number.parseFloat(part))
  return [parts[0] || 0, parts[1] || 0, parts[2] || 0]
}

function mix(from: RGB, to: RGB, amount: number): string {
  const t = Math.max(0, Math.min(1, amount))
  const red = Math.round(from[0] + (to[0] - from[0]) * t)
  const green = Math.round(from[1] + (to[1] - from[1]) * t)
  const blue = Math.round(from[2] + (to[2] - from[2]) * t)
  return `rgb(${red}, ${green}, ${blue})`
}

export function SensorFrame({
  values,
  rows,
  cols,
  selected = null,
  onSelect,
  maxWidth = 420,
  maxHeight = 220,
  label = "Sensor window",
}: {
  values: number[]
  rows: number
  cols: number
  selected?: number | null
  onSelect?: (index: number) => void
  maxWidth?: number
  maxHeight?: number
  label?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const layoutRef = useRef({ cellW: 1, cellH: 1, cols, rows })
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || values.length !== rows * cols || rows < 1 || cols < 1) return
    const maxW = canvas.parentElement?.clientWidth ?? maxWidth
    const cellW = Math.max(1, Math.min(8, (Math.min(maxW, maxWidth) - 4) / cols))
    const cellH = Math.max(3, Math.min(14, maxHeight / rows))
    layoutRef.current = { cellW, cellH, cols, rows }
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const width = cols * cellW
    const height = rows * cellH
    canvas.width = Math.floor(width * dpr)
    canvas.height = Math.floor(height * dpr)
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    const context = canvas.getContext("2d")
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    let peak = 1e-6
    for (const value of values) peak = Math.max(peak, Math.abs(value))
    const positive = readColor("--pos", "#b86a2e")
    const negative = readColor("--neg", "#3d6fbf")
    const paper = readColor("--stage", "#10120c")
    context.fillStyle = mix(paper, paper, 1)
    context.fillRect(0, 0, width, height)
    for (let index = 0; index < values.length; index += 1) {
      const row = Math.floor(index / cols)
      const col = index % cols
      const value = values[index] ?? 0
      const amount = Math.min(1, Math.abs(value) / peak)
      context.fillStyle = mix(paper, value >= 0 ? positive : negative, amount)
      context.fillRect(col * cellW, row * cellH, Math.max(1, cellW), Math.max(1, cellH))
    }
    context.strokeStyle = "rgba(245, 245, 247, 0.28)"
    context.lineWidth = 1
    for (let row = 3; row < rows; row += 3) {
      context.beginPath()
      context.moveTo(0, row * cellH + 0.5)
      context.lineTo(width, row * cellH + 0.5)
      context.stroke()
    }
    if (selected !== null && selected >= 0 && selected < values.length) {
      const row = Math.floor(selected / cols)
      const col = selected % cols
      const ink = readColor("--text", "#f5f5f7")
      context.strokeStyle = `rgb(${ink[0]}, ${ink[1]}, ${ink[2]})`
      context.strokeRect(col * cellW + 0.5, row * cellH + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1))
    }
  }, [values, rows, cols, selected, maxWidth, maxHeight])

  function pick(event: ReactPointerEvent<HTMLCanvasElement>) {
    const layout = layoutRef.current
    if (!selectRef.current) return
    const rect = event.currentTarget.getBoundingClientRect()
    const col = Math.min(layout.cols - 1, Math.max(0, Math.floor((event.clientX - rect.left) / layout.cellW)))
    const row = Math.min(layout.rows - 1, Math.max(0, Math.floor((event.clientY - rect.top) / layout.cellH)))
    selectRef.current(row * layout.cols + col)
  }

  return (
    <canvas
      ref={canvasRef}
      className="sensor-frame"
      role="img"
      aria-label={label}
      onPointerDown={(event) => {
        if (!selectRef.current) return
        event.currentTarget.setPointerCapture(event.pointerId)
        pick(event)
      }}
    />
  )
}
