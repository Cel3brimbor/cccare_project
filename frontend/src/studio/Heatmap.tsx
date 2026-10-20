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

function mix(from: RGB, to: RGB, amount: number): RGB {
  const t = Math.max(0, Math.min(1, amount))
  return [
    Math.round(from[0] + (to[0] - from[0]) * t),
    Math.round(from[1] + (to[1] - from[1]) * t),
    Math.round(from[2] + (to[2] - from[2]) * t),
  ]
}

function css(color: RGB): string {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`
}

export function Heatmap({
  matrix,
  selected,
  cursor,
  mode,
  onSelect,
}: {
  matrix: number[][]
  selected: number | null
  cursor: number | null
  mode: "diverging" | "magnitude"
  onSelect?: (column: number) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const layoutRef = useRef({ cell: 1, cols: 0 })
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rows = matrix.length
    const cols = matrix[0]?.length ?? 0
    layoutRef.current = { cell: 1, cols }
    if (!rows || !cols) {
      canvas.width = 0
      canvas.height = 0
      return
    }
    const maxWidth = canvas.parentElement?.clientWidth ?? 360
    const cell = Math.max(2, Math.min(13, Math.floor(maxWidth / cols), Math.floor(220 / rows)))
    layoutRef.current = { cell, cols }
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    canvas.width = Math.floor(cols * cell * dpr)
    canvas.height = Math.floor(rows * cell * dpr)
    canvas.style.width = `${cols * cell}px`
    canvas.style.height = `${rows * cell}px`
    const context = canvas.getContext("2d")
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    let peak = 1e-6
    for (const row of matrix) {
      for (const value of row) peak = Math.max(peak, Math.abs(value))
    }
    const positive = readColor("--pos", "#b86a2e")
    const negative = readColor("--neg", "#3d6fbf")
    const accent = readColor("--accent", "#0071e3")
    const paper = readColor("--panel", "#ffffff")
    const idle = readColor("--node", "#d2d2d7")
    const gap = cell >= 6 ? 1 : 0
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const value = matrix[row][col] ?? 0
        const color =
          mode === "diverging"
            ? mix(paper, value >= 0 ? positive : negative, Math.min(1, Math.abs(value) / peak))
            : mix(idle, accent, Math.min(1, Math.abs(value) / peak))
        context.fillStyle = css(color)
        context.fillRect(col * cell, row * cell, cell - gap, cell - gap)
      }
    }
    context.strokeStyle = css(accent)
    context.lineWidth = 1.5
    if (selected !== null && selected >= 0 && selected < cols) {
      context.strokeRect(selected * cell + 0.5, 0.5, cell - 1, rows * cell - 1)
    }
    if (cursor !== null && cursor >= 0 && cursor < rows) {
      context.globalAlpha = 0.9
      context.strokeRect(0.5, cursor * cell + 0.5, cols * cell - 1, cell - 1)
      context.globalAlpha = 1
    }
  }, [matrix, selected, cursor, mode])

  function pick(event: ReactPointerEvent<HTMLCanvasElement>) {
    const layout = layoutRef.current
    if (!layout.cols || !selectRef.current) return
    const rect = event.currentTarget.getBoundingClientRect()
    const x = event.clientX - rect.left
    const column = Math.min(layout.cols - 1, Math.max(0, Math.floor(x / layout.cell)))
    selectRef.current(column)
  }

  return (
    <canvas
      ref={canvasRef}
      className="heatmap"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        pick(event)
      }}
      onPointerMove={(event) => {
        if (event.buttons === 1) pick(event)
      }}
      role="img"
      aria-label="Matrix"
    />
  )
}
