import { useEffect, useRef, type PointerEvent as ReactPointerEvent } from "react"
import { usePrefers } from "./media"

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

export function PixelGrid({
  values,
  side,
  mode = "ink",
  selected = null,
  onSelect,
  maxSide = 220,
}: {
  values: number[]
  side: number
  mode?: "ink" | "diverging"
  selected?: number | null
  onSelect?: (index: number) => void
  maxSide?: number
}) {
  const dark = usePrefers("(prefers-color-scheme: dark)")
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const layoutRef = useRef({ cell: 1 })
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || values.length !== side * side) return
    const maxWidth = canvas.parentElement?.clientWidth ?? maxSide
    const cell = Math.max(2, Math.min(14, Math.floor(Math.min(maxWidth, maxSide) / side)))
    layoutRef.current = { cell }
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const gap = cell >= 6 ? 1 : 0
    canvas.width = Math.floor(side * cell * dpr)
    canvas.height = Math.floor(side * cell * dpr)
    canvas.style.width = `${side * cell}px`
    canvas.style.height = `${side * cell}px`
    const context = canvas.getContext("2d")
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    let peak = 1e-6
    if (mode === "diverging") {
      for (const value of values) peak = Math.max(peak, Math.abs(value))
    }
    const positive = readColor("--pos", "#b86a2e")
    const negative = readColor("--neg", "#3d6fbf")
    const paper = readColor("--panel", "#ffffff")
    const accent = readColor("--accent", "#0071e3")
    context.fillStyle = mode === "ink" ? "#101012" : mix(paper, paper, 1)
    context.fillRect(0, 0, side * cell, side * cell)
    for (let index = 0; index < values.length; index += 1) {
      const row = Math.floor(index / side)
      const col = index % side
      const value = values[index] ?? 0
      context.fillStyle =
        mode === "ink"
          ? mix([16, 16, 18], [245, 245, 247], value)
          : mix(paper, value >= 0 ? positive : negative, Math.min(1, Math.abs(value) / peak))
      context.fillRect(col * cell, row * cell, cell - gap, cell - gap)
    }
    if (selected !== null && selected >= 0 && selected < values.length) {
      const row = Math.floor(selected / side)
      const col = selected % side
      context.strokeStyle = mode === "ink" ? "#f5f5f7" : mix(accent, accent, 1)
      context.lineWidth = 1.5
      context.strokeRect(col * cell + 0.5, row * cell + 0.5, cell - 1, cell - 1)
    }
  }, [values, side, mode, selected, maxSide, dark])

  function pick(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!selectRef.current) return
    const cell = layoutRef.current.cell
    const rect = event.currentTarget.getBoundingClientRect()
    const col = Math.min(side - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * side)))
    const row = Math.min(side - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * side)))
    if (cell <= 0) return
    selectRef.current(row * side + col)
  }

  return (
    <canvas
      ref={canvasRef}
      className="pixel-grid"
      aria-label={`${side} by ${side} pixels`}
      onPointerDown={onSelect ? pick : undefined}
    />
  )
}

export function FilterBank({
  weight,
  side,
  selected,
  onSelect,
  units: unitIds,
}: {
  weight: number[][]
  side: number
  selected: number
  onSelect: (unit: number) => void
  units?: number[]
}) {
  const dark = usePrefers("(prefers-color-scheme: dark)")
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const layoutRef = useRef({ cols: 1, pitch: 1, units: 0 })
  const idsRef = useRef<number[]>([])
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect

  useEffect(() => {
    const ids = unitIds ?? Array.from({ length: weight[0]?.length ?? 0 }, (_, index) => index)
    idsRef.current = ids
    const units = ids.length
    const canvas = canvasRef.current
    if (!canvas || !units || weight.length !== side * side) return
    const maxWidth = canvas.parentElement?.clientWidth ?? 320
    const gap = 4
    const cell = Math.max(1, Math.min(3, Math.floor(maxWidth / (8 * (side + 1)))))
    const unitSize = side * cell
    const cols = Math.max(1, Math.floor((maxWidth + gap) / (unitSize + gap)))
    const rows = Math.ceil(units / cols)
    layoutRef.current = { cols, pitch: unitSize + gap, units }
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    canvas.width = Math.floor((cols * (unitSize + gap) - gap) * dpr)
    canvas.height = Math.floor((rows * (unitSize + gap) - gap) * dpr)
    canvas.style.width = `${cols * (unitSize + gap) - gap}px`
    canvas.style.height = `${rows * (unitSize + gap) - gap}px`
    const context = canvas.getContext("2d")
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    let peak = 1e-6
    for (const row of weight) {
      for (const value of row) peak = Math.max(peak, Math.abs(value))
    }
    const positive = readColor("--pos", "#b86a2e")
    const negative = readColor("--neg", "#3d6fbf")
    const paper = readColor("--panel", "#ffffff")
    const accent = readColor("--accent", "#0071e3")
    context.clearRect(0, 0, canvas.width, canvas.height)
    for (let local = 0; local < units; local += 1) {
      const originX = (local % cols) * (unitSize + gap)
      const originY = Math.floor(local / cols) * (unitSize + gap)
      for (let index = 0; index < side * side; index += 1) {
        const value = weight[index]?.[local] ?? 0
        context.fillStyle = mix(paper, value >= 0 ? positive : negative, Math.min(1, Math.abs(value) / peak))
        context.fillRect(originX + (index % side) * cell, originY + Math.floor(index / side) * cell, cell, cell)
      }
      if (ids[local] === selected) {
        context.strokeStyle = mix(accent, accent, 1)
        context.lineWidth = 1.5
        context.strokeRect(originX + 0.5, originY + 0.5, unitSize - 1, unitSize - 1)
      }
    }
  }, [weight, side, selected, unitIds, dark])

  function pick(event: ReactPointerEvent<HTMLCanvasElement>) {
    const layout = layoutRef.current
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * rect.width
    const y = ((event.clientY - rect.top) / rect.height) * rect.height
    const col = Math.floor(x / layout.pitch)
    const row = Math.floor(y / layout.pitch)
    const local = row * layout.cols + col
    if (local < 0 || local >= layout.units) return
    selectRef.current(idsRef.current[local] ?? local)
  }

  return <canvas ref={canvasRef} className="pixel-grid" aria-label="Incoming weights as images" onPointerDown={pick} />
}
