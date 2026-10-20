import { useEffect, useRef, type KeyboardEvent as ReactKeyboardEvent } from "react"
import { weightLookup } from "./linalg"
import { latestTrace } from "./store"
import type { Selection, Trace } from "./types"

const STRONG_PER_UNIT = 6
const CANVAS_MONO = '11px "JetBrains Mono", ui-monospace, monospace'

type RGB = [number, number, number]
type Node = { x: number; y: number; layer: number; unit: number }
type Particle = {
  x0: number
  y0: number
  x1: number
  y1: number
  t: number
  speed: number
  alpha: number
}

const MAX_SHOWN = 96
const MAX_EDGES = 8000
const TOP_K = 8

function parseColor(value: string): RGB {
  const text = value.trim()
  if (text.startsWith("#")) {
    const hex = text.slice(1)
    const full = hex.length === 3 ? hex.split("").map((part) => part + part).join("") : hex
    const parsed = Number.parseInt(full, 16)
    return [(parsed >> 16) & 255, (parsed >> 8) & 255, parsed & 255]
  }
  return [140, 140, 140]
}

function cssColor(name: string, fallback: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
}

function shownUnits(count: number): number[] {
  if (count <= MAX_SHOWN) return Array.from({ length: count }, (_, index) => index)
  return Array.from({ length: MAX_SHOWN }, (_, index) => Math.round((index * (count - 1)) / (MAX_SHOWN - 1)))
}

function percentile(values: number[], fraction: number): number {
  if (!values.length) return 1
  const sorted = values.slice().sort((a, b) => a - b)
  const index = Math.min(sorted.length - 1, Math.floor(fraction * (sorted.length - 1)))
  return Math.max(sorted[index], 1e-6)
}

function mix(from: RGB, to: RGB, amount: number): string {
  const t = Math.max(0, Math.min(1, amount))
  const red = Math.round(from[0] + (to[0] - from[0]) * t)
  const green = Math.round(from[1] + (to[1] - from[1]) * t)
  const blue = Math.round(from[2] + (to[2] - from[2]) * t)
  return `rgb(${red}, ${green}, ${blue})`
}

function activationAt(trace: Trace | null, layer: number, unit: number): number {
  if (!trace) return 0
  if (layer === 0) {
    if (trace.task === "digits") return trace.pixels[unit] ?? 0
    return trace.pooled[unit] ?? 0
  }
  return trace.layers[layer - 1]?.output[unit] ?? 0
}

export function NetworkCanvas({
  sizes,
  titles,
  selection,
  reducedMotion,
  imageSide,
  onSelect,
}: {
  sizes: number[]
  titles: string[]
  selection: Selection
  reducedMotion: boolean
  imageSide: number | null
  onSelect: (selection: Selection) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  const selectionRef = useRef(selection)
  const reducedRef = useRef(reducedMotion)
  const onSelectRef = useRef(onSelect)
  const sizesRef = useRef(sizes)
  const titlesRef = useRef(titles)
  const imageRef = useRef(imageSide)
  selectionRef.current = selection
  reducedRef.current = reducedMotion
  onSelectRef.current = onSelect
  sizesRef.current = sizes
  titlesRef.current = titles
  imageRef.current = imageSide

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return
    const context = canvas.getContext("2d")
    if (!context) return

    let frame = 0
    let width = 0
    let height = 0
    let traceSeq = -1
    let sizeKey = ""
    let nodes: Node[] = []
    let buckets: { path: Path2D; color: string; alpha: number }[] = []
    const particles: Particle[] = []
    let last = performance.now()
    const hover = { layer: -1, unit: -1 }

    let labelY = 112
    let grid = 0
    let cell = 0
    const rebuild = () => {
      const rawTrace = latestTrace
      const layerSizes = sizesRef.current
      const side = imageRef.current
      grid = side && layerSizes[0] === side * side ? side : 0
      const padX = grid ? 28 : 72
      const toolbar = document.querySelector(".toolbar")
      const top = wrap.getBoundingClientRect().top
      const occupied = toolbar ? toolbar.getBoundingClientRect().bottom - top : 96
      const padY = Math.max(48, occupied + 28)
      labelY = padY - 16
      const bottom = 36
      const availH = Math.max(80, height - padY - bottom)
      cell = grid ? Math.max(3, Math.min(12, availH / grid, ((width - 56) * 0.36) / grid)) : 0
      const gridSize = grid * cell
      const contentLeft = grid ? padX + gridSize + 48 : padX
      const contentRight = width - (grid ? 56 : padX)
      const laterLayers = grid ? Math.max(1, layerSizes.length - 1) : layerSizes.length
      nodes = []
      const anchors: Node[][] = []
      layerSizes.forEach((count, layer) => {
        if (grid && layer === 0) {
          const originX = padX
          const originY = padY + (availH - gridSize) / 2
          const column: Node[] = []
          for (let unit = 0; unit < count; unit += 1) {
            const node = {
              x: originX + ((unit % grid) + 0.5) * cell,
              y: originY + (Math.floor(unit / grid) + 0.5) * cell,
              layer,
              unit,
            }
            nodes.push(node)
            column.push(node)
          }
          anchors.push(column)
          return
        }
        const units = shownUnits(count)
        const laterIndex = grid ? layer - 1 : layer
        const x =
          laterLayers === 1
            ? (contentLeft + contentRight) / 2
            : contentLeft + ((contentRight - contentLeft) * laterIndex) / (laterLayers - 1)
        const step = units.length === 1 ? 0 : availH / (units.length - 1)
        const column: Node[] = units.map((unit, index) => {
          const node = { x, y: units.length === 1 ? padY + availH / 2 : padY + step * index, layer, unit }
          nodes.push(node)
          return node
        })
        anchors.push(column)
      })
      buckets = []
      const trace = rawTrace && ((grid && rawTrace.task === "digits") || (!grid && rawTrace.task !== "digits")) ? rawTrace : null
      if (!trace) return
      const positive = cssColor("--pos", "#b86a2e")
      const negative = cssColor("--neg", "#3d6fbf")
      const fullEdges = layerSizes.slice(0, -1).reduce((sum, count, index) => sum + count * layerSizes[index + 1], 0)
      const drawAll = fullEdges <= MAX_EDGES && layerSizes.every((count) => count <= MAX_SHOWN)
      trace.layers.forEach((layer, index) => {
        const sources = anchors[index]
        const dests = anchors[index + 1]
        if (!sources || !dests) return
        const at = weightLookup(layer)
        const mags: number[] = []
        for (const source of sources) {
          for (const dest of dests) mags.push(Math.abs(at(source.unit, dest.unit)))
        }
        const scale = percentile(mags, 0.95)
        const groups = new Map<string, { path: Path2D; color: string; alpha: number }>()
        const add = (source: Node, dest: Node, value: number, faint = false) => {
          const alpha = faint ? 0.035 : Math.max(0.3, Math.pow(Math.min(1, Math.abs(value) / scale), 0.8) * 0.85)
          const color = value >= 0 ? positive : negative
          const bucket = faint ? alpha : Math.round(alpha * 5) / 5
          const key = `${color}:${bucket}`
          let group = groups.get(key)
          if (!group) {
            group = { path: new Path2D(), color, alpha: bucket }
            groups.set(key, group)
          }
          group.path.moveTo(source.x, source.y)
          group.path.lineTo(dest.x, dest.y)
        }
        if (drawAll) {
          dests.forEach((dest) => {
            sources
              .map((source) => ({ source, value: at(source.unit, dest.unit) }))
              .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
              .forEach((entry, rank) => add(entry.source, dest, entry.value, rank >= STRONG_PER_UNIT))
          })
        } else {
          const fromImage = sources.length > MAX_SHOWN
          dests.forEach((dest) => {
            const ranked = sources
              .map((source) => ({ source, value: at(source.unit, dest.unit) }))
              .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
            const inked = fromImage
              ? ranked.filter((entry) => Math.abs(activationAt(trace, entry.source.layer, entry.source.unit)) > 0.08)
              : ranked
            const chosen = (inked.length ? inked : ranked).slice(0, fromImage ? 3 : TOP_K)
            chosen.forEach((entry) => add(entry.source, dest, entry.value))
          })
        }
        buckets.push(...groups.values())
      })
    }

    const spawn = (dt: number) => {
      const trace = latestTrace
      if (reducedRef.current || !trace || particles.length >= 70) return
      const births = dt > 0.05 ? 1 : 2
      for (let n = 0; n < births && particles.length < 70; n += 1) {
        const boundary = Math.floor(Math.random() * trace.layers.length)
        const layer = trace.layers[boundary]
        const sources = nodes.filter((node) => node.layer === boundary)
        const dests = nodes.filter((node) => node.layer === boundary + 1)
        if (!sources.length || !dests.length) continue
        const sourceWeights = sources.map((node) => Math.abs(activationAt(trace, boundary, node.unit)) + 0.02)
        const source = pickWeighted(sources, sourceWeights)
        const destWeights = dests.map((node) => {
          const activation = activationAt(trace, boundary, source.unit)
          return Math.abs(activation * weightLookup(layer)(source.unit, node.unit)) + 0.002
        })
        const dest = pickWeighted(dests, destWeights)
        const strength = destWeights[dests.indexOf(dest)]
        particles.push({
          x0: source.x,
          y0: source.y,
          x1: dest.x,
          y1: dest.y,
          t: 0,
          speed: 0.85 + Math.random() * 0.55,
          alpha: Math.max(0.25, Math.min(0.9, strength * 4)),
        })
      }
    }

    const draw = (now: number) => {
      const rect = wrap.getBoundingClientRect()
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const nextW = Math.max(1, rect.width)
      const nextH = Math.max(1, rect.height)
      if (canvas.width !== Math.floor(nextW * dpr) || canvas.height !== Math.floor(nextH * dpr)) {
        canvas.width = Math.floor(nextW * dpr)
        canvas.height = Math.floor(nextH * dpr)
        width = nextW
        height = nextH
        traceSeq = -1
      } else {
        width = nextW
        height = nextH
      }
      const rawTrace = latestTrace
      const key = `${sizesRef.current.join(",")}@${imageRef.current ?? 0}`
      if (traceSeq !== (rawTrace?.seq ?? 0) || key !== sizeKey) {
        traceSeq = rawTrace?.seq ?? 0
        sizeKey = key
        rebuild()
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0)
      context.clearRect(0, 0, width, height)
      const stage = cssColor("--stage", "#111113")
      context.fillStyle = stage
      context.fillRect(0, 0, width, height)
      context.strokeStyle = cssColor("--grid", "rgba(0, 0, 0, 0.05)")
      context.lineWidth = 1
      context.beginPath()
      for (let x = 0.5; x < width; x += 40) {
        context.moveTo(x, 0)
        context.lineTo(x, height)
      }
      for (let y = 0.5; y < height; y += 40) {
        context.moveTo(0, y)
        context.lineTo(width, y)
      }
      context.stroke()
      if (grid) {
        const origin = nodes.find((node) => node.layer === 0)
        if (origin) {
          const left = origin.x - cell / 2
          const top = origin.y - cell / 2
          context.fillStyle = "#0e110b"
          context.beginPath()
          context.roundRect(left - 8, top - 8, grid * cell + 16, grid * cell + 16, 2)
          context.fill()
        }
      }
      buckets.forEach((bucket) => {
        context.globalAlpha = bucket.alpha
        context.strokeStyle = bucket.color
        context.lineWidth = 1
        context.stroke(bucket.path)
      })
      context.globalAlpha = 1
      const trace = rawTrace && ((grid && rawTrace.task === "digits") || (!grid && rawTrace.task !== "digits")) ? rawTrace : null
      const accent = parseColor(cssColor("--accent", "#0a84ff"))
      const idle = parseColor(cssColor("--node", "#3a3a3c"))
      const ink = cssColor("--secondary", "#a1a1a6")
      const text = cssColor("--text", "#f5f5f7")
      const selected = selectionRef.current
      nodes.forEach((node) => {
        const value = activationAt(trace, node.layer, node.unit)
        const hot = selected.layer === node.layer && selected.unit === node.unit
        const over = hover.layer === node.layer && hover.unit === node.unit
        if (grid && node.layer === 0) {
          const ink = Math.max(0, Math.min(1, value))
          const gap = cell >= 6 ? 1 : 0
          const size = Math.max(1, cell - gap)
          context.fillStyle = mix([16, 16, 18], [245, 245, 247], ink)
          context.fillRect(node.x - cell / 2, node.y - cell / 2, size, size)
          if (hot || over) {
            context.strokeStyle = text
            context.globalAlpha = hot ? 0.95 : 0.45
            context.lineWidth = 1
            context.strokeRect(node.x - cell / 2 + 0.5, node.y - cell / 2 + 0.5, size - 1, size - 1)
            context.globalAlpha = 1
          }
          return
        }
        const layerValues = nodes.filter((item) => item.layer === node.layer).map((item) => Math.abs(activationAt(trace, item.layer, item.unit)))
        const scale = Math.max(0.2, percentile(layerValues, 0.9))
        const columnCount = nodes.filter((item) => item.layer === node.layer).length || 1
        const span = columnCount === 1 ? 40 : (height - 168) / (columnCount - 1)
        const radius = node.layer === sizesRef.current.length - 1 ? Math.min(12, span * 0.45) : Math.max(1.7, Math.min(6.5, span * 0.32))
        context.beginPath()
        context.fillStyle = mix(idle, accent, Math.min(1, Math.abs(value) / scale))
        context.arc(node.x, node.y, radius, 0, Math.PI * 2)
        context.fill()
        if (grid && node.layer === sizesRef.current.length - 1) {
          context.fillStyle = text
          context.font = CANVAS_MONO
          context.textAlign = "left"
          context.textBaseline = "middle"
          context.fillText(String(node.unit), node.x + radius + 6, node.y)
          context.textAlign = "center"
          context.textBaseline = "alphabetic"
        }
        if (hot || over) {
          context.beginPath()
          context.arc(node.x, node.y, radius, 0, Math.PI * 2)
          context.strokeStyle = text
          context.globalAlpha = hot ? 0.9 : 0.45
          context.lineWidth = hot ? 2 : 1
          context.stroke()
          context.globalAlpha = 1
        }
      })
      context.font = CANVAS_MONO
      context.textAlign = "center"
      titlesRef.current.forEach((title, layer) => {
        const column = nodes.filter((node) => node.layer === layer)
        if (!column.length) return
        const labelX =
          grid && layer === 0 ? column.reduce((sum, node) => sum + node.x, 0) / column.length : column[0].x
        context.fillStyle = selected.layer === layer ? text : ink
        context.fillText(`${title}  ${sizesRef.current[layer]}`, labelX, labelY)
      })
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      if (!reducedRef.current) {
        spawn(dt)
        for (let index = particles.length - 1; index >= 0; index -= 1) {
          const particle = particles[index]
          particle.t += dt * particle.speed
          if (particle.t >= 1) particles.splice(index, 1)
        }
        context.fillStyle = cssColor("--accent", "#0a84ff")
        particles.forEach((particle) => {
          const x = particle.x0 + (particle.x1 - particle.x0) * particle.t
          const y = particle.y0 + (particle.y1 - particle.y0) * particle.t
          context.globalAlpha = particle.alpha * Math.sin(Math.PI * particle.t)
          context.beginPath()
          context.arc(x, y, 2.1, 0, Math.PI * 2)
          context.fill()
        })
        context.globalAlpha = 1
      }
      context.textAlign = "left"
      context.font = CANVAS_MONO
      context.fillStyle = ink
      const noun = trace?.task === "digits" ? "digit" : "review"
      const caption = !trace
        ? "Weights appear with the first step"
        : trace.kind === "probe"
          ? `Your ${noun}`
          : trace.untrained
            ? `Initial weights · one ${noun}`
            : `Step ${trace.step.toLocaleString()} · one ${noun}`
      context.fillText(caption, 20, height - 16)
      if (hover.layer >= 0) {
        const node = nodes.find((item) => item.layer === hover.layer && item.unit === hover.unit)
        if (node) {
          const value = activationAt(trace, node.layer, node.unit)
          const label =
            grid && node.layer === 0
              ? `Pixel ${Math.floor(node.unit / grid) + 1}, ${(node.unit % grid) + 1} · ${value.toFixed(2)}`
              : grid && node.layer === sizesRef.current.length - 1
                ? `Digit ${node.unit} · ${value.toFixed(3)}`
                : `${titlesRef.current[node.layer] ?? "Unit"} ${node.unit + 1} · ${value.toFixed(3)}`
          context.font = CANVAS_MONO
          const box = context.measureText(label).width + 16
          const x = Math.min(width - box - 12, Math.max(12, node.x + 12))
          const y = Math.max(labelY + 8, node.y - 18)
          context.fillStyle = cssColor("--panel", "#1c1c1e")
          context.beginPath()
          context.roundRect(x, y - 14, box, 22, 2)
          context.fill()
          context.fillStyle = text
          context.fillText(label, x + 8, y + 1)
        }
      }
      frame = requestAnimationFrame(draw)
    }

    const hit = (event: PointerEvent): Node | null => {
      const rect = canvas.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      let best: Node | null = null
      let bestDistance = 16
      for (const node of nodes) {
        const distance = Math.hypot(node.x - x, node.y - y)
        if (distance < bestDistance) {
          best = node
          bestDistance = distance
        }
      }
      return best
    }

    const onPointer = (event: PointerEvent) => {
      const node = hit(event)
      hover.layer = node ? node.layer : -1
      hover.unit = node ? node.unit : -1
      canvas.style.cursor = node ? "pointer" : "default"
      if (event.type === "pointerdown" && node) onSelectRef.current({ layer: node.layer, unit: node.unit })
    }
    canvas.addEventListener("pointermove", onPointer)
    canvas.addEventListener("pointerdown", onPointer)
    frame = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(frame)
      canvas.removeEventListener("pointermove", onPointer)
      canvas.removeEventListener("pointerdown", onPointer)
    }
  }, [])

  function onKey(event: ReactKeyboardEvent<HTMLCanvasElement>) {
    if (event.altKey || event.metaKey || event.ctrlKey) return
    const keys = ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]
    if (!keys.includes(event.key)) return
    event.preventDefault()
    const widths = sizes.length ? sizes : [1]
    let layer = Math.min(selection.layer, widths.length - 1)
    let unit = selection.unit
    if (event.key === "ArrowLeft") layer = Math.max(0, layer - 1)
    if (event.key === "ArrowRight") layer = Math.min(widths.length - 1, layer + 1)
    const count = Math.max(1, widths[layer] ?? 1)
    if (event.key === "ArrowUp") unit = Math.max(0, unit - 1)
    if (event.key === "ArrowDown") unit = Math.min(count - 1, unit + 1)
    if (event.key === "Home") unit = 0
    if (event.key === "End") unit = count - 1
    unit = Math.min(unit, count - 1)
    if (layer !== selection.layer || unit !== selection.unit) onSelect({ layer, unit })
  }

  const spoken = describeSelection(titles, sizes, selection, imageSide)

  return (
    <div ref={wrapRef} className="canvas-wrap">
      <canvas
        ref={canvasRef}
        className="network"
        tabIndex={0}
        aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight"
        aria-label="Network graph. Arrow keys move the selected unit. The inspector follows."
        onKeyDown={onKey}
      />
      <p className="sr" aria-live="polite">
        {spoken}
      </p>
    </div>
  )
}

function describeSelection(titles: string[], sizes: number[], selection: Selection, imageSide: number | null) {
  const title = titles[selection.layer] ?? "Unit"
  const last = sizes.length > 0 && selection.layer === sizes.length - 1
  if (imageSide && selection.layer === 0) {
    const row = Math.floor(selection.unit / imageSide) + 1
    const col = (selection.unit % imageSide) + 1
    return `Pixel row ${row}, column ${col}`
  }
  if (imageSide && last) return `Digit ${selection.unit}`
  return `${title} ${selection.unit + 1}`
}

function pickWeighted<T>(items: T[], weights: number[]): T {
  const total = weights.reduce((sum, value) => sum + value, 0)
  let cursor = Math.random() * total
  for (let index = 0; index < items.length; index += 1) {
    cursor -= weights[index]
    if (cursor <= 0) return items[index]
  }
  return items[items.length - 1]
}
