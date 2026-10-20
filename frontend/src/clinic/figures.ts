import { HIDDEN, INPUTS, OUTPUTS, W1, W2 } from "./model"
import type { Measure, Reading } from "./types"

export type ChartPoint = {
  date: string
  value: number
  kind: "recorded" | "projected"
}

export type PlacedPoint = ChartPoint & { x: number; y: number }

export type ChartLayout = {
  width: number
  height: number
  plot: { left: number; right: number; top: number; bottom: number }
  yTicks: { y: number; label: string }[]
  xTicks: { x: number; label: string }[]
  recorded: PlacedPoint[]
  projected: PlacedPoint[]
  goal: { y: number; value: number } | null
  active: PlacedPoint | null
  place: (values: number[]) => { x: number; y: number }[]
}

const CHART_WIDTH = 960
const CHART_HEIGHT = 360

export function timeline(recorded: Reading[], forecast: { date: string; value: number }[]): ChartPoint[] {
  return [
    ...recorded.map((point) => ({ ...point, kind: "recorded" as const })),
    ...forecast.map((point) => ({ ...point, kind: "projected" as const })),
  ]
}

export function layoutChart(
  points: ChartPoint[],
  goalValue: number | null,
  activeIndex: number,
  tickLabel: (iso: string) => string,
  valueLabel: (value: number) => string,
  width = CHART_WIDTH,
  alternatives: number[][] = [],
): ChartLayout {
  const narrow = width < 560
  const height = narrow ? 300 : CHART_HEIGHT
  const values = points.map((point) => point.value)
  if (goalValue != null) values.push(goalValue)
  alternatives.forEach((series) => values.push(...series))
  const min = Math.min(...values)
  const max = Math.max(...values)
  const pad = (max - min) * 0.18 || 1
  const yMin = min - pad
  const yMax = max + pad
  const plot = { left: 40, right: narrow ? 104 : 136, top: 32, bottom: 40 }
  const innerW = width - plot.left - plot.right
  const innerH = height - plot.top - plot.bottom
  const yOf = (value: number) => plot.top + ((yMax - value) / (yMax - yMin)) * innerH
  const xOf = (index: number) => plot.left + (points.length === 1 ? innerW / 2 : (index / (points.length - 1)) * innerW)
  const placed = points.map((point, index) => ({ ...point, x: xOf(index), y: yOf(point.value) }))
  const recorded = placed.filter((point) => point.kind === "recorded")
  const lastRecorded = recorded[recorded.length - 1]
  const projected = lastRecorded ? [lastRecorded, ...placed.filter((point) => point.kind === "projected")] : []
  const yTicks = [yMax, (yMax + yMin) / 2, yMin].map((value) => ({ y: yOf(value), label: valueLabel(value) }))
  const tickIndexes = [0, Math.max(0, recorded.length - 1), points.length - 1]
  const xTicks = [...new Set(tickIndexes)].map((index) => ({
    x: xOf(index),
    label: tickLabel(points[index].date),
  }))
  return {
    width,
    height,
    plot,
    yTicks,
    xTicks,
    recorded,
    projected,
    goal: goalValue == null ? null : { y: yOf(goalValue), value: goalValue },
    active: placed[clampIndex(activeIndex, placed.length)] ?? null,
    place: (series) => {
      const start = Math.max(0, recorded.length - 1)
      const head = lastRecorded ? [{ x: lastRecorded.x, y: lastRecorded.y }] : []
      return [...head, ...series.map((value, index) => ({ x: xOf(start + 1 + index), y: yOf(value) }))]
    },
  }
}

function clampIndex(index: number, length: number): number {
  if (length === 0) return 0
  return Math.min(length - 1, Math.max(0, index))
}

export function pointsAttr(points: PlacedPoint[]): string {
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ")
}

export type DiagramNode = {
  id: string
  label: string
  x: number
  y: number
  anchor: "start" | "end" | "middle"
  textX: number
  textY: number
}

export type DiagramEdge = {
  x1: number
  y1: number
  x2: number
  y2: number
  weight: number
  from: string
  to: string
}

export type DiagramLayout = {
  width: number
  height: number
  nodes: DiagramNode[]
  edges: DiagramEdge[]
}

export function layoutDiagram(): DiagramLayout {
  const outputLabels = ["Change", "Response"]
  const inputs: DiagramNode[] = INPUTS.map((input, index) => ({
    id: `in-${index}`,
    label: input.label,
    x: 128,
    y: 28 + index * 48,
    anchor: "end",
    textX: 104,
    textY: 32 + index * 48,
  }))
  const hidden: DiagramNode[] = HIDDEN.map((unit, index) => ({
    id: `hid-${index}`,
    label: unit.label,
    x: 286,
    y: 52 + index * 64,
    anchor: "middle",
    textX: 286,
    textY: 80 + index * 64,
  }))
  const outputs: DiagramNode[] = outputLabels.map((label, index) => ({
    id: `out-${index}`,
    label,
    x: 444,
    y: 116 + index * 96,
    anchor: "middle",
    textX: 444,
    textY: 144 + index * 96,
  }))
  const edges: DiagramEdge[] = []
  W1.forEach((row, hiddenIndex) => {
    row.forEach((weight, inputIndex) => {
      edges.push({
        x1: inputs[inputIndex].x,
        y1: inputs[inputIndex].y,
        x2: hidden[hiddenIndex].x,
        y2: hidden[hiddenIndex].y,
        weight,
        from: inputs[inputIndex].id,
        to: hidden[hiddenIndex].id,
      })
    })
  })
  W2.forEach((row, outputIndex) => {
    row.forEach((weight, hiddenIndex) => {
      edges.push({
        x1: hidden[hiddenIndex].x,
        y1: hidden[hiddenIndex].y,
        x2: outputs[outputIndex].x,
        y2: outputs[outputIndex].y,
        weight,
        from: hidden[hiddenIndex].id,
        to: outputs[outputIndex].id,
      })
    })
  })
  return { width: 520, height: 316, nodes: [...inputs, ...hidden, ...outputs], edges }
}

export function edgeWidth(weight: number): number {
  return 1 + 3.2 * (Math.abs(weight) / 1.25)
}

export const diagramUnits = [
  ...INPUTS.map((input, index) => ({ id: `in-${index}`, group: "Inputs", label: input.label, detail: input.detail })),
  ...HIDDEN.map((unit, index) => ({ id: `hid-${index}`, group: "Hidden units", label: unit.label, detail: unit.detail })),
  ...OUTPUTS.map((output, index) => ({ id: `out-${index}`, group: "Outputs", label: output.label, detail: output.detail })),
]

export function measureTick(value: number, measure: Measure): string {
  return value.toFixed(measure.decimals)
}
