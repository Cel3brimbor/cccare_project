import type { AffineLayer, Trace } from "./types"

export type LayerGrad = {
  delta: number[]
  gradW: number[][]
  gradB: number[]
  dInput: number[]
}

export function columnForUnit(layer: AffineLayer, unit: number): number[] | null {
  if (layer.rowIndex) return null
  const col = layer.colIndex ? layer.colIndex.indexOf(unit) : unit
  if (col < 0) return null
  return column(layer.weight, col)
}

export function weightLookup(layer: AffineLayer) {
  const rowMap = layer.rowIndex ? new Map(layer.rowIndex.map((unit, index) => [unit, index])) : null
  const colMap = layer.colIndex ? new Map(layer.colIndex.map((unit, index) => [unit, index])) : null
  const frameCols = layer.frameCols
  return (row: number, col: number) => {
    const source = frameCols ? Math.floor(row / frameCols) : row
    const mappedRow = rowMap ? rowMap.get(source) : source
    const mappedCol = colMap ? colMap.get(col) : col
    if (mappedRow === undefined || mappedCol === undefined) return 0
    return layer.weight[mappedRow]?.[mappedCol] ?? 0
  }
}

function layerGrad(layer: Trace["layers"][number], delta: number[], dInput: number[]): LayerGrad {
  const cells = layer.input.length * delta.length
  const gradW = cells > 96 * 256 ? [] : layer.input.map((activation) => delta.map((entry) => activation * entry))
  return { delta, gradW, gradB: delta.slice(), dInput }
}

export function exampleBackward(trace: Trace): LayerGrad[] | null {
  if (trace.label === null) return null
  const start = trace.layers.findIndex((layer) => layer.activation !== "conv")
  if (start < 0) return null
  const layers = trace.layers.slice(start)
  const sampled = layers.some((layer) => layer.rowIndex || layer.colIndex)
  const grads: LayerGrad[] = []
  if (sampled) {
    if (layers.some((layer) => !layer.delta || !layer.dInput)) return null
    for (const layer of layers) grads.push(layerGrad(layer, layer.delta ?? [], layer.dInput ?? []))
  } else {
    const last = layers[layers.length - 1]
    let upstream =
      last?.activation === "softmax"
        ? last.output.map((probability, index) => probability - (index === trace.label ? 1 : 0))
        : [trace.prob - trace.label]
    const reversed: LayerGrad[] = []
    for (let index = layers.length - 1; index >= 0; index -= 1) {
      const layer = layers[index]
      const delta =
        layer.activation === "relu" ? upstream.map((value, unit) => (layer.pre[unit] > 0 ? value : 0)) : upstream
      const dInput = layer.weight.map((row) => row.reduce((sum, weight, unit) => sum + weight * (delta[unit] ?? 0), 0))
      reversed.push(layerGrad(layer, delta, dInput))
      upstream = dInput
    }
    reversed.reverse()
    grads.push(...reversed)
  }
  const full = trace.layers.map(() => ({ delta: [] as number[], gradW: [] as number[][], gradB: [] as number[], dInput: [] as number[] }))
  grads.forEach((grad, index) => {
    full[start + index] = grad
  })
  return full
}

export function column(weight: number[][], unit: number): number[] {
  return weight.map((row) => row[unit] ?? 0)
}

export function dotPrefix(left: number[], right: number[], count: number): { sum: number; term: number } {
  const end = Math.max(0, Math.min(count, left.length, right.length))
  let sum = 0
  for (let index = 0; index < end; index += 1) sum += left[index] * right[index]
  const term = end > 0 ? left[end - 1] * right[end - 1] : 0
  return { sum, term }
}

export function meanPrefix(rows: number[][], unit: number, count: number, total: number): { sum: number; term: number } {
  const end = Math.max(0, Math.min(count, rows.length))
  const denom = Math.max(1, total)
  let raw = 0
  for (let index = 0; index < end; index += 1) raw += rows[index][unit] ?? 0
  const term = end > 0 ? (rows[end - 1][unit] ?? 0) / denom : 0
  return { sum: raw / denom, term }
}
