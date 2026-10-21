export function formatCompact(value: number): string {
  if (!Number.isFinite(value)) return "—"
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 0.001 || abs >= 10000)) return value.toExponential(2)
  const text = String(value)
  return text.startsWith("-0") && Number(text) === 0 ? text.slice(1) : text
}

export function formatNum(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—"
  const abs = Math.abs(value)
  if (abs !== 0 && (abs < 0.001 || abs >= 10000)) return value.toExponential(2)
  const text = value.toFixed(digits)
  return text.startsWith("-0") && Number(text) === 0 ? text.slice(1) : text
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let amount = value
  let index = 0
  while (amount >= 1024 && index < units.length - 1) {
    amount /= 1024
    index += 1
  }
  const digits = index === 0 ? 0 : amount >= 10 ? 0 : 1
  return `${amount.toFixed(digits)} ${units[index]}`
}

export function formatPct(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—"
  return `${Math.round(value)}%`
}

export function layerTitles(hiddenCount: number, task: "gait" | "reviews" | "digits" = "gait"): string[] {
  const hidden = Array.from({ length: hiddenCount }, (_, index) =>
    hiddenCount === 1 ? "Hidden" : `Hidden ${index + 1}`,
  )
  if (task === "digits") return ["Pixels", ...hidden, "Digits"]
  if (task === "gait") {
    const head = hiddenCount === 1 ? ["Head"] : hidden.map((_, index) => `Head ${index + 1}`)
    return ["Sensors", "Stem", "Mid", "Deep", ...head, "Activity"]
  }
  return ["Pool", ...hidden, "Logit"]
}

export function imageSide(count: number): number | null {
  if (count < 64) return null
  const side = Math.sqrt(count)
  return Number.isInteger(side) ? side : null
}

export function networkParameters(input: number, hidden: number[], outputs: number): number {
  let total = 0
  let incoming = input
  for (const width of hidden) {
    total += incoming * width + width
    incoming = width
  }
  return total + incoming * outputs + outputs
}

export function parameterCount(embedDim: number, hidden: number[], vocab: number): number {
  let total = vocab * embedDim
  let incoming = embedDim
  for (const width of hidden) {
    total += incoming * width + width
    incoming = width
  }
  return total + incoming + 1
}
