import { conditionById } from "./data"
import { formatDate, formatDay, formatGoal, formatMeasure, fullName } from "./format"
import { edgeWidth, layoutChart, layoutDiagram, pointsAttr, timeline } from "./figures"
import { B1, B2, HIDDEN, INPUTS, W1, W2, projectAll, sumLine } from "./model"
import type { Explanation, UnitStep } from "./model"
import type { Measure, Patient } from "./types"

export function reportFilename(patient: Patient): string {
  const slug = `${patient.family}-${patient.given}`
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
  return `meridian-${slug || "patient"}-projection.html`
}

export function buildReport(patient: Patient, treatmentId: string): string {
  const condition = conditionById(patient.conditionId)
  const ranked = projectAll(patient, condition)
  const selected = ranked.find((item) => item.treatment.id === treatmentId) ?? ranked[0]
  const measure = condition.measure
  const latest = patient.readings[patient.readings.length - 1]
  const name = fullName(patient)
  const goal = formatGoal(measure)
  const drop = latest.value - selected.explanation.futureNative
  const reachesGoal = drop + 1e-6 >= measure.goal
  const points = timeline(patient.readings, selected.forecast)
  const chart = layoutChart(points, latest.value - measure.goal, patient.readings.length - 1, formatDay, (value) =>
    value.toFixed(measure.decimals),
  )
  const recordedEnd = chart.recorded[chart.recorded.length - 1]
  const projectedEnd = chart.projected[chart.projected.length - 1]

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Projection for ${esc(name)}</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #eef0e7; color: #1b2016; font: 16px/1.5 Archivo, "Helvetica Neue", Arial, sans-serif; }
    main { max-width: 800px; margin: 0 auto; padding: 32px 20px 64px; }
    h1, h2, h3 { font-family: Archivo, "Helvetica Neue", Arial, sans-serif; line-height: 1.2; letter-spacing: -0.02em; margin: 0 0 8px; }
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.25rem; margin-top: 32px; }
    h3 { font-size: 1.05rem; margin-top: 20px; }
    p { margin: 0 0 12px; }
    .meta { color: #535a4a; }
    .note { background: #fff; border: 2px solid #c2c6b6; border-radius: 2px; padding: 12px 16px; }
    .result { background: #fff; border: 2px solid #1b2016; border-radius: 2px; padding: 16px; }
    .result strong { font-size: 1.4rem; }
    table { width: 100%; border-collapse: collapse; background: #fff; margin: 8px 0 16px; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #c2c6b6; vertical-align: top; }
    th { font-size: 0.875rem; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    figure { margin: 12px 0; background: #fff; border: 2px solid #c2c6b6; border-radius: 2px; padding: 12px; }
    svg { width: 100%; height: auto; display: block; }
    .legend { display: flex; flex-wrap: wrap; gap: 12px 20px; margin: 0; padding: 0; list-style: none; }
    .legend i { display: inline-block; width: 28px; height: 0; border-top: 3px solid #1b2016; margin-right: 6px; vertical-align: middle; }
    .legend .dashed { border-top-style: dashed; border-top-color: #91550d; }
    .legend .dotted { border-top-style: dotted; border-top-color: #3f5d1b; }
    .eq { font-variant-numeric: tabular-nums; background: #dfe1d6; padding: 8px 12px; border-radius: 8px; }
    details { background: #fff; border: 2px solid #c2c6b6; border-radius: 2px; padding: 8px 16px; margin: 12px 0; }
    summary { cursor: pointer; font-weight: 650; min-height: 44px; display: flex; align-items: center; }
    a { color: #1b2016; }
    @media print {
      body { background: #fff; }
      main { max-width: none; padding: 0; }
      details { break-inside: avoid; }
      details > *:not(summary) { display: block !important; }
    }
  </style>
</head>
<body>
<main>
  <p class="meta">Meridian Clinical · patient copy · ${esc(formatDate(latest.date))}</p>
  <h1>Projection for ${esc(name)}</h1>
  <p class="note">This file is a plain-language copy of a demonstration chart. The people and results are synthetic. The network uses fixed weights so the arithmetic can be checked. It is not a medical device, it was not trained on these records, and it does not decide treatment.</p>
  <h2>Who this is about</h2>
  <table>
    <tbody>
      ${row("Name", name)}
      ${row("Record number", patient.mrn)}
      ${row("Age", `${patient.age} years`)}
      ${row("Sex", patient.sex)}
      ${row("Condition", condition.name)}
      ${row("Other problems", patient.problems.length ? patient.problems.join(", ") : "None listed")}
      ${row("Medicines", patient.medications.join("; "))}
      ${row("Allergies", patient.allergies)}
      ${row("Adherence entered", `${Math.round(patient.adherence * 100)}%`)}
      ${patient.note ? row("Clinician note", patient.note) : ""}
    </tbody>
  </table>
  <h2>The result in words</h2>
  <div class="result">
    <p>Latest ${esc(measure.name)}: <strong>${esc(formatMeasure(latest.value, measure))}</strong> on ${esc(formatDate(latest.date))}.</p>
    <p>For <strong>${esc(selected.treatment.name)}</strong>, the network projects <strong>${esc(formatMeasure(selected.explanation.futureNative, measure))}</strong> in 12 weeks.</p>
    <p>${reachesGoal ? `The central projection meets a drop of ${esc(goal)}.` : `The central projection does not fall by ${esc(goal)}.`} The chance of a drop at least that large is <strong>${pct(selected.explanation.probability)}%</strong>. Higher values on this measure are worse.</p>
    <p>${esc(selected.treatment.summary)}</p>
    ${selected.caution ? `<p><strong>Allergy:</strong> ${esc(selected.caution)}</p>` : ""}
  </div>
  <h2>How the options compare</h2>
  <p>The same network scores every option. Only adherence and intensity change. Medicines on the list above are not separate inputs.</p>
  <table>
    <thead>
      <tr>
        <th scope="col">Option</th>
        <th scope="col" class="num">In 12 weeks</th>
        <th scope="col" class="num">Chance of a drop of at least ${esc(goal)}</th>
      </tr>
    </thead>
    <tbody>
      ${ranked
        .map(
          (item) => `<tr>
            <th scope="row">${esc(item.treatment.name)}${item.treatment.id === selected.treatment.id ? " (this copy)" : ""}</th>
            <td class="num">${esc(formatMeasure(item.explanation.futureNative, measure))}</td>
            <td class="num">${pct(item.explanation.probability)}%</td>
          </tr>`,
        )
        .join("")}
    </tbody>
  </table>
  <h2>Graph</h2>
  <p>The solid line is what was recorded. The dashed line is the projection for ${esc(selected.treatment.name)}. The dotted line is the goal, a drop of ${esc(goal)} from the latest result. The table under the graph has every value.</p>
  <figure>
    ${chartSvg(chart, recordedEnd, projectedEnd, formatMeasure(latest.value - measure.goal, measure))}
    <ul class="legend">
      <li><i></i> Recorded, latest ${esc(formatMeasure(latest.value, measure))}</li>
      <li><i class="dashed"></i> Projected, 12 weeks ${esc(formatMeasure(selected.explanation.futureNative, measure))}</li>
      <li><i class="dotted"></i> Goal ${esc(formatMeasure(latest.value - measure.goal, measure))}</li>
    </ul>
  </figure>
  <h3>Recorded values</h3>
  ${valueTable(patient.readings.map((reading) => [formatDate(reading.date), formatMeasure(reading.value, measure)]))}
  <h3>Projected values for ${esc(selected.treatment.name)}</h3>
  ${valueTable(selected.forecast.map((point) => [formatDate(point.date), formatMeasure(point.value, measure)]))}
  <h2>The network</h2>
  <p>Six numbers go in. Four hidden units turn on only when their weighted sum is above zero. Two numbers come out: the change in severity, and the chance of a response. Ochre lines are positive weights. Blue lines are negative weights. Thicker lines are larger weights.</p>
  <figure>
    ${diagramSvg(`out-0`)}
  </figure>
  <h3>Inputs for ${esc(selected.treatment.name)}</h3>
  <table>
    <thead><tr><th scope="col">Input</th><th scope="col" class="num">Value</th><th scope="col">What it means</th></tr></thead>
    <tbody>
      ${selected.explanation.inputs
        .map(
          (input) =>
            `<tr><th scope="row">${esc(input.label)}</th><td class="num">${input.value.toFixed(4)}</td><td>${esc(input.detail)}</td></tr>`,
        )
        .join("")}
    </tbody>
  </table>
  <h2>Equations for ${esc(selected.treatment.name)}</h2>
  <p>Each line is bias plus weight times input. The total is the model's sum before display rounding, so a hand total of the rounded products can differ by about 0.001.</p>
  ${hiddenBlock(selected.explanation)}
  ${outputBlock(selected.explanation, measure)}
  <h2>Weights</h2>
  <p>These weights are the same for every patient on the panel. A negative weight pulls the unit down when that input is high.</p>
  <h3>Hidden layer</h3>
  ${matrixTable(["Unit", ...INPUTS.map((input) => input.label), "Bias"], HIDDEN.map((unit, index) => [unit.label, ...W1[index].map(weightText), B1[index].toFixed(2)]))}
  <h3>Output layer</h3>
  ${matrixTable(["Output", ...HIDDEN.map((unit) => unit.label), "Bias"], [
    ["Change in severity", ...W2[0].map(weightText), B2[0].toFixed(2)],
    ["Response", ...W2[1].map(weightText), B2[1].toFixed(2)],
  ])}
  <h2>The same arithmetic for the other options</h2>
  ${ranked
    .filter((item) => item.treatment.id !== selected.treatment.id)
    .map(
      (item) => `<details>
        <summary>${esc(item.treatment.name)}: ${esc(formatMeasure(item.explanation.futureNative, measure))}, ${pct(item.explanation.probability)}%</summary>
        ${hiddenBlock(item.explanation)}
        ${outputBlock(item.explanation, measure)}
      </details>`,
    )
    .join("")}
  <p class="meta">Generated from Meridian Clinical for a conversation with the patient. Panel date ${esc(formatDate(latest.date))}.</p>
</main>
</body>
</html>`
}

function chartSvg(
  chart: ReturnType<typeof layoutChart>,
  recordedEnd: { x: number; y: number } | undefined,
  projectedEnd: { x: number; y: number } | undefined,
  goalLabel: string,
): string {
  const axis = `M ${chart.plot.left} ${chart.plot.top} V ${chart.height - chart.plot.bottom} H ${chart.width - chart.plot.right}`
  const goal = chart.goal
    ? `<line x1="${chart.plot.left}" y1="${chart.goal.y.toFixed(1)}" x2="${chart.width - chart.plot.right}" y2="${chart.goal.y.toFixed(1)}" stroke="#3f5d1b" stroke-width="2" stroke-dasharray="2 4" />
       <text x="${chart.plot.left + 6}" y="${(chart.goal.y - 6).toFixed(1)}" fill="#3f5d1b" font-size="13">Goal ${esc(goalLabel)}</text>`
    : ""
  return `<svg viewBox="0 0 ${chart.width} ${chart.height}" role="img" aria-label="Recorded results as a solid line, projection as a dashed line, and the goal as a dotted line. Higher is worse.">
    <path d="${axis}" fill="none" stroke="#c2c6b6" stroke-width="1.5" />
    ${chart.yTicks.map((tick) => `<text x="4" y="${(tick.y + 4).toFixed(1)}" fill="#1b2016" font-size="13">${esc(tick.label)}</text>`).join("")}
    ${chart.xTicks.map((tick) => `<text x="${tick.x.toFixed(1)}" y="${chart.height - 8}" text-anchor="middle" fill="#1b2016" font-size="13">${esc(tick.label)}</text>`).join("")}
    ${goal}
    <polyline points="${pointsAttr(chart.recorded)}" fill="none" stroke="#1b2016" stroke-width="2.5" />
    <polyline points="${pointsAttr(chart.projected)}" fill="none" stroke="#91550d" stroke-width="2.5" stroke-dasharray="6 4" />
    ${recordedEnd ? `<circle cx="${recordedEnd.x.toFixed(1)}" cy="${recordedEnd.y.toFixed(1)}" r="4" fill="#1b2016" />` : ""}
    ${projectedEnd ? `<circle cx="${projectedEnd.x.toFixed(1)}" cy="${projectedEnd.y.toFixed(1)}" r="4" fill="#91550d" />` : ""}
  </svg>`
}

function diagramSvg(selected: string): string {
  const diagram = layoutDiagram()
  const edges = diagram.edges
    .map((edge) => {
      const hot = edge.from === selected || edge.to === selected
      const color = edge.weight < 0 ? "#3d5a6c" : "#4f6420"
      return `<line x1="${edge.x1}" y1="${edge.y1}" x2="${edge.x2}" y2="${edge.y2}" stroke="${color}" stroke-width="${edgeWidth(edge.weight).toFixed(2)}" opacity="${hot ? 0.95 : 0.28}" />`
    })
    .join("")
  const nodes = diagram.nodes
    .map((node) => {
      const on = node.id === selected
      const chip = node.anchor === "middle" ? `<rect x="${node.textX - 62}" y="${node.textY - 12}" width="124" height="16" fill="#ffffff" />` : ""
      return `<circle cx="${node.x}" cy="${node.y}" r="16" fill="#fff" stroke="${on ? "#1b2016" : "#c2c6b6"}" stroke-width="${on ? 3 : 1.5}" />
        ${chip}
        <text x="${node.textX}" y="${node.textY}" text-anchor="${node.anchor}" fill="#1b2016" font-size="13">${esc(node.label)}</text>`
    })
    .join("")
  return `<svg viewBox="0 0 ${diagram.width} ${diagram.height}" role="img" aria-label="Network with six inputs, four hidden units, and two outputs. Ochre is a positive weight and blue is a negative weight. Thicker lines are larger weights.">
    ${edges}${nodes}
  </svg>`
}

function hiddenBlock(explanation: Explanation): string {
  return explanation.hidden
    .map((step) => {
      const on = step.activation ?? 0
      return `<h3>${esc(step.label)}</h3>
        <p class="eq">z = ${esc(sumLine(step.bias, step.terms))}</p>
        <p>Model total z = ${step.z.toFixed(4)}. Activation = max(0, z) = ${on.toFixed(4)}. ${on > 0 ? "This unit is on." : "This unit is off, so it adds nothing downstream."}</p>
        ${termTable(step)}`
    })
    .join("")
}

function outputBlock(explanation: Explanation, measure: Measure): string {
  const change = explanation.change
  const response = explanation.response
  return `<h3>Change in severity</h3>
    <p class="eq">change = ${esc(sumLine(change.bias, change.terms))}</p>
    <p>Model total = ${change.z.toFixed(4)}. Future severity = clamp(today's severity + change, 0, 1) = clamp(${explanation.severity.toFixed(4)} + ${change.z.toFixed(4)}, 0, 1) = ${explanation.futureSeverity.toFixed(4)}.</p>
    <p>That scales back to ${esc(formatMeasure(explanation.futureNative, measure))}. Today's recorded result is ${esc(formatMeasure(explanation.currentNative, measure))}.</p>
    ${termTable(change)}
    <h3>Chance of a response</h3>
    <p class="eq">logit = ${esc(sumLine(response.bias, response.terms))}</p>
    <p>Model total = ${response.z.toFixed(4)}. Chance = 1 / (1 + e<sup>−logit</sup>) = ${pct(explanation.probability)}%.</p>
    ${termTable(response)}`
}

function termTable(step: UnitStep): string {
  const rows = step.terms.map(
    (term) =>
      `<tr><th scope="row">${esc(term.label)}</th><td class="num">${weightText(term.weight)}</td><td class="num">${term.value.toFixed(4)}</td><td class="num">${term.product.toFixed(4)}</td></tr>`,
  )
  rows.push(`<tr><th scope="row">Bias</th><td class="num">${step.bias.toFixed(2)}</td><td class="num"></td><td class="num">${step.bias.toFixed(2)}</td></tr>`)
  rows.push(`<tr><th scope="row">Total</th><td></td><td></td><td class="num">${step.z.toFixed(4)}</td></tr>`)
  return `<table>
    <thead><tr><th scope="col">Term</th><th scope="col" class="num">Weight</th><th scope="col" class="num">Value</th><th scope="col" class="num">Product</th></tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`
}

function valueTable(rows: [string, string][]): string {
  return `<table>
    <thead><tr><th scope="col">Date</th><th scope="col" class="num">Value</th></tr></thead>
    <tbody>${rows.map(([date, value]) => `<tr><th scope="row">${esc(date)}</th><td class="num">${esc(value)}</td></tr>`).join("")}</tbody>
  </table>`
}

function matrixTable(headers: string[], rows: string[][]): string {
  return `<table>
    <thead><tr>${headers.map((header, index) => `<th scope="col"${index ? ` class="num"` : ""}>${esc(header)}</th>`).join("")}</tr></thead>
    <tbody>${rows
      .map(
        (row) =>
          `<tr>${row.map((cell, index) => (index === 0 ? `<th scope="row">${esc(cell)}</th>` : `<td class="num">${esc(cell)}</td>`)).join("")}</tr>`,
      )
      .join("")}</tbody>
  </table>`
}

function row(label: string, value: string): string {
  return `<tr><th scope="row">${esc(label)}</th><td>${esc(value)}</td></tr>`
}

function weightText(weight: number): string {
  return weight.toFixed(2)
}

function pct(probability: number): string {
  return Math.round(probability * 100).toString()
}

function esc(value: string): string {
  return value.replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char] ?? char)
}
