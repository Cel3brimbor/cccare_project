import { diagramUnits, edgeWidth, layoutDiagram } from "./figures"
import { formatGoal, formatMeasure } from "./format"
import { B1, B2, HIDDEN, INPUTS, W1, W2, sumLine } from "./model"
import type { Projection, UnitStep } from "./model"
import type { Measure } from "./types"

export function NetworkPanel({
  projection,
  measure,
  unitId,
  onUnit,
}: {
  projection: Projection
  measure: Measure
  unitId: string
  onUnit: (id: string) => void
}) {
  const diagram = layoutDiagram()
  const explanation = projection.explanation
  const unit = diagramUnits.find((item) => item.id === unitId) ?? diagramUnits[0]
  const groups = ["Inputs", "Hidden units", "Outputs"]

  return (
    <section className="c-sheet c-network-sheet" aria-labelledby="network-heading">
      <h2 id="network-heading">How the projection is worked out</h2>
      <p className="c-meta c-measure">
        The projection is this network, not a separate score. Six inputs, four hidden units, two outputs. A hidden unit
        is on only when its weighted sum is above zero. The weights are fixed for every patient. Medicines listed on the
        chart are not inputs. Only the six numbers below are.
      </p>
      <div className="c-network">
        <div className="c-diagram" tabIndex={0} role="region" aria-label="Network diagram. Scroll sideways if the figure is clipped.">
          <svg viewBox={`0 0 ${diagram.width} ${diagram.height}`} role="img" aria-label="Six inputs connect to four hidden units, which connect to a change in severity and a response chance. Ochre lines are positive weights. Blue lines are negative weights. Thicker lines are larger weights.">
            {diagram.edges.map((edge) => {
              const hot = edge.from === unit.id || edge.to === unit.id
              return (
                <line
                  key={`${edge.from}-${edge.to}`}
                  x1={edge.x1}
                  y1={edge.y1}
                  x2={edge.x2}
                  y2={edge.y2}
                  strokeWidth={edgeWidth(edge.weight)}
                  className={edge.weight < 0 ? "c-edge neg" : "c-edge pos"}
                  opacity={hot ? 0.95 : 0.28}
                />
              )
            })}
            {diagram.nodes.map((node) => (
              <g key={node.id} className="c-node-hit" onClick={() => onUnit(node.id)}>
                <title>Inspect {node.label}</title>
                <circle cx={node.x} cy={node.y} r={16} className={node.id === unit.id ? "c-node on" : "c-node"} />
                {node.anchor === "middle" ? (
                  <rect x={node.textX - 62} y={node.textY - 12} width={124} height={16} className="c-label-chip" />
                ) : null}
                <text x={node.textX} y={node.textY} textAnchor={node.anchor} className="c-node-label">
                  {node.label}
                </text>
              </g>
            ))}
          </svg>
          <ul className="c-legend">
            <li>
              <i className="c-swatch pos" aria-hidden="true" /> Positive weight
            </li>
            <li>
              <i className="c-swatch neg" aria-hidden="true" /> Negative weight
            </li>
            <li>Thicker lines carry larger weights. Select a node to inspect it.</li>
          </ul>
        </div>
        <div className="c-inspect">
          <label className="c-field" htmlFor="inspect-unit">
            <span>Inspect a unit</span>
            <select id="inspect-unit" value={unit.id} onChange={(event) => onUnit(event.target.value)}>
              {groups.map((group) => (
                <optgroup key={group} label={group}>
                  {diagramUnits
                    .filter((item) => item.group === group)
                    .map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.label}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          <h3>{unit.label}</h3>
          <p>{unit.detail}</p>
          <UnitMath unitId={unit.id} projection={projection} measure={measure} />
        </div>
      </div>
      <details className="c-disclose">
        <summary>Inputs and weights for {projection.treatment.name}</summary>
        <h3>Inputs used for {projection.treatment.name}</h3>
        <div className="c-table-wrap">
          <table className="c-table">
            <thead>
              <tr>
                <th scope="col">Input</th>
                <th scope="col" className="c-num">
                  Value
                </th>
                <th scope="col">Meaning</th>
              </tr>
            </thead>
            <tbody>
              {explanation.inputs.map((input) => (
                <tr key={input.label} className={unit.label === input.label ? "is-on" : undefined}>
                  <th scope="row">{input.label}</th>
                  <td className="c-num">{input.value.toFixed(4)}</td>
                  <td>{input.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h3>Hidden-layer weights</h3>
        <p>Rows are units. Columns are inputs. The bias is added after the products.</p>
        <WeightTable
          caption="Hidden layer weights"
          headers={["Unit", ...INPUTS.map((input) => input.label), "Bias"]}
          rows={HIDDEN.map((hidden, index) => ({
            id: `hid-${index}`,
            label: hidden.label,
            values: [...W1[index], B1[index]],
          }))}
          selected={unit.id}
        />
        <h3>Output weights</h3>
        <WeightTable
          caption="Output weights"
          headers={["Output", ...HIDDEN.map((hidden) => hidden.label), "Bias"]}
          rows={[
            { id: "out-0", label: "Change in severity", values: [...W2[0], B2[0]] },
            { id: "out-1", label: "Response", values: [...W2[1], B2[1]] },
          ]}
          selected={unit.id}
        />
      </details>
    </section>
  )
}

function UnitMath({
  unitId,
  projection,
  measure,
}: {
  unitId: string
  projection: Projection
  measure: Measure
}) {
  const explanation = projection.explanation
  if (unitId.startsWith("in-")) {
    const input = explanation.inputs[Number(unitId.slice(3))]
    if (!input) return null
    return (
      <p className="c-eq">
        {input.label} = {input.value.toFixed(4)}
      </p>
    )
  }
  if (unitId.startsWith("hid-")) {
    const step = explanation.hidden[Number(unitId.slice(4))]
    if (!step) return null
    return <StepMath step={step} symbol="z" showActivation />
  }
  if (unitId === "out-0") {
    return (
      <>
        <StepMath step={explanation.change} symbol="change" />
        <p>
          Future severity = clamp({explanation.severity.toFixed(4)} + {explanation.delta.toFixed(4)}, 0, 1) ={" "}
          {explanation.futureSeverity.toFixed(4)}. That is {formatMeasure(explanation.futureNative, measure)} in 12 weeks.
          Today is {formatMeasure(explanation.currentNative, measure)}.
        </p>
      </>
    )
  }
  return (
    <>
      <StepMath step={explanation.response} symbol="logit" />
      <p>
        Chance = 1 / (1 + e<sup>−logit</sup>) = {Math.round(explanation.probability * 100)}%. Response means the result
        falls by at least {formatGoal(measure)}.
      </p>
    </>
  )
}

function StepMath({
  step,
  symbol,
  showActivation = false,
}: {
  step: UnitStep
  symbol: string
  showActivation?: boolean
}) {
  return (
    <>
      <p className="c-eq">
        {symbol} = {sumLine(step.bias, step.terms)}
      </p>
      <p>
        Model total = {step.z.toFixed(4)}.
        {showActivation ? ` Activation = max(0, z) = ${(step.activation ?? 0).toFixed(4)}.` : null}
      </p>
      <div className="c-table-wrap">
        <table className="c-table">
          <caption className="sr">{step.label} arithmetic</caption>
          <thead>
            <tr>
              <th scope="col">Term</th>
              <th scope="col" className="c-num">Weight</th>
              <th scope="col" className="c-num">Value</th>
              <th scope="col" className="c-num">Product</th>
            </tr>
          </thead>
          <tbody>
            {step.terms.map((term) => (
              <tr key={term.label}>
                <th scope="row">{term.label}</th>
                <td className={tone(term.weight)}>{term.weight.toFixed(2)}</td>
                <td className="c-num">{term.value.toFixed(4)}</td>
                <td className="c-num">{term.product.toFixed(4)}</td>
              </tr>
            ))}
            <tr>
              <th scope="row">Bias</th>
              <td className={tone(step.bias)}>{step.bias.toFixed(2)}</td>
              <td />
              <td className="c-num">{step.bias.toFixed(2)}</td>
            </tr>
            <tr>
              <th scope="row">Total</th>
              <td />
              <td />
              <td className="c-num">{step.z.toFixed(4)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </>
  )
}

function WeightTable({
  caption,
  headers,
  rows,
  selected,
}: {
  caption: string
  headers: string[]
  rows: { id: string; label: string; values: number[] }[]
  selected: string
}) {
  return (
    <div className="c-table-wrap">
      <table className="c-table">
        <caption className="sr">{caption}</caption>
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={header} scope="col" className={index ? "c-num" : undefined}>
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id} className={row.id === selected ? "is-on" : undefined}>
              <th scope="row">{row.label}</th>
              {row.values.map((value, index) => (
                <td key={`${row.id}-${index}`} className={tone(value)}>
                  {value.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function tone(weight: number): string {
  if (weight < 0) return "c-num c-neg"
  if (weight > 0) return "c-num c-pos"
  return "c-num"
}
