import { useEffect, useRef, useState } from "react"
import { Formula } from "./Formula"
import { Heatmap } from "./Heatmap"
import { columnForUnit, dotPrefix, exampleBackward, meanPrefix } from "./linalg"
import { formatCompact, formatNum, imageSide, layerTitles } from "./format"
import { usePrefers } from "./media"
import { FilterBank, PixelGrid } from "./PixelGrid"
import { Segmented, segmentTabId } from "./Segmented"
import { send, useStudio } from "./store"
import type { AffineLayer, DigitDataset, ReviewTrace, Selection, Trace } from "./types"

const tabs = ["Forward", "Weights", "Backward", "Update", "Data"] as const
type Tab = (typeof tabs)[number]

export function Inspector({
  selection,
  onSelect,
}: {
  selection: Selection
  onSelect: (selection: Selection) => void
}) {
  const studio = useStudio()
  const [tab, setTab] = useState<Tab>("Forward")
  const digits = studio.task === "digits"
  const side = studio.dataset?.task === "digits" ? studio.dataset.width : 28
  const titles = layerTitles(studio.config.hidden.length, studio.task)
  const widths = digits ? [side * side, ...studio.config.hidden, 10] : [studio.config.embedDim, ...studio.config.hidden, 1]
  const layer = Math.min(selection.layer, titles.length - 1)
  const unit = Math.min(selection.unit, Math.max(0, widths[layer] - 1))
  const trace = studio.trace && studio.trace.task === studio.task ? studio.trace : null
  const affine = layer > 0 ? trace?.layers[layer - 1] : undefined

  return (
    <aside className="inspector">
      <div className="inspector-head">
        <Segmented
          label="Layer"
          className="layer-switch"
          value={String(layer)}
          options={titles.map((title, index) => ({ id: String(index), label: title }))}
          onChange={(id) => onSelect({ layer: Number(id), unit: 0 })}
        />
        <Segmented
          label="Lesson"
          mode="tabs"
          group="lesson"
          controls="lesson-panel"
          value={tab}
          options={tabs.map((name) => ({ id: name, label: name }))}
          onChange={(id) => setTab(id as Tab)}
        />
      </div>
      <div className="inspector-body" id="lesson-panel" role="tabpanel" aria-labelledby={segmentTabId("lesson", tab)}>
        {tab === "Forward" && (
          <Forward trace={trace} layer={layer} unit={unit} affine={affine} widths={widths} digits={digits} onSelect={onSelect} />
        )}
        {tab === "Weights" && <Weights trace={trace} layer={layer} unit={unit} affine={affine} digits={digits} onSelect={onSelect} />}
        {tab === "Backward" && (
          <Backward trace={trace} layer={layer} unit={unit} affine={affine} digits={digits} trainedSteps={studio.status.trainedSteps} onSelect={onSelect} />
        )}
        {tab === "Update" && (
          <Update trace={trace} layer={layer} affine={affine} digits={digits} trainedSteps={studio.status.trainedSteps} />
        )}
        {tab === "Data" && <Data onEval={() => send({ type: "eval" })} evaluating={studio.status.phase === "evaluating"} />}
      </div>
    </aside>
  )
}

function Forward({
  trace,
  layer,
  unit,
  affine,
  widths,
  digits,
  onSelect,
}: {
  trace: Trace | null
  layer: number
  unit: number
  affine: AffineLayer | undefined
  widths: number[]
  digits: boolean
  onSelect: (selection: Selection) => void
}) {
  if (layer === 0) {
    if (digits) return <PixelsForward trace={trace} unit={unit} onSelect={onSelect} />
    return <PoolForward trace={trace?.task === "reviews" ? trace : null} unit={unit} width={widths[0]} onSelect={onSelect} />
  }
  const input = affine?.input.length ?? widths[layer - 1]
  const output = affine?.output.length ?? widths[layer]
  const activation = affine?.activation ?? (layer === widths.length - 1 ? (digits ? "softmax" : "logit") : "relu")
  return (
    <section>
      <Formula tex={`z = hW + b`} />
      <p className="lesson">
        Unit {digits && layer === widths.length - 1 ? unit : unit + 1} is the dot product of h with column {unit + 1} of W, plus a
        bias. h is 1×{input}, W is {input}×{output}.
      </p>
      {activation === "relu" ? (
        <Formula tex={`a = \\max(0, z)`} />
      ) : activation === "softmax" ? (
        <>
          <Formula tex={`p_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}`} />
          <Formula tex={`L = -\\log p_y`} />
          <p className="lesson">Ten logits become a probability over the digits. The loss is the negative log of the true digit’s probability.</p>
        </>
      ) : (
        <>
          <Formula tex={`\\hat{y} = \\sigma(z) = \\frac{1}{1+e^{-z}}`} />
          <Formula tex={`L = \\log(1+e^{z}) - yz`} />
          <p className="lesson">That loss is binary cross-entropy on the logit. Its derivative in z is the probability minus the label.</p>
        </>
      )}
      {affine ? (
        <AffineProduct affine={affine} unit={unit} onSelect={(next) => onSelect({ layer, unit: next })} />
      ) : (
        <p className="lesson">{digits ? "Train or step once to fill these matrices with a real digit." : "Train or step once to fill these matrices with a real review."}</p>
      )}
    </section>
  )
}

function PixelsForward({
  trace,
  unit,
  onSelect,
}: {
  trace: Trace | null
  unit: number
  onSelect: (selection: Selection) => void
}) {
  const digit = trace?.task === "digits" ? trace : null
  const side = digit?.width ?? 28
  const safe = Math.min(unit, side * side - 1)
  const row = Math.floor(safe / side) + 1
  const col = (safe % side) + 1
  return (
    <section>
      <Formula tex={`x \\in [0,1]^{${side} \\times ${side}}`} />
      <p className="lesson">
        The input is the image itself, flattened row by row. Pixel {row}, {col} is {formatNum(digit?.pixels[safe] ?? 0, 3)}. Bright
        ink is 1, the background is 0.
      </p>
      {digit && (
        <PixelGrid
          values={digit.pixels}
          side={side}
          mode="ink"
          selected={safe}
          onSelect={(index) => onSelect({ layer: 0, unit: index })}
        />
      )}
    </section>
  )
}

function PoolForward({
  trace,
  unit,
  width,
  onSelect,
}: {
  trace: ReviewTrace | null
  unit: number
  width: number
  onSelect: (selection: Selection) => void
}) {
  const rows = trace?.gathered ?? []
  const total = Math.max(1, rows.length)
  const [k, setK] = useState(0)
  const shown = Math.min(k, total)
  const partial = trace ? meanPrefix(rows, unit, shown, total) : { sum: 0, term: 0 }
  return (
    <section>
      <Formula tex={`e_t = E_{x_t}, \\qquad \\bar{e} = \\frac{1}{T}\\sum_{t=1}^{T} e_t`} />
      <p className="lesson">
        Each token gathers one row of the embedding. The pool is their average, a vector of length {width}. Padding is left out,
        so T is {trace ? total : "the number of real tokens"}.
      </p>
      {trace && (
        <>
          <MatrixBlock label="Gathered rows" matrix={rows.slice(0, 48)} selected={unit} cursor={shown > 0 ? Math.min(shown - 1, 47) : null} onSelect={(column) => onSelect({ layer: 0, unit: column })} />
          {rows.length > 48 && <p className="fine">Showing 48 of {rows.length} token rows. The average below uses every row.</p>}
          <Scrubber length={total} value={shown} onChange={setK} />
          <Arithmetic
            lines={[
              `T = ${total}`,
              shown > 0 ? `term ${shown} / T = ${formatNum(partial.term, 4)}` : "Scrub to add token rows",
              `mean so far = ${formatNum(partial.sum, 4)}`,
              `pool coordinate = ${formatNum(trace.pooled[unit] ?? 0, 4)}`,
            ]}
          />
        </>
      )}
    </section>
  )
}

function AffineProduct({
  affine,
  unit,
  onSelect,
}: {
  affine: AffineLayer
  unit: number
  onSelect: (unit: number) => void
}) {
  const length = affine.input.length
  const [k, setK] = useState(length)
  const safeUnit = Math.min(unit, affine.output.length - 1)
  const weights = columnForUnit(affine, safeUnit)
  const localCol = affine.colIndex ? affine.colIndex.indexOf(safeUnit) : safeUnit
  const selectColumn = (next: number) => onSelect(affine.colIndex ? (affine.colIndex[next] ?? safeUnit) : next)
  const partial = dotPrefix(affine.input, weights ?? [], Math.min(k, length))
  const bias = affine.bias[safeUnit] ?? 0
  const computed = partial.sum + (k >= length ? bias : 0)
  const stored = affine.pre[safeUnit] ?? 0
  const playingLength = length
  const side = imageSide(length)
  const cursor = k > 0 ? Math.min(k, length) - 1 : null
  return (
    <>
      {weights && side ? (
        <>
          <figure className="matrix-block">
            <figcaption>h, the pixels</figcaption>
            <PixelGrid values={affine.input} side={side} mode="ink" selected={cursor} />
          </figure>
          <figure className="matrix-block">
            <figcaption>Column {safeUnit + 1} of W</figcaption>
            <PixelGrid values={weights} side={side} mode="diverging" selected={cursor} />
          </figure>
        </>
      ) : weights ? (
        <>
          <MatrixBlock label="h" matrix={[affine.input]} selected={Math.min(k, length) - 1} cursor={null} mode="diverging" />
          <MatrixBlock
            label="W"
            matrix={affine.weight}
            selected={localCol >= 0 ? localCol : null}
            cursor={cursor}
            onSelect={selectColumn}
          />
        </>
      ) : (
        <p className="fine">
          This unit sits outside the drawn sample of the {length}×{affine.output.length} matrix. z = {formatNum(stored, 4)}.
        </p>
      )}
      <MatrixBlock label="b" matrix={[affine.bias]} selected={safeUnit} cursor={null} onSelect={onSelect} />
      <MatrixBlock
        label={affine.activation === "relu" ? "a = ReLU(z)" : affine.activation === "softmax" ? "logits z" : "z"}
        matrix={[affine.activation === "relu" ? affine.output : affine.pre]}
        selected={safeUnit}
        cursor={null}
        mode={affine.activation === "relu" ? "magnitude" : "diverging"}
        onSelect={onSelect}
      />
      {affine.activation === "softmax" && (
        <MatrixBlock label="softmax p" matrix={[affine.output]} selected={safeUnit} cursor={null} mode="magnitude" onSelect={onSelect} />
      )}
      {weights && (
        <>
          <Scrubber length={playingLength} value={Math.min(k, playingLength)} onChange={setK} />
          <Arithmetic
            lines={[
              k > 0 && k <= length
                ? `h_${k} W_${k},${safeUnit + 1} = ${formatNum(affine.input[k - 1], 4)} × ${formatNum(weights[k - 1], 4)} = ${formatNum(partial.term, 4)}`
                : "Scrub along the dot product",
              `running sum = ${formatNum(partial.sum, 4)}`,
              `bias = ${formatNum(bias, 4)}`,
              `z on screen = ${formatNum(k >= length ? computed : partial.sum, 4)}`,
              Math.abs(computed - stored) > 0.002 && k >= length ? `stored forward = ${formatNum(stored, 4)}` : "",
            ].filter(Boolean)}
          />
        </>
      )}
    </>
  )
}

function Weights({
  trace,
  layer,
  unit,
  affine,
  digits,
  onSelect,
}: {
  trace: Trace | null
  layer: number
  unit: number
  affine: AffineLayer | undefined
  digits: boolean
  onSelect: (selection: Selection) => void
}) {
  if (layer === 0) {
    if (digits) {
      return (
        <section>
          <h3>Pixels</h3>
          <p className="lesson">These intensities are the input, not learned weights. Learning starts at the first hidden layer.</p>
          {trace?.task === "digits" && (
            <PixelGrid
              values={trace.pixels}
              side={trace.width}
              mode="ink"
              selected={Math.min(unit, trace.pixels.length - 1)}
              onSelect={(index) => onSelect({ layer: 0, unit: index })}
            />
          )}
        </section>
      )
    }
    const embed = trace?.task === "reviews" ? trace.embed : undefined
    return (
      <section>
        <h3>Embedding table</h3>
        <p className="lesson">
          {embed ? `${embed.vocab.toLocaleString()} × ${embed.dim}` : "Vocabulary × dimensions"}. A token selects a row. The canvas
          never draws those rows as neurons; it draws the pooled vector.
        </p>
        {embed && (
          <>
            <Histogram counts={embed.normHist.counts} edges={embed.normHist.edges} />
            <Stats
              rows={[
                ["Mean", formatNum(embed.weightMean, 4)],
                ["Std", formatNum(embed.weightStd, 4)],
                ["Row RMS", formatNum(embed.rowRms, 4)],
                ["Batch grad RMS", formatNum(embed.batchGradRms, 4)],
              ]}
            />
          </>
        )}
      </section>
    )
  }
  if (!affine) return <p className="lesson">Step once to see this weight matrix.</p>
  const safeUnit = Math.min(unit, affine.bias.length - 1)
  const filters = columnForUnit(affine, safeUnit)
  const side = filters ? imageSide(filters.length) : null
  const localCol = affine.colIndex ? affine.colIndex.indexOf(safeUnit) : safeUnit
  const sampled = Boolean(affine.rowIndex || affine.colIndex)
  return (
    <section>
      <h3>
        {affine.title} · {affine.input.length} × {affine.output.length}
      </h3>
      <p className="lesson">
        {side
          ? "Each hidden unit looks at the whole image. Its column of W, drawn as pixels, is the pattern it adds up."
          : "Each column is one unit’s incoming weights. A line on the canvas is one of these cells."}
        {sampled ? " Wider layers are drawn from an even sample of units." : ""}
      </p>
      {side && filters ? (
        <>
          <PixelGrid values={filters} side={side} mode="diverging" />
          <FilterBank
            weight={affine.weight}
            side={side}
            selected={safeUnit}
            units={affine.colIndex}
            onSelect={(next) => onSelect({ layer, unit: next })}
          />
        </>
      ) : (
        <MatrixBlock
          label="W"
          matrix={affine.weight}
          selected={localCol >= 0 ? localCol : null}
          cursor={null}
          onSelect={(next) => onSelect({ layer, unit: affine.colIndex ? (affine.colIndex[next] ?? safeUnit) : next })}
        />
      )}
      <MatrixBlock label="b" matrix={[affine.bias]} selected={safeUnit} cursor={null} onSelect={(next) => onSelect({ layer, unit: next })} />
      <Stats
        rows={[
          ["Mean", formatNum(affine.weightMean, 4)],
          ["Std", formatNum(affine.weightStd, 4)],
          ["Bias", formatNum(affine.bias[safeUnit], 4)],
          ["Batch grad RMS", formatNum(affine.batchGradRms, 4)],
          ["Bias grad RMS", formatNum(affine.biasGradRms, 4)],
        ]}
      />
    </section>
  )
}

function Backward({
  trace,
  layer,
  unit,
  affine,
  digits,
  trainedSteps,
  onSelect,
}: {
  trace: Trace | null
  layer: number
  unit: number
  affine: AffineLayer | undefined
  digits: boolean
  trainedSteps: number
  onSelect: (selection: Selection) => void
}) {
  const grads = trace ? exampleBackward(trace) : null
  const noun = digits ? "digit" : "review"
  if (!trace) return <p className="lesson">A backward pass needs a labeled {noun} from a training step.</p>
  if (!grads) {
    return <p className="lesson">A probe has no label, so there is no loss to differentiate. Step the trainer to see a backward pass.</p>
  }
  if (layer === 0) {
    const incoming = grads[0]?.dInput ?? []
    if (trace.task === "digits") {
      return (
        <section>
          <Formula tex={`\\frac{\\partial L}{\\partial x} = W^{\\top}\\delta`} />
          <p className="lesson">The loss reaches every pixel through the first weight matrix. Bright and dark here are the two signs of that gradient.</p>
          <PixelGrid
            values={incoming}
            side={trace.width}
            mode="diverging"
            selected={Math.min(unit, incoming.length - 1)}
            onSelect={(index) => onSelect({ layer: 0, unit: index })}
          />
          <p className="fine">Pixel {Math.floor(unit / trace.width) + 1}, {(unit % trace.width) + 1} is credited {formatNum(incoming[unit] ?? 0, 4)}.</p>
        </section>
      )
    }
    const share = incoming.map((value) => value / Math.max(1, trace.gathered.length))
    return (
      <section>
        <Formula tex={`\\frac{\\partial L}{\\partial e_t} = \\frac{1}{T}\\frac{\\partial L}{\\partial \\bar{e}}`} />
        <p className="lesson">
          The pool is a mean, so each gathered row receives the same slice of the upstream gradient, divided by T = {trace.gathered.length}.
        </p>
        <MatrixBlock label="Gradient into the pool" matrix={[incoming]} selected={Math.min(unit, incoming.length - 1)} cursor={null} />
        <p className="fine">Each token row is credited {formatNum(share[unit] ?? 0, 4)} on coordinate {unit + 1}.</p>
        <BatchNote rms={trace.embed.batchGradRms} trainedSteps={trainedSteps} noun="review" />
      </section>
    )
  }
  if (!affine) return null
  const grad = grads[layer - 1]
  const safeUnit = Math.min(unit, grad.delta.length - 1)
  const upstream = layer === trace.layers.length ? null : grad.delta[safeUnit]
  return (
    <section>
      <Formula tex={`\\frac{\\partial L}{\\partial W} = h^{\\top}\\delta, \\qquad \\frac{\\partial L}{\\partial h} = W\\delta`} />
      {affine.activation === "relu" ? (
        <Formula tex={`\\delta = \\frac{\\partial L}{\\partial a} \\odot \\mathbf{1}_{z > 0}`} />
      ) : affine.activation === "softmax" ? (
        <Formula tex={`\\delta = p - y`} />
      ) : (
        <Formula tex={`\\delta = \\sigma(z) - y`} />
      )}
      <p className="lesson">
        {affine.activation === "logit"
          ? `δ = ${formatNum(trace.prob, 4)} − ${trace.label} = ${formatNum(grad.delta[0] ?? 0, 4)}.`
          : affine.activation === "softmax"
            ? `Digit ${safeUnit} has δ = ${formatNum(grad.delta[safeUnit] ?? 0, 4)}, its probability minus ${trace.label === safeUnit ? "1" : "0"}.`
            : `Unit ${safeUnit + 1} has δ = ${formatNum(grad.delta[safeUnit] ?? 0, 4)}${affine.pre[safeUnit] > 0 ? "" : ", blocked by ReLU"}.`}
      </p>
      {imageSide(affine.input.length) ? (
        <figure className="matrix-block">
          <figcaption>This {noun}’s ∂L/∂W, column {safeUnit + 1}</figcaption>
          <PixelGrid
            values={affine.input.map((value) => value * (grad.delta[safeUnit] ?? 0))}
            side={imageSide(affine.input.length) ?? 28}
            mode="diverging"
          />
        </figure>
      ) : grad.gradW.length ? (
        <MatrixBlock label={`This ${noun}’s ∂L/∂W`} matrix={grad.gradW} selected={safeUnit} cursor={null} />
      ) : (
        <MatrixBlock
          label={`This ${noun}’s ∂L/∂W, column ${safeUnit + 1}`}
          matrix={[affine.input.map((value) => value * (grad.delta[safeUnit] ?? 0))]}
          selected={null}
          cursor={null}
        />
      )}
      <Stats
        rows={[
          ["δ", formatNum(grad.delta[safeUnit], 4)],
          ["∂L/∂b", formatNum(grad.gradB[safeUnit], 4)],
          ["Upstream sample", formatNum(upstream, 4)],
        ]}
      />
      <BatchNote rms={affine.batchGradRms} trainedSteps={trainedSteps} noun={noun} />
    </section>
  )
}

function Update({
  trace,
  layer,
  affine,
  digits,
  trainedSteps,
}: {
  trace: Trace | null
  layer: number
  affine: AffineLayer | undefined
  digits: boolean
  trainedSteps: number
}) {
  const { config } = useStudio()
  const optimizer = trace?.optimizer
  if (digits && layer === 0) {
    return (
      <section>
        <p className="lesson">Pixels are data, not parameters. The learning step is on the layers to the right.</p>
      </section>
    )
  }
  const target = layer === 0 && trace?.task === "reviews" ? trace.embed : affine
  const name = config.optimizer
  const decay = config.weightDecay
  return (
    <section>
      {name === "adam" ? (
        <>
          <Formula tex={`m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t`} />
          <Formula tex={`v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2`} />
          <Formula tex={`\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}, \\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}`} />
          <Formula
            tex={
              decay > 0
                ? `W \\leftarrow (1-\\eta\\lambda)W - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}`
                : `W \\leftarrow W - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}`
            }
          />
          <p className="lesson">
            Adam keeps a moving average of the gradient and of its square, then steps in that direction, scaled down where the
            gradient has been noisy.
            {decay > 0 ? " λ is the L2 weight decay, applied to the weights before that step." : ""}
          </p>
        </>
      ) : (
        <>
          <Formula tex={decay > 0 ? `W \\leftarrow W - \\eta (g + \\lambda W)` : `W \\leftarrow W - \\eta g`} />
          <p className="lesson">
            SGD moves every weight by the learning rate times its gradient. One rate, shared by the whole matrix.
            {decay > 0 ? " λ adds an L2 pull toward zero." : ""}
          </p>
        </>
      )}
      <Stats
        rows={[
          ["Rule", name === "adam" ? "Adam" : "SGD"],
          ["Learning rate", formatCompact(config.lr)],
          ["L2", formatCompact(decay)],
          ["Optimizer step", optimizer ? String(optimizer.step) : "—"],
          ["β1, β2, ε", `${formatCompact(config.beta1)}, ${formatCompact(config.beta2)}, ${formatCompact(config.eps)}`],
          ["Batch grad RMS", formatNum(target?.batchGradRms, 4)],
          ["Moment RMS", formatNum(target?.momentRms, 4)],
          ["Second moment RMS", formatNum(target?.secondMomentRms, 4)],
        ]}
      />
      {trainedSteps > 0 && (optimizer?.step ?? 0) === 0 && (
        <p className="fine">The checkpoint restored the weights. Adam’s moments started over at zero.</p>
      )}
    </section>
  )
}

const PAGE = 12

type DatasetRow = {
  index: number
  label: 0 | 1
  text: string
  tokens: number
  tokenCount: number
  truncated: boolean
  clipped: boolean
}

type RowPage = {
  split: "train" | "test"
  offset: number
  total: number
  rows: DatasetRow[]
}

function RowBrowser() {
  const [split, setSplit] = useState<"train" | "test">("train")
  const [rows, setRows] = useState<DatasetRow[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const request = useRef(0)

  async function load(nextSplit: "train" | "test", offset: number, replace: boolean) {
    const id = request.current + 1
    request.current = id
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/rows?split=${nextSplit}&offset=${offset}&limit=${PAGE}`)
      if (!response.ok) throw new Error("Could not read that page of reviews.")
      const page = (await response.json()) as RowPage
      if (request.current !== id) return
      setTotal(page.total)
      setRows((current) => (replace ? page.rows : [...current, ...page.rows]))
    } catch (reason) {
      if (request.current !== id) return
      setError(reason instanceof Error ? reason.message : "Could not read that page of reviews.")
    } finally {
      if (request.current === id) setLoading(false)
    }
  }

  useEffect(() => {
    setRows([])
    setTotal(0)
    void load(split, 0, true)
  }, [split])

  const remaining = Math.max(0, total - rows.length)
  const nextCount = Math.min(PAGE, remaining)

  return (
    <section aria-busy={loading}>
      <h3>Rows</h3>
      <Segmented
        label="Split"
        value={split}
        options={[
          { id: "train", label: "Training" },
          { id: "test", label: "Test" },
        ]}
        onChange={(id) => setSplit(id as "train" | "test")}
      />
      <p className="fine">
        {total === 0 && loading
          ? "Reading the first page."
          : `Showing ${rows.length.toLocaleString()} of ${total.toLocaleString()}.`}
      </p>
      <ol className="row-list">
        {rows.map((row) => (
          <li key={`${split}-${row.index}`}>
            <div className="row-meta">
              <span>Row {row.index + 1}</span>
              <span>{row.label === 1 ? "1 · Positive" : "0 · Negative"}</span>
            </div>
            <p>{row.text}</p>
            <p className="fine">
              {row.truncated
                ? `${row.tokenCount.toLocaleString()} words, cut to ${row.tokens} tokens.`
                : `${row.tokens.toLocaleString()} tokens.`}
              {row.clipped ? " Long review, showing the first part." : ""}
            </p>
          </li>
        ))}
      </ol>
      {error && (
        <p className="alert inline" role="alert">
          {error}
        </p>
      )}
      {remaining > 0 && (
        <div className="row-more">
          <button
            type="button"
            className="btn"
            disabled={loading}
            onClick={() => void load(split, rows.length, false)}
          >
            {loading ? "Loading" : `Load ${nextCount} more`}
          </button>
        </div>
      )}
    </section>
  )
}

function Data({ onEval, evaluating }: { onEval: () => void; evaluating: boolean }) {
  const { dataset, trace, metrics, task } = useStudio()
  const view = trace && trace.task === task ? trace : null
  if (task === "digits") return <DigitData dataset={dataset?.task === "digits" ? dataset : null} trace={view?.task === "digits" ? view : null} metrics={metrics} onEval={onEval} evaluating={evaluating} />
  return (
    <section>
      <h3>IMDB</h3>
      {dataset?.task === "reviews" ? (
        <>
          <p className="lesson">
            {dataset.train.toLocaleString()} training reviews and {dataset.test.toLocaleString()} test reviews. 1 is positive, 0 is
            negative.
          </p>
          <div className="balance" aria-hidden="true">
            <span style={{ flex: dataset.positive }} />
            <span style={{ flex: dataset.negative }} />
          </div>
          <p className="fine">
            {dataset.positive.toLocaleString()} positive · {dataset.negative.toLocaleString()} negative in training. Vocabulary{" "}
            {dataset.vocab.toLocaleString()}, window {dataset.seqLen}.
          </p>
          <p className="fine">Common tokens: {dataset.common.join(", ")}.</p>
          <RowBrowser />
        </>
      ) : (
        <p className="lesson">The reviews are still being prepared.</p>
      )}
      {view?.task === "reviews" && (
        <>
          <h3>{view.kind === "probe" ? "Your review" : "Review in view"}</h3>
          <p className="review">{view.text}</p>
          <Stats
            rows={[
              ["Tokens", String(view.tokens.length)],
              ["Label", view.label === null ? "—" : view.label === 1 ? "1 positive" : "0 negative"],
              ["Prediction", `${view.pred} ${view.pred === 1 ? "positive" : "negative"}`],
              ["Probability", formatNum(view.prob, 4)],
              ["Logit", formatNum(view.logit, 4)],
              ["Example loss", formatNum(view.exampleLoss, 4)],
              ["Batch loss", formatNum(view.batchLoss, 4)],
            ]}
          />
          {view.truncated && <p className="fine">Cut to the token window.</p>}
        </>
      )}
      <div className="eval-row">
        <button type="button" className="btn" onClick={onEval} disabled={!dataset || evaluating}>
          {evaluating ? "Scoring" : "Score the test set"}
        </button>
        <span>
          {metrics?.testAccuracy == null
            ? `Full ${dataset?.task === "reviews" ? dataset.test.toLocaleString() : "25,000"} when you ask`
            : `Test accuracy ${Math.round(metrics.testAccuracy * 100)}%`}
        </span>
      </div>
    </section>
  )
}

function MatrixBlock({
  label,
  matrix,
  selected,
  cursor,
  mode = "diverging",
  onSelect,
}: {
  label: string
  matrix: number[][]
  selected: number | null
  cursor: number | null
  mode?: "diverging" | "magnitude"
  onSelect?: (column: number) => void
}) {
  if (!matrix.length || !matrix[0]?.length) return null
  return (
    <figure className="matrix-block">
      <figcaption>{label}</figcaption>
      <Heatmap matrix={matrix} selected={selected} cursor={cursor} mode={mode} onSelect={onSelect} />
    </figure>
  )
}

function Scrubber({ length, value, onChange }: { length: number; value: number; onChange: (value: number) => void }) {
  const reduced = usePrefers("(prefers-reduced-motion: reduce)")
  const [playing, setPlaying] = useState(false)
  useEffect(() => {
    if (!playing || reduced) return
    let raf = 0
    let last = performance.now()
    let bank = 0
    const tick = (now: number) => {
      bank += now - last
      last = now
      if (bank > 46) {
        bank = 0
        onChange(value >= length ? 0 : value + 1)
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, reduced, value, length, onChange])

  return (
    <div className="scrubber">
      {!reduced && (
        <button type="button" className="btn" aria-pressed={playing} onClick={() => setPlaying((on) => !on)}>
          {playing ? "Pause multiply" : "Play multiply"}
        </button>
      )}
      <input
        type="range"
        min={0}
        max={length}
        value={Math.min(value, length)}
        aria-label="Dot product index"
        onPointerDown={() => setPlaying(false)}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span>
        {Math.min(value, length)} / {length}
      </span>
    </div>
  )
}

function Arithmetic({ lines }: { lines: string[] }) {
  return (
    <div className="arithmetic">
      {lines.map((line) => (
        <p key={line}>{line}</p>
      ))}
    </div>
  )
}

function Stats({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="stats">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  )
}

function Histogram({ counts, edges }: { counts: number[]; edges: number[] }) {
  const peak = Math.max(1, ...counts)
  return (
    <div className="hist" aria-hidden="true">
      {counts.map((count, index) => (
        <span key={edges[index] ?? index} style={{ height: `${Math.max(4, (count / peak) * 100)}%` }} />
      ))}
    </div>
  )
}

function DigitData({
  dataset,
  trace,
  metrics,
  onEval,
  evaluating,
}: {
  dataset: DigitDataset | null
  trace: Extract<Trace, { task: "digits" }> | null
  metrics: ReturnType<typeof useStudio>["metrics"]
  onEval: () => void
  evaluating: boolean
}) {
  const peak = Math.max(1, ...(dataset?.counts ?? [1]))
  return (
    <section>
      <h3>MNIST</h3>
      {dataset ? (
        <>
          <p className="lesson">
            {dataset.train.toLocaleString()} training digits and {dataset.test.toLocaleString()} test digits. Each one is a{" "}
            {dataset.width}×{dataset.height} grayscale image, labeled 0 through 9.
          </p>
          <div className="class-bars" aria-hidden="true">
            {dataset.counts.map((count, digit) => (
              <div key={digit}>
                <i style={{ height: `${Math.max(8, (count / peak) * 100)}%` }} />
                <small>{digit}</small>
              </div>
            ))}
          </div>
          <p className="fine">Training counts for digits 0 through 9.</p>
          <DigitBrowser side={dataset.width} />
        </>
      ) : (
        <p className="lesson">The digits are still being prepared.</p>
      )}
      {trace && (
        <>
          <h3>{trace.kind === "probe" ? "Your digit" : "Digit in view"}</h3>
          <PixelGrid values={trace.pixels} side={trace.width} mode="ink" />
          <Stats
            rows={[
              ["Label", trace.label === null ? "—" : String(trace.label)],
              ["Prediction", String(trace.pred)],
              ["Probability", formatNum(trace.prob, 4)],
              ["Example loss", formatNum(trace.exampleLoss, 4)],
              ["Batch loss", formatNum(trace.batchLoss, 4)],
            ]}
          />
        </>
      )}
      <div className="eval-row">
        <button type="button" className="btn" onClick={onEval} disabled={!dataset || evaluating}>
          {evaluating ? "Scoring" : "Score the test set"}
        </button>
        <span>
          {metrics?.testAccuracy == null
            ? `Full ${dataset ? dataset.test.toLocaleString() : "10,000"} when you ask`
            : `Test accuracy ${Math.round(metrics.testAccuracy * 100)}%`}
        </span>
      </div>
    </section>
  )
}

function DigitBrowser({ side }: { side: number }) {
  const [split, setSplit] = useState<"train" | "test">("train")
  const [rows, setRows] = useState<{ index: number; label: number; pixels: number[] }[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const request = useRef(0)

  async function load(nextSplit: "train" | "test", offset: number, replace: boolean) {
    const id = request.current + 1
    request.current = id
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`/api/rows?split=${nextSplit}&offset=${offset}&limit=${PAGE}`)
      if (!response.ok) throw new Error("Could not read that page of digits.")
      const page = (await response.json()) as {
        total: number
        rows: { index: number; label: number; pixels: number[] }[]
      }
      if (request.current !== id) return
      setTotal(page.total)
      setRows((current) => (replace ? page.rows : [...current, ...page.rows]))
    } catch (reason) {
      if (request.current !== id) return
      setError(reason instanceof Error ? reason.message : "Could not read that page of digits.")
    } finally {
      if (request.current === id) setLoading(false)
    }
  }

  useEffect(() => {
    setRows([])
    setTotal(0)
    void load(split, 0, true)
  }, [split])

  const remaining = Math.max(0, total - rows.length)
  const nextCount = Math.min(PAGE, remaining)
  return (
    <section aria-busy={loading}>
      <h3>Rows</h3>
      <Segmented
        label="Split"
        value={split}
        options={[
          { id: "train", label: "Training" },
          { id: "test", label: "Test" },
        ]}
        onChange={(id) => setSplit(id as "train" | "test")}
      />
      <p className="fine">
        {total === 0 && loading ? "Reading the first page." : `Showing ${rows.length.toLocaleString()} of ${total.toLocaleString()}.`}
      </p>
      <ol className="row-list">
        {rows.map((row) => (
          <li key={`${split}-${row.index}`} className="digit-row">
            <PixelGrid values={row.pixels.map((value) => value / 255)} side={side} maxSide={56} />
            <div>
              <div className="row-meta">
                <span>Row {row.index + 1}</span>
                <span>{row.label}</span>
              </div>
            </div>
          </li>
        ))}
      </ol>
      {error && (
        <p className="alert inline" role="alert">
          {error}
        </p>
      )}
      {remaining > 0 && (
        <div className="row-more">
          <button type="button" className="btn" disabled={loading} onClick={() => void load(split, rows.length, false)}>
            {loading ? "Loading" : `Load ${nextCount} more`}
          </button>
        </div>
      )}
    </section>
  )
}

function BatchNote({ rms, trainedSteps, noun = "review" }: { rms: number | null; trainedSteps: number; noun?: string }) {
  return (
    <p className="fine">
      {rms == null || trainedSteps === 0
        ? `These cells are the gradient of this one ${noun}. The training step uses the mean gradient of the batch.`
        : `These cells are this ${noun}. The batch gradient RMS, what the optimizer steps on, is ${formatNum(rms, 4)}.`}
    </p>
  )
}
