import { useEffect, useMemo, useState } from "react"
import { Charts } from "./Charts"
import { Inspector } from "./Inspector"
import { layerTitles } from "./format"
import { usePrefers } from "./media"
import { ModelPopover } from "./ModelPopover"
import { NetworkCanvas } from "./NetworkCanvas"
import { ProbeSheet } from "./ProbeSheet"
import { connect, disconnect, send, useStudio } from "./store"
import { Toolbar } from "./Toolbar"
import type { Selection } from "./types"
import "katex/dist/katex.min.css"

export default function Studio({ onReturn }: { onReturn: () => void }) {
  const studio = useStudio()
  useEffect(() => {
    connect()
    return disconnect
  }, [])
  useEffect(() => {
    const previous = document.title
    document.title = "Training bench · Meridian"
    return () => {
      document.title = previous
    }
  }, [])
  const reduced = usePrefers("(prefers-reduced-motion: reduce)")
  const [selection, setSelection] = useState<Selection>({ layer: 1, unit: 0 })
  const [modelOpen, setModelOpen] = useState(false)
  const [probeOpen, setProbeOpen] = useState(false)
  const digits = studio.task === "digits"
  const gait = studio.task === "gait"
  const side = studio.dataset?.task === "digits" ? studio.dataset.width : 28
  const frame =
    gait && studio.config.frameRows && studio.config.frameCols
      ? { rows: studio.config.frameRows, cols: studio.config.frameCols }
      : null
  const titles = useMemo(
    () => (gait && studio.config.titles?.length ? studio.config.titles : layerTitles(studio.config.hidden.length, studio.task)),
    [gait, studio.config.hidden.length, studio.config.titles, studio.task],
  )
  const sizes = useMemo(() => {
    if (gait && studio.config.widths?.length) return studio.config.widths
    if (digits) return [side * side, ...studio.config.hidden, 10]
    return [studio.config.embedDim, ...studio.config.hidden, 1]
  }, [digits, gait, side, studio.config])

  useEffect(() => {
    const widths =
      gait && studio.config.widths?.length
        ? studio.config.widths
        : digits
          ? [side * side, ...studio.config.hidden, 10]
          : [studio.config.embedDim, ...studio.config.hidden, 1]
    setSelection((current) => {
      const layer = Math.min(current.layer, widths.length - 1)
      const unit = Math.min(current.unit, widths[layer] - 1)
      if (layer === current.layer && unit === current.unit) return current
      return { layer, unit }
    })
  }, [studio.config, digits, gait, side])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (!target?.closest(".network")) return
      if (event.key === " ") {
        event.preventDefault()
        send({ type: studio.status.phase === "running" ? "pause" : "start" })
      } else if (event.key === "." && studio.status.phase !== "running") {
        send({ type: "step" })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [studio.status.phase])

  return (
    <div className="app">
      <div className="sr" aria-live="polite">
        {studio.status.message}
      </div>
      <main className={gait ? "stage gait" : "stage"}>
        <div className="chrome">
          <Toolbar
            modelOpen={modelOpen}
            probeOpen={probeOpen}
            onReturn={onReturn}
            onToggleModel={() => {
              setModelOpen((open) => !open)
              setProbeOpen(false)
            }}
            onOpenProbe={() => {
              setProbeOpen(true)
              setModelOpen(false)
            }}
          />
          <ModelPopover open={modelOpen} onClose={() => setModelOpen(false)} />
        </div>
        <h2 className="sr">Network</h2>
        <NetworkCanvas
          sizes={sizes}
          titles={titles}
          selection={selection}
          reducedMotion={reduced}
          imageSide={digits ? side : null}
          frame={frame}
          onSelect={setSelection}
        />
        <div className="legend">
          <span>
            <i className="swatch pos" aria-hidden="true" /> Positive weight
          </span>
          <span>
            <i className="swatch neg" aria-hidden="true" /> Negative weight
          </span>
        </div>
        <h2 className="sr">Training curves</h2>
        <Charts />
        <ProbeSheet open={probeOpen} onClose={() => setProbeOpen(false)} />
      </main>
      <Inspector selection={selection} onSelect={setSelection} />
    </div>
  )
}
