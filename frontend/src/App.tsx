import { useEffect, useMemo, useState } from "react"
import { Clinic } from "./clinic/Clinic"
import { Charts } from "./studio/Charts"
import { Inspector } from "./studio/Inspector"
import { layerTitles } from "./studio/format"
import { usePrefers } from "./studio/media"
import { ModelPopover } from "./studio/ModelPopover"
import { NetworkCanvas } from "./studio/NetworkCanvas"
import { ProbeSheet } from "./studio/ProbeSheet"
import { send, useStudio } from "./studio/store"
import { Toolbar } from "./studio/Toolbar"
import type { Selection } from "./studio/types"

export function App() {
  const [bench, setBench] = useState(() => window.location.hash.startsWith("#/bench"))

  useEffect(() => {
    const sync = () => setBench(window.location.hash.startsWith("#/bench"))
    window.addEventListener("hashchange", sync)
    return () => window.removeEventListener("hashchange", sync)
  }, [])

  if (bench) {
    return (
      <Studio
        onReturn={() => {
          window.location.hash = sessionStorage.getItem("meridian.return") || "#/"
        }}
      />
    )
  }

  return (
    <Clinic
      onOpenBench={() => {
        sessionStorage.setItem("meridian.return", window.location.hash || "#/")
        window.location.hash = "#/bench"
      }}
    />
  )
}

function Studio({ onReturn }: { onReturn: () => void }) {
  const studio = useStudio()
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
  const side = studio.dataset?.task === "digits" ? studio.dataset.width : 28
  const titles = useMemo(
    () => layerTitles(studio.config.hidden.length, studio.task),
    [studio.config.hidden.length, studio.task],
  )
  const sizes = useMemo(
    () => (digits ? [side * side, ...studio.config.hidden, 10] : [studio.config.embedDim, ...studio.config.hidden, 1]),
    [digits, side, studio.config],
  )

  useEffect(() => {
    const widths = digits ? [side * side, ...studio.config.hidden, 10] : [studio.config.embedDim, ...studio.config.hidden, 1]
    setSelection((current) => {
      const layer = Math.min(current.layer, widths.length - 1)
      const unit = Math.min(current.unit, widths[layer] - 1)
      if (layer === current.layer && unit === current.unit) return current
      return { layer, unit }
    })
  }, [studio.config, digits, side])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const tag = target?.tagName
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || tag === "BUTTON" || target?.isContentEditable) return
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
      <main className="stage">
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
        <NetworkCanvas
          sizes={sizes}
          titles={titles}
          selection={selection}
          reducedMotion={reduced}
          imageSide={digits ? side : null}
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
        <Charts />
        <ProbeSheet open={probeOpen} onClose={() => setProbeOpen(false)} />
      </main>
      <Inspector selection={selection} onSelect={setSelection} />
    </div>
  )
}
