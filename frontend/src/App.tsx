import { Component, Suspense, lazy, useEffect, useState, type ReactNode } from "react"
import { Clinic } from "./clinic/Clinic"

const Studio = lazy(() => import("./studio/StudioScreen"))

function remember(key: string, value?: string): string | null {
  try {
    if (value !== undefined) sessionStorage.setItem(key, value)
    return sessionStorage.getItem(key)
  } catch {
    return null
  }
}

class BenchBoundary extends Component<{ children: ReactNode; onReturn: () => void }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <div className="loading" role="alert">
        <p>The training bench did not load. The app may have been updated.</p>
        <p>
          <button type="button" className="btn primary" onClick={() => window.location.reload()}>
            Reload
          </button>{" "}
          <button type="button" className="btn" onClick={this.props.onReturn}>
            Back to the clinic
          </button>
        </p>
      </div>
    )
  }
}

export function App() {
  const [bench, setBench] = useState(() => window.location.hash.startsWith("#/bench"))

  useEffect(() => {
    const sync = () => setBench(window.location.hash.startsWith("#/bench"))
    window.addEventListener("hashchange", sync)
    return () => window.removeEventListener("hashchange", sync)
  }, [])

  if (bench) {
    const back = () => {
      window.location.hash = remember("meridian.return") || "#/"
    }
    return (
      <BenchBoundary onReturn={back}>
        <Suspense fallback={<p className="loading">Opening the training bench</p>}>
          <Studio onReturn={back} />
        </Suspense>
      </BenchBoundary>
    )
  }

  return (
    <Clinic
      onOpenBench={() => {
        remember("meridian.return", window.location.hash || "#/")
        window.location.hash = "#/bench"
      }}
    />
  )
}
