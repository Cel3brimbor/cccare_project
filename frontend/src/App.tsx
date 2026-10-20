import { Suspense, lazy, useEffect, useState } from "react"
import { Clinic } from "./clinic/Clinic"

const Studio = lazy(() => import("./studio/StudioScreen"))

export function App() {
  const [bench, setBench] = useState(() => window.location.hash.startsWith("#/bench"))

  useEffect(() => {
    const sync = () => setBench(window.location.hash.startsWith("#/bench"))
    window.addEventListener("hashchange", sync)
    return () => window.removeEventListener("hashchange", sync)
  }, [])

  if (bench) {
    return (
      <Suspense fallback={<p className="loading">Opening the training bench</p>}>
        <Studio
          onReturn={() => {
            window.location.hash = sessionStorage.getItem("meridian.return") || "#/"
          }}
        />
      </Suspense>
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
