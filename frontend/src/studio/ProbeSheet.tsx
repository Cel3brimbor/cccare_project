import { useEffect, useId, useRef, useState, type PointerEvent as ReactPointerEvent } from "react"
import { animate, motion, useMotionValue, type AnimationPlaybackControls } from "motion/react"
import { useDialog } from "./dialog"
import { DigitPad } from "./DigitPad"
import { formatNum } from "./format"
import { usePrefers } from "./media"
import { send, useStudio } from "./store"

const samples = [
  "A remarkable film, tender and exact, with an ending that earns it.",
  "Dull, empty, and a waste of the evening.",
]

export function ProbeSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { messages, dataset, status, connected, task } = useStudio()
  const reduced = usePrefers("(prefers-reduced-motion: reduce)")
  const y = useMotionValue(typeof window === "undefined" ? 800 : window.innerHeight)
  const controls = useRef<AnimationPlaybackControls | null>(null)
  const keep = useRef(false)
  const sheetRef = useRef<HTMLDivElement>(null)
  const heightRef = useRef(640)
  const endRef = useRef<HTMLDivElement>(null)
  const titleId = useId()
  const fieldId = useId()
  const [draft, setDraft] = useState("")
  const ready = connected && Boolean(dataset) && status.metal && status.phase !== "preparing"
  const dismissRef = useRef<() => void>(() => onClose())
  useDialog(open, sheetRef, () => dismissRef.current())

  useEffect(() => {
    const element = sheetRef.current
    if (!element) return
    const measure = () => {
      heightRef.current = element.getBoundingClientRect().height || 640
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  function hiddenOffset() {
    return Math.max(sheetRef.current?.offsetHeight ?? 0, heightRef.current, window.innerHeight)
  }

  function dismiss() {
    keep.current = true
    controls.current?.stop()
    y.set(hiddenOffset())
    onClose()
  }
  dismissRef.current = dismiss

  useEffect(() => {
    if (keep.current) {
      keep.current = false
      return
    }
    const destination = open ? 0 : hiddenOffset()
    if (!open) y.set(destination)
    const animation = animate(
      y,
      destination,
      reduced ? { duration: 0.16, ease: "easeOut" } : { type: "spring", bounce: open ? 0.16 : 0, duration: open ? 0.42 : 0.01 },
    )
    controls.current = animation
    return () => animation.stop()
  }, [open, reduced, y])

  useEffect(() => {
    if (!open) return
    const scroller = endRef.current?.parentElement
    if (!scroller) return
    scroller.scrollTop = scroller.scrollHeight
  }, [messages.length, open])

  function submit(text: string) {
    const cleaned = text.trim()
    if (!cleaned || !ready) return
    send({ type: "predict", id: `${Date.now()}`, text: cleaned })
    setDraft("")
  }

  function onDown(event: ReactPointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture(event.pointerId)
    controls.current?.stop()
    const drag = {
      last: event.clientY,
      samples: [{ t: performance.now(), y: event.clientY }],
    }
    const move = (ev: PointerEvent) => {
      const dy = ev.clientY - drag.last
      drag.last = ev.clientY
      drag.samples.push({ t: performance.now(), y: ev.clientY })
      let next = y.get() + dy
      if (next < 0) next = -rubberband(-next, heightRef.current)
      y.set(Math.min(heightRef.current + 40, next))
    }
    const up = () => {
      event.currentTarget.removeEventListener("pointermove", move)
      event.currentTarget.removeEventListener("pointerup", up)
      const velocity = releaseVelocity(drag.samples)
      const projected = y.get() + project(velocity)
      const close = projected > heightRef.current * 0.34
      keep.current = true
      controls.current = animate(
        y,
        close ? hiddenOffset() : 0,
        reduced
          ? { duration: 0.16, ease: "easeOut" }
          : { type: "spring", bounce: 0.2, duration: 0.45, velocity },
      )
      if (close) onClose()
    }
    event.currentTarget.addEventListener("pointermove", move)
    event.currentTarget.addEventListener("pointerup", up)
  }

  return (
    <motion.div
      ref={sheetRef}
      id="probe-sheet"
      className={task === "digits" ? "sheet digits" : "sheet"}
      style={{ y }}
      data-open={open ? "true" : "false"}
      role="dialog"
      aria-modal={open}
      aria-labelledby={titleId}
      aria-hidden={!open}
      inert={!open}
      tabIndex={-1}
    >
      <div className="grabber" onPointerDown={onDown} aria-hidden="true">
        <span />
      </div>
      <div className="sheet-head">
        <div>
          <h2 id={titleId}>{task === "digits" ? "Draw a digit" : "Try a review"}</h2>
          <p>
            {task === "digits"
              ? "Paint the grid. The reply is the digit with the highest probability."
              : "1 is positive, 0 is negative. The reply is the sigmoid of one logit."}
          </p>
        </div>
        <button
          type="button"
          className="btn"
          data-initial-focus={task === "digits" ? true : undefined}
          onClick={dismiss}
          onPointerDown={(event) => {
            event.preventDefault()
            event.stopPropagation()
            dismiss()
          }}
        >
          Close
        </button>
      </div>
      {task === "digits" ? (
        <DigitPad ready={ready} />
      ) : (
      <>
      <div className="transcript">
        {messages.length === 0 && <p className="fine">Send a sentence. The network answers with a probability and a hard 0 or 1.</p>}
        {messages.map((message) => (
          <div key={message.id} className="exchange">
            <p className="bubble user">{message.text}</p>
            <div className={message.pred === 1 ? "bubble net is-positive" : "bubble net is-negative"}>
              <strong>{message.pred === 1 ? "Positive" : "Negative"}</strong>
              <i className="prob-bar" aria-hidden="true">
                <b style={{ width: `${Math.round(message.prob * 100)}%` }} />
              </i>
              <span>
                Output {message.pred}, {Math.round(message.prob * 100)}% positive
              </span>
              <small>
                σ({formatNum(message.logit, 3)}) = {formatNum(message.prob, 3)}
              </small>
              {message.untrained && <small>Untrained weights. This is the random initialization.</small>}
              {message.truncated && <small>Cut to the token window.</small>}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="suggestions" role="group" aria-label="Example reviews">
        {samples.filter((sample) => !messages.some((message) => message.text === sample)).map((sample) => (
          <button key={sample} type="button" className="sample" disabled={!ready} onClick={() => submit(sample)}>
            {sample}
          </button>
        ))}
      </div>
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault()
          submit(draft)
        }}
      >
        <label className="sr" htmlFor={fieldId}>
          Review
        </label>
        <textarea
          id={fieldId}
          data-initial-focus
          value={draft}
          placeholder={ready ? "Write a short movie review" : "Preparing the reviews"}
          disabled={!ready}
          rows={2}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault()
              submit(draft)
            }
          }}
        />
        <button type="submit" className="btn primary" disabled={!ready || !draft.trim()}>
          Send
        </button>
      </form>
      </>
      )}
    </motion.div>
  )
}

function project(velocity: number, deceleration = 0.998) {
  return (velocity / 1000) * deceleration / (1 - deceleration)
}

function rubberband(overshoot: number, dimension: number, constant = 0.55) {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot))
}

function releaseVelocity(samples: { t: number; y: number }[]) {
  if (samples.length < 2) return 0
  const last = samples[samples.length - 1]
  const cutoff = last.t - 80
  let first = samples[0]
  for (const sample of samples) {
    if (sample.t >= cutoff) {
      first = sample
      break
    }
  }
  const dt = last.t - first.t
  if (dt < 8) return 0
  return ((last.y - first.y) / dt) * 1000
}
