import { useSyncExternalStore } from "react"
import { defaultConfig, type ServerMessage, type Studio, type TaskName, type Trace } from "./types"

const initial: Studio = {
  connected: false,
  task: "reviews",
  device: null,
  system: null,
  status: {
    type: "status",
    phase: "preparing",
    message: "Connecting",
    progress: null,
    locked: false,
    trainedSteps: 0,
    metal: false,
    task: "reviews",
  },
  config: defaultConfig,
  dataset: null,
  metrics: null,
  trace: null,
  messages: [],
  error: null,
}

let state = initial
const listeners = new Set<() => void>()
let socket: WebSocket | null = null
let reconnect: number | null = null
let retryDelay = 800

export let latestTrace: Trace | null = null

function emit() {
  listeners.forEach((listener) => listener())
}

function commit(next: Studio) {
  state = next
  emit()
}

export function getState() {
  return state
}

export function subscribe(listener: () => void) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useStudio() {
  return useSyncExternalStore(subscribe, getState)
}

export function clearError() {
  if (state.error === null) return
  commit({ ...state, error: null })
}

function adoptTask(task: TaskName | undefined, patch: Partial<Studio>): Studio {
  if (!task || task === state.task) return { ...state, ...patch }
  const trace = patch.trace !== undefined ? patch.trace : state.trace
  const keepTrace = trace?.task === task
  if (!keepTrace) latestTrace = null
  return {
    ...state,
    ...patch,
    task,
    messages: [],
    trace: keepTrace ? trace : null,
  }
}

function apply(message: ServerMessage) {
  switch (message.type) {
    case "device":
      commit({ ...state, device: message })
      break
    case "system":
      commit({ ...state, system: message })
      break
    case "status":
      commit(adoptTask(message.task, { status: message }))
      break
    case "config":
      commit(
        adoptTask(message.task, {
          config: {
            ...defaultConfig,
            ...message.config,
            hidden: message.config.hidden?.length ? message.config.hidden : defaultConfig.hidden,
          },
        }),
      )
      break
    case "dataset":
      commit(adoptTask(message.task, { dataset: message }))
      break
    case "metrics":
      commit({ ...state, metrics: message })
      break
    case "trace":
      latestTrace = message.trace
      commit({
        ...state,
        trace: message.trace,
        messages: message.trace === null ? [] : state.messages,
      })
      break
    case "predict":
      commit({
        ...state,
        task: message.task,
        messages: message.task === "digits" ? [message] : [...state.messages, message],
      })
      break
    case "error":
      commit({ ...state, error: message.message })
      break
    default:
      break
  }
}

export function send(message: object) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(message))
  }
}

export function connect() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return
  }
  const protocol = location.protocol === "https:" ? "wss" : "ws"
  const next = new WebSocket(`${protocol}://${location.host}/ws`)
  socket = next
  next.onopen = () => {
    if (socket !== next) return
    retryDelay = 800
    commit({ ...state, connected: true })
  }
  next.onmessage = (event) => {
    if (socket !== next) return
    apply(JSON.parse(String(event.data)) as ServerMessage)
  }
  next.onclose = () => {
    if (socket !== next) return
    socket = null
    commit({ ...state, connected: false })
    if (reconnect !== null) window.clearTimeout(reconnect)
    reconnect = window.setTimeout(connect, retryDelay)
    retryDelay = Math.min(15000, retryDelay * 2)
  }
}
