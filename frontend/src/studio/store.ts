import { useSyncExternalStore } from "react"
import { defaultConfig, type ServerMessage, type Studio, type TaskName, type Trace } from "./types"

function nest(values: number[], shape: number[]): number[] | number[][] | number[][][] {
  if (shape.length <= 1) return values
  const [rows, ...rest] = shape
  const width = rest.reduce((product, item) => product * item, 1)
  const nested: Array<number[] | number[][] | number[][][]> = []
  for (let row = 0; row < rows; row += 1) {
    const slice = values.slice(row * width, (row + 1) * width)
    nested.push(rest.length === 1 ? slice : nest(slice, rest))
  }
  return nested as number[][] | number[][][]
}

function inflate(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(inflate)
  if (!value || typeof value !== "object") return value
  const record = value as Record<string, unknown>
  if (record.dtype === "f32" && typeof record.data === "string" && Array.isArray(record.shape)) {
    const shape = record.shape.map((item) => Number(item))
    const binary = Uint8Array.from(atob(record.data), (char) => char.charCodeAt(0))
    const view = new DataView(binary.buffer, binary.byteOffset, binary.byteLength)
    const count = shape.reduce((product, item) => product * item, 1)
    const flat = new Array<number>(count)
    for (let index = 0; index < count; index += 1) flat[index] = view.getFloat32(index * 4, true)
    return nest(flat, shape)
  }
  const next: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(record)) next[key] = inflate(item)
  return next
}

const initial: Studio = {
  connected: false,
  task: "gait",
  tasks: ["gait"],
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
    task: "gait",
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
    case "catalog":
      commit({ ...state, tasks: message.tasks, task: message.task })
      break
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
    case "trace": {
      const trace = message.trace ? (inflate(message.trace) as Trace) : null
      latestTrace = trace
      commit({
        ...state,
        trace,
        messages: trace === null ? [] : state.messages,
      })
      break
    }
    case "predict":
      commit({
        ...state,
        task: message.task,
        messages: message.task === "reviews" ? [...state.messages, message] : [message],
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

const REQUIRED: Record<string, string> = {
  config: "config",
  trace: "trace",
  error: "message",
}

function isMessage(value: unknown): value is ServerMessage {
  if (!value || typeof value !== "object") return false
  const type = (value as { type?: unknown }).type
  if (typeof type !== "string") return false
  const field = REQUIRED[type]
  return !field || field in value
}

export function disconnect() {
  if (reconnect !== null) window.clearTimeout(reconnect)
  reconnect = null
  retryDelay = 800
  const current = socket
  socket = null
  current?.close()
  commit({ ...state, connected: false })
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
    let message: unknown
    try {
      message = JSON.parse(String(event.data))
    } catch {
      return
    }
    if (!isMessage(message)) return
    try {
      apply(message)
    } catch (error) {
      console.warn("Ignored a malformed server message", error)
    }
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
