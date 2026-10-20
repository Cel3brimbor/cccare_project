export type TaskName = "reviews" | "digits"

export type Config = {
  embedDim: number
  hidden: number[]
  lr: number
  batchSize: number
  epochs: number
  optimizer: "adam" | "sgd"
  beta1: number
  beta2: number
  eps: number
  weightDecay: number
  seed: number
}

export type DeviceInfo = {
  type: "device"
  available: boolean
  name: string | null
  architecture: string | null
  memory: number | null
  cores: number | null
}

export type SystemInfo = {
  type: "system"
  cpu: number
  processCpu: number
  memoryUsed: number
  memoryTotal: number
  processMemory: number
  metalActive: number
  metalPeak: number
  metalCache: number
  gpuUtil: number | null
}

export type Status = {
  type: "status"
  phase: "preparing" | "ready" | "running" | "paused" | "evaluating" | "error"
  message: string
  progress: number | null
  locked: boolean
  trainedSteps: number
  metal: boolean
  task: TaskName
}

export type ReviewDataset = {
  type: "dataset"
  task: "reviews"
  train: number
  test: number
  positive: number
  negative: number
  vocab: number
  seqLen: number
  common: string[]
}

export type DigitDataset = {
  type: "dataset"
  task: "digits"
  train: number
  test: number
  width: number
  height: number
  classes: number
  counts: number[]
}

export type DatasetInfo = ReviewDataset | DigitDataset

export type HistoryPoint = {
  step: number
  loss: number
  accuracy: number
  val: number | null
}

export type Metrics = {
  type: "metrics"
  step: number
  epoch: number
  epochs: number
  loss: number | null
  accuracy: number | null
  valAccuracy: number | null
  valLoss: number | null
  testAccuracy: number | null
  testLoss: number | null
  examplesPerSec: number
  history: HistoryPoint[]
}

export type AffineLayer = {
  name: string
  title: string
  activation: "relu" | "logit" | "softmax"
  input: number[]
  weight: number[][]
  rowIndex?: number[]
  colIndex?: number[]
  bias: number[]
  pre: number[]
  output: number[]
  delta?: number[]
  dInput?: number[]
  weightMean: number
  weightStd: number
  batchGradRms: number | null
  biasGradRms: number | null
  momentRms: number | null
  secondMomentRms: number | null
}

type TraceCore = {
  kind: "preview" | "train" | "probe"
  seq: number
  step: number
  layers: AffineLayer[]
  logit: number
  prob: number
  pred: number
  label: number | null
  exampleLoss: number | null
  batchLoss: number | null
  batchAccuracy: number | null
  untrained: boolean
  optimizer: {
    name: "adam" | "sgd"
    lr: number
    beta1: number
    beta2: number
    eps: number
    weightDecay?: number
    biasCorrection: boolean
    step: number
  }
}

export type ReviewTrace = TraceCore & {
  task: "reviews"
  text: string
  tokens: string[]
  truncated: boolean
  gathered: number[][]
  pooled: number[]
  embed: {
    vocab: number
    dim: number
    weightMean: number
    weightStd: number
    rowRms: number
    batchGradRms: number | null
    momentRms: number | null
    secondMomentRms: number | null
    normHist: { counts: number[]; edges: number[] }
  }
}

export type DigitTrace = TraceCore & {
  task: "digits"
  pixels: number[]
  width: number
  height: number
  logits: number[]
  probs: number[]
}

export type Trace = ReviewTrace | DigitTrace

export type Prediction = {
  type: "predict"
  id: string
  task: TaskName
  text: string
  prob: number
  logit: number
  pred: number
  probs?: number[]
  untrained: boolean
  truncated: boolean
  tokens: string[]
}

export type ServerMessage =
  | DeviceInfo
  | SystemInfo
  | Status
  | DatasetInfo
  | Metrics
  | { type: "config"; config: Config; task?: TaskName }
  | { type: "trace"; trace: Trace | null }
  | Prediction
  | { type: "error"; message: string }

export type Studio = {
  connected: boolean
  task: TaskName
  device: DeviceInfo | null
  system: SystemInfo | null
  status: Status
  config: Config
  dataset: DatasetInfo | null
  metrics: Metrics | null
  trace: Trace | null
  messages: Prediction[]
  error: string | null
}

export type Selection = { layer: number; unit: number }

export const defaultConfig: Config = {
  embedDim: 48,
  hidden: [64, 32],
  lr: 0.001,
  batchSize: 64,
  epochs: 3,
  optimizer: "adam",
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 0,
  seed: 0,
}
