from __future__ import annotations

import math
from typing import Any

import numpy as np


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    ez = math.exp(value)
    return ez / (1.0 + ez)


def to_unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size and float(array.max()) > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


TRACE_UNITS = 96


def _round_list(values: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=np.float32)
    rounded = np.round(array, 5)
    return rounded.tolist()


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def _sample_indices(count: int, limit: int = TRACE_UNITS) -> list[int] | None:
    if count <= limit:
        return None
    if limit <= 1:
        return [0]
    return [int(round(index * (count - 1) / (limit - 1))) for index in range(limit)]


def _pack_matrix(weight: np.ndarray, keep_rows: bool) -> tuple[np.ndarray, list[int] | None, list[int] | None]:
    row_index = None if keep_rows else _sample_indices(int(weight.shape[0]))
    col_index = _sample_indices(int(weight.shape[1]))
    sampled = weight
    if row_index is not None:
        sampled = sampled[np.asarray(row_index, dtype=np.int32)]
    if col_index is not None:
        sampled = sampled[:, np.asarray(col_index, dtype=np.int32)]
    return sampled, row_index, col_index


def _backward(
    weights: list[np.ndarray],
    pres: list[np.ndarray],
    outputs: list[np.ndarray],
    activations: list[str],
    label: int,
    prob: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if activations[-1] == "softmax":
        upstream = outputs[-1].astype(np.float64, copy=True)
        if 0 <= label < upstream.size:
            upstream[label] -= 1.0
    else:
        upstream = np.array([prob - float(label)], dtype=np.float64)
    packed: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(weights) - 1, -1, -1):
        if activations[index] == "relu":
            delta = np.where(pres[index] > 0, upstream, 0.0)
        else:
            delta = upstream
        delta = np.asarray(delta, dtype=np.float64).reshape(-1)
        d_input = np.asarray(weights[index] @ delta, dtype=np.float64).reshape(-1)
        packed.append((delta, d_input))
        upstream = d_input
    packed.reverse()
    return packed


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0:
        return np.full_like(logits, 1.0 / max(1, logits.size))
    return exp / total


def _cross_entropy(logits: np.ndarray, label: int) -> float:
    shifted = logits - np.max(logits)
    return float(-shifted[label] + np.log(np.sum(np.exp(shifted))))


def _trace_layers(
    hidden: np.ndarray,
    affine: list[Any],
    *,
    last_name: str,
    last_title: str,
    last_activation: str,
    label: int | None,
    keep_first_rows: bool,
    sample: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    np_weights: list[np.ndarray] = []
    np_pres: list[np.ndarray] = []
    np_outputs: list[np.ndarray] = []
    activations: list[str] = []
    last = len(affine) - 1
    cursor = np.asarray(hidden, dtype=np.float32)
    for index, layer in enumerate(affine):
        raw_w = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32).reshape(-1)
        weight = raw_w.T
        pre = cursor @ weight + bias
        if index == last:
            output = _softmax(pre) if last_activation == "softmax" else pre
            activation = last_activation
            title = last_title
            name = last_name
        else:
            output = np.maximum(pre, 0)
            activation = "relu"
            title = "Hidden" if last == 1 else f"Hidden {index + 1}"
            name = f"hidden-{index + 1}"
        np_weights.append(weight)
        np_pres.append(np.asarray(pre, dtype=np.float32))
        np_outputs.append(np.asarray(output, dtype=np.float32))
        activations.append(activation)
        records.append(
            {
                "name": name,
                "title": title,
                "activation": activation,
                "input": _round_list(cursor),
                "bias": _round_list(bias),
                "pre": _round_list(pre),
                "output": _round_list(output),
                "weightMean": float(weight.mean()),
                "weightStd": float(weight.std()),
                "batchGradRms": None,
                "biasGradRms": None,
                "momentRms": None,
                "secondMomentRms": None,
            }
        )
        cursor = output.astype(np.float32)

    if sample:
        packed = [
            _pack_matrix(weight, keep_rows=keep_first_rows and index == 0) for index, weight in enumerate(np_weights)
        ]
    else:
        packed = [(weight, None, None) for weight in np_weights]
    sampled = any(row is not None or col is not None for _, row, col in packed)
    grads = None
    if label is not None and sampled:
        prob = sigmoid(float(records[-1]["pre"][0])) if last_activation == "logit" else 0.0
        grads = _backward(np_weights, np_pres, np_outputs, activations, int(label), prob)
    for record, (matrix, row_index, col_index), grad in zip(records, packed, grads or [None] * len(records)):
        record["weight"] = _round_list(matrix)
        if row_index is not None:
            record["rowIndex"] = row_index
        if col_index is not None:
            record["colIndex"] = col_index
        if grad is not None:
            delta, d_input = grad
            record["delta"] = _round_list(delta)
            record["dInput"] = _round_list(d_input)
    return records
