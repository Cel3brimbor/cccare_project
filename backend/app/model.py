from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

PAD_ID = 0


class SentimentNet(nn.Module):
    def __init__(self, vocab: int, embed_dim: int, hidden: list[int], pad_id: int = PAD_ID):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, embed_dim)
        widths = [embed_dim, *hidden]
        self.hidden = [nn.Linear(widths[i], widths[i + 1]) for i in range(len(hidden))]
        self.output = nn.Linear(widths[-1], 1)

    def __call__(self, tokens: mx.array) -> mx.array:
        embedded = self.embed(tokens)
        mask = (tokens != self.pad_id).astype(mx.float32)
        denom = mx.maximum(mx.sum(mask, axis=1, keepdims=True), 1)
        hidden = mx.sum(embedded * mask[..., None], axis=1) / denom
        for layer in self.hidden:
            hidden = mx.maximum(layer(hidden), 0)
        return mx.squeeze(self.output(hidden), axis=-1)


class DigitNet(nn.Module):
    def __init__(self, hidden: list[int], pixels: int = 784, classes: int = 10):
        super().__init__()
        widths = [pixels, *hidden]
        self.hidden = [nn.Linear(widths[i], widths[i + 1]) for i in range(len(hidden))]
        self.output = nn.Linear(widths[-1], classes)

    def __call__(self, pixels: mx.array) -> mx.array:
        hidden = pixels
        for layer in self.hidden:
            hidden = mx.maximum(layer(hidden), 0)
        return self.output(hidden)


def digit_loss(model: DigitNet, pixels: mx.array, labels: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(pixels), labels, reduction="mean")


def bce_with_logits(logits: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.binary_cross_entropy(logits, targets, with_logits=True, reduction="mean")


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    ez = math.exp(value)
    return ez / (1.0 + ez)


def example_bce(logit: float, label: float) -> float:
    if logit > 0:
        return logit - logit * label + math.log1p(math.exp(-logit))
    return -logit * label + math.log1p(math.exp(logit))


def _round_list(values: np.ndarray) -> list[Any]:
    array = np.asarray(values, dtype=np.float32)
    rounded = np.round(array, 5)
    return rounded.tolist()


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64))))


def explain_example(
    weights: dict[str, Any],
    token_ids: np.ndarray,
    itos: list[str],
    label: int | None,
) -> dict[str, Any]:
    embed = np.asarray(weights["embed"], dtype=np.float32)
    ids = [int(i) for i in token_ids.tolist() if int(i) != PAD_ID]
    truncated = int(token_ids[-1]) != PAD_ID if token_ids.size else False
    if not ids:
        ids = [PAD_ID]
        tokens = ["unk"]
        rows = np.zeros((1, embed.shape[1]), dtype=np.float32)
    else:
        tokens = []
        for index in ids:
            word = itos[index] if 0 <= index < len(itos) else "unk"
            tokens.append("unk" if word in ("<unk>", "<pad>") else word)
        rows = embed[np.asarray(ids, dtype=np.int32)]
    pooled = rows.mean(axis=0).astype(np.float32)

    layers = _trace_layers(
        pooled,
        list(weights["hidden"]) + [weights["output"]],
        last_name="logit",
        last_title="Logit",
        last_activation="logit",
        label=label,
        keep_first_rows=False,
    )

    logit = float(layers[-1]["pre"][0])
    probability = sigmoid(logit)
    prediction = 1 if probability >= 0.5 else 0
    norms = np.linalg.norm(embed, axis=1)
    counts, edges = np.histogram(norms, bins=24)
    return {
        "tokens": tokens,
        "truncated": truncated,
        "gathered": _round_list(rows),
        "pooled": _round_list(pooled),
        "layers": layers,
        "logit": round(logit, 6),
        "prob": round(probability, 6),
        "pred": prediction,
        "label": None if label is None else int(label),
        "exampleLoss": None if label is None else round(example_bce(logit, float(label)), 6),
        "embed": {
            "vocab": int(embed.shape[0]),
            "dim": int(embed.shape[1]),
            "weightMean": float(embed.mean()),
            "weightStd": float(embed.std()),
            "rowRms": _rms(embed),
            "batchGradRms": None,
            "momentRms": None,
            "secondMomentRms": None,
            "normHist": {
                "counts": [int(c) for c in counts.tolist()],
                "edges": [round(float(e), 5) for e in edges.tolist()],
            },
        },
    }


def to_unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size and float(array.max()) > 1.5:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


TRACE_UNITS = 96


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


def _trace_layers(
    hidden: np.ndarray,
    affine: list[Any],
    *,
    last_name: str,
    last_title: str,
    last_activation: str,
    label: int | None,
    keep_first_rows: bool,
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

    packed = [
        _pack_matrix(weight, keep_rows=keep_first_rows and index == 0) for index, weight in enumerate(np_weights)
    ]
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


def explain_digit(
    weights: dict[str, Any],
    pixels: np.ndarray,
    label: int | None,
    side: int = 28,
) -> dict[str, Any]:
    image = to_unit_interval(pixels)
    if image.size != side * side:
        fitted = np.zeros(side * side, dtype=np.float32)
        count = min(fitted.size, image.size)
        fitted[:count] = image[:count]
        image = fitted

    layers = _trace_layers(
        image,
        list(weights["hidden"]) + [weights["output"]],
        last_name="digits",
        last_title="Digits",
        last_activation="softmax",
        label=label,
        keep_first_rows=True,
    )

    logits = np.asarray(layers[-1]["pre"], dtype=np.float32)
    probabilities = np.asarray(layers[-1]["output"], dtype=np.float32)
    prediction = int(np.argmax(probabilities))
    return {
        "pixels": _round_list(image),
        "width": side,
        "height": side,
        "layers": layers,
        "logits": [round(float(value), 6) for value in logits.tolist()],
        "probs": [round(float(value), 6) for value in probabilities.tolist()],
        "logit": round(float(logits[prediction]), 6),
        "prob": round(float(probabilities[prediction]), 6),
        "pred": prediction,
        "label": None if label is None else int(label),
        "exampleLoss": None if label is None else round(_cross_entropy(logits, int(label)), 6),
    }
