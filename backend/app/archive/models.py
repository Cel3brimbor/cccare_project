from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.app.model import _cross_entropy, _rms, _round_list, _softmax, _trace_layers, sigmoid, to_unit_interval

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


def example_bce(logit: float, label: float) -> float:
    if logit > 0:
        return logit - logit * label + math.log1p(math.exp(-logit))
    return -logit * label + math.log1p(math.exp(logit))


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
