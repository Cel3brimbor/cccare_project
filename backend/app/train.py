from __future__ import annotations

import copy
import hashlib
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import psutil

from backend.app.archive import ENABLE
from backend.app.bus import Bus
from backend.app.gait import CHANNELS, WINDOW, GaitData, GaitNet, activity_name, explain_gait, layout
from backend.app.system import device_message, gpu_registry, host_memory, metal_available, metal_memory

if ENABLE:
    from backend.app.archive.digits import CLASSES, PIXELS, SIDE, Digits
    from backend.app.archive.imdb import Corpus
    from backend.app.archive.models import DigitNet, SentimentNet, bce_with_logits, digit_loss, explain_digit, explain_example
    from backend.app.model import to_unit_interval

log = logging.getLogger("studio")

DEFAULT_CONFIG: dict[str, Any] = {
    "embedDim": 48,
    "hidden": [64, 32],
    "lr": 1e-3,
    "batchSize": 64,
    "epochs": 3,
    "optimizer": "adam",
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weightDecay": 0.0,
    "seed": 0,
}

GAIT_CONFIG: dict[str, Any] = {
    "embedDim": 48,
    "hidden": [128],
    "lr": 1e-3,
    "batchSize": 256,
    "epochs": 4,
    "optimizer": "adam",
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weightDecay": 0.01,
    "seed": 0,
}

TRAIN_KEYS = ("lr", "batchSize", "epochs", "optimizer", "beta1", "beta2", "eps", "weightDecay")
DIGITS_CONFIG: dict[str, Any] = {
    "embedDim": 48,
    "hidden": [64, 32],
    "lr": 1e-3,
    "batchSize": 128,
    "epochs": 2,
    "optimizer": "adam",
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weightDecay": 0.0,
    "seed": 0,
}


def sanitize(raw: dict[str, Any], task: str = "gait") -> dict[str, Any]:
    gait = task == "gait"
    hidden_in = raw.get("hidden") if isinstance(raw.get("hidden"), list) else ([128] if gait else [64, 32])
    hidden: list[int] = []
    width_lo = 32 if gait else 4
    layer_cap = 2 if gait else 3
    for value in hidden_in[:layer_cap]:
        try:
            hidden.append(min(1024, max(width_lo, int(value))))
        except (TypeError, ValueError):
            continue
    if not hidden:
        hidden = [128] if gait else [64, 32]

    def clamp_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            return min(hi, max(lo, int(raw.get(key, default))))
        except (TypeError, ValueError):
            return default

    def clamp_float(key: str, default: float, lo: float, hi: float) -> float:
        try:
            return min(hi, max(lo, float(raw.get(key, default))))
        except (TypeError, ValueError):
            return default

    optimizer = raw.get("optimizer", "adam")
    if optimizer not in ("adam", "sgd"):
        optimizer = "adam"
    return {
        "embedDim": clamp_int("embedDim", 48, 8, 96),
        "hidden": hidden,
        "lr": clamp_float("lr", 1e-3, 1e-5, 1.0),
        "batchSize": clamp_int("batchSize", 256 if gait else 64, 32 if gait else 8, 768 if gait else 256),
        "epochs": clamp_int("epochs", 4 if gait else 3, 1, 30),
        "optimizer": optimizer,
        "beta1": clamp_float("beta1", 0.9, 0.0, 0.9999),
        "beta2": clamp_float("beta2", 0.999, 0.0, 0.9999),
        "eps": clamp_float("eps", 1e-8, 0.0, 1e-2),
        "weightDecay": clamp_float("weightDecay", 0.01 if gait else 0.0, 0.0, 0.1),
        "seed": clamp_int("seed", 0, 0, 2_147_483_647),
    }


def loss_only(model: nn.Module, tokens: mx.array, labels: mx.array) -> mx.array:
    return bce_with_logits(model(tokens), labels)


def _as_numpy(array: mx.array) -> np.ndarray:
    return np.array(array)


def _mx_rms(array: mx.array) -> float:
    value = mx.sqrt(mx.mean(mx.square(array)))
    mx.eval(value)
    return float(value.item())


def _leaf(tree: Any, path: tuple[Any, ...]) -> Any:
    cursor = tree
    for key in path:
        if isinstance(cursor, dict) and key in cursor:
            cursor = cursor[key]
        elif isinstance(cursor, (list, tuple)) and isinstance(key, int) and 0 <= key < len(cursor):
            cursor = cursor[key]
        else:
            return None
    return cursor


def snapshot_weights(model: nn.Module) -> dict[str, Any]:
    params = model.parameters()
    mx.eval(params)
    payload: dict[str, Any] = {
        "hidden": [
            {"weight": _as_numpy(layer["weight"]), "bias": _as_numpy(layer["bias"])}
            for layer in params["hidden"]
        ],
        "output": {
            "weight": _as_numpy(params["output"]["weight"]),
            "bias": _as_numpy(params["output"]["bias"]),
        },
    }
    if "embed" in params:
        payload["embed"] = _as_numpy(params["embed"]["weight"])
    return payload


def _count_parameters(model: nn.Module) -> int:
    total = 0

    def walk(tree: Any) -> None:
        nonlocal total
        if isinstance(tree, mx.array):
            total += int(tree.size)
        elif isinstance(tree, dict):
            for value in tree.values():
                walk(value)
        elif isinstance(tree, (list, tuple)):
            for value in tree:
                walk(value)

    walk(model.parameters())
    return total


class Trainer:
    def __init__(self, root: Path):
        self.root = root
        self.task = "gait"
        self.gait = GaitData(root / "data")
        self.corpus = Corpus(root / "data") if ENABLE else None
        self.digits = Digits(root / "data") if ENABLE else None
        self._saved: dict[str, dict[str, Any]] = {}
        self._compiled_step = None
        self._class_weight: mx.array | None = None
        self.parameter_count = 0
        self.bus = Bus()
        self.commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self.config = dict(GAIT_CONFIG)
        self.config["hidden"] = list(GAIT_CONFIG["hidden"])
        self.mode = "preparing"
        self.metal = False
        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None
        self._loss_and_grad = None
        self.trained_steps = 0
        self.passes = 0
        self.run_origin = 0
        self.pass_limit = 0
        self.perm: np.ndarray | None = None
        self.cursor = 0
        self.history: list[dict[str, Any]] = []
        self.val_acc: float | None = None
        self.val_loss: float | None = None
        self.test_acc: float | None = None
        self.test_loss: float | None = None
        self._locked = False
        self._trace_seq = 1
        self._last_trace = 0.0
        self._last_metric = 0.0
        self._rate_examples = 0
        self._rate_time = time.perf_counter()
        self._examples_per_sec = 0.0
        self._message = "Starting"
        self._progress: float | None = None
        self.stop_flag = False
        self._device: dict[str, Any] = {"type": "device", "available": False}
        self.ckpt = root / "data" / "checkpoints"

    @property
    def weights_path(self) -> Path:
        if self.task == "gait":
            name = "gait.safetensors"
        elif self.task == "digits":
            name = "digits.safetensors"
        else:
            name = "model.safetensors"
        return self.ckpt / name

    @property
    def meta_path(self) -> Path:
        if self.task == "gait":
            name = "gait.meta.json"
        elif self.task == "digits":
            name = "digits.meta.json"
        else:
            name = "meta.json"
        return self.ckpt / name

    @property
    def data_ready(self) -> bool:
        if self.task == "gait":
            return self.gait.ready
        if self.task == "digits":
            return self.digits is not None and self.digits.ready
        return self.corpus is not None and self.corpus.ready

    def tasks(self) -> list[str]:
        names = ["gait"]
        if ENABLE:
            names.extend(["reviews", "digits"])
        return names

    def start(self) -> None:
        threading.Thread(target=self._loop, name="trainer", daemon=True).start()
        threading.Thread(target=self._stats_loop, name="stats", daemon=True).start()

    def stop(self) -> None:
        self.stop_flag = True

    def submit(self, command: dict[str, Any]) -> None:
        self.commands.put(command)

    def _loop(self) -> None:
        self.metal = metal_available()
        registry = gpu_registry()
        cores = registry.get("cores")
        self._device = device_message(cores if isinstance(cores, int) else None)
        self.bus.publish(self._device)
        self._publish_catalog()
        self._publish_config()
        if not self.metal:
            self._status("error", "Metal is not available on this machine.")
        else:
            self._try_prepare()
        while not self.stop_flag:
            if self.mode == "running":
                self._run_slice()
            else:
                command = self._wait_command(0.25)
                if command:
                    self._handle(command)

    def _stats_loop(self) -> None:
        process = psutil.Process()
        process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        gpu: dict[str, int | None] = {"utilization": None, "cores": None}
        last_gpu = 0.0
        while not self.stop_flag:
            try:
                now = time.monotonic()
                if now - last_gpu >= 2.0:
                    gpu = gpu_registry()
                    last_gpu = now
                memory = host_memory()
                metal = metal_memory()
                self.bus.publish(
                    {
                        "type": "system",
                        "cpu": psutil.cpu_percent(interval=None),
                        "processCpu": process.cpu_percent(interval=None),
                        "memoryUsed": memory["used"],
                        "memoryTotal": memory["total"],
                        "processMemory": memory["process"],
                        "metalActive": metal["active"],
                        "metalPeak": metal["peak"],
                        "metalCache": metal["cache"],
                        "gpuUtil": gpu.get("utilization"),
                    }
                )
            except Exception:
                log.exception("system sample failed")
            time.sleep(0.5)

    def _try_prepare(self) -> None:
        try:
            if self.task == "gait":
                if not self.gait.ready:
                    self.gait.prepare(lambda message, value: self._status("preparing", message, value))
            elif self.task == "digits":
                assert self.digits is not None
                if not self.digits.ready:
                    self.digits.prepare(lambda message, value: self._status("preparing", message, value))
            else:
                assert self.corpus is not None
                if not self.corpus.ready:
                    self.corpus.prepare(lambda message, value: self._status("preparing", message, value))
        except Exception as error:
            log.exception("dataset prepare failed")
            self._status("error", str(error))
            return
        self._publish_dataset()
        try:
            self._load_checkpoint()
        except Exception:
            log.exception("checkpoint load failed")
            self._clear_checkpoint_files()
            self.model = None
            self.optimizer = None
        if self.model is None:
            self._ensure_model()
        self._publish_metrics()
        if self.trained_steps:
            self.mode = "paused"
            self._status("paused", "Checkpoint restored")
        else:
            self.mode = "ready"
            self._status("ready", "Ready on Metal")

    def _run_slice(self) -> None:
        self._drain_nowait()
        if self.mode != "running":
            return
        try:
            self._one_step(capture=self._trace_due(), honor_budget=True)
        except Exception:
            log.exception("training step failed")
            self.mode = "paused"
            self._status("paused", "Training stopped on a step error.")
            return
        self._maybe_publish_metrics()

    def _drain_nowait(self) -> None:
        while True:
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            self._handle(command)

    def _wait_command(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self.commands.get(timeout=timeout)
        except queue.Empty:
            return None

    def _handle(self, command: dict[str, Any]) -> None:
        kind = command.get("type")
        if kind == "prepare":
            if self.mode != "running" and self.metal:
                self._try_prepare()
            return
        if kind == "task":
            self._switch_task(str(command.get("task") or ""))
            return
        if kind == "configure":
            self._configure(command.get("config") or {})
            return
        if not self.metal:
            self.bus.emit({"type": "error", "message": "Metal is not available, so training stays off."})
            return
        if not self.data_ready:
            self.bus.emit({"type": "error", "message": "The dataset is still being prepared."})
            return
        if kind == "start":
            self._start()
        elif kind == "pause":
            self._pause()
        elif kind == "step":
            self._step_once()
        elif kind == "reset":
            self._reset()
        elif kind == "predict":
            if self.task == "gait":
                self._predict_gait(command.get("index"), str(command.get("id") or ""))
            elif self.task == "digits":
                self._predict_digits(command.get("pixels"), str(command.get("id") or ""))
            else:
                self._predict(str(command.get("text") or ""), str(command.get("id") or ""))
        elif kind == "eval":
            self._full_eval()

    def _shape_differs(self, merged: dict[str, Any]) -> bool:
        hidden_changed = [int(value) for value in merged["hidden"]] != [int(value) for value in self.config["hidden"]]
        seed_changed = int(merged["seed"]) != int(self.config["seed"])
        if self.task in ("digits", "gait"):
            return hidden_changed or seed_changed
        return hidden_changed or seed_changed or int(merged["embedDim"]) != int(self.config["embedDim"])

    def _configure(self, incoming: dict[str, Any]) -> None:
        merged = sanitize({**self.config, **incoming}, self.task)
        if self._locked and self._shape_differs(merged):
            for key in TRAIN_KEYS:
                self.config[key] = merged[key]
            self._apply_training_config()
            self._publish_config()
            self._publish_metrics(reset_rate=False)
            self.bus.emit({"type": "error", "message": "Reset to change the network shape."})
            return
        shape_changed = self._shape_differs(merged)
        self.config = merged
        self._publish_config()
        if shape_changed:
            self.model = None
            self.optimizer = None
            self._loss_and_grad = None
            if self.data_ready and self.metal:
                self._ensure_model()
            return
        self._apply_training_config()
        self._publish_metrics(reset_rate=False)

    def _apply_training_config(self) -> None:
        if self.mode == "running":
            self.pass_limit = self.run_origin + int(self.config["epochs"])
        self._sync_optimizer()

    def _sync_optimizer(self) -> None:
        if self.model is None:
            return
        kind = "adam" if isinstance(self.optimizer, optim.Adam) else "sgd" if self.optimizer is not None else None
        if self.optimizer is None or kind != self.config["optimizer"]:
            self.optimizer = self._new_optimizer()
            self.optimizer.init(self.model.trainable_parameters())
        else:
            self.optimizer.learning_rate = float(self.config["lr"])
            self.optimizer.weight_decay = float(self.config["weightDecay"])
            if self.config["optimizer"] == "adam":
                self.optimizer.betas = [float(self.config["beta1"]), float(self.config["beta2"])]
                self.optimizer.eps = float(self.config["eps"])
        mx.eval(self.optimizer.state)
        if self.task == "gait" and self.model is not None:
            self._bind_gait_step()

    def _start(self) -> None:
        self._ensure_model()
        self.run_origin = self.passes
        self.pass_limit = self.passes + int(self.config["epochs"])
        self.perm = None
        self.cursor = 0
        self._rate_examples = 0
        self._rate_time = time.perf_counter()
        self.mode = "running"
        self._status("running", "Training on Metal")

    def _pause(self) -> None:
        if self.mode != "running":
            return
        self.mode = "paused"
        self._checkpoint()
        self._publish_metrics()
        self._status("paused", "Paused")

    def _step_once(self) -> None:
        self._ensure_model()
        self.mode = "paused"
        self._one_step(capture=True, honor_budget=False)
        self._publish_metrics()
        self._status("paused", "Held on this step")

    def _reset(self) -> None:
        self.mode = "ready"
        self.model = None
        self.optimizer = None
        self._loss_and_grad = None
        self.trained_steps = 0
        self.passes = 0
        self.run_origin = 0
        self.pass_limit = 0
        self.perm = None
        self.cursor = 0
        self.history = []
        self.val_acc = None
        self.val_loss = None
        self.test_acc = None
        self.test_loss = None
        self._locked = False
        self._clear_checkpoint_files()
        self.bus.publish({"type": "trace", "trace": None})
        self._publish_metrics()
        self._status("ready", "Weights reset")
        if self.data_ready and self.metal:
            self._ensure_model()

    def _predict(self, text: str, request_id: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            self.bus.emit({"type": "error", "message": "Write a review first."})
            return
        self._ensure_model()
        assert self.model is not None
        ids, truncated, _tokens = self.corpus.encode(cleaned)
        trace = explain_example(snapshot_weights(self.model), ids, self.corpus.itos, None)
        trace["truncated"] = truncated
        trace["text"] = cleaned[:1500]
        self._finish_trace(trace, "probe", None, None)
        self.bus.emit(
            {
                "type": "predict",
                "id": request_id,
                "task": "reviews",
                "text": cleaned[:1500],
                "prob": trace["prob"],
                "logit": trace["logit"],
                "pred": trace["pred"],
                "untrained": self.trained_steps == 0,
                "truncated": truncated,
                "tokens": trace["tokens"][:96],
            }
        )

    def _predict_digits(self, raw: Any, request_id: str) -> None:
        if not isinstance(raw, list) or len(raw) != PIXELS:
            self.bus.emit({"type": "error", "message": "Draw on the 28 by 28 grid first."})
            return
        pixels = np.zeros(PIXELS, dtype=np.float32)
        for index, value in enumerate(raw):
            try:
                pixels[index] = float(value)
            except (TypeError, ValueError):
                pixels[index] = 0.0
        self._ensure_model()
        assert self.model is not None
        trace = explain_digit(snapshot_weights(self.model), pixels, None, SIDE)
        self._finish_trace(trace, "probe", None, None)
        self.bus.emit(
            {
                "type": "predict",
                "id": request_id,
                "task": "digits",
                "text": "",
                "prob": trace["prob"],
                "logit": trace["logit"],
                "pred": trace["pred"],
                "probs": trace["probs"],
                "untrained": self.trained_steps == 0,
                "truncated": False,
                "tokens": [],
            }
        )

    def _full_eval(self) -> None:
        self._ensure_model()
        test_x, test_y = self._split("test")
        assert test_x is not None and test_y is not None
        previous = self.mode
        self._status("evaluating", "Scoring the test set")
        accuracy, loss = self._score(test_x, test_y)
        self.test_acc = round(accuracy, 5)
        self.test_loss = round(loss, 5)
        self._publish_metrics()
        self._checkpoint()
        if previous == "running":
            self.mode = "running"
            self._status("running", "Training on Metal")
        else:
            self.mode = previous if previous in ("paused", "ready") else "paused"
            self._status(self.mode, f"Test accuracy {accuracy:.1%}")

    def _one_step(self, capture: bool, honor_budget: bool) -> bool:
        assert self.model is not None and self.optimizer is not None
        assert self._loss_and_grad is not None
        batch = self._next_batch(honor_budget)
        if batch is None:
            return False
        batch_x, labels, picked = batch
        if self.task == "gait":
            assert isinstance(self.model, GaitNet)
            trace = None
            if capture:
                person = int(self.gait.train_person[int(picked[0])]) if self.gait.train_person is not None else None
                trace = explain_gait(
                    self.model,
                    batch_x[0].astype(np.float32),
                    int(labels[0]),
                    person=person,
                    index=int(picked[0]),
                )
            mx_x = mx.array(batch_x.astype(np.float32))
            mx_y = mx.array(labels.astype(np.int32))
            (loss, accuracy), grads = self._loss_and_grad(self.model, mx_x, mx_y)
            grads, _norm = optim.clip_grad_norm(grads, 1.0)
            self.optimizer.update(self.model, grads)
            mx.eval(loss, accuracy, self.model.parameters())
            loss_v = float(loss.item())
            acc_v = float(accuracy.item())
            self.trained_steps += 1
            self._locked = True
            self._rate_examples += int(batch_x.shape[0])
            self.history.append(
                {
                    "step": self.trained_steps,
                    "loss": round(loss_v, 5),
                    "accuracy": round(acc_v, 5),
                    "val": None,
                }
            )
            if len(self.history) > 4000:
                self.history = self.history[-4000:]
            if trace is not None:
                self._finish_trace(trace, "train", loss_v, acc_v)
            return True
        if self.task == "digits":
            mx_x = mx.array(to_unit_interval(batch_x).reshape(batch_x.shape))
            mx_y = mx.array(labels.astype(np.int32))
            loss, grads = self._loss_and_grad(self.model, mx_x, mx_y)
            logits = self.model(mx_x)
            accuracy = mx.mean((mx.argmax(logits, axis=-1) == mx_y).astype(mx.float32))
        else:
            mx_x = mx.array(batch_x.astype(np.int32))
            mx_y = mx.array(labels.astype(np.float32))
            loss, grads = self._loss_and_grad(self.model, mx_x, mx_y)
            logits = self.model(mx_x)
            accuracy = mx.mean(
                (mx.sigmoid(logits) >= 0.5).astype(mx.float32) == (mx_y >= 0.5).astype(mx.float32)
            )
        mx.eval(loss, accuracy, grads)
        loss_v = float(loss.item())
        acc_v = float(accuracy.item())
        trace = None
        if capture:
            if self.task == "digits":
                trace = explain_digit(snapshot_weights(self.model), batch_x[0], int(labels[0]), SIDE)
            else:
                assert self.corpus is not None
                trace = explain_example(
                    snapshot_weights(self.model),
                    batch_x[0],
                    self.corpus.itos,
                    int(labels[0]),
                )
                trace["text"] = " ".join(trace["tokens"])[:1500]
            self._attach_grads(trace, grads)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        self.trained_steps += 1
        self._locked = True
        self._rate_examples += int(batch_x.shape[0])
        self.history.append(
            {
                "step": self.trained_steps,
                "loss": round(loss_v, 5),
                "accuracy": round(acc_v, 5),
                "val": None,
            }
        )
        if len(self.history) > 4000:
            self.history = self.history[-4000:]
        if trace is not None:
            self._finish_trace(trace, "train", loss_v, acc_v)
        return True

    def _next_batch(self, honor_budget: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        train_x, train_y = self._split("train")
        assert train_x is not None and train_y is not None
        count = int(train_x.shape[0])
        if self.perm is None or self.cursor >= count:
            if self.perm is not None:
                self.passes += 1
                self._validate()
                self._checkpoint()
                if honor_budget and self.passes >= self.pass_limit:
                    self.mode = "paused"
                    self._publish_metrics()
                    self._status("paused", f"Finished {self.config['epochs']} epochs.")
                    self.perm = None
                    self.cursor = 0
                    return None
            self.perm = np.random.default_rng(self.config["seed"] + self.passes + 1).permutation(count)
            self.cursor = 0
        batch = int(self.config["batchSize"])
        index = self.perm[self.cursor : self.cursor + batch]
        self.cursor += int(index.shape[0])
        return train_x[index], train_y[index], index

    def _ensure_model(self, preview: bool = True) -> None:
        if self.model is not None:
            return
        assert self.data_ready
        mx.random.seed(int(self.config["seed"]))
        if self.task == "gait":
            assert self.gait.class_weight is not None
            self.model = GaitNet(list(self.config["hidden"]))
            self._class_weight = mx.array(self.gait.class_weight.astype(np.float32))
            self._loss_and_grad = None
            self._compiled_step = None
        elif self.task == "digits":
            self.model = DigitNet(list(self.config["hidden"]), PIXELS, CLASSES)
            self._loss_and_grad = nn.value_and_grad(self.model, digit_loss)
            self._compiled_step = None
        else:
            assert self.corpus is not None
            self.model = SentimentNet(len(self.corpus.itos), int(self.config["embedDim"]), list(self.config["hidden"]))
            self._loss_and_grad = nn.value_and_grad(self.model, loss_only)
            self._compiled_step = None
        self.optimizer = self._new_optimizer()
        mx.eval(self.model.parameters())
        self.parameter_count = _count_parameters(self.model)
        self.optimizer.init(self.model.trainable_parameters())
        mx.eval(self.optimizer.state)
        if self.task == "gait":
            self._bind_gait_step()
        self._publish_config()
        if preview:
            self._publish_preview()

    def _new_optimizer(self) -> optim.Optimizer:
        learning_rate = float(self.config["lr"])
        weight_decay = float(self.config["weightDecay"])
        if self.config["optimizer"] == "sgd":
            return optim.SGD(learning_rate=learning_rate, weight_decay=weight_decay)
        return optim.AdamW(
            learning_rate=learning_rate,
            betas=[float(self.config["beta1"]), float(self.config["beta2"])],
            eps=float(self.config["eps"]),
            weight_decay=weight_decay,
            bias_correction=True,
        )

    def _publish_preview(self) -> None:
        assert self.model is not None
        train_x, train_y = self._split("train")
        assert train_x is not None and train_y is not None
        if self.task == "gait":
            assert isinstance(self.model, GaitNet)
            person = int(self.gait.train_person[0]) if self.gait.train_person is not None and self.gait.train_person.size else None
            trace = explain_gait(self.model, train_x[0].astype(np.float32), int(train_y[0]), person=person, index=0)
        elif self.task == "digits":
            trace = explain_digit(snapshot_weights(self.model), train_x[0], int(train_y[0]), SIDE)
        else:
            assert self.corpus is not None
            trace = explain_example(snapshot_weights(self.model), train_x[0], self.corpus.itos, int(train_y[0]))
            trace["text"] = " ".join(trace["tokens"])[:1500]
        self._finish_trace(trace, "preview", None, None)

    def _bind_gait_step(self) -> None:
        assert isinstance(self.model, GaitNet)
        assert self._class_weight is not None
        model = self.model
        weights = self._class_weight

        def loss_fn(model: GaitNet, x: mx.array, y: mx.array) -> tuple[mx.array, mx.array]:
            logits = model(x)
            per = nn.losses.cross_entropy(logits, y, reduction="none")
            chosen = weights[y]
            loss = mx.sum(per * chosen) / mx.sum(chosen)
            accuracy = mx.mean((mx.argmax(logits, axis=-1) == y).astype(mx.float32))
            return loss, accuracy

        self._compiled_step = None
        self._loss_and_grad = nn.value_and_grad(model, loss_fn)

    def _predict_gait(self, raw_index: Any, request_id: str) -> None:
        self._ensure_model()
        assert isinstance(self.model, GaitNet)
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            index = 0
        window, label, person = self.gait.window("test", index)
        total = int(self.gait.test_y.shape[0]) if self.gait.test_y is not None else 0
        chosen = index % total if total else 0
        trace = explain_gait(self.model, window, label, person=person, index=chosen)
        self._finish_trace(trace, "probe", None, None)
        self.bus.emit(
            {
                "type": "predict",
                "id": request_id,
                "task": "gait",
                "text": "",
                "index": chosen,
                "total": total,
                "prob": trace["prob"],
                "logit": trace["logit"],
                "pred": trace["pred"],
                "label": label,
                "activity": activity_name(int(trace["pred"])),
                "labelName": activity_name(label),
                "probs": trace["probs"],
                "untrained": self.trained_steps == 0,
                "truncated": False,
                "tokens": [],
            }
        )

    def _finish_trace(self, trace: dict[str, Any], kind: str, batch_loss: float | None, batch_accuracy: float | None) -> None:
        self._attach_moments(trace)
        trace["task"] = self.task
        trace["kind"] = kind
        trace["step"] = self.trained_steps
        trace["seq"] = self._trace_seq
        self._trace_seq += 1
        trace["batchLoss"] = None if batch_loss is None else round(batch_loss, 5)
        trace["batchAccuracy"] = None if batch_accuracy is None else round(batch_accuracy, 5)
        trace["untrained"] = self.trained_steps == 0
        trace["optimizer"] = self._optimizer_payload()
        self.bus.publish({"type": "trace", "trace": trace})

    def _attach_grads(self, trace: dict[str, Any], grads: dict[str, Any]) -> None:
        if "embed" in grads and "embed" in trace:
            trace["embed"]["batchGradRms"] = _mx_rms(grads["embed"]["weight"])
        hidden_grads = grads["hidden"]
        for index, layer in enumerate(trace["layers"][:-1]):
            layer["batchGradRms"] = _mx_rms(hidden_grads[index]["weight"])
            layer["biasGradRms"] = _mx_rms(hidden_grads[index]["bias"])
        trace["layers"][-1]["batchGradRms"] = _mx_rms(grads["output"]["weight"])
        trace["layers"][-1]["biasGradRms"] = _mx_rms(grads["output"]["bias"])

    def _attach_moments(self, trace: dict[str, Any]) -> None:
        if self.optimizer is None or self.config["optimizer"] != "adam":
            return
        state = self.optimizer.state
        mx.eval(state)

        def paint(target: dict[str, Any], path: tuple[Any, ...]) -> None:
            leaf = _leaf(state, path)
            if not isinstance(leaf, dict) or "m" not in leaf:
                return
            target["momentRms"] = _mx_rms(leaf["m"])
            if "v" in leaf:
                target["secondMomentRms"] = _mx_rms(leaf["v"])

        if "embed" in trace:
            paint(trace["embed"], ("embed", "weight"))
        for index, layer in enumerate(trace["layers"][:-1]):
            paint(layer, ("hidden", index, "weight"))
        paint(trace["layers"][-1], ("output", "weight"))

    def _optimizer_payload(self) -> dict[str, Any]:
        step = 0
        learning_rate = float(self.config["lr"])
        if self.optimizer is not None:
            raw_step = self.optimizer.state.get("step", 0)
            step = int(raw_step.item()) if hasattr(raw_step, "item") else int(raw_step)
            raw_lr = self.optimizer.learning_rate
            learning_rate = float(raw_lr.item()) if hasattr(raw_lr, "item") else float(raw_lr)
        return {
            "name": self.config["optimizer"],
            "lr": learning_rate,
            "beta1": float(self.config["beta1"]),
            "beta2": float(self.config["beta2"]),
            "eps": float(self.config["eps"]),
            "weightDecay": float(self.config["weightDecay"]),
            "biasCorrection": self.config["optimizer"] == "adam",
            "step": step,
        }

    def _validate(self) -> None:
        test_x, test_y = self._split("test")
        assert test_x is not None and test_y is not None
        take = min(2000, int(test_x.shape[0]))
        picked = np.random.default_rng(int(self.config["seed"]) + 17).choice(test_x.shape[0], size=take, replace=False)
        phase = self.mode if self.mode in ("running", "paused", "ready", "evaluating") else "running"
        noun = "windows" if self.task == "gait" else "digits" if self.task == "digits" else "reviews"
        self._status(phase, f"Checking 2,000 test {noun}")
        accuracy, loss = self._score(test_x[picked], test_y[picked])
        self.val_acc = round(accuracy, 5)
        self.val_loss = round(loss, 5)
        if self.history:
            last = dict(self.history[-1])
            last["val"] = self.val_acc
            self.history[-1] = last

    def _score(self, features: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        assert self.model is not None
        total = int(features.shape[0])
        if total == 0:
            return 0.0, 0.0
        correct = 0.0
        loss_sum = 0.0
        seen = 0
        digits = self.task == "digits"
        gait = self.task == "gait"
        for start in range(0, total, 256):
            batch_x = features[start : start + 256]
            batch_y = labels[start : start + 256]
            if gait:
                logits = self.model(mx.array(batch_x.astype(np.float32)))
                targets = mx.array(batch_y.astype(np.int32))
                loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
                hits = mx.sum(mx.argmax(logits, axis=-1) == targets)
            elif digits:
                logits = self.model(mx.array(to_unit_interval(batch_x).reshape(batch_x.shape)))
                targets = mx.array(batch_y.astype(np.int32))
                loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
                hits = mx.sum(mx.argmax(logits, axis=-1) == targets)
            else:
                logits = self.model(mx.array(batch_x.astype(np.int32)))
                targets = mx.array(batch_y.astype(np.float32))
                loss = nn.losses.binary_cross_entropy(logits, targets, with_logits=True, reduction="sum")
                hits = mx.sum((mx.sigmoid(logits) >= 0.5).astype(mx.float32) == targets)
            mx.eval(loss, hits)
            loss_sum += float(loss.item())
            correct += float(hits.item())
            seen += int(batch_x.shape[0])
        return correct / seen, loss_sum / seen

    def _trace_due(self) -> bool:
        now = time.perf_counter()
        interval = 0.5 if self.task == "gait" else 0.25
        if now - self._last_trace >= interval:
            self._last_trace = now
            return True
        return False

    def _maybe_publish_metrics(self) -> None:
        now = time.perf_counter()
        if now - self._last_metric < 0.2:
            return
        self._last_metric = now
        self._publish_metrics()
        if self.mode == "running":
            self._status("running", "Training on Metal")

    def _publish_metrics(self, reset_rate: bool = True) -> None:
        if reset_rate:
            now = time.perf_counter()
            elapsed = now - self._rate_time
            if 0.05 < elapsed < 3 and self._rate_examples:
                self._examples_per_sec = self._rate_examples / elapsed
            self._rate_examples = 0
            self._rate_time = now
        last = self.history[-1] if self.history else None
        self.bus.publish(
            {
                "type": "metrics",
                "step": self.trained_steps,
                "epoch": self._epoch_display(),
                "epochs": int(self.config["epochs"]),
                "loss": None if last is None else last["loss"],
                "accuracy": None if last is None else last["accuracy"],
                "valAccuracy": self.val_acc,
                "valLoss": self.val_loss,
                "testAccuracy": self.test_acc,
                "testLoss": self.test_loss,
                "examplesPerSec": round(self._examples_per_sec, 1),
                "history": self._chart_history(),
            }
        )

    def _epoch_display(self) -> int:
        within = self.passes - self.run_origin + 1
        if (
            self.mode != "running"
            and self.pass_limit > self.run_origin
            and self.passes >= self.pass_limit
        ):
            within = self.passes - self.run_origin
        return max(1, within)

    def _epoch_progress(self) -> float:
        train_x, _train_y = self._split("train")
        count = 0 if train_x is None else int(train_x.shape[0])
        if count <= 0 or self.perm is None:
            return 0.0
        return round(min(1.0, self.cursor / count), 4)

    def _chart_history(self) -> list[dict[str, Any]]:
        points = self.history
        limit = 480
        if len(points) <= limit:
            return points
        stride = len(points) / limit
        picked = [points[int(i * stride)] for i in range(limit - 1)]
        picked.append(points[-1])
        return picked

    def _status(self, phase: str, message: str, progress: float | None = None) -> None:
        if progress is None and phase == "running":
            progress = self._epoch_progress()
        if phase != self.mode or message != self._message:
            log.info("%s: %s", phase, message)
        if phase in ("preparing", "ready", "running", "paused", "evaluating", "error"):
            self.mode = phase
        self._message = message
        self._progress = progress
        self.bus.publish(
            {
                "type": "status",
                "phase": self.mode,
                "message": message,
                "progress": progress,
                "locked": self._locked,
                "trainedSteps": self.trained_steps,
                "metal": self.metal,
                "task": self.task,
            }
        )

    def _checkpoint(self) -> None:
        if self.model is None or self.trained_steps <= 0:
            return
        self.ckpt.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(self.weights_path))
        payload = {
            "task": self.task,
            "config": self.config,
            "steps": self.trained_steps,
            "passes": self.passes,
            "history": self._chart_history(),
            "valAccuracy": self.val_acc,
            "valLoss": self.val_loss,
            "testAccuracy": self.test_acc,
            "testLoss": self.test_loss,
        }
        if self.task == "gait":
            payload["dataset"] = "hugadb-v2-acc-emg"
            payload["window"] = WINDOW
            payload["channels"] = CHANNELS
        elif self.task == "digits":
            payload["dataset"] = "mnist28"
        else:
            assert self.corpus is not None
            payload["vocab"] = hashlib.sha256("\n".join(self.corpus.itos).encode()).hexdigest()[:16]
        self.meta_path.write_text(json.dumps(payload))

    def _load_checkpoint(self) -> None:
        if not self.weights_path.exists() or not self.meta_path.exists():
            return
        meta = json.loads(self.meta_path.read_text())
        if meta.get("task", "gait") != self.task:
            self._clear_checkpoint_files()
            return
        if self.task == "gait":
            if meta.get("dataset") != "hugadb-v2-acc-emg" or int(meta.get("window") or 0) != WINDOW:
                self._clear_checkpoint_files()
                return
        elif self.task == "digits":
            if meta.get("dataset") != "mnist28":
                self._clear_checkpoint_files()
                return
        else:
            assert self.corpus is not None
            vocab_id = hashlib.sha256("\n".join(self.corpus.itos).encode()).hexdigest()[:16]
            if meta.get("vocab") != vocab_id:
                self._clear_checkpoint_files()
                return
        self.config = sanitize(meta.get("config") or self.config, self.task)
        self._publish_config()
        self._ensure_model(preview=False)
        assert self.model is not None
        self.model.load_weights(str(self.weights_path))
        mx.eval(self.model.parameters())
        self.trained_steps = int(meta.get("steps") or 0)
        self.passes = int(meta.get("passes") or 0)
        self.run_origin = self.passes
        self.pass_limit = self.passes
        self.history = list(meta.get("history") or [])
        self.val_acc = meta.get("valAccuracy")
        self.val_loss = meta.get("valLoss")
        self.test_acc = meta.get("testAccuracy")
        self.test_loss = meta.get("testLoss")
        self._locked = self.trained_steps > 0
        self._publish_preview()

    def _clear_checkpoint_files(self) -> None:
        for path in (self.weights_path, self.meta_path):
            if path.exists():
                path.unlink()

    def page(self, split: str, offset: int, limit: int) -> dict:
        if self.task == "gait":
            return self.gait.page(split, offset, limit)
        if self.task == "digits":
            assert self.digits is not None
            return self.digits.page(split, offset, limit)
        assert self.corpus is not None
        return self.corpus.page(split, offset, limit)

    def _publish_catalog(self) -> None:
        self.bus.publish({"type": "catalog", "tasks": self.tasks(), "task": self.task})

    def _publish_config(self) -> None:
        payload = dict(self.config)
        if self.task == "gait":
            widths, titles = layout(list(self.config["hidden"]))
            payload["widths"] = widths
            payload["titles"] = titles
            payload["frameRows"] = CHANNELS
            payload["frameCols"] = WINDOW
            if self.parameter_count:
                payload["parameters"] = self.parameter_count
        self.bus.publish({"type": "config", "config": payload, "task": self.task})

    def _publish_dataset(self) -> None:
        if self.task == "gait":
            self.bus.publish(self.gait.dataset_message())
            return
        if self.task == "digits":
            assert self.digits is not None
            self.bus.publish(self.digits.dataset_message())
            return
        assert self.corpus is not None
        self.bus.publish(self.corpus.dataset_message())

    def _split(self, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.task == "gait":
            if name == "test":
                return self.gait.test_x, self.gait.test_y
            return self.gait.train_x, self.gait.train_y
        if self.task == "digits":
            assert self.digits is not None
            if name == "test":
                return self.digits.test_x, self.digits.test_y
            return self.digits.train_x, self.digits.train_y
        assert self.corpus is not None
        if name == "test":
            return self.corpus.test_x, self.corpus.test_y
        return self.corpus.train_x, self.corpus.train_y

    def _switch_task(self, name: str) -> None:
        if name not in self.tasks() or name == self.task:
            return
        if not self.metal:
            self.bus.emit({"type": "error", "message": "Metal is not available, so training stays off."})
            return
        if self.mode == "evaluating":
            return
        if self.mode == "running":
            self._pause()
        elif self.trained_steps > 0:
            self._checkpoint()
        self._saved[self.task] = self._pack()
        self.task = name
        slot = self._saved.get(name)
        if slot is not None:
            self._unpack(slot)
            self._publish_dataset()
            self._publish_config()
            self._publish_metrics()
            if self.model is not None:
                self._publish_preview()
            phase = "paused" if self.trained_steps else "ready"
            self._status(phase, "Paused" if phase == "paused" else "Ready on Metal")
            return
        if name == "digits":
            fresh = DIGITS_CONFIG
        elif name == "gait":
            fresh = GAIT_CONFIG
        else:
            fresh = DEFAULT_CONFIG
        self._blank_runtime(fresh)
        self._try_prepare()

    def _pack(self) -> dict[str, Any]:
        return {
            "config": copy.deepcopy(self.config),
            "model": self.model,
            "optimizer": self.optimizer,
            "loss_and_grad": self._loss_and_grad,
            "trained_steps": self.trained_steps,
            "passes": self.passes,
            "run_origin": self.run_origin,
            "pass_limit": self.pass_limit,
            "perm": self.perm,
            "cursor": self.cursor,
            "history": self.history,
            "val_acc": self.val_acc,
            "val_loss": self.val_loss,
            "test_acc": self.test_acc,
            "test_loss": self.test_loss,
            "locked": self._locked,
        }

    def _unpack(self, slot: dict[str, Any]) -> None:
        self.config = copy.deepcopy(slot["config"])
        self.model = slot["model"]
        self.optimizer = slot["optimizer"]
        self._loss_and_grad = slot["loss_and_grad"]
        self.trained_steps = int(slot["trained_steps"])
        self.passes = int(slot["passes"])
        self.run_origin = int(slot["run_origin"])
        self.pass_limit = int(slot["pass_limit"])
        self.perm = None
        self.cursor = 0
        self.history = slot["history"]
        self.val_acc = slot["val_acc"]
        self.val_loss = slot["val_loss"]
        self.test_acc = slot["test_acc"]
        self.test_loss = slot["test_loss"]
        self._locked = bool(slot["locked"])

    def _blank_runtime(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.config["hidden"] = list(config["hidden"])
        self.model = None
        self.optimizer = None
        self._loss_and_grad = None
        self.trained_steps = 0
        self.passes = 0
        self.run_origin = 0
        self.pass_limit = 0
        self.perm = None
        self.cursor = 0
        self.history = []
        self.val_acc = None
        self.val_loss = None
        self.test_acc = None
        self.test_loss = None
        self._locked = False
        self.bus.publish({"type": "trace", "trace": None})
        self._publish_config()
