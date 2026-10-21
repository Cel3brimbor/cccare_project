"""Leg acceleration and quadriceps EMG, from HuGaDB.

HuGaDB records both legs with accelerometers on the feet, shins, and thighs,
plus a muscle sensor on each quadriceps. Windows are labeled with the activity
at the middle of the window. Gyroscopes are in the files and are left out:
several recordings have corrupted gyro channels, and this network is the
accelerometer plus muscle path.

The trunk is a temporal residual network sized so a training step fills an
Apple M3 Max GPU. The head is a narrow ReLU layer so one column can still be
read as a dot product.
"""

from __future__ import annotations

import base64
import zipfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.app.model import _cross_entropy, _pack_matrix, _round_list, _trace_layers

WINDOW = 128
STRIDE = 64
CHANNELS = 20
STEM = 768
MID = 1024
DEEP = 1280
BLOCKS = 2
KERNEL = 5
RATE = 56.35
TEST_PEOPLE = (16, 17, 18)
CACHE_TAG = "g3"
HEAD_FULL_LIMIT = 220_000
URLS = ("https://www.dropbox.com/s/406z5oqjlr6fcy9/HuGaDB%20v2.zip?dl=1",)

ACTIVITY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11)
ACTIVITIES: tuple[tuple[str, str], ...] = (
    ("Walking", "Walk"),
    ("Running", "Run"),
    ("Stairs up", "Up"),
    ("Stairs down", "Down"),
    ("Sitting", "Sit"),
    ("Sitting down", "Lower"),
    ("Standing up", "Rise"),
    ("Standing", "Stand"),
    ("Elevator up", "Lift+"),
    ("Elevator down", "Lift-"),
)
ID_TO_INDEX = {activity_id: index for index, activity_id in enumerate(ACTIVITY_IDS)}

SITES = (
    "Right foot",
    "Right shin",
    "Right thigh",
    "Left foot",
    "Left shin",
    "Left thigh",
)
CHANNEL_NAMES = [f"{site} accel {axis}" for site in SITES for axis in "XYZ"] + [
    "Right quadriceps",
    "Left quadriceps",
]
CHANNEL_KINDS = ["accel"] * 18 + ["muscle"] * 2
FEATURE_INDEX = [block * 6 + axis for block in range(6) for axis in range(3)] + [36, 37]

Progress = Callable[[str, float | None], None]


def _emit(values: np.ndarray) -> Any:
    array = np.ascontiguousarray(values, dtype=np.float32)
    if array.size <= 4096:
        return _round_list(array)
    return {
        "dtype": "f32",
        "shape": [int(dim) for dim in array.shape],
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("dtype") == "f32":
            return value
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, list):
        if not value or isinstance(value[0], (dict, str)):
            return [_compact(item) for item in value]
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            return [_compact(item) for item in value]
        if array.dtype == np.dtype("object") or array.size <= 4096:
            return [_compact(item) for item in value]
        return _emit(array)
    return value


def layout(head: list[int]) -> tuple[list[int], list[str]]:
    titles = ["Sensors", "Stem", "Mid", "Deep"]
    if len(head) == 1:
        titles.append("Head")
    else:
        titles.extend(f"Head {index + 1}" for index in range(len(head)))
    titles.append("Activity")
    widths = [WINDOW * CHANNELS, STEM, MID, DEEP, *[int(width) for width in head], len(ACTIVITIES)]
    return widths, titles


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm1 = nn.LayerNorm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=pad)
        self.norm2 = nn.LayerNorm(channels)

    def __call__(self, x: mx.array) -> mx.array:
        y = nn.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return nn.gelu(x + y)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = KERNEL):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(cin, cout, kernel, stride=2, padding=pad)
        self.norm = nn.LayerNorm(cout)
        self.skip = nn.Conv1d(cin, cout, 1, stride=2)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.norm(self.conv(x)) + self.skip(x))


class GaitNet(nn.Module):
    def __init__(self, head: list[int]):
        super().__init__()
        self.stem = nn.Conv1d(CHANNELS, STEM, 9, padding=4)
        self.stem_norm = nn.LayerNorm(STEM)
        self.blocks1 = [ResBlock(STEM) for _ in range(BLOCKS)]
        self.down1 = Down(STEM, MID)
        self.blocks2 = [ResBlock(MID) for _ in range(BLOCKS)]
        self.down2 = Down(MID, DEEP)
        self.blocks3 = [ResBlock(DEEP) for _ in range(BLOCKS)]
        widths = [DEEP, *[int(width) for width in head]]
        self.head = [nn.Linear(widths[index], widths[index + 1]) for index in range(len(head))]
        self.output = nn.Linear(widths[-1], len(ACTIVITIES))

    def __call__(self, x: mx.array) -> mx.array:
        pooled = mx.mean(self.trunk(x), axis=1)
        hidden = pooled
        for layer in self.head:
            hidden = mx.maximum(layer(hidden), 0)
        return self.output(hidden)

    def trunk(self, x: mx.array) -> mx.array:
        hidden = nn.gelu(self.stem_norm(self.stem(x)))
        for block in self.blocks1:
            hidden = block(hidden)
        hidden = self.down1(hidden)
        for block in self.blocks2:
            hidden = block(hidden)
        hidden = self.down2(hidden)
        for block in self.blocks3:
            hidden = block(hidden)
        return hidden

    def inspect(self, window: mx.array) -> dict[str, mx.array]:
        if window.ndim == 2:
            window = window[None]
        hidden = nn.gelu(self.stem_norm(self.stem(window)))
        for block in self.blocks1:
            hidden = block(hidden)
        stem = hidden
        hidden = self.down1(hidden)
        for block in self.blocks2:
            hidden = block(hidden)
        mid = hidden
        hidden = self.down2(hidden)
        for block in self.blocks3:
            hidden = block(hidden)
        pooled = mx.mean(hidden, axis=1)
        return {
            "stem": mx.mean(mx.abs(stem), axis=1)[0],
            "mid": mx.mean(mx.abs(mid), axis=1)[0],
            "pooled": pooled[0],
        }


def _coupling(weight: mx.array) -> np.ndarray:
    rms = mx.sqrt(mx.mean(mx.square(weight), axis=1))
    return np.array(rms.T, dtype=np.float32)


def _take_outputs(weight: mx.array, col_index: list[int] | None) -> mx.array:
    if col_index is None:
        return weight
    return weight[mx.array(np.asarray(col_index, dtype=np.int32))]


def _conv_record(
    name: str,
    title: str,
    detail: str,
    output: np.ndarray,
    bias: mx.array,
    weight: mx.array,
    *,
    frame_cols: int | None,
    with_kernel: bool,
) -> dict[str, Any]:
    coupling = _coupling(weight)
    matrix, row_index, col_index = _pack_matrix(coupling, keep_rows=frame_cols is not None)
    temporal = mx.sqrt(mx.mean(mx.square(weight), axis=2))
    temporal = _take_outputs(temporal, col_index)
    record: dict[str, Any] = {
        "name": name,
        "title": title,
        "activation": "conv",
        "detail": detail,
        "input": _round_list(np.mean(np.abs(output), keepdims=True)),
        "bias": _round_list(np.array(bias, dtype=np.float32)),
        "pre": _round_list(output),
        "output": _round_list(output),
        "weight": _round_list(matrix),
        "weightMean": float(coupling.mean()),
        "weightStd": float(coupling.std()),
        "batchGradRms": None,
        "biasGradRms": None,
        "momentRms": None,
        "secondMomentRms": None,
        "taps": int(weight.shape[1]),
        "temporal": _round_list(np.array(temporal, dtype=np.float32)),
    }
    if frame_cols is not None:
        record["frameCols"] = int(frame_cols)
    if row_index is not None:
        record["rowIndex"] = row_index
    if col_index is not None:
        record["colIndex"] = col_index
    if with_kernel:
        kernel = _take_outputs(weight, col_index)
        record["kernel"] = _round_list(np.array(kernel, dtype=np.float32))
        record["kernelIndex"] = col_index if col_index is not None else list(range(int(weight.shape[0])))
    return record


def explain_gait(
    model: GaitNet,
    window: np.ndarray,
    label: int | None,
    *,
    person: int | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    frame = np.asarray(window, dtype=np.float32)
    if frame.shape != (WINDOW, CHANNELS):
        fitted = np.zeros((WINDOW, CHANNELS), dtype=np.float32)
        rows = min(WINDOW, frame.shape[0])
        cols = min(CHANNELS, frame.shape[1] if frame.ndim == 2 else 0)
        if frame.ndim == 1:
            fitted.reshape(-1)[: min(fitted.size, frame.size)] = frame[: min(fitted.size, frame.size)]
        else:
            fitted[:rows, :cols] = frame[:rows, :cols]
        frame = fitted
    viewed = mx.array(frame)
    parts = model.inspect(viewed)
    mx.eval(parts["stem"], parts["mid"], parts["pooled"])
    pooled = np.array(parts["pooled"], dtype=np.float32)
    stem = np.array(parts["stem"], dtype=np.float32)
    mid = np.array(parts["mid"], dtype=np.float32)
    params = model.parameters()
    head_weights = [{"weight": np.array(layer["weight"]), "bias": np.array(layer["bias"])} for layer in params["head"]]
    output_weight = {"weight": np.array(params["output"]["weight"]), "bias": np.array(params["output"]["bias"])}
    widest = 0
    cursor = DEEP
    for layer in head_weights + [output_weight]:
        widest = max(widest, cursor * int(np.asarray(layer["weight"]).shape[0]))
        cursor = int(np.asarray(layer["weight"]).shape[0])
    linear = _trace_layers(
        pooled,
        head_weights + [output_weight],
        last_name="activity",
        last_title="Activity",
        last_activation="softmax",
        label=label,
        keep_first_rows=False,
        sample=widest > HEAD_FULL_LIMIT,
    )
    if linear:
        linear[0]["title"] = "Head" if len(head_weights) == 1 else "Head 1"
        for index_layer, layer in enumerate(linear[:-1]):
            layer["name"] = f"head-{index_layer + 1}"
            if len(head_weights) > 1:
                layer["title"] = f"Head {index_layer + 1}"
    convs = [
        _conv_record(
            "stem",
            "Stem",
            "A 9-sample convolution into 768 channels, then two residual blocks. The value is the mean absolute activation over the window.",
            stem,
            params["stem"]["bias"],
            params["stem"]["weight"],
            frame_cols=WINDOW,
            with_kernel=True,
        ),
        _conv_record(
            "mid",
            "Mid",
            "A stride-2 convolution into 1024 channels, then two residual blocks. The value is the mean absolute activation over the window.",
            mid,
            params["down1"]["conv"]["bias"],
            params["down1"]["conv"]["weight"],
            frame_cols=None,
            with_kernel=False,
        ),
        _conv_record(
            "deep",
            "Deep",
            "A stride-2 convolution into 1280 channels, then two residual blocks. The value is the signed average over the window. That average is the vector the head multiplies.",
            pooled,
            params["down2"]["conv"]["bias"],
            params["down2"]["conv"]["weight"],
            frame_cols=None,
            with_kernel=False,
        ),
    ]
    layers = convs + linear
    logits = np.asarray(linear[-1]["pre"], dtype=np.float32) if linear else np.zeros(len(ACTIVITIES), dtype=np.float32)
    probabilities = np.asarray(linear[-1]["output"], dtype=np.float32) if linear else _softmax_np(logits)
    prediction = int(np.argmax(probabilities))
    flat = np.ascontiguousarray(frame.T).reshape(-1)
    return _compact({
        "channels": CHANNEL_NAMES,
        "frame": _round_list(flat),
        "width": WINDOW,
        "height": CHANNELS,
        "activityNames": [name for name, _short in ACTIVITIES],
        "activityShort": [short for _name, short in ACTIVITIES],
        "layers": layers,
        "logits": [round(float(value), 6) for value in logits.tolist()],
        "probs": [round(float(value), 6) for value in probabilities.tolist()],
        "logit": round(float(logits[prediction]), 6),
        "prob": round(float(probabilities[prediction]), 6),
        "pred": prediction,
        "label": None if label is None else int(label),
        "exampleLoss": None if label is None else round(_cross_entropy(logits, int(label)), 6),
        "person": None if person is None else int(person),
        "index": None if index is None else int(index),
    })


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0:
        return np.full_like(logits, 1.0 / max(1, logits.size))
    return exp / total


class GaitData:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw" / "hugadb"
        self.cache = root / "cache"
        self.archive = root / "raw" / "HuGaDB_v2.zip"
        self.train_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.train_person: np.ndarray | None = None
        self.test_x: np.ndarray | None = None
        self.test_y: np.ndarray | None = None
        self.test_person: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.class_weight: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.train_x is not None

    def cache_path(self) -> Path:
        return self.cache / f"hugadb_v2_w{WINDOW}_s{STRIDE}_{CACHE_TAG}.npz"

    def load_cache(self) -> bool:
        path = self.cache_path()
        if not path.exists():
            return False
        payload = np.load(path)
        self.train_x = payload["train_x"]
        self.train_y = payload["train_y"]
        self.train_person = payload["train_person"]
        self.test_x = payload["test_x"]
        self.test_y = payload["test_y"]
        self.test_person = payload["test_person"]
        self.mean = payload["mean"]
        self.std = payload["std"]
        self.class_weight = payload["class_weight"]
        return (
            self.train_x.ndim == 3
            and self.train_x.shape[1:] == (WINDOW, CHANNELS)
            and self.train_y.shape[0] == self.train_x.shape[0]
            and int(self.class_weight.shape[0]) == len(ACTIVITIES)
        )

    def prepare(self, progress: Progress) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        if self.load_cache():
            progress("HuGaDB cache ready", 1)
            return
        files = self._recordings()
        if not files:
            if not self.archive.exists() or self.archive.stat().st_size < 1_000_000:
                self._download(progress)
            progress("Extracting HuGaDB", 0.08)
            self._extract()
            files = self._recordings()
        if not files:
            raise RuntimeError("HuGaDB extracted, but no recording files were found.")
        recordings = self._read_all(files, progress)
        if not recordings:
            raise RuntimeError("HuGaDB files did not contain usable windows.")
        progress("Scaling sensors", 0.86)
        train_rows = [features for person, features, _labels in recordings if person not in TEST_PEOPLE]
        if not train_rows:
            raise RuntimeError("No training recordings were left after holding out the test people.")
        stacked = np.concatenate(train_rows, axis=0)
        self.mean = stacked.mean(axis=0).astype(np.float32)
        self.std = np.maximum(stacked.std(axis=0).astype(np.float32), 1e-3)
        progress("Cutting windows", 0.9)
        train_x: list[np.ndarray] = []
        train_y: list[int] = []
        train_person: list[int] = []
        test_x: list[np.ndarray] = []
        test_y: list[int] = []
        test_person: list[int] = []
        for person, features, labels in recordings:
            scaled = ((features - self.mean) / self.std).astype(np.float32)
            holdout = person in TEST_PEOPLE
            count = int(scaled.shape[0])
            if count < WINDOW:
                continue
            for start in range(0, count - WINDOW + 1, STRIDE):
                mapped = ID_TO_INDEX.get(int(labels[start + WINDOW // 2]))
                if mapped is None:
                    continue
                window = scaled[start : start + WINDOW]
                bucket_x = test_x if holdout else train_x
                bucket_y = test_y if holdout else train_y
                bucket_p = test_person if holdout else train_person
                bucket_x.append(window.astype(np.float16))
                bucket_y.append(mapped)
                bucket_p.append(person)
        if not train_x or not test_x:
            raise RuntimeError("HuGaDB windows were empty for training or test.")
        self.train_x = np.stack(train_x)
        self.train_y = np.asarray(train_y, dtype=np.uint8)
        self.train_person = np.asarray(train_person, dtype=np.uint8)
        self.test_x = np.stack(test_x)
        self.test_y = np.asarray(test_y, dtype=np.uint8)
        self.test_person = np.asarray(test_person, dtype=np.uint8)
        counts = np.bincount(self.train_y.astype(np.int64), minlength=len(ACTIVITIES)).astype(np.float64)
        self.class_weight = (counts.sum() / np.maximum(counts, 1.0) / len(ACTIVITIES)).astype(np.float32)
        np.savez_compressed(
            self.cache_path(),
            train_x=self.train_x,
            train_y=self.train_y,
            train_person=self.train_person,
            test_x=self.test_x,
            test_y=self.test_y,
            test_person=self.test_person,
            mean=self.mean,
            std=self.std,
            class_weight=self.class_weight,
        )
        progress("Leg sensors ready", 1)

    def dataset_message(self) -> dict[str, Any]:
        assert self.train_y is not None and self.test_y is not None and self.train_person is not None
        counts = np.bincount(self.train_y.astype(np.int64), minlength=len(ACTIVITIES))
        people = sorted({int(person) for person in self.train_person.tolist()})
        return {
            "type": "dataset",
            "task": "gait",
            "name": "HuGaDB",
            "train": int(self.train_y.shape[0]),
            "test": int(self.test_y.shape[0]),
            "window": WINDOW,
            "stride": STRIDE,
            "rate": RATE,
            "seconds": round(WINDOW / RATE, 2),
            "channels": [{"name": name, "kind": kind} for name, kind in zip(CHANNEL_NAMES, CHANNEL_KINDS)],
            "activities": [
                {"name": name, "short": short, "count": int(counts[index])}
                for index, (name, short) in enumerate(ACTIVITIES)
            ],
            "trainPeople": people,
            "testPeople": list(TEST_PEOPLE),
        }

    def page(self, split: str, offset: int, limit: int) -> dict[str, Any]:
        if split not in ("train", "test"):
            split = "train"
        windows = self.train_x if split == "train" else self.test_x
        labels = self.train_y if split == "train" else self.test_y
        people = self.train_person if split == "train" else self.test_person
        assert windows is not None and labels is not None and people is not None
        total = int(windows.shape[0])
        start = min(max(0, offset), total)
        count = min(12, max(1, limit))
        end = min(total, start + count)
        rows = []
        for index in range(start, end):
            frame = np.ascontiguousarray(windows[index].astype(np.float32).T).reshape(-1)
            label = int(labels[index])
            rows.append(
                {
                    "index": index,
                    "label": label,
                    "activity": ACTIVITIES[label][0],
                    "person": int(people[index]),
                    "frame": np.round(frame, 3).tolist(),
                    "width": WINDOW,
                    "height": CHANNELS,
                }
            )
        return {
            "task": "gait",
            "split": split,
            "offset": start,
            "total": total,
            "width": WINDOW,
            "height": CHANNELS,
            "rows": rows,
        }

    def window(self, split: str, index: int) -> tuple[np.ndarray, int, int]:
        windows = self.train_x if split == "train" else self.test_x
        labels = self.train_y if split == "train" else self.test_y
        people = self.train_person if split == "train" else self.test_person
        assert windows is not None and labels is not None and people is not None
        total = int(windows.shape[0])
        if total == 0:
            raise RuntimeError("There are no windows in that split.")
        chosen = int(index) % total
        return windows[chosen].astype(np.float32), int(labels[chosen]), int(people[chosen])

    def _recordings(self) -> list[Path]:
        return sorted(self.raw.glob("HuGaDB_v2_*.txt"))

    def _read_all(self, files: list[Path], progress: Progress) -> list[tuple[int, np.ndarray, np.ndarray]]:
        rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        total = max(1, len(files))
        columns = np.asarray(FEATURE_INDEX, dtype=np.int32)
        for index, path in enumerate(files):
            person = _person_id(path)
            if person is None:
                continue
            try:
                table = np.loadtxt(path, delimiter="\t", skiprows=4, dtype=np.float32)
            except (OSError, ValueError):
                continue
            if table.ndim != 2 or table.shape[1] < 39 or table.shape[0] < WINDOW:
                continue
            raw = table[:, columns]
            labels = table[:, 38].astype(np.int16)
            features = np.empty((table.shape[0], CHANNELS), dtype=np.float32)
            features[:, :18] = raw[:, :18] / 32768.0 * 2.0
            features[:, 18:] = (raw[:, 18:] - 128.0) / 128.0
            rows.append((person, features, labels))
            if index % 25 == 0:
                progress(f"Reading recordings ({index + 1} of {total})", 0.12 + 0.7 * (index / total))
        return rows

    def _download(self, progress: Progress) -> None:
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = self.archive.with_suffix(".zip.part")
        last_error: Exception | None = None
        for url in URLS:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "nn-visualiser/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    total = int(response.headers.get("Content-Length") or 0)
                    got = 0
                    with partial.open("wb") as handle:
                        while True:
                            chunk = response.read(1024 * 256)
                            if not chunk:
                                break
                            handle.write(chunk)
                            got += len(chunk)
                            frac = (got / total) if total else None
                            progress(
                                f"Downloading HuGaDB ({got / 1_048_576:.0f} MB)",
                                None if frac is None else 0.08 * frac,
                            )
                partial.replace(self.archive)
                return
            except Exception as error:
                last_error = error
                if partial.exists():
                    partial.unlink()
        raise RuntimeError(f"Could not download HuGaDB: {last_error}")

    def _extract(self) -> None:
        with zipfile.ZipFile(self.archive) as bundle:
            for member in bundle.infolist():
                name = Path(member.filename).name
                if not name.startswith("HuGaDB_v2_") or not name.endswith(".txt"):
                    continue
                target = self.raw / name
                if target.exists() and target.stat().st_size > 100:
                    continue
                with bundle.open(member) as source, target.open("wb") as handle:
                    handle.write(source.read())


def _person_id(path: Path) -> int | None:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    try:
        return int(parts[-2])
    except ValueError:
        return None


def activity_name(index: int) -> str:
    if 0 <= index < len(ACTIVITIES):
        return ACTIVITIES[index][0]
    return "Unknown"
