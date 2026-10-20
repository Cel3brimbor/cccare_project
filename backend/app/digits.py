from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

SIDE = 28
PIXELS = SIDE * SIDE
CLASSES = 10
MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

Progress = Callable[[str, float | None], None]


def read_idx(path: Path) -> np.ndarray:
    raw = gzip.decompress(path.read_bytes())
    magic = struct.unpack(">I", raw[:4])[0]
    if magic == 2051:
        _, count, rows, cols = struct.unpack(">IIII", raw[:16])
        data = np.frombuffer(raw, dtype=np.uint8, offset=16)
        return data.reshape(count, rows * cols).copy()
    if magic == 2049:
        _, count = struct.unpack(">II", raw[:8])
        data = np.frombuffer(raw, dtype=np.uint8, offset=8)
        return data.reshape(count).copy()
    raise RuntimeError(f"Unexpected MNIST file {path.name}")


class Digits:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw" / "mnist"
        self.cache = root / "cache"
        self.train_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.test_x: np.ndarray | None = None
        self.test_y: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.train_x is not None

    def cache_path(self) -> Path:
        return self.cache / "mnist28.npz"

    def load_cache(self) -> bool:
        path = self.cache_path()
        if not path.exists():
            return False
        payload = np.load(path)
        self.train_x = payload["train_x"]
        self.train_y = payload["train_y"]
        self.test_x = payload["test_x"]
        self.test_y = payload["test_y"]
        return (
            self.train_x.ndim == 2
            and self.train_x.shape[1] == PIXELS
            and self.train_y.shape[0] == self.train_x.shape[0]
        )

    def prepare(self, progress: Progress) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        if self.load_cache():
            progress("MNIST cache ready", 1)
            return
        names = list(FILES)
        for index, key in enumerate(names):
            target = self.raw / FILES[key]
            span = index / len(names)
            if not target.exists() or target.stat().st_size < 100:
                self._download(FILES[key], target, progress, span, span + 0.8 / len(names))
        progress("Reading digits", 0.86)
        train_x = read_idx(self.raw / FILES["train_images"])
        train_y = read_idx(self.raw / FILES["train_labels"])
        test_x = read_idx(self.raw / FILES["test_images"])
        test_y = read_idx(self.raw / FILES["test_labels"])
        if train_x.shape[0] != train_y.shape[0] or test_x.shape[0] != test_y.shape[0]:
            raise RuntimeError("MNIST images and labels do not match.")
        if train_x.shape[1] != PIXELS:
            raise RuntimeError("MNIST images are not 28 by 28.")
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        np.savez_compressed(
            self.cache_path(),
            train_x=self.train_x,
            train_y=self.train_y,
            test_x=self.test_x,
            test_y=self.test_y,
        )
        progress("Digits ready", 1)

    def dataset_message(self) -> dict:
        assert self.train_y is not None and self.test_y is not None
        counts = np.bincount(self.train_y.astype(np.int64), minlength=CLASSES)
        return {
            "type": "dataset",
            "task": "digits",
            "train": int(self.train_y.shape[0]),
            "test": int(self.test_y.shape[0]),
            "width": SIDE,
            "height": SIDE,
            "classes": CLASSES,
            "counts": [int(count) for count in counts.tolist()],
        }

    def page(self, split: str, offset: int, limit: int) -> dict:
        if split not in ("train", "test"):
            split = "train"
        images = self.train_x if split == "train" else self.test_x
        labels = self.train_y if split == "train" else self.test_y
        assert images is not None and labels is not None
        total = int(images.shape[0])
        start = min(max(0, offset), total)
        count = min(40, max(1, limit))
        end = min(total, start + count)
        rows = []
        for index in range(start, end):
            rows.append(
                {
                    "index": index,
                    "label": int(labels[index]),
                    "pixels": images[index].astype(int).tolist(),
                }
            )
        return {
            "task": "digits",
            "split": split,
            "offset": start,
            "total": total,
            "width": SIDE,
            "height": SIDE,
            "rows": rows,
        }

    def _download(self, name: str, target: Path, progress: Progress, start: float, end: float) -> None:
        partial = target.with_suffix(target.suffix + ".part")
        last_error: Exception | None = None
        for mirror in MIRRORS:
            url = mirror + name
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
                            frac = (got / total) if total else 0.5
                            progress(
                                f"Downloading MNIST ({got / 1_048_576:.1f} MB)",
                                start + (end - start) * frac,
                            )
                partial.replace(target)
                return
            except Exception as error:
                last_error = error
                if partial.exists():
                    partial.unlink()
        raise RuntimeError(f"Could not download MNIST {name}: {last_error}")
