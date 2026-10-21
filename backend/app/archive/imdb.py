from __future__ import annotations

import json
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

VOCAB_SIZE = 12_000
SEQ_LEN = 256
PAD_ID = 0
UNK_ID = 1
TOKEN_RE = re.compile(r"[a-z0-9']+")
TAG_RE = re.compile(r"<[^>]+>")
CACHE_TAG = "t2"
URLS = (
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
)

Progress = Callable[[str, float | None], None]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(TAG_RE.sub(" ", text).lower())


class Corpus:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw"
        self.cache = root / "cache"
        self.train_x: np.ndarray | None = None
        self.train_y: np.ndarray | None = None
        self.test_x: np.ndarray | None = None
        self.test_y: np.ndarray | None = None
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}
        self._paths: dict[str, list[tuple[int, Path]]] = {}

    @property
    def ready(self) -> bool:
        return self.train_x is not None

    def cache_paths(self) -> tuple[Path, Path]:
        stem = f"imdb_v{VOCAB_SIZE}_s{SEQ_LEN}_{CACHE_TAG}"
        return self.cache / f"{stem}.npz", self.cache / f"{stem}.vocab.json"

    def load_cache(self) -> bool:
        arrays, vocab = self.cache_paths()
        if not arrays.exists() or not vocab.exists():
            return False
        payload = np.load(arrays)
        self.train_x = payload["train_x"]
        self.train_y = payload["train_y"]
        self.test_x = payload["test_x"]
        self.test_y = payload["test_y"]
        self.itos = json.loads(vocab.read_text())
        self.stoi = {word: index for index, word in enumerate(self.itos)}
        return True

    def prepare(self, progress: Progress) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        if self.load_cache():
            progress("IMDB cache ready", 1)
            return
        archive = self.raw / "aclImdb_v1.tar.gz"
        if not archive.exists() or archive.stat().st_size < 1_000_000:
            self._download(archive, progress)
        extracted = self.raw / "aclImdb" / "train" / "pos"
        if not extracted.exists():
            progress("Extracting reviews", 0.16)
            try:
                self._extract(archive)
            except (tarfile.TarError, EOFError, OSError):
                if archive.exists():
                    archive.unlink()
                self._download(archive, progress)
                self._extract(archive)
        progress("Reading reviews", 0.22)
        train_rows = self._read_split("train", progress, 0.22, 0.48)
        test_rows = self._read_split("test", progress, 0.48, 0.62)
        progress("Building the vocabulary", 0.64)
        counts: Counter[str] = Counter()
        train_tokens: list[list[str]] = []
        for text, _label in train_rows:
            tokens = tokenize(text)
            counts.update(tokens)
            train_tokens.append(tokens)
        most = counts.most_common(VOCAB_SIZE - 2)
        self.itos = ["<pad>", "<unk>"] + [word for word, _count in most]
        self.stoi = {word: index for index, word in enumerate(self.itos)}
        progress("Encoding reviews", 0.72)
        self.train_x = self._encode_many(train_tokens)
        self.train_y = np.asarray([label for _text, label in train_rows], dtype=np.uint8)
        test_tokens = [tokenize(text) for text, _label in test_rows]
        self.test_x = self._encode_many(test_tokens)
        self.test_y = np.asarray([label for _text, label in test_rows], dtype=np.uint8)
        arrays, vocab = self.cache_paths()
        np.savez_compressed(
            arrays,
            train_x=self.train_x,
            train_y=self.train_y,
            test_x=self.test_x,
            test_y=self.test_y,
        )
        vocab.write_text(json.dumps(self.itos))
        progress("Dataset ready", 1)

    def dataset_message(self) -> dict:
        assert self.train_y is not None and self.test_y is not None
        positive = int(self.train_y.astype(np.int64).sum())
        return {
            "type": "dataset",
            "task": "reviews",
            "train": int(self.train_y.shape[0]),
            "test": int(self.test_y.shape[0]),
            "positive": positive,
            "negative": int(self.train_y.shape[0] - positive),
            "vocab": len(self.itos),
            "seqLen": SEQ_LEN,
            "common": [word for word in self.itos[2:10]],
        }

    def page(self, split: str, offset: int, limit: int) -> dict:
        if split not in ("train", "test"):
            split = "train"
        labels = self.train_y if split == "train" else self.test_y
        paths = self._split_paths(split)
        total = int(labels.shape[0]) if labels is not None else len(paths)
        start = min(max(0, offset), total)
        count = min(40, max(1, limit))
        end = min(total, start + count)
        rows = []
        for index in range(start, end):
            if index >= len(paths):
                break
            label, path = paths[index]
            raw = path.read_text(encoding="utf-8", errors="replace")
            words = tokenize(raw)
            text = TAG_RE.sub("\n", raw).strip()
            clipped = len(text) > 1800
            if clipped:
                text = text[:1800].rstrip() + "…"
            stored = int(labels[index]) if labels is not None else int(label)
            rows.append(
                {
                    "index": index,
                    "label": stored,
                    "text": text,
                    "tokens": min(len(words), SEQ_LEN),
                    "tokenCount": len(words),
                    "truncated": len(words) > SEQ_LEN,
                    "clipped": clipped,
                }
            )
        return {"split": split, "offset": start, "total": total, "rows": rows}

    def _split_paths(self, split: str) -> list[tuple[int, Path]]:
        cached = self._paths.get(split)
        if cached is not None:
            return cached
        paths: list[tuple[int, Path]] = []
        for label, name in ((1, "pos"), (0, "neg")):
            found = sorted((self.raw / "aclImdb" / split / name).glob("*.txt"))
            paths.extend((label, path) for path in found)
        self._paths[split] = paths
        return paths

    def encode(self, text: str) -> tuple[np.ndarray, bool, list[str]]:
        words = tokenize(text)
        truncated = len(words) > SEQ_LEN
        chosen = words[:SEQ_LEN]
        ids = np.zeros(SEQ_LEN, dtype=np.uint16)
        tokens: list[str] = []
        for index, word in enumerate(chosen):
            mapped = self.stoi.get(word, UNK_ID)
            ids[index] = mapped
            tokens.append("unk" if mapped == UNK_ID else word)
        return ids, truncated, tokens

    def _encode_many(self, rows: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(rows), SEQ_LEN), dtype=np.uint16)
        for row_index, words in enumerate(rows):
            for col, word in enumerate(words[:SEQ_LEN]):
                out[row_index, col] = self.stoi.get(word, UNK_ID)
        return out

    def _download(self, archive: Path, progress: Progress) -> None:
        partial = archive.with_suffix(".tar.gz.part")
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
                            span = 0.14 * frac if frac is not None else None
                            progress(
                                f"Downloading IMDB ({got / 1_048_576:.0f} MB)",
                                None if span is None else span,
                            )
                partial.replace(archive)
                return
            except Exception as error:
                last_error = error
                if partial.exists():
                    partial.unlink()
        raise RuntimeError(f"Could not download IMDB: {last_error}")

    def _extract(self, archive: Path) -> None:
        with tarfile.open(archive, "r:gz") as bundle:
            base = self.raw.resolve()
            for member in bundle.getmembers():
                target = (self.raw / member.name).resolve()
                if base not in target.parents and target != base:
                    continue
                bundle.extract(member, self.raw, filter="data")

    def _read_split(self, split: str, progress: Progress, start: float, end: float) -> list[tuple[str, int]]:
        rows: list[tuple[str, int]] = []
        folders = [(1, "pos"), (0, "neg")]
        paths: list[tuple[int, Path]] = []
        for label, name in folders:
            found = sorted((self.raw / "aclImdb" / split / name).glob("*.txt"))
            paths.extend((label, path) for path in found)
        total = max(1, len(paths))
        for index, (label, path) in enumerate(paths):
            text = path.read_text(encoding="utf-8", errors="replace")
            rows.append((text, label))
            if index % 1000 == 0:
                frac = start + (end - start) * (index / total)
                progress(f"Reading {split} reviews", frac)
        return rows
