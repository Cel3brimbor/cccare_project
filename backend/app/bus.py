from __future__ import annotations

import threading
from typing import Any


class Bus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._channels: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, int] = {}
        self._events: list[tuple[int, dict[str, Any]]] = []
        self._next_event = 1

    def publish(self, message: dict[str, Any]) -> None:
        kind = message["type"]
        with self._lock:
            self._versions[kind] = self._versions.get(kind, 0) + 1
            self._channels[kind] = message

    def emit(self, message: dict[str, Any]) -> None:
        with self._lock:
            event_id = self._next_event
            self._next_event += 1
            self._events.append((event_id, message))
            if len(self._events) > 40:
                self._events = self._events[-40:]

    def tip(self) -> int:
        with self._lock:
            return self._next_event - 1

    def take(self, seen: dict[str, int], event_cursor: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for kind, version in self._versions.items():
                if seen.get(kind) != version:
                    seen[kind] = version
                    out.append(self._channels[kind])
            fresh = [message for event_id, message in self._events if event_id > event_cursor]
            if fresh:
                event_cursor = self._events[-1][0]
            out.extend(fresh)
            return out, event_cursor
