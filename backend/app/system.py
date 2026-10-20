from __future__ import annotations

import re
import subprocess
from typing import Any

import mlx.core as mx
import psutil

_UTIL = re.compile(r'"Device Utilization %"=(\d+)')
_CORES = re.compile(r'"gpu-core-count" = (\d+)')


def metal_available() -> bool:
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False


def device_message(cores: int | None = None) -> dict[str, Any]:
    available = metal_available()
    info: dict[str, Any] = {}
    if available:
        info = dict(mx.device_info())
    return {
        "type": "device",
        "available": available,
        "name": info.get("device_name"),
        "architecture": info.get("architecture"),
        "memory": info.get("memory_size"),
        "cores": cores,
    }


def metal_memory() -> dict[str, int]:
    if not metal_available():
        return {"active": 0, "peak": 0, "cache": 0}
    return {
        "active": int(mx.get_active_memory()),
        "peak": int(mx.get_peak_memory()),
        "cache": int(mx.get_cache_memory()),
    }


def gpu_registry() -> dict[str, int | None]:
    try:
        out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {"utilization": None, "cores": None}
    util = _UTIL.search(out)
    cores = _CORES.search(out)
    return {
        "utilization": int(util.group(1)) if util else None,
        "cores": int(cores.group(1)) if cores else None,
    }


def host_memory() -> dict[str, int]:
    mem = psutil.virtual_memory()
    return {
        "used": int(mem.total - mem.available),
        "total": int(mem.total),
        "process": int(psutil.Process().memory_info().rss),
    }
