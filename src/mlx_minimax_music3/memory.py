"""MLX and macOS memory telemetry for inference stage boundaries."""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

_BYTE_UNITS = {
    "B": 1,
    "K": 1 << 10,
    "M": 1 << 20,
    "G": 1 << 30,
    "T": 1 << 40,
}
_SWAP_USED_RE = re.compile(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([BKMGT])\b")
_FREE_PERCENT_RE = re.compile(r"System-wide memory free percentage:\s*([0-9]+)%")
_PAGE_SIZE_RE = re.compile(r"page size of\s+([0-9]+) bytes")
_VM_COUNTER_RE = re.compile(r"^(Swapins|Swapouts):\s*([0-9]+)\.", re.MULTILINE)
_FOOTPRINT_RE = re.compile(r"phys_footprint:\s*([0-9]+) B")
_FOOTPRINT_PEAK_RE = re.compile(r"phys_footprint_peak:\s*([0-9]+) B")


@dataclass(frozen=True, slots=True)
class MLXMemory:
    """Bytes reported by the MLX allocator."""

    active_bytes: int
    cache_bytes: int
    peak_bytes: int


@dataclass(frozen=True, slots=True)
class SystemMemory:
    """Process and system-wide macOS memory signals."""

    process_rss_bytes: int
    process_footprint_bytes: int | None
    process_footprint_peak_bytes: int | None
    swap_used_bytes: int
    swapins_bytes: int
    swapouts_bytes: int
    free_percent: int


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """One timestamped memory sample."""

    label: str
    monotonic_ns: int
    mlx: MLXMemory
    system: SystemMemory


@dataclass(frozen=True, slots=True)
class MemoryDelta:
    """Signed changes between two memory samples."""

    active_bytes: int
    cache_bytes: int
    process_rss_bytes: int
    process_footprint_bytes: int | None
    swap_used_bytes: int
    swapins_bytes: int
    swapouts_bytes: int


def _run(command: tuple[str, ...]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout


def parse_swap_used(output: str) -> int:
    """Parse `sysctl -n vm.swapusage` into used bytes."""

    match = _SWAP_USED_RE.search(output)
    if match is None:
        raise ValueError("could not parse vm.swapusage output")
    return round(float(match.group(1)) * _BYTE_UNITS[match.group(2)])


def parse_free_percent(output: str) -> int:
    """Parse the free-memory percentage reported by `memory_pressure -Q`."""

    match = _FREE_PERCENT_RE.search(output)
    if match is None:
        raise ValueError("could not parse memory_pressure output")
    return int(match.group(1))


def parse_vm_swap_io(output: str) -> tuple[int, int]:
    """Parse cumulative swap-in and swap-out bytes from `vm_stat`."""

    page_size_match = _PAGE_SIZE_RE.search(output)
    counters = {name: int(value) for name, value in _VM_COUNTER_RE.findall(output)}
    if page_size_match is None or set(counters) != {"Swapins", "Swapouts"}:
        raise ValueError("could not parse vm_stat swap counters")
    page_size = int(page_size_match.group(1))
    return counters["Swapins"] * page_size, counters["Swapouts"] * page_size


def parse_footprint(output: str) -> tuple[int, int | None]:
    """Parse current and peak physical footprint from macOS `footprint`."""

    footprint_match = _FOOTPRINT_RE.search(output)
    if footprint_match is None:
        raise ValueError("could not parse footprint output")
    peak_match = _FOOTPRINT_PEAK_RE.search(output)
    peak = int(peak_match.group(1)) if peak_match is not None else None
    return int(footprint_match.group(1)), peak


def capture_mlx_memory() -> MLXMemory:
    """Capture allocator telemetry without importing MLX at module import time."""

    import mlx.core as mx

    return MLXMemory(
        active_bytes=int(mx.get_active_memory()),
        cache_bytes=int(mx.get_cache_memory()),
        peak_bytes=int(mx.get_peak_memory()),
    )


def capture_system_memory(*, include_footprint: bool = False) -> SystemMemory:
    """Capture macOS memory signals using fixed system commands."""

    rss_output = _run(("/bin/ps", "-o", "rss=", "-p", str(os.getpid())))
    swap_output = _run(("/usr/sbin/sysctl", "-n", "vm.swapusage"))
    pressure_output = _run(("/usr/bin/memory_pressure", "-Q"))
    vm_stat_output = _run(("/usr/bin/vm_stat",))
    swapins_bytes, swapouts_bytes = parse_vm_swap_io(vm_stat_output)

    footprint = None
    footprint_peak = None
    if include_footprint:
        footprint_output = _run(
            (
                "/usr/bin/footprint",
                "-f",
                "bytes",
                "--noCategories",
                "--pid",
                str(os.getpid()),
            )
        )
        footprint, footprint_peak = parse_footprint(footprint_output)

    return SystemMemory(
        process_rss_bytes=int(rss_output.strip()) * 1024,
        process_footprint_bytes=footprint,
        process_footprint_peak_bytes=footprint_peak,
        swap_used_bytes=parse_swap_used(swap_output),
        swapins_bytes=swapins_bytes,
        swapouts_bytes=swapouts_bytes,
        free_percent=parse_free_percent(pressure_output),
    )


def capture_memory(label: str, *, include_footprint: bool = False) -> MemorySnapshot:
    """Capture a combined MLX/macOS sample at an inference boundary."""

    return MemorySnapshot(
        label=label,
        monotonic_ns=time.monotonic_ns(),
        mlx=capture_mlx_memory(),
        system=capture_system_memory(include_footprint=include_footprint),
    )


def memory_delta(before: MemorySnapshot, after: MemorySnapshot) -> MemoryDelta:
    """Calculate signed allocator, process, and swap changes."""

    footprint_delta = None
    if (
        before.system.process_footprint_bytes is not None
        and after.system.process_footprint_bytes is not None
    ):
        footprint_delta = (
            after.system.process_footprint_bytes - before.system.process_footprint_bytes
        )
    return MemoryDelta(
        active_bytes=after.mlx.active_bytes - before.mlx.active_bytes,
        cache_bytes=after.mlx.cache_bytes - before.mlx.cache_bytes,
        process_rss_bytes=after.system.process_rss_bytes - before.system.process_rss_bytes,
        process_footprint_bytes=footprint_delta,
        swap_used_bytes=after.system.swap_used_bytes - before.system.swap_used_bytes,
        swapins_bytes=after.system.swapins_bytes - before.system.swapins_bytes,
        swapouts_bytes=after.system.swapouts_bytes - before.system.swapouts_bytes,
    )


__all__ = [
    "MLXMemory",
    "MemoryDelta",
    "MemorySnapshot",
    "SystemMemory",
    "capture_memory",
    "capture_mlx_memory",
    "capture_system_memory",
    "memory_delta",
]
