"""Tests for memory telemetry parsing and deltas."""

from mlx_minimax_music3.memory import (
    MemorySnapshot,
    MLXMemory,
    SystemMemory,
    memory_delta,
    parse_footprint,
    parse_free_percent,
    parse_swap_used,
    parse_vm_swap_io,
)


def test_parse_macos_memory_outputs() -> None:
    assert parse_swap_used(
        "total = 1024.00M  used = 1.81M  free = 1022.19M  (encrypted)"
    ) == 1_897_923
    assert parse_free_percent("System-wide memory free percentage: 92%") == 92
    assert parse_vm_swap_io(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes).\n"
        "Swapins: 24.\nSwapouts: 116.\n"
    ) == (393_216, 1_900_544)
    assert parse_footprint(
        "Auxiliary data:\n    phys_footprint: 9175448 B\n"
        "    phys_footprint_peak: 9437184 B\n"
    ) == (9_175_448, 9_437_184)


def test_memory_delta_keeps_swap_growth_visible() -> None:
    before = MemorySnapshot(
        label="before",
        monotonic_ns=1,
        mlx=MLXMemory(10, 20, 30),
        system=SystemMemory(100, 120, 130, 1_000, 2_000, 3_000, 90),
    )
    after = MemorySnapshot(
        label="after",
        monotonic_ns=2,
        mlx=MLXMemory(40, 50, 60),
        system=SystemMemory(170, 210, 230, 1_500, 2_300, 4_000, 80),
    )

    assert memory_delta(before, after).swapouts_bytes == 1_000
    assert memory_delta(before, after).swap_used_bytes == 500
    assert memory_delta(before, after).process_footprint_bytes == 90
