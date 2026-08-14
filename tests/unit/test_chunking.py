"""Tests for acoustic window boundaries and waveform cropping."""

import pytest

from mlx_minimax_music3.chunking import (
    ChunkWindow,
    chunk_windows,
    crop_sample_bounds,
    overlap_latent_length,
)


@pytest.mark.parametrize(
    ("frames", "bounds"),
    [
        (0, []),
        (1, [(0, 1)]),
        (200, [(0, 200)]),
        (201, [(0, 200), (100, 201)]),
        (300, [(0, 200), (100, 300)]),
        (301, [(0, 200), (100, 300), (200, 301)]),
    ],
)
def test_chunk_windows_matches_reference_boundaries(
    frames: int, bounds: list[tuple[int, int]]
) -> None:
    windows = chunk_windows(frames)

    assert [(window.start, window.end) for window in windows] == bounds
    if windows:
        assert windows[0].is_first
        assert windows[-1].is_last
        assert [window.index for window in windows] == list(range(len(windows)))


def test_crop_sample_bounds_tiles_neighboring_windows() -> None:
    first = ChunkWindow(0, 0, 200, True, False)
    middle = ChunkWindow(1, 100, 300, False, False)
    last = ChunkWindow(2, 200, 301, False, True)

    assert overlap_latent_length() == 172
    assert crop_sample_bounds(first) == (0, 132_096)
    assert crop_sample_bounds(middle) == (44_032, 132_096)
    assert crop_sample_bounds(last) == (44_032, 0)


@pytest.mark.parametrize("frames", [-1, True, 1.5])
def test_chunk_windows_rejects_invalid_frame_counts(frames: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        chunk_windows(frames)  # type: ignore[arg-type]
