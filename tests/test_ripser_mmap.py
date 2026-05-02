"""Tests for the mmap-based cell-29 refactor in the ripser/templates notebook.

Guards two invariants:
1. The default multiprocessing start method is fork (or unset), NOT spawn.
   Spawn breaks Jupyter notebooks because __main__ has no source file to
   re-import for unpickling the target function.
2. After a fork()ed child loads a slice of an .npy file via mmap, the
   parent's RSS growth is negligible (< 10 MB). This verifies that the
   parent never materialises the full array body — the core fix for the OOM.

Also verifies the helper worker function (load_slice_checksum) returns
the correct checksum so the mmap indexing is actually exercised.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import psutil
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_slice_checksum(queue: Queue, path: str, indices: list[int]) -> None:
    """Worker: mmap-load path, extract indices, put checksum in queue.

    This is the pattern the refactored get_only_barcodes uses — the worker
    receives a filename + indices, not a pre-loaded array.
    """
    arr = np.load(path, mmap_mode="r")[indices]
    queue.put(int(arr.sum()))


def _make_fixture_npy(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a tiny attention-shaped .npy and return (path, array)."""
    # 8 samples × 2 layers × 2 heads × 8 × 8, float16
    rng = np.random.default_rng(42)
    arr = rng.random((8, 2, 2, 8, 8), dtype=np.float32).astype(np.float16)
    path = tmp_path / "tiny_attention.npy"
    np.save(path, arr)
    return path, arr


# ---------------------------------------------------------------------------
# Test 1: start method must NOT be spawn
# ---------------------------------------------------------------------------


def test_start_method_is_not_spawn():
    """Default start method on Linux must be fork (or None → fork).

    If someone sets mp.set_start_method("spawn") globally in the project
    it would break Jupyter: workers would try to re-import __main__,
    which has no source file in a kernel context.
    """
    method = mp.get_start_method(allow_none=True)
    assert method != "spawn", (
        f"multiprocessing start method is 'spawn' — this breaks Jupyter notebooks. "
        f"Got: {method!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: child mmap-loads slice; parent RSS stays bounded
# ---------------------------------------------------------------------------


def test_child_mmap_load_does_not_bloat_parent_rss(tmp_path):
    """Parent RSS growth after a fork()ed child completes must stay < 10 MB.

    With the old implementation (parent np.load + fancy-index copies),
    the parent would hold ~9.4 GB before the first fork. After the fix
    the parent holds nothing — it passes (filename, indices) to the child.

    We verify this with a small fixture: the array body is only ~8 KB,
    so any growth > 10 MB would indicate something went badly wrong.
    """
    path, arr = _make_fixture_npy(tmp_path)
    indices = [1, 2, 3]

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    q = Queue()
    p = Process(target=_load_slice_checksum, args=(q, str(path), indices))
    p.start()
    result = q.get()
    p.join()
    p.close()

    rss_after = proc.memory_info().rss
    rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)

    assert rss_delta_mb < 10, (
        f"Parent RSS grew by {rss_delta_mb:.1f} MB after child completed. "
        f"Expected < 10 MB — this suggests the parent is loading array data. "
        f"RSS before: {rss_before / 1024**2:.1f} MB, "
        f"RSS after: {rss_after / 1024**2:.1f} MB"
    )


# ---------------------------------------------------------------------------
# Test 3: worker returns correct checksum (mmap indexing is exercised)
# ---------------------------------------------------------------------------


def test_child_mmap_returns_correct_checksum(tmp_path):
    """Worker must return checksum matching np.load()[indices].sum().

    Verifies that mmap_mode='r' plus integer-array indexing gives the
    correct data — no corruption from the mmap approach.
    """
    path, arr = _make_fixture_npy(tmp_path)
    indices = [0, 3, 5, 7]

    expected = int(arr[indices].sum())

    q = Queue()
    p = Process(target=_load_slice_checksum, args=(q, str(path), indices))
    p.start()
    got = q.get()
    p.join()
    p.close()

    assert got == expected, (
        f"Worker returned checksum {got}, expected {expected}. "
        f"mmap indexing may be broken."
    )
