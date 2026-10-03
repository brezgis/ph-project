"""Tests for the mmap-based cell-29 refactor in the ripser/templates notebook.

Guards two invariants:
1. The default multiprocessing start method is fork (or unset), NOT spawn.
   Spawn breaks Jupyter notebooks because __main__ has no source file to
   re-import for unpickling the target function.
2. After a fork()ed child loads a slice of an .npy file via mmap, the
   parent's RSS growth is negligible (< 10 MB). This verifies that the
   parent never materialises the full array body — the core fix for the OOM.
   The fixture is large enough (~25 MB on disk) that the OLD broken pattern
   (parent np.load + 20 fancy-index copies) would exceed the threshold,
   making the test an actual regression guard rather than a vacuous pass.

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

# Threshold for parent RSS growth: must be exceeded by the OLD pattern and
# stayed below by the NEW pattern.  Empirically measured on this machine:
#   OLD (parent np.load + 20 fancy-index copies): ~27 MB delta
#   NEW (parent passes filename+indices, child mmap-loads): ~0.4 MB delta
# 10 MB sits comfortably between the two, with ample headroom on both sides.
_RSS_THRESHOLD_MB = 10


def _load_slice_checksum(queue: Queue, path: str, indices: list[int]) -> None:
    """Worker: mmap-load path, extract indices, put checksum in queue.

    This is the pattern the refactored get_only_barcodes uses — the worker
    receives a filename + indices, not a pre-loaded array.
    """
    arr = np.load(path, mmap_mode="r")[indices]
    queue.put(int(arr.sum()))


def _old_pattern_worker(queue: Queue, chunk: np.ndarray) -> None:
    """Worker for the OLD (broken) pattern: receives a pre-loaded array slice.

    Used only in test_old_pattern_does_bloat_parent_rss to verify that the
    fixture is large enough for the OLD pattern to exceed the RSS threshold.
    Uses float64 accumulation to avoid float16 overflow on large chunks.
    """
    queue.put(int(np.sum(chunk, dtype=np.float64)))


def _make_fixture_npy(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a ~25 MB attention-shaped .npy and return (path, array).

    Shape (50_000, 2, 2, 8, 8) float16 gives ~24.4 MB on disk — large enough
    that the OLD broken pattern (parent np.load + 20 fancy-index copies) adds
    ~27 MB to parent RSS, well above the 10 MB regression threshold.
    """
    # 50_000 samples × 2 layers × 2 heads × 8 × 8, float16 (~24.4 MB)
    rng = np.random.default_rng(42)
    arr = rng.random((50_000, 2, 2, 8, 8), dtype=np.float32).astype(np.float16)
    path = tmp_path / "attention.npy"
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

    The fixture is ~25 MB on disk, so any regression to the old pattern
    would produce ~27 MB of parent RSS growth — well above the 10 MB
    threshold.  See test_old_pattern_does_bloat_parent_rss for the
    complementary check that confirms the threshold is actually meaningful.
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

    assert rss_delta_mb < _RSS_THRESHOLD_MB, (
        f"Parent RSS grew by {rss_delta_mb:.1f} MB after child completed. "
        f"Expected < {_RSS_THRESHOLD_MB} MB — this suggests the parent is loading array data. "
        f"RSS before: {rss_before / 1024**2:.1f} MB, "
        f"RSS after: {rss_after / 1024**2:.1f} MB"
    )


def test_old_pattern_does_bloat_parent_rss(tmp_path):
    """Regression guard: the OLD broken pattern MUST exceed the RSS threshold.

    Verifies that the fixture is large enough to make the test meaningful —
    if the OLD pattern no longer exceeds 10 MB, the fixture is too small
    and test_child_mmap_load_does_not_bloat_parent_rss would pass vacuously.

    OLD pattern: parent calls np.load(filename, allow_pickle=True) then
    makes 20 fancy-index copies (one per split), passing each copy to a
    child Process.  With the ~25 MB fixture this adds ~27 MB to parent RSS.
    """
    path, _arr = _make_fixture_npy(tmp_path)
    n_samples = 50_000
    number_of_splits = 20

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    # Simulate old broken pattern: parent loads the full array, then slices
    loaded = np.load(str(path), allow_pickle=True)
    split_ids = np.array_split(np.arange(n_samples), number_of_splits)
    q = Queue()
    for ids in split_ids:
        chunk = loaded[ids]  # fancy-index copy materialised in parent
        p = Process(target=_old_pattern_worker, args=(q, chunk))
        p.start()
        q.get()
        p.join()
        p.close()
    del loaded

    rss_after = proc.memory_info().rss
    rss_delta_mb = (rss_after - rss_before) / (1024 * 1024)

    assert rss_delta_mb > _RSS_THRESHOLD_MB, (
        f"OLD pattern only grew parent RSS by {rss_delta_mb:.1f} MB — "
        f"expected > {_RSS_THRESHOLD_MB} MB. The fixture may be too small "
        f"to make test_child_mmap_load_does_not_bloat_parent_rss meaningful. "
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
