"""Tests for draganov_replication/pd_distances.py.

Tests per issue spec (ph-project-inu.3):

  Synthetic (always run, not gated):
    1. test_bars_statistics_vector_shape_dim1  — 5-feature PD → length 40
    2. test_bars_statistics_vector_shape_dim0  — 5-feature PD + only_death → length 10
    3. test_bars_statistics_empty_pd           — empty PD falls back to zero-vector
    4. test_persistence_image_smoke            — 5-feature PD → flat float vector, no error

  Gated by PH_REQUIRE_DRAGANOV_PD_DISTANCES=1 (real-data tests against
  the built cache in data/phase3/draganov_pd_distances/):
    5. test_grid_shape        — each of 8 matrices is (34, 34), float32
    6. test_symmetric         — np.allclose(M, M.T, atol=1e-5) for all 8
    7. test_diagonal_zero     — np.allclose(np.diag(M), 0, atol=1e-5) for all 8
    8. test_finite            — no inf/nan in any matrix
    9. test_cells_match_pointcloud_manifest — cells array order matches pointclouds manifest
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Gate setup
# ---------------------------------------------------------------------------

REQUIRE = os.environ.get("PH_REQUIRE_DRAGANOV_PD_DISTANCES") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# data/phase3/ is gitignored. Gated tests (PH_REQUIRE_DRAGANOV_PD_DISTANCES=1)
# require the cache to be built locally first; ungated tests do not need data.
PD_DISTANCES_DIR = REPO_ROOT / "data" / "phase3" / "draganov_pd_distances"
POINTCLOUDS_DIR = REPO_ROOT / "data" / "phase3" / "draganov_pointclouds"

DISTANCES = ("bottleneck", "sliced_wasserstein", "persistence_image", "bars_statistics")
DIMS = (0, 1)

try:
    from draganov_replication.pd_distances import (
        compute_pd_distance_grid,
        vectorise_bars_statistics,
        vectorise_persistence_image,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _require_import() -> None:
    if not _IMPORT_OK:
        pytest.fail(
            "Could not import draganov_replication.pd_distances — module not yet created."
        )


def _skip_or_fail(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason + " (PH_REQUIRE_DRAGANOV_PD_DISTANCES=1)")
    pytest.skip(reason)


def _cache_ready() -> bool:
    """Return True when the distance cache is populated (at least one file present)."""
    return (PD_DISTANCES_DIR / "bottleneck_d0.npz").exists()


def _synthetic_pd(n_features: int = 5) -> np.ndarray:
    """Return a synthetic (n, 2) float64 PD with non-trivial birth/death pairs."""
    rng = np.random.default_rng(42)
    births = rng.uniform(0.0, 0.5, size=n_features)
    deaths = births + rng.uniform(0.01, 0.3, size=n_features)
    return np.column_stack([births, deaths])


# ---------------------------------------------------------------------------
# Synthetic test 1-3: vectorise_bars_statistics
# ---------------------------------------------------------------------------

def test_bars_statistics_vector_shape_dim1():
    """5-feature PD → vector length 40 for dim >= 1 (all_death=False).

    vectorise_bars_statistics returns 10 statistics over each of 4 quantities
    (deaths, births, lifespans, midpoints) = 40 values total.
    Reproduces draganov/B run_compute_pd_distances.py:80-108 contract.
    """
    _require_import()
    pd_arr = _synthetic_pd(n_features=5)
    vec = vectorise_bars_statistics(pd_arr, only_death=False)
    assert vec.shape == (40,), (
        f"Expected shape (40,) for only_death=False, got {vec.shape}"
    )
    assert vec.dtype == np.float64, f"Expected float64, got {vec.dtype}"


def test_bars_statistics_vector_shape_dim0():
    """5-feature PD + only_death=True → vector length 10.

    only_death=True restricts to death values only (1 quantity × 10 stats = 10).
    This is the dim=0 path in Draganov's script.
    """
    _require_import()
    pd_arr = _synthetic_pd(n_features=5)
    vec = vectorise_bars_statistics(pd_arr, only_death=True)
    assert vec.shape == (10,), (
        f"Expected shape (10,) for only_death=True, got {vec.shape}"
    )
    assert vec.dtype == np.float64, f"Expected float64, got {vec.dtype}"


def test_bars_statistics_empty_pd():
    """Empty PD does not raise; falls back gracefully (zeros via the np.zeros(1) branch).

    Draganov's script guards empty PDs by substituting np.zeros(1).
    """
    _require_import()
    empty_pd = np.empty((0, 2), dtype=np.float64)
    # Should not raise; should return 10 or 40 element vectors
    vec_d0 = vectorise_bars_statistics(empty_pd, only_death=True)
    assert vec_d0.shape == (10,), f"Expected (10,) for empty PD only_death=True, got {vec_d0.shape}"

    vec_d1 = vectorise_bars_statistics(empty_pd, only_death=False)
    assert vec_d1.shape == (40,), f"Expected (40,) for empty PD only_death=False, got {vec_d1.shape}"


# ---------------------------------------------------------------------------
# Synthetic test 4: vectorise_persistence_image
# ---------------------------------------------------------------------------

def test_persistence_image_smoke():
    """5-feature PD vectorises to a flat float vector without raising.

    Does not assert exact shape (pixel_size=0.1, range=(0,1) → 10×10=100 pixels)
    but verifies: 1-D, finite float values, non-zero.
    """
    _require_import()
    pd_arr = _synthetic_pd(n_features=5)

    for dim in (0, 1):
        vec = vectorise_persistence_image(pd_arr, dim=dim)
        assert vec.ndim == 1, (
            f"Expected 1-D vector from vectorise_persistence_image(dim={dim}), "
            f"got ndim={vec.ndim}"
        )
        assert np.issubdtype(vec.dtype, np.floating), (
            f"Expected float dtype, got {vec.dtype}"
        )
        assert np.all(np.isfinite(vec)), (
            f"vectorise_persistence_image(dim={dim}) produced non-finite values"
        )
        # Should be non-trivially non-zero (the PD has real features)
        assert np.any(vec > 0), (
            f"vectorise_persistence_image(dim={dim}) produced all-zero vector for non-empty PD"
        )


def test_persistence_image_empty_pd():
    """Empty PD does not raise; returns a flat finite zero vector.

    Real (lang, term) cells with sparse H_1 features can hit this branch.
    """
    _require_import()
    empty_pd = np.empty((0, 2), dtype=np.float64)
    for dim in (0, 1):
        vec = vectorise_persistence_image(empty_pd, dim=dim)
        assert vec.ndim == 1, f"Expected 1-D vector for empty PD, got ndim={vec.ndim}"
        assert np.all(np.isfinite(vec)), "Empty PD produced non-finite values"
        assert np.all(vec == 0.0), "Empty PD vector should be all zeros"


# ---------------------------------------------------------------------------
# Error path tests (not gated)
# ---------------------------------------------------------------------------

def test_compute_grid_missing_manifest_raises(tmp_path: pathlib.Path):
    """compute_pd_distance_grid raises FileNotFoundError when manifest.csv is absent."""
    _require_import()
    diag_dir = tmp_path / "diagrams"
    diag_dir.mkdir()  # no manifest.csv inside
    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match="manifest"):
        compute_pd_distance_grid(
            diagrams_dir=diag_dir,
            out_dir=out_dir,
            distances=("bottleneck",),
            dims=(0,),
        )


def test_compute_grid_invalid_distance_raises(tmp_path: pathlib.Path):
    """compute_pd_distance_grid raises ValueError for an unrecognised distance name."""
    _require_import()

    diag_dir = tmp_path / "diagrams"
    diag_dir.mkdir()
    # Write a minimal manifest so we get past the manifest check
    pd.DataFrame(columns=["lang", "term", "n_h0", "n_h1", "file"]).to_csv(
        diag_dir / "manifest.csv", index=False
    )
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="Unknown distance"):
        compute_pd_distance_grid(
            diagrams_dir=diag_dir,
            out_dir=out_dir,
            distances=("not_a_real_distance",),
            dims=(0,),
        )


# ---------------------------------------------------------------------------
# Gated real-data tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("PH_REQUIRE_DRAGANOV_PD_DISTANCES") != "1",
    reason="PH_REQUIRE_DRAGANOV_PD_DISTANCES not set to 1",
)
def test_grid_shape():
    """Each of the 8 output matrices is shape (34, 34) and dtype float32."""
    _require_import()
    if not _cache_ready():
        _skip_or_fail(f"PD distances cache not built yet: {PD_DISTANCES_DIR}")

    for distance in DISTANCES:
        for dim in DIMS:
            fname = PD_DISTANCES_DIR / f"{distance}_d{dim}.npz"
            assert fname.exists(), f"Missing cache file: {fname}"
            data = np.load(fname)
            M = data["matrix"]
            assert M.shape == (34, 34), (
                f"{distance}_d{dim}: expected shape (34, 34), got {M.shape}"
            )
            assert M.dtype == np.float32, (
                f"{distance}_d{dim}: expected float32, got {M.dtype}"
            )


@pytest.mark.skipif(
    os.environ.get("PH_REQUIRE_DRAGANOV_PD_DISTANCES") != "1",
    reason="PH_REQUIRE_DRAGANOV_PD_DISTANCES not set to 1",
)
def test_symmetric():
    """Each matrix is symmetric: np.allclose(M, M.T, atol=1e-5)."""
    _require_import()
    if not _cache_ready():
        _skip_or_fail(f"PD distances cache not built yet: {PD_DISTANCES_DIR}")

    for distance in DISTANCES:
        for dim in DIMS:
            fname = PD_DISTANCES_DIR / f"{distance}_d{dim}.npz"
            data = np.load(fname)
            M = data["matrix"].astype(np.float64)
            assert np.allclose(M, M.T, atol=1e-5), (
                f"{distance}_d{dim}: matrix is not symmetric "
                f"(max asymmetry: {np.abs(M - M.T).max():.2e})"
            )


@pytest.mark.skipif(
    os.environ.get("PH_REQUIRE_DRAGANOV_PD_DISTANCES") != "1",
    reason="PH_REQUIRE_DRAGANOV_PD_DISTANCES not set to 1",
)
def test_diagonal_zero():
    """Each matrix has zero diagonal: np.allclose(np.diag(M), 0, atol=1e-5)."""
    _require_import()
    if not _cache_ready():
        _skip_or_fail(f"PD distances cache not built yet: {PD_DISTANCES_DIR}")

    for distance in DISTANCES:
        for dim in DIMS:
            fname = PD_DISTANCES_DIR / f"{distance}_d{dim}.npz"
            data = np.load(fname)
            M = data["matrix"]
            diag = np.diag(M)
            assert np.allclose(diag, 0, atol=1e-5), (
                f"{distance}_d{dim}: diagonal is not zero "
                f"(max |diag|: {np.abs(diag).max():.2e})"
            )


@pytest.mark.skipif(
    os.environ.get("PH_REQUIRE_DRAGANOV_PD_DISTANCES") != "1",
    reason="PH_REQUIRE_DRAGANOV_PD_DISTANCES not set to 1",
)
def test_finite():
    """No inf or nan in any matrix."""
    _require_import()
    if not _cache_ready():
        _skip_or_fail(f"PD distances cache not built yet: {PD_DISTANCES_DIR}")

    for distance in DISTANCES:
        for dim in DIMS:
            fname = PD_DISTANCES_DIR / f"{distance}_d{dim}.npz"
            data = np.load(fname)
            M = data["matrix"]
            assert np.all(np.isfinite(M)), (
                f"{distance}_d{dim}: matrix contains inf/nan values"
            )


@pytest.mark.skipif(
    os.environ.get("PH_REQUIRE_DRAGANOV_PD_DISTANCES") != "1",
    reason="PH_REQUIRE_DRAGANOV_PD_DISTANCES not set to 1",
)
def test_cells_match_pointcloud_manifest():
    """cells array in each .npz matches data/phase3/draganov_pointclouds/manifest.csv order."""
    _require_import()
    if not _cache_ready():
        _skip_or_fail(f"PD distances cache not built yet: {PD_DISTANCES_DIR}")

    pc_manifest_path = POINTCLOUDS_DIR / "manifest.csv"
    if not pc_manifest_path.exists():
        _skip_or_fail(f"Pointclouds manifest not found: {pc_manifest_path}")

    import pandas as pd
    pc_manifest = pd.read_csv(pc_manifest_path)
    # Expected labels: f"{lang}/{term}" sorted by (lang, term)
    expected_labels = [
        f"{row.lang}/{row.term}"
        for row in pc_manifest.sort_values(["lang", "term"]).itertuples()
    ]

    for distance in DISTANCES:
        for dim in DIMS:
            fname = PD_DISTANCES_DIR / f"{distance}_d{dim}.npz"
            data = np.load(fname, allow_pickle=True)
            cells = data["cells"]
            actual_labels = list(cells)
            assert actual_labels == expected_labels, (
                f"{distance}_d{dim}: cells array does not match pointclouds manifest order.\n"
                f"  Expected: {expected_labels[:5]}...\n"
                f"  Got:      {actual_labels[:5]}..."
            )
