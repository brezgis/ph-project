"""Tests for draganov_replication/diagrams.py.

Tests per issue spec:
  Synthetic (always run):
    1. test_synthetic_pointcloud  — ring of 10 points → H_1 non-trivial bar
    2. test_diagrams_shape        — .npz keys exist; arrays are 2D float64
    3. test_h0_no_inf             — no inf values in h0 or h1
    4. test_birth_le_death        — every (b, d) row satisfies b <= d
    5. test_overwrite_false_skips — overwrite=False skips already-written .npz
    6. test_overwrite_true_rewrites — overwrite=True rewrites existing .npz
    7. test_manifest_columns      — returned DataFrame has expected columns
    8. test_manifest_csv_written  — manifest.csv is written to out_dir

  Gated by PH_REQUIRE_DRAGANOV_DIAGRAMS=1 (real-data tests):
    9. test_canon_term_coverage   — 34 .npz files present
    10. test_cells_match_pointcloud_manifest — diagram manifest ↔ pointcloud manifest
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

REQUIRE = os.environ.get("PH_REQUIRE_DRAGANOV_DIAGRAMS") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = REPO_ROOT / "data" / "phase3" / "draganov_diagrams"
POINTCLOUDS_DIR = REPO_ROOT / "data" / "phase3" / "draganov_pointclouds"

try:
    from draganov_replication.diagrams import compute_diagrams
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _require_import() -> None:
    if not _IMPORT_OK:
        pytest.fail(
            "Could not import draganov_replication.diagrams — module not yet created."
        )


def _skip_or_fail(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason + " (PH_REQUIRE_DRAGANOV_DIAGRAMS=1)")
    pytest.skip(reason)


def _diagrams_ready() -> bool:
    """Return True when draganov_diagrams/ is non-empty (manifest.csv present)."""
    return (DIAGRAMS_DIR / "manifest.csv").exists()


def _load_diagrams_manifest() -> pd.DataFrame:
    return pd.read_csv(DIAGRAMS_DIR / "manifest.csv")


def _load_pointclouds_manifest() -> pd.DataFrame:
    return pd.read_csv(POINTCLOUDS_DIR / "manifest.csv")


# ---------------------------------------------------------------------------
# Helpers for building synthetic point clouds
# ---------------------------------------------------------------------------

def _ring_points(n: int = 10, radius: float = 1.0) -> np.ndarray:
    """Return n evenly-spaced points on a circle in 2D (embedded in R^2)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return pts.astype(np.float32)


def _write_synthetic_pointcloud(
    out_dir: pathlib.Path,
    lang: str,
    term: str,
    points: np.ndarray,
) -> None:
    """Write a synthetic .npy point cloud + manifest.csv to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{lang}_{term}.npy"
    np.save(npy_path, points)
    manifest = pd.DataFrame(
        [{"lang": lang, "term": term, "n_samples": len(points), "file": str(npy_path)}]
    )
    manifest.to_csv(out_dir / "manifest.csv", index=False)


# ---------------------------------------------------------------------------
# Synthetic test 1: H_1 non-trivial bar from a ring
# ---------------------------------------------------------------------------

def test_synthetic_pointcloud(tmp_path: pathlib.Path):
    """A ring of 10 points in 2D produces at least one H_1 bar with non-trivial persistence.

    This is the key correctness check: Vietoris-Rips on a ring should detect
    one 1-cycle (the loop itself).  Not gated — runs on any machine.
    """
    _require_import()

    ring = _ring_points(n=10, radius=1.0)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "en", "blue", ring)

    out_dir = tmp_path / "diagrams"
    summary = compute_diagrams(
        pointcloud_dir=pc_dir,
        out_dir=out_dir,
        max_dim=1,
        overwrite=False,
    )

    assert len(summary) == 1, f"Expected 1 row in summary, got {len(summary)}"
    npz_path = out_dir / "en_blue.npz"
    assert npz_path.exists(), f"Expected .npz at {npz_path}"

    d = np.load(npz_path)
    h1 = d["h1"]
    assert h1.ndim == 2, f"h1 should be 2D, got ndim={h1.ndim}"
    assert h1.shape[1] == 2, f"h1 should have 2 columns (birth, death), got {h1.shape[1]}"
    assert len(h1) >= 1, (
        "H_1 barcode for a 10-point ring should have at least one bar "
        f"(the loop); got {len(h1)} bars"
    )
    # The ring should have a persistent H_1 bar with non-trivial persistence
    max_persistence = float(np.max(h1[:, 1] - h1[:, 0]))
    assert max_persistence > 0.01, (
        f"Longest H_1 bar has persistence {max_persistence:.4f}; "
        "expected > 0.01 for a unit circle ring"
    )


# ---------------------------------------------------------------------------
# Synthetic tests 2-8: shape, dtype, invariants, behavior
# ---------------------------------------------------------------------------

def test_diagrams_shape(tmp_path: pathlib.Path):
    """Each .npz has h0 and h1 keys; both are 2D float64 with 2 columns; n may be 0."""
    _require_import()

    ring = _ring_points(n=8)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "ru", "синий", ring)

    out_dir = tmp_path / "diagrams"
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)

    npz_path = out_dir / "ru_синий.npz"
    assert npz_path.exists()

    d = np.load(npz_path)
    assert "h0" in d, "npz must contain 'h0' key"
    assert "h1" in d, "npz must contain 'h1' key"

    for key in ("h0", "h1"):
        arr = d[key]
        assert arr.ndim == 2, f"{key}: expected 2D array, got ndim={arr.ndim}"
        assert arr.shape[1] == 2, f"{key}: expected shape (n, 2), got {arr.shape}"
        assert arr.dtype == np.float64, (
            f"{key}: expected float64, got {arr.dtype}"
        )


def test_h0_no_inf(tmp_path: pathlib.Path):
    """No np.inf values in h0 or h1 arrays (infinite bars must be stripped)."""
    _require_import()

    ring = _ring_points(n=10)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "es", "azul", ring)

    out_dir = tmp_path / "diagrams"
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)

    d = np.load(out_dir / "es_azul.npz")
    for key in ("h0", "h1"):
        arr = d[key]
        assert not np.any(np.isinf(arr)), (
            f"{key}: found inf values — infinite bars should be stripped before saving"
        )


def test_birth_le_death(tmp_path: pathlib.Path):
    """Every (b, d) row in h0 and h1 satisfies b <= d."""
    _require_import()

    ring = _ring_points(n=12)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "en", "red", ring)

    out_dir = tmp_path / "diagrams"
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)

    d = np.load(out_dir / "en_red.npz")
    for key in ("h0", "h1"):
        arr = d[key]
        if len(arr) > 0:
            assert np.all(arr[:, 0] <= arr[:, 1]), (
                f"{key}: found bars with birth > death: "
                f"{arr[arr[:, 0] > arr[:, 1]]}"
            )


def test_overwrite_false_skips(tmp_path: pathlib.Path):
    """overwrite=False skips cells whose .npz already exists (returns same data)."""
    _require_import()

    ring = _ring_points(n=6)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "en", "green", ring)
    out_dir = tmp_path / "diagrams"

    # First run — writes the .npz
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)
    npz_path = out_dir / "en_green.npz"
    assert npz_path.exists()
    mtime_1 = npz_path.stat().st_mtime

    # Second run with overwrite=False — must not rewrite
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1, overwrite=False)
    mtime_2 = npz_path.stat().st_mtime
    assert mtime_1 == mtime_2, (
        "overwrite=False should not rewrite an existing .npz "
        f"(mtime changed from {mtime_1} to {mtime_2})"
    )


def test_overwrite_true_rewrites(tmp_path: pathlib.Path):
    """overwrite=True rewrites an existing .npz even if it already exists."""
    _require_import()

    ring = _ring_points(n=6)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "en", "white", ring)
    out_dir = tmp_path / "diagrams"

    # First run
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)
    npz_path = out_dir / "en_white.npz"
    mtime_1 = npz_path.stat().st_mtime

    # Force mtime to differ by at least 0.01s
    import time; time.sleep(0.02)

    # Second run with overwrite=True — must rewrite
    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1, overwrite=True)
    mtime_2 = npz_path.stat().st_mtime
    assert mtime_2 > mtime_1, (
        "overwrite=True should rewrite the .npz "
        f"(mtime unchanged: {mtime_1} == {mtime_2})"
    )


def test_manifest_columns(tmp_path: pathlib.Path):
    """Returned DataFrame has columns [lang, term, n_h0, n_h1, file]."""
    _require_import()

    ring = _ring_points(n=8)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "ru", "красный", ring)
    out_dir = tmp_path / "diagrams"

    summary = compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)

    expected_cols = {"lang", "term", "n_h0", "n_h1", "file"}
    assert expected_cols.issubset(set(summary.columns)), (
        f"Expected columns {expected_cols}, got {list(summary.columns)}"
    )
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["lang"] == "ru"
    assert row["term"] == "красный"
    assert isinstance(row["n_h0"], (int, np.integer))
    assert isinstance(row["n_h1"], (int, np.integer))


def test_missing_pointcloud_manifest_raises(tmp_path: pathlib.Path):
    """compute_diagrams raises FileNotFoundError when manifest.csv is absent."""
    _require_import()

    pc_dir = tmp_path / "empty_pointclouds"
    pc_dir.mkdir()  # no manifest.csv
    out_dir = tmp_path / "diagrams"

    with pytest.raises(FileNotFoundError, match="manifest"):
        compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir)


def test_manifest_csv_written(tmp_path: pathlib.Path):
    """manifest.csv is written to out_dir after compute_diagrams."""
    _require_import()

    ring = _ring_points(n=8)
    pc_dir = tmp_path / "pointclouds"
    _write_synthetic_pointcloud(pc_dir, "es", "verde", ring)
    out_dir = tmp_path / "diagrams"

    compute_diagrams(pointcloud_dir=pc_dir, out_dir=out_dir, max_dim=1)

    manifest_path = out_dir / "manifest.csv"
    assert manifest_path.exists(), f"manifest.csv not written to {out_dir}"

    df = pd.read_csv(manifest_path)
    assert len(df) == 1
    assert set(df.columns).issuperset({"lang", "term", "n_h0", "n_h1", "file"})


# ---------------------------------------------------------------------------
# Gated real-data tests
# ---------------------------------------------------------------------------

def test_canon_term_coverage():
    """34 .npz files match the 34 entries in the pointclouds manifest.

    Gated by PH_REQUIRE_DRAGANOV_DIAGRAMS=1.
    """
    _require_import()
    if not _diagrams_ready():
        _skip_or_fail(f"draganov_diagrams not built yet: {DIAGRAMS_DIR}")

    pc_manifest = _load_pointclouds_manifest()
    assert len(pc_manifest) == 34, (
        f"Expected 34 rows in pointclouds manifest, got {len(pc_manifest)}"
    )

    for _, row in pc_manifest.iterrows():
        lang, term = row["lang"], row["term"]
        expected_npz = DIAGRAMS_DIR / f"{lang}_{term}.npz"
        assert expected_npz.exists(), (
            f"Missing diagram file for ({lang}, {term!r}): {expected_npz}"
        )


def test_h0_no_inf_real():
    """No inf values in any h0/h1 arrays in the real cache.

    Gated by PH_REQUIRE_DRAGANOV_DIAGRAMS=1.
    """
    _require_import()
    if not _diagrams_ready():
        _skip_or_fail(f"draganov_diagrams not built yet: {DIAGRAMS_DIR}")

    manifest = _load_diagrams_manifest()
    for _, row in manifest.iterrows():
        npz_path = DIAGRAMS_DIR / f"{row['lang']}_{row['term']}.npz"
        d = np.load(npz_path)
        for key in ("h0", "h1"):
            arr = d[key]
            assert not np.any(np.isinf(arr)), (
                f"({row['lang']}, {row['term']!r}) {key}: found inf values"
            )


def test_birth_le_death_real():
    """Every (b, d) row in real diagrams satisfies b <= d.

    Gated by PH_REQUIRE_DRAGANOV_DIAGRAMS=1.
    """
    _require_import()
    if not _diagrams_ready():
        _skip_or_fail(f"draganov_diagrams not built yet: {DIAGRAMS_DIR}")

    manifest = _load_diagrams_manifest()
    for _, row in manifest.iterrows():
        npz_path = DIAGRAMS_DIR / f"{row['lang']}_{row['term']}.npz"
        d = np.load(npz_path)
        for key in ("h0", "h1"):
            arr = d[key]
            if len(arr) > 0:
                assert np.all(arr[:, 0] <= arr[:, 1]), (
                    f"({row['lang']}, {row['term']!r}) {key}: birth > death in some bars: "
                    f"{arr[arr[:, 0] > arr[:, 1]]}"
                )


def test_cells_match_pointcloud_manifest():
    """(lang, term) set in diagrams manifest exactly matches pointclouds manifest.

    Gated by PH_REQUIRE_DRAGANOV_DIAGRAMS=1.
    """
    _require_import()
    if not _diagrams_ready():
        _skip_or_fail(f"draganov_diagrams not built yet: {DIAGRAMS_DIR}")

    pc_manifest = _load_pointclouds_manifest()
    diag_manifest = _load_diagrams_manifest()

    pc_cells = set(zip(pc_manifest["lang"], pc_manifest["term"]))
    diag_cells = set(zip(diag_manifest["lang"], diag_manifest["term"]))

    missing_in_diag = pc_cells - diag_cells
    extra_in_diag = diag_cells - pc_cells

    assert not missing_in_diag, (
        f"Cells in pointclouds manifest but missing from diagrams manifest: "
        f"{sorted(missing_in_diag)}"
    )
    assert not extra_in_diag, (
        f"Cells in diagrams manifest but not in pointclouds manifest: "
        f"{sorted(extra_in_diag)}"
    )
