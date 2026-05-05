"""Tests for replication/diagram_distances.py — compute functions.

Covers:
  - compute_per_head_distances: symmetry, zero diagonal, dtype, shape
  - compute_full_distance_tensor: dtype, shape (12, 12, N, N, 2)
  - Determinism given same input + seed
  - cache round-trip: save_distance_tensor / load_distance_tensor
  - Integration test (gated by PH_REQUIRE_DIAGRAM_DISTANCES): tiny synthetic run in <30s

PH_REQUIRE_DIAGRAM_DISTANCES=1 flips skips into hard failures, mirroring
the pattern in test_mbert_attention_ripser_features.py and
test_diagram_distances_loader.py.
"""
from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import pandas as pd
import pytest

REQUIRE = os.environ.get("PH_REQUIRE_DIAGRAM_DISTANCES") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Conditional import — tests fail on missing production code in Phase 1
# ---------------------------------------------------------------------------
try:
    from replication.diagram_distances import (
        compute_per_head_distances,
        compute_full_distance_tensor,
        cache_path,
        save_distance_tensor,
        load_distance_tensor,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _skip_or_fail(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason + " (PH_REQUIRE_DIAGRAM_DISTANCES=1)")
    pytest.skip(reason)


def _require_import():
    if not _IMPORT_OK:
        pytest.fail(
            "Could not import compute/cache functions from replication.diagram_distances — "
            "module does not yet expose compute_per_head_distances, compute_full_distance_tensor, "
            "cache_path, save_distance_tensor, load_distance_tensor."
        )


# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_per_layer_head_diagrams(
    n_layers: int = 2,
    n_heads: int = 2,
    n_samples: int = 10,
    seed: int = 7,
) -> dict:
    """Build a per_layer_head_diagrams dict in the parsed (int-key) format.

    Each sample has H_0 and H_1 features as float64 arrays of shape (n_features, 2).
    This is the format produced by load_barcode_json / load_lang_barcodes — not the
    raw JSON string-key format.
    """
    rng = np.random.default_rng(seed)
    diagrams: dict = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            samples = []
            for _ in range(n_samples):
                n_h0 = int(rng.integers(1, 5))
                n_h1 = int(rng.integers(0, 4))
                h0 = rng.random((n_h0, 2)) * 0.5 + np.array([0.0, 0.3])
                # Ensure birth <= death
                h0[:, 1] = h0[:, 0] + rng.random(n_h0) * 0.3
                if n_h1 > 0:
                    h1 = rng.random((n_h1, 2)) * 0.5 + np.array([0.0, 0.3])
                    h1[:, 1] = h1[:, 0] + rng.random(n_h1) * 0.3
                else:
                    h1 = np.empty((0, 2), dtype=np.float64)
                samples.append({
                    0: h0.astype(np.float64),
                    1: h1.astype(np.float64),
                })
            diagrams[(layer, head)] = samples
    return diagrams


def _make_synthetic_metadata(n_samples: int = 10, terms=("red", "blue"), lang: str = "en") -> pd.DataFrame:
    """Build a synthetic metadata DataFrame."""
    rows = []
    n_per_term = n_samples // len(terms)
    for term in terms:
        for i in range(n_per_term):
            rows.append({
                "lang": lang,
                "term": term,
                "sentence_idx_within_term": i,
                "source_file": "synthetic.json",
                "source_part": 1,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# cache_path tests
# ---------------------------------------------------------------------------

class TestCachePath:
    def test_import(self):
        _require_import()

    def test_returns_path_with_metric_name(self, tmp_path):
        _require_import()
        p = cache_path(tmp_path, "wasserstein")
        assert p == tmp_path / "wasserstein.npz"

    def test_bottleneck(self, tmp_path):
        _require_import()
        p = cache_path(tmp_path, "bottleneck")
        assert p == tmp_path / "bottleneck.npz"

    def test_returns_path_type(self, tmp_path):
        _require_import()
        p = cache_path(tmp_path, "wasserstein")
        assert isinstance(p, pathlib.Path)


# ---------------------------------------------------------------------------
# compute_per_head_distances tests
# ---------------------------------------------------------------------------

class TestComputePerHeadDistances:
    def test_import(self):
        _require_import()

    def test_output_dtype_float32(self):
        """Output array must be float32."""
        _require_import()
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=6)
        indices = np.arange(6)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0, 1),
        )
        assert result.dtype == np.float32, (
            f"Expected float32, got {result.dtype}"
        )

    def test_output_shape_N_N_ndims(self):
        """Shape must be (N, N, len(dims))."""
        _require_import()
        n = 6
        dims = (0, 1)
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n)
        indices = np.arange(n)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=dims,
        )
        assert result.shape == (n, n, len(dims)), (
            f"Expected shape {(n, n, len(dims))}, got {result.shape}"
        )

    def test_symmetry(self):
        """Distance matrix must be symmetric: D[i, j, d] == D[j, i, d] for each dim d."""
        _require_import()
        n = 8
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n)
        indices = np.arange(n)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0, 1),
        )
        for d in range(result.shape[2]):
            mat = result[:, :, d]
            np.testing.assert_allclose(
                mat, mat.T, atol=1e-5,
                err_msg=f"Distance matrix not symmetric for dim index {d}",
            )

    def test_zero_diagonal(self):
        """Diagonal entries must be zero: D[i, i, d] == 0 for each dim d."""
        _require_import()
        n = 8
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n)
        indices = np.arange(n)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0, 1),
        )
        for d in range(result.shape[2]):
            diag = np.diag(result[:, :, d])
            np.testing.assert_allclose(
                diag, 0.0, atol=1e-6,
                err_msg=f"Diagonal not zero for dim index {d}",
            )

    def test_deterministic(self):
        """Same input and seed must produce identical output."""
        _require_import()
        n = 6
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n, seed=42)
        indices = np.arange(n)
        result1 = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0, 1),
        )
        result2 = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0, 1),
        )
        np.testing.assert_array_equal(result1, result2, err_msg="Results differ across identical calls")

    def test_single_dim(self):
        """dims=(0,) alone should produce shape (N, N, 1)."""
        _require_import()
        n = 5
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n)
        indices = np.arange(n)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0,),
        )
        assert result.shape == (n, n, 1), f"Expected (N, N, 1), got {result.shape}"

    def test_nonnegative(self):
        """All distance values must be >= 0."""
        _require_import()
        n = 6
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n)
        indices = np.arange(n)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="wasserstein", dims=(0, 1),
        )
        assert (result >= 0).all(), "Distance matrix has negative entries"

    def test_bottleneck_metric(self):
        """Bottleneck metric path: shape, dtype, symmetry, zero diagonal, nonnegative.

        The CLI runs both metrics by default; without a direct test, a regression
        in the bottleneck code path (e.g., metric_params handling) would be invisible.
        """
        _require_import()
        n = 6
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=1, n_heads=1, n_samples=n)
        indices = np.arange(n)
        result = compute_per_head_distances(
            diagrams, indices, layer=0, head=0,
            metric="bottleneck", dims=(0, 1),
        )
        assert result.shape == (n, n, 2), f"Expected (N, N, 2), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        assert (result >= 0).all(), "Bottleneck distance matrix has negative entries"
        for d in range(result.shape[2]):
            mat = result[:, :, d]
            np.testing.assert_allclose(mat, mat.T, atol=1e-5,
                err_msg=f"Bottleneck not symmetric for dim index {d}")
            np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-6,
                err_msg=f"Bottleneck diagonal not zero for dim index {d}")


# ---------------------------------------------------------------------------
# compute_full_distance_tensor tests
# ---------------------------------------------------------------------------

class TestComputeFullDistanceTensor:
    def test_import(self):
        _require_import()

    def test_output_shape_12x12_N_N_ndims(self):
        """Full tensor shape must be (12, 12, N, N, len(dims))."""
        _require_import()
        n = 4
        dims = (0, 1)
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=12, n_heads=12, n_samples=n)
        indices = np.arange(n)
        result = compute_full_distance_tensor(
            diagrams, indices, metric="wasserstein", dims=dims,
            layers=range(12), heads=range(12), progress=False,
        )
        expected_shape = (12, 12, n, n, len(dims))
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result.shape}"
        )

    def test_output_dtype_float32(self):
        """Output tensor must be float32."""
        _require_import()
        n = 4
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=2, n_heads=2, n_samples=n)
        indices = np.arange(n)
        result = compute_full_distance_tensor(
            diagrams, indices, metric="wasserstein", dims=(0, 1),
            layers=range(2), heads=range(2), progress=False,
        )
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_subset_layers_heads(self):
        """Passing a subset of layers/heads should produce a smaller tensor."""
        _require_import()
        n = 4
        dims = (0, 1)
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=12, n_heads=12, n_samples=n)
        indices = np.arange(n)
        result = compute_full_distance_tensor(
            diagrams, indices, metric="wasserstein", dims=dims,
            layers=range(3), heads=range(4), progress=False,
        )
        # layers=range(3) → 3 layers, heads=range(4) → 4 heads
        assert result.shape == (3, 4, n, n, len(dims)), (
            f"Expected (3, 4, {n}, {n}, {len(dims)}), got {result.shape}"
        )

    def test_per_head_slice_matches_compute_per_head(self):
        """Each (layer, head) slice of compute_full_distance_tensor must match compute_per_head_distances."""
        _require_import()
        n = 5
        dims = (0, 1)
        diagrams = _make_synthetic_per_layer_head_diagrams(n_layers=2, n_heads=2, n_samples=n)
        indices = np.arange(n)

        full = compute_full_distance_tensor(
            diagrams, indices, metric="wasserstein", dims=dims,
            layers=range(2), heads=range(2), progress=False,
        )

        for layer in range(2):
            for head in range(2):
                per_head = compute_per_head_distances(
                    diagrams, indices, layer=layer, head=head,
                    metric="wasserstein", dims=dims,
                )
                np.testing.assert_allclose(
                    full[layer, head], per_head, atol=1e-6,
                    err_msg=f"Mismatch at (layer={layer}, head={head})",
                )


# ---------------------------------------------------------------------------
# save/load round-trip tests
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:
    def test_import(self):
        _require_import()

    def test_round_trip_identical_tensor(self, tmp_path):
        """save_distance_tensor then load_distance_tensor returns identical tensor."""
        _require_import()
        n = 4
        dims = (0, 1)
        tensor = np.random.default_rng(0).random((2, 2, n, n, len(dims))).astype(np.float32)
        metadata = _make_synthetic_metadata(n_samples=n)
        metric = "wasserstein"
        path = tmp_path / "wasserstein.npz"

        save_distance_tensor(tensor, metadata, dims, metric, path)
        loaded_tensor, loaded_meta, loaded_metric, loaded_dims = load_distance_tensor(path)

        np.testing.assert_array_equal(tensor, loaded_tensor, err_msg="Tensor changed after round-trip")

    def test_round_trip_identical_metadata(self, tmp_path):
        """Metadata DataFrame survives round-trip through save/load."""
        _require_import()
        n = 6
        dims = (0, 1)
        tensor = np.zeros((2, 2, n, n, len(dims)), dtype=np.float32)
        metadata = _make_synthetic_metadata(n_samples=n)
        path = tmp_path / "bottleneck.npz"

        save_distance_tensor(tensor, metadata, dims, "bottleneck", path)
        _, loaded_meta, _, _ = load_distance_tensor(path)

        pd.testing.assert_frame_equal(
            metadata.reset_index(drop=True),
            loaded_meta.reset_index(drop=True),
            check_like=True,
        )

    def test_round_trip_metric_string(self, tmp_path):
        """Metric string is preserved through round-trip."""
        _require_import()
        n = 4
        dims = (0, 1)
        tensor = np.zeros((2, 2, n, n, 2), dtype=np.float32)
        metadata = _make_synthetic_metadata(n_samples=n)
        path = tmp_path / "wasserstein.npz"

        for metric in ("wasserstein", "bottleneck"):
            p = tmp_path / f"{metric}.npz"
            save_distance_tensor(tensor, metadata, dims, metric, p)
            _, _, loaded_metric, _ = load_distance_tensor(p)
            assert loaded_metric == metric, (
                f"Metric changed: saved '{metric}', loaded '{loaded_metric}'"
            )

    def test_round_trip_homology_dimensions(self, tmp_path):
        """Homology dimensions tuple is preserved through round-trip."""
        _require_import()
        n = 4
        for dims in [(0, 1), (0,), (1,)]:
            tensor = np.zeros((2, 2, n, n, len(dims)), dtype=np.float32)
            metadata = _make_synthetic_metadata(n_samples=n)
            path = tmp_path / f"test_dims_{'_'.join(map(str, dims))}.npz"

            save_distance_tensor(tensor, metadata, dims, "wasserstein", path)
            _, _, _, loaded_dims = load_distance_tensor(path)

            assert tuple(loaded_dims) == tuple(dims), (
                f"Dims changed: saved {dims}, loaded {loaded_dims}"
            )

    def test_load_returns_float32_tensor(self, tmp_path):
        """Loaded tensor must be float32 (not upcast during save/load)."""
        _require_import()
        n = 4
        dims = (0, 1)
        tensor = np.ones((2, 2, n, n, 2), dtype=np.float32)
        metadata = _make_synthetic_metadata(n_samples=n)
        path = tmp_path / "wasserstein.npz"

        save_distance_tensor(tensor, metadata, dims, "wasserstein", path)
        loaded_tensor, _, _, _ = load_distance_tensor(path)

        assert loaded_tensor.dtype == np.float32, (
            f"Expected float32 after round-trip, got {loaded_tensor.dtype}"
        )

    def test_cache_path_used_by_save(self, tmp_path):
        """cache_path returns a path that is written by save_distance_tensor."""
        _require_import()
        n = 4
        tensor = np.zeros((2, 2, n, n, 2), dtype=np.float32)
        metadata = _make_synthetic_metadata(n_samples=n)
        metric = "wasserstein"
        p = cache_path(tmp_path, metric)

        save_distance_tensor(tensor, metadata, (0, 1), metric, p)
        assert p.exists(), f"Expected {p} to exist after save_distance_tensor"


# ---------------------------------------------------------------------------
# Integration test (gated by PH_REQUIRE_DIAGRAM_DISTANCES)
# ---------------------------------------------------------------------------

def test_integration_tiny_synthetic_run(tmp_path):
    """Integration test: compute + save + load on tiny synthetic data.

    Runs compute_per_head_distances on 20 synthetic samples, 1 layer, 1 head,
    wasserstein metric. Must complete in < 30s. Validates the full pipeline:
    compute → save → load → check.

    Gated by PH_REQUIRE_DIAGRAM_DISTANCES=1.
    """
    if not os.environ.get("PH_REQUIRE_DIAGRAM_DISTANCES") == "1":
        pytest.skip("Set PH_REQUIRE_DIAGRAM_DISTANCES=1 to run integration tests")

    _require_import()

    n = 20
    dims = (0, 1)
    diagrams = _make_synthetic_per_layer_head_diagrams(
        n_layers=1, n_heads=1, n_samples=n, seed=123
    )
    indices = np.arange(n)
    metadata = _make_synthetic_metadata(n_samples=n)

    t0 = time.time()
    result = compute_per_head_distances(
        diagrams, indices, layer=0, head=0,
        metric="wasserstein", dims=dims,
    )
    elapsed = time.time() - t0

    assert elapsed < 30.0, f"Integration test took {elapsed:.1f}s, must be < 30s"
    assert result.shape == (n, n, len(dims))
    assert result.dtype == np.float32

    # Symmetry and zero diagonal. compute_per_head_distances symmetrizes in
    # float64 before downcasting, so D[i,j] == D[j,i] should hold exactly in
    # float32 — atol=0.0 would also pass but a tiny atol guards against any
    # future numeric tweaks.
    for d in range(len(dims)):
        mat = result[:, :, d]
        np.testing.assert_allclose(mat, mat.T, atol=1e-7)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-6)

    # Cache round-trip
    path = cache_path(tmp_path, "wasserstein")
    # Build a (1, 1, N, N, 2) tensor from the per-head result
    full_tensor = result[np.newaxis, np.newaxis, ...]  # (1, 1, N, N, 2)
    save_distance_tensor(full_tensor, metadata, dims, "wasserstein", path)
    loaded_tensor, loaded_meta, loaded_metric, loaded_dims = load_distance_tensor(path)

    np.testing.assert_array_equal(full_tensor, loaded_tensor)
    assert loaded_metric == "wasserstein"
    assert tuple(loaded_dims) == tuple(dims)
