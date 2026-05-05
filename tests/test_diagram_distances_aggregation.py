"""Tests for aggregation helpers in replication/diagram_distances.py.

Covers:
  - rank_heads_by_effect: top_k count, sort order, rank column, passes_bh preserved
  - effect_heatmap_data: shape (12, 12), layer/head indexing, missing cells → NaN

Uses deterministic synthetic DataFrames; no real-data dependencies.

PH_REQUIRE_DIAGRAM_DISTANCES=1 flips skips into hard failures, mirroring the
pattern in test_diagram_distances_perm.py and test_diagram_distances_per_term.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

REQUIRE = os.environ.get("PH_REQUIRE_DIAGRAM_DISTANCES") == "1"

# ---------------------------------------------------------------------------
# Conditional import — tests must fail (ImportError) before Phase 2
# ---------------------------------------------------------------------------
try:
    from replication.diagram_distances import (
        rank_heads_by_effect,
        effect_heatmap_data,
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
            "Could not import aggregation helpers from replication.diagram_distances — "
            "rank_heads_by_effect and effect_heatmap_data are not yet defined."
        )


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_per_head_df(n_layers: int = 12, n_heads: int = 12, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic per-head results DataFrame matching permutation_test_per_head output.

    Columns: [layer, head, observed, p_value, effect_size, passes_bh]
    """
    rng = np.random.default_rng(seed)
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            effect_size = float(rng.normal(0.0, 2.0))
            p_value = float(rng.uniform(0.0, 1.0))
            passes_bh = bool(p_value < 0.05)
            rows.append({
                "layer": layer,
                "head": head,
                "observed": float(rng.normal(0.0, 0.5)),
                "p_value": p_value,
                "effect_size": effect_size,
                "passes_bh": passes_bh,
            })
    df = pd.DataFrame(rows)
    df["passes_bh"] = df["passes_bh"].astype(bool)
    return df


def _make_small_df() -> pd.DataFrame:
    """Build a small deterministic per-head DataFrame with known effect_size values.

    Used to test rank ordering without seed-dependent randomness.
    """
    rows = [
        # layer, head, observed, p_value, effect_size, passes_bh
        {"layer": 0, "head": 0, "observed": 0.1, "p_value": 0.01, "effect_size":  3.0, "passes_bh": True},
        {"layer": 0, "head": 1, "observed": 0.2, "p_value": 0.10, "effect_size": -5.0, "passes_bh": False},
        {"layer": 1, "head": 0, "observed": 0.3, "p_value": 0.04, "effect_size":  1.5, "passes_bh": True},
        {"layer": 1, "head": 1, "observed": 0.4, "p_value": 0.50, "effect_size": -0.5, "passes_bh": False},
        {"layer": 2, "head": 0, "observed": 0.5, "p_value": 0.02, "effect_size":  4.0, "passes_bh": True},
        {"layer": 2, "head": 1, "observed": 0.6, "p_value": 0.70, "effect_size": -2.0, "passes_bh": False},
    ]
    df = pd.DataFrame(rows)
    df["passes_bh"] = df["passes_bh"].astype(bool)
    return df


# ---------------------------------------------------------------------------
# rank_heads_by_effect tests
# ---------------------------------------------------------------------------

class TestRankHeadsByEffect:
    def test_import(self):
        _require_import()

    def test_top_k_count(self):
        """rank_heads_by_effect with top_k=5 returns exactly 5 rows."""
        _require_import()
        df = _make_per_head_df(seed=1)
        result = rank_heads_by_effect(df, top_k=5)
        assert len(result) == 5, f"Expected 5 rows, got {len(result)}"

    def test_top_k_count_10(self):
        """rank_heads_by_effect with top_k=10 returns exactly 10 rows."""
        _require_import()
        df = _make_per_head_df(seed=2)
        result = rank_heads_by_effect(df, top_k=10)
        assert len(result) == 10, f"Expected 10 rows, got {len(result)}"

    def test_sorted_by_abs_effect_size_descending(self):
        """Rows are sorted by |effect_size| in descending order."""
        _require_import()
        df = _make_small_df()
        # 6 rows available, request all 6
        result = rank_heads_by_effect(df, top_k=6)
        abs_effects = result["effect_size"].abs().values
        assert (abs_effects[:-1] >= abs_effects[1:]).all(), (
            f"Rows are not sorted by |effect_size| descending: {abs_effects.tolist()}"
        )

    def test_rank_column_is_one_indexed(self):
        """rank column is 1-indexed: [1, 2, 3, 4, 5]."""
        _require_import()
        df = _make_per_head_df(seed=3)
        result = rank_heads_by_effect(df, top_k=5)
        expected_ranks = list(range(1, 6))
        assert result["rank"].tolist() == expected_ranks, (
            f"Expected rank=[1,2,3,4,5], got {result['rank'].tolist()}"
        )

    def test_rank_column_default_top_k(self):
        """Default top_k=20 produces rank column [1..20]."""
        _require_import()
        df = _make_per_head_df(seed=4)
        result = rank_heads_by_effect(df)  # default top_k=20
        expected_ranks = list(range(1, 21))
        assert result["rank"].tolist() == expected_ranks, (
            f"Expected rank=[1..20], got {result['rank'].tolist()}"
        )

    def test_passes_bh_preserved(self):
        """passes_bh column is preserved and matches original values for selected rows."""
        _require_import()
        df = _make_small_df()
        result = rank_heads_by_effect(df, top_k=3)
        # Check that passes_bh values match the source DataFrame
        assert "passes_bh" in result.columns, "passes_bh column missing from result"
        assert result["passes_bh"].dtype == bool, (
            f"passes_bh must be bool, got {result['passes_bh'].dtype}"
        )
        # Each returned row's passes_bh should match the corresponding source row
        for _, row in result.iterrows():
            src = df[(df["layer"] == row["layer"]) & (df["head"] == row["head"])]
            assert len(src) == 1
            assert bool(row["passes_bh"]) == bool(src.iloc[0]["passes_bh"]), (
                f"passes_bh mismatch at layer={row['layer']}, head={row['head']}"
            )

    def test_correct_top_rows_by_abs_effect(self):
        """The returned rows are actually the top-K by |effect_size|."""
        _require_import()
        df = _make_small_df()
        # Known |effect_size| order (descending): 5.0, 4.0, 3.0, 2.0, 1.5, 0.5
        # heads: (0,1), (2,0), (0,0), (2,1), (1,0), (1,1)
        result = rank_heads_by_effect(df, top_k=3)
        top3_abs = result["effect_size"].abs().tolist()
        expected_top3 = sorted(df["effect_size"].abs().tolist(), reverse=True)[:3]
        assert top3_abs == pytest.approx(expected_top3), (
            f"Top-3 |effect_size|: expected {expected_top3}, got {top3_abs}"
        )

    def test_output_columns_include_rank(self):
        """Output DataFrame has a 'rank' column in addition to input columns."""
        _require_import()
        df = _make_per_head_df(seed=5)
        result = rank_heads_by_effect(df, top_k=5)
        assert "rank" in result.columns, f"'rank' column missing. Columns: {list(result.columns)}"

    def test_top_k_larger_than_df(self):
        """If top_k > len(df), returns all rows sorted (no error)."""
        _require_import()
        df = _make_small_df()  # 6 rows
        result = rank_heads_by_effect(df, top_k=100)
        assert len(result) == len(df), (
            f"Expected {len(df)} rows when top_k > len(df), got {len(result)}"
        )

    def test_top_k_zero_returns_empty(self):
        """top_k=0 returns an empty DataFrame with the rank column present."""
        _require_import()
        df = _make_small_df()
        result = rank_heads_by_effect(df, top_k=0)
        assert len(result) == 0, f"Expected 0 rows for top_k=0, got {len(result)}"
        assert "rank" in result.columns, (
            f"'rank' column missing from empty result. Columns: {list(result.columns)}"
        )

    def test_empty_dataframe(self):
        """Empty input DataFrame returns an empty result without error."""
        _require_import()
        df = pd.DataFrame(
            columns=["layer", "head", "observed", "p_value", "effect_size", "passes_bh"]
        )
        df["passes_bh"] = df["passes_bh"].astype(bool)
        result = rank_heads_by_effect(df, top_k=5)
        assert len(result) == 0, f"Expected 0 rows for empty input, got {len(result)}"
        assert "rank" in result.columns

    def test_ties_produce_contiguous_ranks(self):
        """Ties in |effect_size| still yield a contiguous 1..N rank column."""
        _require_import()
        # Two rows with identical |effect_size|=3.0 (one positive, one negative)
        rows = [
            {"layer": 0, "head": 0, "observed": 0.1, "p_value": 0.01, "effect_size":  3.0, "passes_bh": True},
            {"layer": 0, "head": 1, "observed": 0.2, "p_value": 0.02, "effect_size": -3.0, "passes_bh": True},
            {"layer": 1, "head": 0, "observed": 0.3, "p_value": 0.10, "effect_size":  1.0, "passes_bh": False},
        ]
        df = pd.DataFrame(rows)
        df["passes_bh"] = df["passes_bh"].astype(bool)
        result = rank_heads_by_effect(df, top_k=3)
        assert result["rank"].tolist() == [1, 2, 3], (
            f"Tied |effect_size| should still produce ranks [1,2,3], got {result['rank'].tolist()}"
        )


# ---------------------------------------------------------------------------
# effect_heatmap_data tests
# ---------------------------------------------------------------------------

class TestEffectHeatmapData:
    def test_import(self):
        _require_import()

    def test_shape_12_12(self):
        """effect_heatmap_data returns a (12, 12) array for the full 12x12 grid."""
        _require_import()
        df = _make_per_head_df(n_layers=12, n_heads=12, seed=0)
        result = effect_heatmap_data(df)
        assert result.shape == (12, 12), (
            f"Expected shape (12, 12), got {result.shape}"
        )

    def test_layer_head_indexing(self):
        """result[layer, head] equals the effect_size for (layer, head) in the input."""
        _require_import()
        df = _make_small_df()
        # df has layers 0-2, heads 0-1
        result = effect_heatmap_data(df)
        for _, row in df.iterrows():
            layer = int(row["layer"])
            head = int(row["head"])
            expected = float(row["effect_size"])
            actual = float(result[layer, head])
            assert actual == pytest.approx(expected), (
                f"result[{layer}, {head}]={actual} != expected {expected}"
            )

    def test_missing_cells_are_nan(self):
        """Cells not present in the input DataFrame become NaN (defensive fill)."""
        _require_import()
        # Build a DataFrame with only 2 rows (layer=0 head=0, layer=1 head=1)
        df = pd.DataFrame([
            {"layer": 0, "head": 0, "observed": 0.1, "p_value": 0.01, "effect_size": 2.0, "passes_bh": True},
            {"layer": 1, "head": 1, "observed": 0.2, "p_value": 0.05, "effect_size": 1.0, "passes_bh": False},
        ])
        df["passes_bh"] = df["passes_bh"].astype(bool)
        result = effect_heatmap_data(df)

        # Filled cells
        assert float(result[0, 0]) == pytest.approx(2.0)
        assert float(result[1, 1]) == pytest.approx(1.0)

        # Missing cells are NaN
        assert np.isnan(result[0, 1]), f"result[0, 1] should be NaN, got {result[0, 1]}"
        assert np.isnan(result[1, 0]), f"result[1, 0] should be NaN, got {result[1, 0]}"
        assert np.isnan(result[2, 0]), f"result[2, 0] should be NaN, got {result[2, 0]}"

    def test_dtype_is_float(self):
        """Output array dtype is float (float64 or float32 acceptable)."""
        _require_import()
        df = _make_per_head_df(seed=6)
        result = effect_heatmap_data(df)
        assert np.issubdtype(result.dtype, np.floating), (
            f"Expected floating dtype, got {result.dtype}"
        )

    def test_full_grid_no_nan(self):
        """Full 12x12 input produces no NaN values in output."""
        _require_import()
        df = _make_per_head_df(n_layers=12, n_heads=12, seed=7)
        result = effect_heatmap_data(df)
        assert not np.isnan(result).any(), (
            f"Expected no NaN in full-grid output, got {np.isnan(result).sum()} NaN values"
        )

    def test_partial_grid_shape_still_12_12(self):
        """Even with only 2 rows in the input, the output shape is (12, 12)."""
        _require_import()
        df = pd.DataFrame([
            {"layer": 3, "head": 7, "observed": 0.1, "p_value": 0.01, "effect_size": 3.5, "passes_bh": True},
        ])
        df["passes_bh"] = df["passes_bh"].astype(bool)
        result = effect_heatmap_data(df)
        assert result.shape == (12, 12), (
            f"Expected shape (12, 12) even for partial input, got {result.shape}"
        )
        assert float(result[3, 7]) == pytest.approx(3.5)

    def test_values_match_input_for_full_grid(self):
        """For a complete 12x12 grid, every cell value matches the source effect_size."""
        _require_import()
        df = _make_per_head_df(n_layers=12, n_heads=12, seed=8)
        result = effect_heatmap_data(df)
        for _, row in df.iterrows():
            layer = int(row["layer"])
            head = int(row["head"])
            expected = float(row["effect_size"])
            actual = float(result[layer, head])
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"Mismatch at result[{layer}, {head}]: {actual} != {expected}"
            )

    def test_empty_dataframe_yields_all_nan(self):
        """Empty input DataFrame produces a (12, 12) array of all NaN."""
        _require_import()
        df = pd.DataFrame(
            columns=["layer", "head", "observed", "p_value", "effect_size", "passes_bh"]
        )
        result = effect_heatmap_data(df)
        assert result.shape == (12, 12)
        assert np.isnan(result).all(), (
            f"Expected all NaN for empty input, got {(~np.isnan(result)).sum()} non-NaN cells"
        )

    def test_out_of_range_layer_raises(self):
        """A layer index ≥ 12 raises IndexError (fail-loud guard for non-mBERT inputs)."""
        _require_import()
        df = pd.DataFrame([
            {"layer": 12, "head": 0, "observed": 0.1, "p_value": 0.01, "effect_size": 1.0, "passes_bh": False},
        ])
        with pytest.raises(IndexError):
            effect_heatmap_data(df)

    def test_out_of_range_head_raises(self):
        """A head index ≥ 12 raises IndexError."""
        _require_import()
        df = pd.DataFrame([
            {"layer": 0, "head": 12, "observed": 0.1, "p_value": 0.01, "effect_size": 1.0, "passes_bh": False},
        ])
        with pytest.raises(IndexError):
            effect_heatmap_data(df)

    def test_duplicate_layer_head_last_write_wins(self):
        """Duplicate (layer, head) rows take the last value (documented behavior)."""
        _require_import()
        df = pd.DataFrame([
            {"layer": 5, "head": 3, "observed": 0.1, "p_value": 0.01, "effect_size": 1.0, "passes_bh": True},
            {"layer": 5, "head": 3, "observed": 0.2, "p_value": 0.02, "effect_size": 9.9, "passes_bh": False},
        ])
        result = effect_heatmap_data(df)
        assert float(result[5, 3]) == pytest.approx(9.9), (
            f"Expected last value 9.9 to win, got {result[5, 3]}"
        )
