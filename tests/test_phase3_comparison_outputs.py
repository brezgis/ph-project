"""Tests for phase3_comparison.ipynb outputs.

Mirrors the shape/no-NaN/file-existence pattern from test_mbert_attention_thresholds_features.py.
Skipped unless the comparison notebook has been executed (outputs exist).

Checks:
  - results/phase3_comparison/summary.csv exists and has required columns + 1 row per pair
  - results/figures/phase3_thresholds_comparison_color.{pdf,png} exist
  - No NaN values in summary CSV numeric columns
  - summary CSV columns are a superset of the required set

Set PH_REQUIRE_COMPARISON=1 to flip all skips into hard failures (CI / post-run mode).
"""
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

REQUIRE = os.environ.get("PH_REQUIRE_COMPARISON") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
COMP_DIR = REPO_ROOT / "results" / "phase3_comparison"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

SUMMARY_CSV = COMP_DIR / "summary.csv"

REQUIRED_COLUMNS = {
    "feature_type",
    "pair",
    "n_sig_features_q05",
    "perm_p_value",
}

EXPECTED_PAIRS = {"en-es", "en-ru", "ru-es"}
EXPECTED_FEATURE_TYPES_MIN = {"thresholds"}  # ripser+template optional

EXPECTED_FIGURES = [
    "phase3_thresholds_comparison_color.pdf",
    "phase3_thresholds_comparison_color.png",
]


def _skip_or_fail(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason + " (PH_REQUIRE_COMPARISON=1)")
    pytest.skip(reason)


# ---------------------------------------------------------------------------
# Summary CSV tests
# ---------------------------------------------------------------------------

def test_summary_csv_exists():
    """Summary CSV must exist after notebook execution."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")


def test_summary_csv_columns():
    """Summary CSV must contain the required columns."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, (
        f"summary CSV is missing required columns: {missing}. "
        f"Found: {list(df.columns)}"
    )


def test_summary_csv_has_rows():
    """Summary CSV must have at least one row per expected pair (catches partial runs)."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    assert len(df) >= len(EXPECTED_PAIRS), (
        f"summary CSV has {len(df)} rows; expected at least {len(EXPECTED_PAIRS)} "
        f"(one per pair: {EXPECTED_PAIRS})."
    )


def test_summary_csv_pairs():
    """All three language pairs must appear in summary CSV."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "pair" not in df.columns:
        _skip_or_fail("'pair' column absent — summary CSV missing required column")
    found_pairs = set(df["pair"].unique())
    assert EXPECTED_PAIRS <= found_pairs, (
        f"Expected language pairs {EXPECTED_PAIRS} not all found in summary CSV. "
        f"Found: {found_pairs}"
    )


def test_summary_csv_thresholds_feature_type():
    """'thresholds' feature type must appear (it's guaranteed — files exist on disk)."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "feature_type" not in df.columns:
        _skip_or_fail("'feature_type' column absent")
    found = set(df["feature_type"].unique())
    assert "thresholds" in found, (
        f"'thresholds' feature type missing from summary CSV. Found: {found}"
    )


def test_summary_csv_no_nan_in_perm_p():
    """Permutation test p-values must not be NaN (test must have produced a result)."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "perm_p_value" not in df.columns:
        _skip_or_fail("'perm_p_value' column absent")
    n_nan = df["perm_p_value"].isna().sum()
    assert n_nan == 0, (
        f"{n_nan} NaN values found in 'perm_p_value' column — "
        "permutation test may have crashed silently."
    )


def test_summary_csv_perm_p_in_range():
    """Permutation test p-values must be in [0, 1]."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "perm_p_value" not in df.columns:
        _skip_or_fail("'perm_p_value' column absent")
    bad = df[(df["perm_p_value"] < 0) | (df["perm_p_value"] > 1)]
    assert len(bad) == 0, (
        f"{len(bad)} rows have perm_p_value outside [0, 1]:\n{bad}"
    )


def test_summary_csv_n_sig_features_nonneg():
    """n_sig_features_q05 must be non-negative integers."""
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "n_sig_features_q05" not in df.columns:
        _skip_or_fail("'n_sig_features_q05' column absent")
    bad = df[df["n_sig_features_q05"] < 0]
    assert len(bad) == 0, (
        f"{len(bad)} rows have negative n_sig_features_q05."
    )


def test_summary_csv_obs_dist_finite():
    """perm_obs_dist must be finite (not inf/nan).

    A non-finite value here indicates float16 overflow in the flatten step —
    the feature tensors are stored as float16 and must be upcast to float64
    before computing L2 norms over 5184 dimensions.
    """
    if not SUMMARY_CSV.exists():
        _skip_or_fail(f"summary CSV not yet produced: {SUMMARY_CSV}")
    df = pd.read_csv(SUMMARY_CSV)
    if "perm_obs_dist" not in df.columns:
        _skip_or_fail("'perm_obs_dist' column absent")
    bad = df[~np.isfinite(df["perm_obs_dist"])]
    assert len(bad) == 0, (
        f"{len(bad)} rows have non-finite perm_obs_dist (inf or nan): "
        f"{bad[['feature_type', 'pair', 'perm_obs_dist']].to_dict('records')}. "
        "Likely float16 overflow — ensure flatten functions cast to float64."
    )


# ---------------------------------------------------------------------------
# Figure file tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fname", EXPECTED_FIGURES)
def test_figure_exists(fname):
    """Required output figures must exist after notebook execution."""
    path = FIGURES_DIR / fname
    if not path.exists():
        _skip_or_fail(f"Expected figure not produced: {path}")
    assert path.stat().st_size > 1024, (
        f"Figure {fname} appears too small ({path.stat().st_size} bytes) — "
        "may be empty or corrupted."
    )


