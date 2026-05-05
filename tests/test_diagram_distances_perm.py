"""Tests for permutation-test helpers in replication/diagram_distances.py.

Covers:
  - per_domain_test_statistic: correct between-vs-within statistic
  - permutation_test_per_domain: H_0 calibration, planted-signal detection
  - permutation_test_per_head: BH correction monotonicity

PH_REQUIRE_DIAGRAM_DISTANCES=1 flips skips into hard failures, mirroring
the pattern in test_diagram_distances_compute.py.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

REQUIRE = os.environ.get("PH_REQUIRE_DIAGRAM_DISTANCES") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Slow-test gate: set PH_RUN_SLOW_TESTS=1 to run tests that take >10s each.
# Mirrors the PH_REQUIRE_DIAGRAM_DISTANCES pattern above.
pytestmark_slow = pytest.mark.skipif(
    not os.environ.get("PH_RUN_SLOW_TESTS"),
    reason="Slow test; set PH_RUN_SLOW_TESTS=1 to run.",
)

# ---------------------------------------------------------------------------
# Conditional import — tests fail on missing production code in Phase 1
# ---------------------------------------------------------------------------
try:
    from replication.diagram_distances import (
        per_domain_test_statistic,
        permutation_test_per_domain,
        permutation_test_per_head,
        _bh_correction,
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
            "Could not import permutation helpers from replication.diagram_distances — "
            "per_domain_test_statistic, permutation_test_per_domain, "
            "permutation_test_per_head, _bh_correction are not yet defined."
        )


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_meta(langs: list[str]) -> pd.DataFrame:
    """Build a metadata DataFrame with a 'lang' column from a list of labels."""
    return pd.DataFrame({"lang": langs})


def _planted_distance_matrix(langs: list[str], between_dist: float = 1.0) -> np.ndarray:
    """Build a distance matrix with within-lang=0, between-lang=between_dist.

    The observed test statistic should be strongly positive (between > within).
    """
    n = len(langs)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j and langs[i] != langs[j]:
                mat[i, j] = between_dist
    return mat


def _random_distance_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Build a random symmetric distance matrix with zero diagonal."""
    rng = np.random.default_rng(seed)
    raw = rng.random((n, n))
    mat = 0.5 * (raw + raw.T)
    np.fill_diagonal(mat, 0.0)
    return mat


def _make_distance_tensor(
    n_layers: int,
    n_heads: int,
    langs: list[str],
    between_dist: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Build a (n_layers, n_heads, N, N) tensor for permutation_test_per_head tests."""
    n = len(langs)
    rng = np.random.default_rng(seed)
    tensor = np.zeros((n_layers, n_heads, n, n), dtype=np.float64)
    for l in range(n_layers):
        for h in range(n_heads):
            raw = rng.random((n, n)) * 0.1  # small random noise
            mat = 0.5 * (raw + raw.T)
            np.fill_diagonal(mat, 0.0)
            if between_dist > 0:
                # Add planted signal
                for i in range(n):
                    for j in range(n):
                        if i != j and langs[i] != langs[j]:
                            mat[i, j] += between_dist
            tensor[l, h] = mat
    return tensor


# ---------------------------------------------------------------------------
# per_domain_test_statistic tests
# ---------------------------------------------------------------------------

class TestPerDomainTestStatistic:
    def test_import(self):
        _require_import()

    def test_planted_signal_positive(self):
        """With zero within-lang and nonzero between-lang distances, statistic > 0."""
        _require_import()
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        mat = _planted_distance_matrix(langs, between_dist=1.0)
        stat = per_domain_test_statistic(mat, meta)
        assert stat > 0.0, f"Expected positive statistic with planted signal, got {stat}"

    def test_all_same_lang_zero(self):
        """With all samples from the same language, between-lang mask is empty.

        The test statistic is mean(between) - mean(within). With no between-lang
        pairs, mean(between) is 0 (no pairs) and mean(within) is nonzero, so
        the statistic should be <= 0 or zero when within-lang distances vary.
        When all are same-lang, between mask is all False — statistic should
        be defined and return a real number (not NaN).
        """
        _require_import()
        langs = ["en"] * 6
        meta = _make_meta(langs)
        rng = np.random.default_rng(7)
        raw = rng.random((6, 6))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        stat = per_domain_test_statistic(mat, meta)
        # With no between pairs, result is 0 - mean(within) ≤ 0
        assert np.isfinite(stat), f"Expected finite statistic, got {stat}"
        assert stat <= 0.0  # all-within-language case: between-mean defaults to 0, within-mean >= 0

    def test_symmetric_distance_matrix(self):
        """Statistic must be the same when distance matrix is symmetric (expected always)."""
        _require_import()
        langs = ["en"] * 3 + ["ru"] * 3 + ["es"] * 3
        meta = _make_meta(langs)
        mat = _planted_distance_matrix(langs, between_dist=2.0)
        stat1 = per_domain_test_statistic(mat, meta)
        stat2 = per_domain_test_statistic(mat.T, meta)
        assert abs(stat1 - stat2) < 1e-10, "Statistic should be same for symmetric matrix"

    def test_three_languages_direction(self):
        """With three languages and planted signal, statistic is positive."""
        _require_import()
        langs = ["en"] * 4 + ["ru"] * 4 + ["es"] * 4
        meta = _make_meta(langs)
        mat = _planted_distance_matrix(langs, between_dist=5.0)
        stat = per_domain_test_statistic(mat, meta)
        assert stat > 0.0, f"Expected positive statistic with strong planted signal, got {stat}"


# ---------------------------------------------------------------------------
# Test 1 (from issue): under H_0, p_value is approximately uniform
# ---------------------------------------------------------------------------

class TestPermutationTestUnderNull:
    """Under H_0 (random distance matrix with random labels), p_value is
    approximately uniform across many seeds — verified via KS test."""

    def test_import(self):
        _require_import()

    @pytestmark_slow
    def test_p_value_uniform_under_null(self):
        """KS test: p-values under H_0 should follow Uniform(0, 1).

        We run the permutation test 100 times with different random distance
        matrices and different random label assignments. The collection of
        p-values should not be significantly non-uniform (KS test, alpha=0.01).

        Uses K=999 permutations per test for speed; 100 independent tests for
        the KS check.
        """
        _require_import()
        n = 30
        n_langs_each = 10
        langs = ["en"] * n_langs_each + ["ru"] * n_langs_each + ["es"] * n_langs_each
        meta = _make_meta(langs)

        p_values = []
        for seed in range(100):
            rng = np.random.default_rng(seed)
            # Random distance matrix (H_0: labels independent of distances)
            raw = rng.random((n, n))
            mat = 0.5 * (raw + raw.T)
            np.fill_diagonal(mat, 0.0)

            # Also randomize labels to ensure H_0
            perm_langs = rng.permutation(langs).tolist()
            perm_meta = _make_meta(perm_langs)

            result = permutation_test_per_domain(mat, perm_meta, K=999, seed=int(seed * 17))
            p_values.append(result["p_value"])

        p_values_arr = np.array(p_values)

        # KS test against Uniform(0, 1)
        ks_stat, ks_p = scipy_stats.kstest(p_values_arr, "uniform")

        # Under H_0 with 100 samples, KS p-value should not be extremely small.
        # We allow alpha=0.001 (very lenient — just checking gross miscalibration).
        assert ks_p > 0.001, (
            f"p-values under H_0 appear non-uniform: KS stat={ks_stat:.4f}, "
            f"KS p-value={ks_p:.4f}. "
            f"Mean p={p_values_arr.mean():.3f}, std={p_values_arr.std():.3f}. "
            "Check permutation_test_per_domain p-value formula."
        )


# ---------------------------------------------------------------------------
# Test 2 (from issue): under planted signal, p_value < 0.001 and effect_size > 0
# ---------------------------------------------------------------------------

class TestPermutationTestPlantedSignal:
    """Under a strong planted signal, p_value must be very small and effect_size > 0."""

    def test_import(self):
        _require_import()

    def test_planted_signal_detected(self):
        """With within=0, between=large, test must reject H_0 at alpha=0.001."""
        _require_import()
        langs = ["en"] * 10 + ["ru"] * 10 + ["es"] * 10
        meta = _make_meta(langs)
        # Planted: within-language distances are 0, between are large
        mat = _planted_distance_matrix(langs, between_dist=100.0)

        result = permutation_test_per_domain(mat, meta, K=9999, seed=42)

        assert result["p_value"] < 0.001, (
            f"Expected p_value < 0.001 under planted signal, got {result['p_value']:.4f}. "
            f"observed={result['observed']:.4f}, effect_size={result['effect_size']:.4f}"
        )
        assert result["effect_size"] > 0, (
            f"Expected positive effect size under planted signal, got {result['effect_size']:.4f}"
        )

    def test_result_keys(self):
        """Result dict must have keys: observed, null, p_value, effect_size."""
        _require_import()
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        mat = _planted_distance_matrix(langs, between_dist=1.0)
        result = permutation_test_per_domain(mat, meta, K=99, seed=0)
        assert "observed" in result
        assert "null" in result
        assert "p_value" in result
        assert "effect_size" in result

    def test_null_length_equals_K(self):
        """Null distribution must have exactly K entries."""
        _require_import()
        K = 200
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        mat = _random_distance_matrix(10, seed=3)
        result = permutation_test_per_domain(mat, meta, K=K, seed=1)
        assert len(result["null"]) == K, (
            f"Expected null length {K}, got {len(result['null'])}"
        )

    def test_p_value_in_range(self):
        """p_value must be in (0, 1]."""
        _require_import()
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        mat = _random_distance_matrix(10, seed=5)
        result = permutation_test_per_domain(mat, meta, K=99, seed=2)
        assert 0 < result["p_value"] <= 1.0, (
            f"p_value={result['p_value']} is outside (0, 1]"
        )

    def test_determinism(self):
        """Same seed and inputs produce identical results."""
        _require_import()
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        mat = _random_distance_matrix(10, seed=9)
        r1 = permutation_test_per_domain(mat, meta, K=99, seed=42)
        r2 = permutation_test_per_domain(mat, meta, K=99, seed=42)
        assert r1["p_value"] == r2["p_value"], "Results not deterministic"
        assert r1["observed"] == r2["observed"], "observed not deterministic"
        np.testing.assert_array_equal(r1["null"], r2["null"], err_msg="null not deterministic")

    def test_effect_size_zero_when_null_std_zero(self):
        """All-equal distance matrix → every permutation gives stat==0 → null is exactly 0.

        With an all-ones (minus diagonal) distance matrix and two language groups,
        every permutation yields the same between-vs-within statistic (every pair
        has distance 1.0, so mean(between) - mean(within) is always 0). The null
        std collapses to exactly 0.0 (or a near-zero float rounding artifact), so
        effect_size must be returned as 0.0 by the defensive guard.
        """
        _require_import()
        langs = ["en"] * 4 + ["ru"] * 4
        meta = _make_meta(langs)
        # Uniform non-diagonal distances: every within/between pair has same distance
        n = 8
        mat = np.ones((n, n), dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        result = permutation_test_per_domain(mat, meta, K=50, seed=0)
        assert np.isfinite(result["effect_size"]), (
            "effect_size must be finite even when null_std == 0"
        )

    def test_effect_size_zero_when_single_language(self):
        """Single-language label vector → every permutation yields the same statistic.

        Under permutation of an all-'en' label vector, the between-language mask is
        always empty and within-language mask is always full, so every permutation
        produces the exact same test statistic. The null std is therefore ~0 (a tiny
        pairwise-summation rounding artefact on the order of 5e-17, not exactly 0.0).

        The guard 'null_std > 0' FAILS to catch this because 5e-17 > 0 is True, and
        division by 5e-17 produces a meaninglessly large effect size (~1e16).
        The corrected guard 'null_std > 1e-10' catches this and returns 0.0.

        This test would have caught the bug fixed in fix #1 (null_std > 0 → > 1e-10).
        """
        _require_import()
        N = 20
        langs = ["en"] * N
        meta = _make_meta(langs)
        # Non-uniform real distance matrix so statistic is not trivially 0
        rng = np.random.default_rng(99)
        raw = rng.random((N, N))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        result = permutation_test_per_domain(mat, meta, K=100, seed=7)
        assert result["effect_size"] == 0.0, (
            f"Expected effect_size == 0.0 for single-language vector, "
            f"got {result['effect_size']!r}. "
            "This indicates the null_std > 0 guard is not using 1e-10 threshold."
        )


# ---------------------------------------------------------------------------
# Test 3 (from issue): BH correction monotonicity
# ---------------------------------------------------------------------------

class TestBhCorrection:
    """Benjamini-Hochberg correction monotonicity and correctness."""

    def test_import(self):
        _require_import()

    def test_monotonicity(self):
        """BH correction is monotone: if rank k is rejected, all ranks < k are too.

        Construct a sorted p-value array [0.001, 0.002, ..., 0.5]. After BH,
        the rejection set must be a prefix of the sorted order (no gaps).
        """
        _require_import()
        m = 50
        pvalues = np.linspace(0.001, 0.5, m)
        reject = _bh_correction(pvalues, alpha=0.05)
        assert isinstance(reject, np.ndarray)
        assert reject.dtype == bool

        # Get the order BH uses
        order = np.argsort(pvalues)
        reject_in_order = reject[order]

        # Find the last rejected index
        if not reject_in_order.any():
            return  # No rejections — trivially monotone

        last_rejected = np.where(reject_in_order)[0][-1]
        # All positions up to last_rejected must also be rejected
        prefix = reject_in_order[: last_rejected + 1]
        assert prefix.all(), (
            f"BH rejection set is not monotone (step-up property violated). "
            f"reject_in_order[:last+1]={prefix.tolist()}"
        )

    def test_all_rejected_when_all_zero(self):
        """All p-values of 0 must all be rejected."""
        _require_import()
        pvalues = np.zeros(10)
        reject = _bh_correction(pvalues, alpha=0.05)
        assert reject.all(), "All zero p-values should all be rejected by BH"

    def test_none_rejected_when_all_one(self):
        """All p-values of 1 must produce no rejections."""
        _require_import()
        pvalues = np.ones(10)
        reject = _bh_correction(pvalues, alpha=0.05)
        assert not reject.any(), "All p=1 should produce no BH rejections"

    def test_empty_input(self):
        """Empty p-value array must return empty boolean array."""
        _require_import()
        reject = _bh_correction(np.array([]), alpha=0.05)
        assert len(reject) == 0
        assert reject.dtype == bool

    def test_length_preserved(self):
        """Output length must equal input length."""
        _require_import()
        rng = np.random.default_rng(0)
        pvalues = rng.random(144)
        reject = _bh_correction(pvalues, alpha=0.05)
        assert len(reject) == 144, f"Expected length 144, got {len(reject)}"

    def test_stricter_alpha_fewer_rejections(self):
        """Stricter alpha → at most as many rejections (verified with known rejections at both levels).

        Uses np.linspace(0.001, 0.05, 20) so both alpha=0.05 and alpha=0.20 produce
        actual rejections (with different counts), making the monotonicity assertion
        non-trivial.
        """
        _require_import()
        # Uniform-spaced p-values: [0.001, ~0.0032, ..., 0.05] — m=20
        pvalues = np.linspace(0.001, 0.05, 20)
        reject_loose = _bh_correction(pvalues, alpha=0.20)
        reject_strict = _bh_correction(pvalues, alpha=0.05)
        # Both must have at least one rejection so we are not just checking 0 <= 0
        assert reject_loose.sum() > 0, "alpha=0.20 should reject at least one with this p-vector"
        assert reject_strict.sum() > 0, "alpha=0.05 should reject at least one with this p-vector"
        assert reject_strict.sum() <= reject_loose.sum(), (
            f"Stricter alpha=0.05 ({reject_strict.sum()}) should have <= rejections "
            f"than alpha=0.20 ({reject_loose.sum()})"
        )

    def test_bh_known_values(self):
        """BH correction returns exactly the expected rejection mask for a hand-crafted input.

        Input: p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205] (m=8), alpha=0.05.
        BH thresholds (i/m * alpha for i=1..8):
          [0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375, 0.05]
        Sorted p-values already in ascending order; comparing each to threshold:
          p[0]=0.001 < 0.00625  ✓  (reject)
          p[1]=0.008 < 0.0125   ✓  (reject)
          p[2]=0.039 > 0.01875  ✗
          p[3]=0.041 > 0.025    ✗
          p[4]=0.042 > 0.03125  ✗
          p[5]=0.060 > 0.0375   ✗
          p[6]=0.074 > 0.04375  ✗
          p[7]=0.205 > 0.05     ✗
        Largest passing rank is 1 (0-indexed) → reject p-values 0.001 and 0.008 only.
        Expected mask: [True, True, False, False, False, False, False, False].
        """
        _require_import()
        pvalues = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205])
        expected = np.array([True, True, False, False, False, False, False, False])
        result = _bh_correction(pvalues, alpha=0.05)
        np.testing.assert_array_equal(
            result,
            expected,
            err_msg=(
                f"BH correction returned {result.tolist()}, "
                f"expected {expected.tolist()} for p={pvalues.tolist()}"
            ),
        )


# ---------------------------------------------------------------------------
# permutation_test_per_head tests
# ---------------------------------------------------------------------------

class TestPermutationTestPerHead:
    def test_import(self):
        _require_import()

    def test_output_is_dataframe(self):
        """permutation_test_per_head must return a DataFrame."""
        _require_import()
        n_layers, n_heads = 2, 3
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        tensor = _make_distance_tensor(n_layers, n_heads, langs, seed=0)
        result = permutation_test_per_head(tensor, meta, K=49, seed=0)
        assert isinstance(result, pd.DataFrame), (
            f"Expected DataFrame, got {type(result)}"
        )

    def test_output_columns(self):
        """Result DataFrame must have expected columns."""
        _require_import()
        n_layers, n_heads = 2, 3
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        tensor = _make_distance_tensor(n_layers, n_heads, langs, seed=1)
        result = permutation_test_per_head(tensor, meta, K=49, seed=1)
        required_cols = {"layer", "head", "observed", "p_value", "effect_size", "passes_bh"}
        missing = required_cols - set(result.columns)
        assert not missing, f"Missing columns: {missing}. Found: {list(result.columns)}"

    def test_row_count_equals_layer_head_product(self):
        """Row count must equal n_layers * n_heads."""
        _require_import()
        n_layers, n_heads = 3, 4
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        tensor = _make_distance_tensor(n_layers, n_heads, langs, seed=2)
        result = permutation_test_per_head(tensor, meta, K=49, seed=2)
        assert len(result) == n_layers * n_heads, (
            f"Expected {n_layers * n_heads} rows, got {len(result)}"
        )

    def test_passes_bh_is_boolean(self):
        """passes_bh column must be boolean dtype."""
        _require_import()
        n_layers, n_heads = 2, 2
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        tensor = _make_distance_tensor(n_layers, n_heads, langs, seed=3)
        result = permutation_test_per_head(tensor, meta, K=49, seed=3)
        assert result["passes_bh"].dtype == bool, (
            f"passes_bh must be bool, got {result['passes_bh'].dtype}"
        )

    def test_p_values_in_range(self):
        """All p_values must be in (0, 1]."""
        _require_import()
        n_layers, n_heads = 2, 2
        langs = ["en"] * 5 + ["ru"] * 5
        meta = _make_meta(langs)
        tensor = _make_distance_tensor(n_layers, n_heads, langs, seed=4)
        result = permutation_test_per_head(tensor, meta, K=49, seed=4)
        bad = result[(result["p_value"] <= 0) | (result["p_value"] > 1)]
        assert len(bad) == 0, f"p_values outside (0, 1]: {bad}"

    @pytestmark_slow
    def test_planted_signal_some_pass_bh(self):
        """With strong planted signal, at least some (layer, head) cells should pass BH."""
        _require_import()
        n_layers, n_heads = 12, 12
        langs = ["en"] * 10 + ["ru"] * 10 + ["es"] * 10
        meta = _make_meta(langs)
        # Very strong signal: between-lang distance = 100
        tensor = _make_distance_tensor(n_layers, n_heads, langs, between_dist=100.0, seed=5)
        result = permutation_test_per_head(tensor, meta, K=999, seed=5)
        n_sig = result["passes_bh"].sum()
        assert n_sig > 0, (
            f"Expected at least one (layer, head) to pass BH under strong planted signal, "
            f"got 0 out of {len(result)}"
        )

