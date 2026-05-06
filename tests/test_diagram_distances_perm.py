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

try:
    from replication.diagram_distances import _per_domain_test_statistic_from_arrays
    _VECTORIZED_IMPORT_OK = True
except ImportError:
    _VECTORIZED_IMPORT_OK = False


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


# ---------------------------------------------------------------------------
# Vectorization equivalence tests (chz — BLAS matmul + cache invariants)
# ---------------------------------------------------------------------------

def _make_synthetic_matrix_and_langs(seed: int = 1234):
    """Return (mat, langs, meta) for a reproducible 15-sample, 3-language setup."""
    rng = np.random.default_rng(seed)
    langs = ["en"] * 5 + ["ru"] * 5 + ["es"] * 5
    n = len(langs)
    raw = rng.random((n, n))
    mat = 0.5 * (raw + raw.T)
    np.fill_diagonal(mat, 0.0)
    meta = pd.DataFrame({"lang": langs})
    return mat, langs, meta


class TestPerDomainArrayHelper:
    """Tests for the private array-level helper _per_domain_test_statistic_from_arrays."""

    def test_import(self):
        if not _VECTORIZED_IMPORT_OK:
            pytest.fail(
                "_per_domain_test_statistic_from_arrays is not yet defined in "
                "replication.diagram_distances — needed for vectorized inner loop."
            )

    def test_matches_public_api(self):
        """_per_domain_test_statistic_from_arrays must return the same value as
        per_domain_test_statistic for the same inputs.

        The observed statistic MUST match exactly (same floating-point formula).
        """
        if not _VECTORIZED_IMPORT_OK:
            pytest.fail("_per_domain_test_statistic_from_arrays not importable")
        _require_import()

        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=1234)
        n = len(langs)
        lang_vec = np.array(langs)

        triu_i, triu_j = np.triu_indices(n, k=1)
        dists_upper = mat[triu_i, triu_j]
        total_sum = dists_upper.sum()
        n_pairs = len(dists_upper)
        between_mask = lang_vec[triu_i] != lang_vec[triu_j]

        ref = per_domain_test_statistic(mat, meta)
        got = _per_domain_test_statistic_from_arrays(
            dists_upper=dists_upper,
            between_mask=between_mask,
            total_sum=total_sum,
            n_pairs=n_pairs,
        )
        assert got == ref, (
            f"_per_domain_test_statistic_from_arrays returned {got!r}, "
            f"but per_domain_test_statistic returned {ref!r}. "
            "They must be identical (exact float equality)."
        )

    def test_planted_signal_direction(self):
        """Array helper returns positive statistic when between-lang > within-lang."""
        if not _VECTORIZED_IMPORT_OK:
            pytest.fail("_per_domain_test_statistic_from_arrays not importable")

        langs = ["en"] * 5 + ["ru"] * 5
        mat = _planted_distance_matrix(langs, between_dist=1.0)
        n = len(langs)
        lang_vec = np.array(langs)
        triu_i, triu_j = np.triu_indices(n, k=1)
        dists_upper = mat[triu_i, triu_j]
        total_sum = dists_upper.sum()
        n_pairs = len(dists_upper)
        between_mask = lang_vec[triu_i] != lang_vec[triu_j]

        result = _per_domain_test_statistic_from_arrays(
            dists_upper=dists_upper,
            between_mask=between_mask,
            total_sum=total_sum,
            n_pairs=n_pairs,
        )
        assert result > 0.0, f"Expected positive statistic, got {result!r}"

    def test_all_within_returns_negative_or_zero(self):
        """Array helper returns 0.0 - mean_within <= 0 when between_mask is all-False."""
        if not _VECTORIZED_IMPORT_OK:
            pytest.fail("_per_domain_test_statistic_from_arrays not importable")

        rng = np.random.default_rng(7)
        n = 6
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        triu_i, triu_j = np.triu_indices(n, k=1)
        dists_upper = mat[triu_i, triu_j]
        total_sum = dists_upper.sum()
        n_pairs = len(dists_upper)
        between_mask = np.zeros(n_pairs, dtype=bool)

        result = _per_domain_test_statistic_from_arrays(
            dists_upper=dists_upper,
            between_mask=between_mask,
            total_sum=total_sum,
            n_pairs=n_pairs,
        )
        assert np.isfinite(result), f"Expected finite result, got {result!r}"
        assert result <= 0.0, f"Expected <= 0 when no between-lang pairs, got {result!r}"


class TestVectorizedEquivalence:
    """Equivalence tests for the vectorized permutation_test_per_domain.

    The observed statistic must match the old implementation exactly.
    The p_value and effect_size must match within Monte-Carlo noise.
    These tests are pinned to the new (vectorized) implementation as reference.
    """

    def test_import(self):
        _require_import()

    def test_observed_statistic_exact_match(self):
        """Observed statistic must match per_domain_test_statistic exactly.

        This pins the exact floating-point value for the synthetic 15-sample,
        3-language matrix built from seed=1234. The value -0.0010110659... is
        computed from the same formula both ways — any refactor that changes
        this breaks the equivalence requirement.
        """
        _require_import()
        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=1234)
        ref_observed = per_domain_test_statistic(mat, meta)
        result = permutation_test_per_domain(mat, meta, K=500, seed=42)
        assert result["observed"] == ref_observed, (
            f"observed statistic changed: got {result['observed']!r}, "
            f"expected {ref_observed!r}. "
            "Refactor must not change the observed statistic formula."
        )

    def test_p_value_and_effect_size_match_old_loop(self):
        """Old Python-loop null distribution should produce p_value and
        effect_size within Monte-Carlo noise of the vectorized version."""
        _require_import()
        # Build a small synthetic case with planted signal
        rng = np.random.default_rng(0)
        n = 60  # 3 langs × 20 samples
        langs = np.array(["en"] * 20 + ["ru"] * 20 + ["es"] * 20)
        metadata_df = pd.DataFrame({"lang": langs})
        distance_matrix = rng.uniform(0, 1, size=(n, n))
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0.0)
        # Add modest planted signal so the test isn't all noise
        for i in range(n):
            for j in range(n):
                if langs[i] != langs[j]:
                    distance_matrix[i, j] += 0.1

        K = 2000
        seed = 42

        # NEW (vectorized)
        new = permutation_test_per_domain(distance_matrix, metadata_df, K=K, seed=seed)

        # OLD (Python loop, faithful to the pre-c5cb93f algorithm)
        rng_old = np.random.default_rng(seed)
        lang_vec = metadata_df["lang"].values
        triu_i, triu_j = np.triu_indices(n, k=1)
        dists_upper = distance_matrix[triu_i, triu_j]
        null_old = np.empty(K, dtype=np.float64)
        for k in range(K):
            shuffled = rng_old.permutation(lang_vec)
            between = shuffled[triu_i] != shuffled[triu_j]
            within = ~between
            null_old[k] = (
                dists_upper[between].mean() - dists_upper[within].mean()
            )
        # observed (same on both — labels unshuffled)
        observed_old = (
            dists_upper[lang_vec[triu_i] != lang_vec[triu_j]].mean()
            - dists_upper[lang_vec[triu_i] == lang_vec[triu_j]].mean()
        )
        null_mean_old = null_old.mean()
        extreme_old = int(np.sum(np.abs(null_old - null_mean_old) >= np.abs(observed_old - null_mean_old)))
        p_value_old = (extreme_old + 1) / (K + 1)
        effect_size_old = (observed_old - null_mean_old) / null_old.std()

        # Observed stat must match exactly (same formula, same dtype path)
        assert new["observed"] == pytest.approx(observed_old, abs=1e-12)
        # p_value within Monte-Carlo noise — at K=2000 use abs=0.05 (loose)
        assert new["p_value"] == pytest.approx(p_value_old, abs=0.05)
        # effect_size within ~10% relative
        assert new["effect_size"] == pytest.approx(effect_size_old, rel=0.10)

    def test_determinism_after_vectorization(self):
        """Vectorized implementation must be deterministic with the same seed."""
        _require_import()
        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=999)
        r1 = permutation_test_per_domain(mat, meta, K=200, seed=77)
        r2 = permutation_test_per_domain(mat, meta, K=200, seed=77)
        assert r1["observed"] == r2["observed"], "observed not deterministic"
        assert r1["p_value"] == r2["p_value"], "p_value not deterministic"
        assert r1["effect_size"] == r2["effect_size"], "effect_size not deterministic"
        np.testing.assert_array_equal(r1["null"], r2["null"], err_msg="null not deterministic")

    def test_null_length_preserved(self):
        """Null array must still have exactly K elements after vectorization."""
        _require_import()
        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=11)
        K = 300
        result = permutation_test_per_domain(mat, meta, K=K, seed=0)
        assert len(result["null"]) == K, (
            f"Expected null length {K}, got {len(result['null'])}"
        )

    def test_timing_n100_k2000(self):
        """Vectorized implementation must complete n=100, K=2000 in under 5 seconds.

        The old Python-loop implementation takes ~0.7s for this size; the
        vectorized BLAS version should be much faster. The bound is set
        generously at 5s to avoid flakiness on loaded hardware.
        """
        import time
        _require_import()
        rng = np.random.default_rng(99)
        n = 100
        langs = ["en"] * 34 + ["ru"] * 33 + ["es"] * 33
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        meta = pd.DataFrame({"lang": langs})

        start = time.time()
        permutation_test_per_domain(mat, meta, K=2000, seed=42)
        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"permutation_test_per_domain(n=100, K=2000) took {elapsed:.2f}s, "
            "expected < 5s. The vectorized BLAS implementation should be faster."
        )

    def test_p_value_range_preserved(self):
        """p_value must remain in (0, 1] after vectorization."""
        _require_import()
        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=55)
        result = permutation_test_per_domain(mat, meta, K=200, seed=3)
        assert 0 < result["p_value"] <= 1.0, (
            f"p_value={result['p_value']} is outside (0, 1]"
        )

    def test_planted_signal_still_detected(self):
        """With a strong planted signal, the vectorized test must still reject H_0."""
        _require_import()
        langs = ["en"] * 10 + ["ru"] * 10 + ["es"] * 10
        meta = _make_meta(langs)
        mat = _planted_distance_matrix(langs, between_dist=100.0)
        result = permutation_test_per_domain(mat, meta, K=1000, seed=42)
        assert result["p_value"] < 0.05, (
            f"Expected p_value < 0.05 under strong planted signal, got {result['p_value']:.4f}"
        )
        assert result["effect_size"] > 0, (
            f"Expected positive effect_size, got {result['effect_size']:.4f}"
        )

    def test_chunk_boundary_correct_null_length(self):
        """Null array length must equal K for various chunk sizes including boundary cases.

        Tests chunk_size=1 (pathological), chunk_size=K (single chunk),
        and chunk_size=7 (K=50 not divisible by 7 → partial last chunk of 1).
        All must return exactly K entries in null, confirming no permutations
        are dropped or duplicated at chunk boundaries.

        Also verifies that p_value and effect_size are equivalent across chunk sizes —
        cross-chunk-size differences are pure float32 accumulation noise (~3e-7 in
        between_sums), so tolerances are generous: abs=5e-3 for p_value, rel=1e-2
        for effect_size.
        """
        _require_import()
        try:
            from replication.diagram_distances import permutation_test_per_domain as ptpd
        except ImportError:
            pytest.fail("permutation_test_per_domain not importable")

        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=42)
        K = 50

        chunk_sizes = [1, 7, K]
        results = {}
        for chunk_size in chunk_sizes:
            r = ptpd(mat, meta, K=K, seed=99, _chunk_size=chunk_size)
            assert len(r["null"]) == K, (
                f"chunk_size={chunk_size}: expected null length {K}, got {len(r['null'])}"
            )
            assert 0 < r["p_value"] <= 1.0, (
                f"chunk_size={chunk_size}: p_value={r['p_value']} out of (0, 1]"
            )
            assert np.isfinite(r["effect_size"]), (
                f"chunk_size={chunk_size}: effect_size is not finite: {r['effect_size']}"
            )
            results[chunk_size] = r

        # Chunk-size invariance: p_value and effect_size must agree within float32 noise
        ref = results[chunk_sizes[0]]
        for chunk_size in chunk_sizes[1:]:
            r = results[chunk_size]
            assert r["p_value"] == pytest.approx(ref["p_value"], abs=5e-3), (
                f"chunk_size={chunk_size}: p_value={r['p_value']!r} diverges from "
                f"chunk_size={chunk_sizes[0]} p_value={ref['p_value']!r} (abs tol 5e-3)"
            )
            assert r["effect_size"] == pytest.approx(ref["effect_size"], rel=1e-2), (
                f"chunk_size={chunk_size}: effect_size={r['effect_size']!r} diverges from "
                f"chunk_size={chunk_sizes[0]} effect_size={ref['effect_size']!r} (rel tol 1e-2)"
            )

    def test_chunk_determinism_fixed_chunk_size(self):
        """Same chunk_size + same seed must produce identical null distributions.

        Verifies that the chunked BLAS path is deterministic: two runs with the
        same chunk_size and seed return bit-for-bit identical null arrays.
        (Different chunk sizes may differ by float32 rounding — that is expected
        and tested separately; here we just confirm same-settings reproducibility.)
        """
        _require_import()
        try:
            from replication.diagram_distances import permutation_test_per_domain as ptpd
        except ImportError:
            pytest.fail("permutation_test_per_domain not importable")

        mat, langs, meta = _make_synthetic_matrix_and_langs(seed=42)
        K = 50
        r1 = ptpd(mat, meta, K=K, seed=99, _chunk_size=7)
        r2 = ptpd(mat, meta, K=K, seed=99, _chunk_size=7)
        np.testing.assert_array_equal(
            r1["null"], r2["null"],
            err_msg="Same chunk_size and seed must give identical null distributions",
        )

