"""Tests for baselines/topology.py — rips_barcode and barcode_features.

Five mandatory test cases per ssa.5 issue spec:
  1. Sanity: two well-separated clusters produce ≥2 H0 bars (one inter-cluster).
  2. Input validation: non-square / non-symmetric / NaN / negative → ValueError.
  3. Empty barcode: barcode_features on all-empty returns 0.0 for every feature.
  4. Regression: fixed synthetic distance matrix → golden feature values.
  5. Degenerate: all-zero distance matrix behaviour matches spec (explicit firewall
     against the prior tda-project silent-zero-output bug, ph-project-mwk.3).

Also includes a smoke test that the public API is importable and callable.
"""

import numpy as np
import pytest

from baselines.topology import barcode_features, rips_barcode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STRUCTURED_DTYPE = np.dtype([("birth", "f8"), ("death", "f8")])


def _empty_barcode() -> dict:
    """Return a barcode with empty structured arrays for H0 and H1."""
    empty = np.empty(0, dtype=_STRUCTURED_DTYPE)
    return {0: empty, 1: empty}


def _to_structured(plain: np.ndarray) -> np.ndarray:
    """Convert a (n,2) plain float array to a structured array with birth/death fields."""
    sa = np.empty(len(plain), dtype=_STRUCTURED_DTYPE)
    sa["birth"] = plain[:, 0]
    sa["death"] = plain[:, 1]
    return sa


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def test_topology_module_importable():
    """Smoke: module is valid and the two public symbols are callable."""
    from baselines.topology import barcode_features, rips_barcode

    assert callable(rips_barcode)
    assert callable(barcode_features)


# ---------------------------------------------------------------------------
# Test 1: Sanity — two well-separated clusters
# ---------------------------------------------------------------------------


def test_rips_barcode_two_cluster_sanity():
    """Two tight clusters separated in cosine space produce ≥1 inter-cluster H0 bar.

    Cluster A: 5 points near direction [10, 0] (pointing right).
    Cluster B: 5 points near direction [0, 10] (pointing up).
    Cosine distance between clusters is ~1.0 (orthogonal directions).
    Intra-cluster cosine distances are tiny (< 0.01).

    After stripping the single infinite H0 bar, at least one finite bar
    should have death > 0.5 (the inter-cluster gap).
    """
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.default_rng(0)
    # Cluster A: tight cloud near direction [10, 0]
    A = rng.normal([10.0, 0.0], 0.1, size=(5, 2))
    # Cluster B: tight cloud near direction [0, 10]
    B = rng.normal([0.0, 10.0], 0.1, size=(5, 2))
    pts = np.vstack([A, B])

    # Use scipy cosine distance (same as the real pipeline's cosine_distance_matrix)
    D = squareform(pdist(pts, metric="cosine"))

    barcode = rips_barcode(D, max_dim=1)

    # Must have both dims present
    assert 0 in barcode and 1 in barcode
    h0 = barcode[0]

    # H0 must have the right dtype
    assert h0.dtype == _STRUCTURED_DTYPE, f"Wrong dtype: {h0.dtype}"

    # After stripping inf, must still have ≥1 finite bar (the inter-cluster one)
    assert len(h0) >= 1, "Expected at least 1 finite H0 bar for two-cluster data"

    # All deaths must be finite — rips_barcode must strip the infinite H0 bar
    assert np.all(np.isfinite(h0["death"])), "rips_barcode must strip the infinite H0 bar"

    # At least one bar should have death > 0.5 (the wide inter-cluster cosine gap)
    inter_cluster_bars = h0[h0["death"] > 0.5]
    assert len(inter_cluster_bars) >= 1, (
        "Expected at least 1 finite H0 bar with death > 0.5 for two-cluster data; "
        f"got deaths={h0['death']}"
    )


# ---------------------------------------------------------------------------
# Test 2: Input validation — each invalid case raises ValueError
# ---------------------------------------------------------------------------


class TestRipsBarcodeInputValidation:
    """rips_barcode must raise ValueError BEFORE calling ripser for bad inputs."""

    def test_non_square_raises(self):
        """Non-square matrix raises ValueError."""
        D = np.zeros((3, 4))
        with pytest.raises(ValueError, match="square"):
            rips_barcode(D)

    def test_non_symmetric_raises(self):
        """Non-symmetric matrix raises ValueError."""
        D = np.array([[0.0, 0.3, 0.5], [0.3, 0.0, 0.7], [0.6, 0.7, 0.0]])
        with pytest.raises(ValueError, match="[Ss]ymmetri"):
            rips_barcode(D)

    def test_nan_raises(self):
        """Matrix containing NaN raises ValueError."""
        D = np.array([[0.0, np.nan], [np.nan, 0.0]])
        with pytest.raises(ValueError, match="[Ff]inite|[Nn]aN|[Nn]an"):
            rips_barcode(D)

    def test_inf_raises(self):
        """Matrix containing inf raises ValueError."""
        D = np.array([[0.0, np.inf], [np.inf, 0.0]])
        with pytest.raises(ValueError, match="[Ff]inite|[Ii]nf"):
            rips_barcode(D)

    def test_negative_values_raises(self):
        """Matrix with negative off-diagonal values raises ValueError."""
        D = np.array([[0.0, -0.1, 0.5], [-0.1, 0.0, 0.3], [0.5, 0.3, 0.0]])
        with pytest.raises(ValueError, match="[Nn]on-negative|[Nn]egative"):
            rips_barcode(D)

    def test_non_2d_raises(self):
        """1-D input raises ValueError mentioning ndim or 2-D."""
        D = np.array([0.0, 0.5, 0.3])
        with pytest.raises(ValueError, match=r"2-D|ndim"):
            rips_barcode(D)


# ---------------------------------------------------------------------------
# Test 3: Empty barcode — barcode_features returns all-zero dict
# ---------------------------------------------------------------------------


def test_barcode_features_empty_returns_zeros():
    """barcode_features on all-empty barcode returns 0.0 for every feature.

    No NaN is allowed in any feature value.
    """
    bc = _empty_barcode()
    features = barcode_features(bc)

    assert isinstance(features, dict), "barcode_features must return a dict"
    assert len(features) > 0, "Feature dict must be non-empty"

    for name, val in features.items():
        assert val == 0.0, (
            f"Feature {name!r} expected 0.0 for empty barcode, got {val!r}"
        )
        # Also guard against NaN (0.0 == 0.0 would pass above, but NaN != 0.0)
        assert not (isinstance(val, float) and np.isnan(val)), (
            f"Feature {name!r} is NaN for empty barcode"
        )


# ---------------------------------------------------------------------------
# Test 4: Regression — golden values on a fixed synthetic distance matrix
# ---------------------------------------------------------------------------

# Fixed 5×5 distance matrix.  All pairwise distances are explicit so that
# any algorithmic regression in the feature extraction would be caught.
_GOLDEN_D = np.array(
    [
        [0.000, 0.200, 0.400, 0.600, 0.800],
        [0.200, 0.000, 0.200, 0.400, 0.600],
        [0.400, 0.200, 0.000, 0.200, 0.400],
        [0.600, 0.400, 0.200, 0.000, 0.200],
        [0.800, 0.600, 0.400, 0.200, 0.000],
    ],
    dtype=float,
)

# Golden values from the initial implementation (computed once, then hardcoded).
# Standard test-first-with-golden-update pattern: run the implementation once,
# record the output at 6 decimal precision, then assert against it on all future runs.
# Any algorithmic regression that changes feature values will fail this test.
#
# The _GOLDEN_D matrix is a 5×5 "ladder": D[i,j] = 0.2 * |i-j|.
# ripser produces 4 finite H0 bars all with birth=0.0 and death=0.2
# (one infinite H0 bar is stripped).
# h0_n_d_m_t0.25 = 0 because deaths are 0.2 < 0.25 (strictly less than threshold).
_GOLDEN_FEATURES: dict[str, float] = {
    # --- H0 features ---
    # 4 finite H0 bars: births=0.0, deaths=0.2 (one infinite stripped)
    "h0_s": 0.800000,   # 4 * 0.2
    "h0_m": 0.200000,   # mean length
    "h0_v": 0.000000,   # std of identical lengths
    "h0_e": 1.386294,   # -4*(0.25*log(0.25)) = log(4)
    "h0_n_d_m_t0.25": 0.0,  # deaths = 0.2 < 0.25; none qualify
    "h0_n_d_m_t0.5": 0.0,
    "h0_n_d_m_t0.75": 0.0,
    "h0_n_b_l_t0.25": 4.0,  # all births = 0.0 ≤ 0.25
    "h0_n_b_l_t0.5": 4.0,
    "h0_n_b_l_t0.75": 4.0,
    "h0_t_b": 0.000000,
    "h0_t_d": 0.200000,
    # --- H1 features ---
    # No loops in a path-like distance matrix
    "h1_s": 0.0,
    "h1_m": 0.0,
    "h1_v": 0.0,
    "h1_e": 0.0,
    "h1_n_d_m_t0.25": 0.0,
    "h1_n_d_m_t0.5": 0.0,
    "h1_n_d_m_t0.75": 0.0,
    "h1_n_b_l_t0.25": 0.0,
    "h1_n_b_l_t0.5": 0.0,
    "h1_n_b_l_t0.75": 0.0,
    "h1_t_b": 0.0,
    "h1_t_d": 0.0,
}


def test_barcode_features_regression_golden():
    """Feature values on a fixed distance matrix must match golden values.

    Golden values were computed from the initial ssa.5 implementation and
    hardcoded here.  Any future algorithmic regression will fail this test.
    """
    barcode = rips_barcode(_GOLDEN_D, max_dim=1)
    features = barcode_features(barcode)

    assert set(features.keys()) == set(_GOLDEN_FEATURES.keys()), (
        f"Feature key mismatch.\n"
        f"Extra: {set(features) - set(_GOLDEN_FEATURES)}\n"
        f"Missing: {set(_GOLDEN_FEATURES) - set(features)}"
    )

    for name, expected in _GOLDEN_FEATURES.items():
        actual = features[name]
        assert abs(actual - expected) < 1e-4, (
            f"Golden regression failure for {name!r}: "
            f"expected {expected:.6f}, got {actual:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: Degenerate — all-zero distance matrix (explicit bug firewall)
# ---------------------------------------------------------------------------


def test_rips_barcode_all_zero_degenerate():
    """All-zero distance matrix (all points coincident) produces correct empty barcodes.

    FIREWALL against ph-project-mwk.3: the prior tda-project had a silent
    ripser zero-output bug where ripserplusplus returned empty barcodes
    without error for degenerate inputs.  This test asserts that:
      1. rips_barcode does NOT raise or silently swallow the result.
      2. H0 is empty after stripping the single infinite bar (all 5 points
         merge at distance 0, producing exactly one infinite H0 bar that
         gets stripped).
      3. H1 is empty (no loops in a point cloud of coincident points).
      4. Both dims are present in the returned dict with the correct dtype.
      5. Both structured arrays have shape (0,) — not None or missing.
    """
    D = np.zeros((5, 5), dtype=float)

    # Must NOT raise — all-zero is a valid (if degenerate) distance matrix.
    barcode = rips_barcode(D, max_dim=1)

    # Both dimensions must be present (not None, not missing key)
    assert 0 in barcode, "H0 key must be present in barcode"
    assert 1 in barcode, "H1 key must be present in barcode"

    h0 = barcode[0]
    h1 = barcode[1]

    # Must be structured arrays with the correct dtype
    assert h0.dtype == _STRUCTURED_DTYPE, (
        f"H0 must have structured dtype [('birth','f8'),('death','f8')], got {h0.dtype}"
    )
    assert h1.dtype == _STRUCTURED_DTYPE, (
        f"H1 must have structured dtype [('birth','f8'),('death','f8')], got {h1.dtype}"
    )

    # After stripping the single infinite H0 bar: H0 should be empty
    # (all 5 points merge at distance 0 → one birth=0 bar that becomes infinite)
    assert h0.shape == (0,), (
        f"H0 should be empty after stripping inf bar for all-zero D; "
        f"got shape {h0.shape}, births={h0['birth'] if len(h0) else 'N/A'}"
    )

    # H1 must also be empty
    assert h1.shape == (0,), (
        f"H1 should be empty for all-zero D; got shape {h1.shape}"
    )


# ---------------------------------------------------------------------------
# Test 6: v_is_std — barcode_features h{d}_v computes std, not variance
# ---------------------------------------------------------------------------


def test_barcode_features_v_is_std():
    """h0_v must equal std (not variance) of bar lengths.

    Uses a barcode with NON-uniform H0 bar lengths [0.1, 0.2, 0.5] so that
    std != var != 0.0.  If h0_v were returning variance this test would fail.
    """
    # Build a synthetic H0 barcode with 3 bars of known lengths: 0.1, 0.2, 0.5
    # births all zero, deaths = lengths
    lengths = np.array([0.1, 0.2, 0.5])
    h0_plain = np.column_stack([np.zeros(3), lengths])  # (3,2) birth/death
    h0_sa = _to_structured(h0_plain)
    barcode = {0: h0_sa, 1: np.empty(0, dtype=_STRUCTURED_DTYPE)}

    features = barcode_features(barcode)
    h0_v = features["h0_v"]

    expected_std = np.std(lengths)
    expected_var = np.var(lengths)

    # Must equal std
    assert h0_v == pytest.approx(expected_std, abs=1e-6), (
        f"h0_v expected std={expected_std:.8f}, got {h0_v:.8f}"
    )
    # Must NOT equal var (they differ for non-uniform lengths)
    assert h0_v != pytest.approx(expected_var, abs=1e-3), (
        f"h0_v == variance {expected_var:.8f}; regression: v silently returns variance"
    )


# ---------------------------------------------------------------------------
# Test 7: key_order_stable — empty and non-empty barcodes have identical key order
# ---------------------------------------------------------------------------

# Documented grouped H1 key order (n_d_m group before n_b_l group)
_EXPECTED_H1_KEY_ORDER = [
    "h1_n_d_m_t0.25",
    "h1_n_d_m_t0.5",
    "h1_n_d_m_t0.75",
    "h1_n_b_l_t0.25",
    "h1_n_b_l_t0.5",
    "h1_n_b_l_t0.75",
]


def test_barcode_features_key_order_stable():
    """Key order from barcode_features must be identical for empty and non-empty barcodes.

    Also asserts that H1 threshold keys are in the documented grouped form
    (all n_d_m thresholds first, then all n_b_l thresholds).
    """
    empty_bc = _empty_barcode()

    # Non-empty barcode: single H0 bar and single H1 bar
    h0_plain = np.array([[0.0, 0.3]])
    h1_plain = np.array([[0.1, 0.6]])
    nonempty_bc = {
        0: _to_structured(h0_plain),
        1: _to_structured(h1_plain),
    }

    empty_keys = list(barcode_features(empty_bc).keys())
    nonempty_keys = list(barcode_features(nonempty_bc).keys())

    assert empty_keys == nonempty_keys, (
        "barcode_features must produce identical key ordering for empty and non-empty barcodes.\n"
        f"Empty keys:    {empty_keys}\n"
        f"Non-empty keys:{nonempty_keys}"
    )

    # Verify that the H1 threshold sub-sequence matches the documented grouped order
    h1_threshold_keys = [k for k in nonempty_keys if k.startswith("h1_n_")]
    assert h1_threshold_keys == _EXPECTED_H1_KEY_ORDER, (
        f"H1 threshold keys must be in grouped order "
        f"(all n_d_m first, then n_b_l).\n"
        f"Expected: {_EXPECTED_H1_KEY_ORDER}\n"
        f"Got:      {h1_threshold_keys}"
    )
