"""Tests for replication/ripser_count_compat.py.

These tests verify that ripser_count_compat exposes the same API surface as
reference/ripser_count.py (the frozen original) and produces outputs in the
same structured-array format the rest of the notebook pipeline expects.

Run with:
    pytest tests/test_ripser_count_compat.py -v
"""
import sys
import os
import numpy as np
import pytest

# Make replication/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "replication"))
# Make reference/ importable so we can use cutoff_matrix without editing it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_distance_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Return a symmetric, zero-diagonal distance matrix of size n×n."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, n)).astype(np.float64)
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


def _make_attention_matrix(n: int, seed: int = 0) -> np.ndarray:
    """Return a random row-stochastic matrix (attention-like)."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, n)).astype(np.float32)
    m /= m.sum(axis=1, keepdims=True)
    return m


# ---------------------------------------------------------------------------
# Import guard: the module must be importable
# ---------------------------------------------------------------------------

def test_module_importable():
    """ripser_count_compat can be imported without error."""
    import ripser_count_compat  # noqa: F401


# ---------------------------------------------------------------------------
# barcode_pop_inf
# ---------------------------------------------------------------------------

def test_barcode_pop_inf_removes_inf_entries():
    from ripser_count_compat import barcode_pop_inf

    barcode = {
        0: np.array([(0.0, 0.5), (0.1, np.inf)], dtype=[("birth", "<f4"), ("death", "<f4")]),
        1: np.array([(0.2, 0.8)], dtype=[("birth", "<f4"), ("death", "<f4")]),
    }
    result = barcode_pop_inf(barcode)
    assert result[0]["death"].shape == (1,), "Infinite entry should be removed from dim 0"
    assert result[1]["death"].shape == (1,), "Non-infinite entry in dim 1 should be kept"


def test_barcode_pop_inf_empty_dim():
    from ripser_count_compat import barcode_pop_inf

    barcode = {0: np.array([], dtype=[("birth", "<f4"), ("death", "<f4")])}
    result = barcode_pop_inf(barcode)
    assert len(result[0]) == 0


# ---------------------------------------------------------------------------
# barcode_sum / barcode_mean / barcode_std
# ---------------------------------------------------------------------------

def test_barcode_sum_known_value():
    from ripser_count_compat import barcode_sum

    barcode = {0: np.array([(0.0, 1.0), (0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    assert barcode_sum(barcode, dim=0) == pytest.approx(1.5, abs=1e-5)


def test_barcode_sum_empty_returns_zero():
    from ripser_count_compat import barcode_sum

    barcode = {0: np.array([], dtype=[("birth", "<f4"), ("death", "<f4")])}
    assert barcode_sum(barcode, dim=0) == 0.0


def test_barcode_mean_known_value():
    from ripser_count_compat import barcode_mean

    barcode = {0: np.array([(0.0, 1.0), (0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    assert barcode_mean(barcode, dim=0) == pytest.approx(0.75, abs=1e-5)


def test_barcode_std_empty_returns_zero():
    from ripser_count_compat import barcode_std

    barcode = {1: np.array([], dtype=[("birth", "<f4"), ("death", "<f4")])}
    assert barcode_std(barcode, dim=1) == 0.0


# ---------------------------------------------------------------------------
# barcode_number
# ---------------------------------------------------------------------------

def test_barcode_number_more_than():
    from ripser_count_compat import barcode_number

    barcode = {
        0: np.array(
            [(0.0, 0.3), (0.0, 0.6), (0.0, 0.9)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        )
    }
    # deaths >= 0.5: two entries (0.6 and 0.9)
    assert barcode_number(barcode, dim=0, bd="death", ml="m", t=0.5) == 2


def test_barcode_number_less_than():
    from ripser_count_compat import barcode_number

    barcode = {
        0: np.array(
            [(0.0, 0.3), (0.0, 0.6), (0.0, 0.9)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        )
    }
    # deaths <= 0.5: one entry (0.3)
    assert barcode_number(barcode, dim=0, bd="death", ml="l", t=0.5) == 1


def test_barcode_number_wrong_ml_raises():
    from ripser_count_compat import barcode_number

    barcode = {0: np.array([(0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    with pytest.raises(Exception):
        barcode_number(barcode, dim=0, bd="death", ml="x", t=0.5)


# ---------------------------------------------------------------------------
# barcode_time
# ---------------------------------------------------------------------------

def test_barcode_time_longest_barcode_death():
    from ripser_count_compat import barcode_time

    barcode = {
        0: np.array(
            [(0.0, 0.2), (0.0, 0.9), (0.1, 0.5)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        )
    }
    # longest barcode: (0.0, 0.9) — length 0.9; death = 0.9
    assert barcode_time(barcode, dim=0, bd="death") == pytest.approx(0.9, abs=1e-5)


def test_barcode_time_empty_returns_zero():
    from ripser_count_compat import barcode_time

    barcode = {0: np.array([], dtype=[("birth", "<f4"), ("death", "<f4")])}
    assert barcode_time(barcode, dim=0) == 0.0


# ---------------------------------------------------------------------------
# barcode_entropy
# ---------------------------------------------------------------------------

def test_barcode_entropy_uniform():
    from ripser_count_compat import barcode_entropy

    # Two equal-length bars → normalized lengths [0.5, 0.5] → entropy = log(2)
    barcode = {
        0: np.array([(0.0, 1.0), (0.0, 1.0)], dtype=[("birth", "<f4"), ("death", "<f4")])
    }
    expected = -2 * (0.5 * np.log(0.5))
    assert barcode_entropy(barcode, dim=0) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# barcode_number_of_barcodes
# ---------------------------------------------------------------------------

def test_barcode_number_of_barcodes():
    from ripser_count_compat import barcode_number_of_barcodes

    barcode = {
        1: np.array(
            [(0.1, 0.8), (0.2, 0.6)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        )
    }
    assert barcode_number_of_barcodes(barcode, dim=1) == 2


# ---------------------------------------------------------------------------
# count_ripser_features
# ---------------------------------------------------------------------------

def test_count_ripser_features_shape():
    from ripser_count_compat import count_ripser_features

    barcodes = []
    for i in range(5):
        barcodes.append({
            0: np.array([(0.0, 0.5 + i * 0.1)], dtype=[("birth", "<f4"), ("death", "<f4")]),
            1: np.array([(0.1, 0.4)], dtype=[("birth", "<f4"), ("death", "<f4")]),
        })
    features = count_ripser_features(barcodes, ["h0_m", "h1_s"])
    assert features.shape == (5, 2), f"Expected (5, 2), got {features.shape}"


def test_count_ripser_features_known_values():
    from ripser_count_compat import count_ripser_features

    barcode = {
        0: np.array([(0.0, 1.0), (0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")]),
        1: np.array([], dtype=[("birth", "<f4"), ("death", "<f4")]),
    }
    # h0_m = mean of lengths = (1.0 + 0.5) / 2 = 0.75
    # h1_s = sum of lengths = 0 (empty)
    features = count_ripser_features([barcode], ["h0_m", "h1_s"])
    assert features.shape == (1, 2)
    assert features[0, 0] == pytest.approx(0.75, abs=1e-5)
    assert features[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_count_ripser_features_nb_feature():
    """'nb' feature type: number of barcodes in a given dimension."""
    from ripser_count_compat import count_ripser_features

    barcode = {
        0: np.array(
            [(0.0, 0.3), (0.0, 0.7), (0.0, 0.9)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        ),
        1: np.array([(0.1, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")]),
    }
    features = count_ripser_features([barcode], ["h0_nb", "h1_nb"])
    assert features.shape == (1, 2)
    assert features[0, 0] == pytest.approx(3.0, abs=1e-5), "h0 should have 3 barcodes"
    assert features[0, 1] == pytest.approx(1.0, abs=1e-5), "h1 should have 1 barcode"


def test_count_ripser_features_v_feature():
    """'v' feature type: variance (std) of bar lengths."""
    from ripser_count_compat import barcode_std, count_ripser_features

    barcode = {
        0: np.array(
            [(0.0, 1.0), (0.0, 0.5), (0.0, 0.75)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        ),
    }
    features = count_ripser_features([barcode], ["h0_v"])
    expected = barcode_std(barcode, dim=0)
    assert features.shape == (1, 1)
    assert features[0, 0] == pytest.approx(expected, rel=1e-4)


def test_count_ripser_features_unknown_type_raises():
    """Unknown feature type raises ValueError."""
    from ripser_count_compat import count_ripser_features

    barcode = {0: np.array([(0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    with pytest.raises((ValueError, KeyError, Exception)):
        count_ripser_features([barcode], ["h0_z"])  # 'z' is not a valid type


# ---------------------------------------------------------------------------
# run_ripser_on_matrix — returns structured-array barcode dict
# ---------------------------------------------------------------------------

def test_run_ripser_on_matrix_dim0_has_entries():
    from ripser_count_compat import run_ripser_on_matrix

    m = _make_distance_matrix(10, seed=2)
    barcode = run_ripser_on_matrix(m, dim=1)
    # For a connected distance matrix H_0 must be non-empty (all points born at 0)
    assert len(barcode[0]) > 0


# ---------------------------------------------------------------------------
# matrix_to_ripser — same logic as reference (pre-processing pipeline)
# ---------------------------------------------------------------------------

def test_matrix_to_ripser_output_shape():
    from ripser_count_compat import matrix_to_ripser

    n = 10
    att = _make_attention_matrix(n)
    ntokens = 8
    result = matrix_to_ripser(att, ntokens)
    assert result.shape == (ntokens, ntokens)


def test_matrix_to_ripser_zero_diagonal():
    from ripser_count_compat import matrix_to_ripser

    att = _make_attention_matrix(12)
    result = matrix_to_ripser(att, 10)
    np.testing.assert_array_equal(np.diag(result), np.zeros(10))


def test_matrix_to_ripser_symmetric():
    from ripser_count_compat import matrix_to_ripser

    att = _make_attention_matrix(12, seed=7)
    result = matrix_to_ripser(att, 10)
    np.testing.assert_allclose(result, result.T)


# ---------------------------------------------------------------------------
# get_barcodes — integration: attention matrices → barcode list
# ---------------------------------------------------------------------------

def test_get_barcodes_returns_list_of_dicts():
    from ripser_count_compat import get_barcodes

    n_samples = 3
    n = 16
    matrices = np.stack([_make_attention_matrix(n, seed=i) for i in range(n_samples)])
    ntokens_array = np.array([8, 10, 12])
    barcodes = get_barcodes(matrices, ntokens_array, dim=1, lower_bound=1e-3)

    assert len(barcodes) == n_samples
    for bc in barcodes:
        assert isinstance(bc, dict)
        assert 0 in bc and 1 in bc


# ---------------------------------------------------------------------------
# calculate_features_r — full pipeline integration
# ---------------------------------------------------------------------------

def test_calculate_features_r_output_shape():
    from ripser_count_compat import calculate_features_r

    n_samples = 4
    n_layers = 2
    n_heads = 2
    n = 16
    # Shape: (samples, layers, heads, n, n)
    adj = np.stack([
        np.stack([
            np.stack([_make_attention_matrix(n, seed=s + l * 10 + h * 100)
                      for h in range(n_heads)])
            for l in range(n_layers)
        ])
        for s in range(n_samples)
    ])
    ntokens_array = np.array([8, 10, 12, 14])
    features = calculate_features_r(adj, dim=1, lower_bound=1e-3,
                                    ripser_features=["h0_m", "h1_s"],
                                    ntokens_array=ntokens_array)
    # Expected: (n_layers, n_heads, n_samples, n_ripser_features)
    assert features.shape == (n_layers, n_heads, n_samples, 2), \
        f"Expected ({n_layers}, {n_heads}, {n_samples}, 2), got {features.shape}"

    # Pin one numeric value to catch silent breakage at any pipeline stage.
    # Captured from current shim output; must stay stable across runs given
    # the fixed seed-based input matrix.
    np.testing.assert_allclose(
        features[0, 0, 0, 0],
        0.7626509,
        rtol=1e-5,
        err_msg="layer0/head0/sample0 h0_m changed — pipeline regression",
    )


# ---------------------------------------------------------------------------
# run_ripser_on_matrix — unit-square topology test (items A, B, C)
# ---------------------------------------------------------------------------

def _unit_square_distance_matrix() -> np.ndarray:
    """Euclidean distance matrix for the 4 corners of a 1×1 square."""
    pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    n = len(pts)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(pts[i] - pts[j])
    return D


def test_run_ripser_on_matrix_unit_square_topology():
    """Known-input/known-output test: unit-square gives the expected barcodes.

    H0: 4 bars (3 finite dying at 1.0, 1 essential with death=inf).
    H1: 1 bar with birth=1.0, death=sqrt(2).
    """
    from ripser_count_compat import run_ripser_on_matrix

    D = _unit_square_distance_matrix()
    barcode = run_ripser_on_matrix(D, dim=1)

    # --- H0 ---
    h0 = barcode[0]
    assert len(h0) == 4, f"H0 should have 4 bars, got {len(h0)}"

    finite_deaths = h0["death"][h0["death"] != np.inf]
    essential_deaths = h0["death"][h0["death"] == np.inf]

    assert len(finite_deaths) == 3, f"H0 should have 3 finite bars, got {len(finite_deaths)}"
    assert len(essential_deaths) == 1, "H0 should have 1 essential bar (death=inf)"
    np.testing.assert_allclose(
        finite_deaths,
        np.ones(3, dtype=np.float32),
        rtol=1e-5,
        err_msg="H0 finite bars should all die at distance 1.0",
    )

    # --- H1 ---
    h1 = barcode[1]
    assert len(h1) == 1, f"H1 should have 1 bar, got {len(h1)}"
    np.testing.assert_allclose(h1["birth"][0], 1.0, rtol=1e-5,
                               err_msg="H1 bar birth should be 1.0")
    np.testing.assert_allclose(h1["death"][0], np.sqrt(2), rtol=1e-5,
                               err_msg="H1 bar death should be sqrt(2)")


def test_run_ripser_on_matrix_essential_h0_bar():
    """Any connected point set produces at least one essential H0 bar (death=inf)."""
    from ripser_count_compat import run_ripser_on_matrix

    # Simple 3-point triangle
    D = np.array([[0.0, 0.5, 0.7],
                  [0.5, 0.0, 0.6],
                  [0.7, 0.6, 0.0]], dtype=np.float64)
    barcode = run_ripser_on_matrix(D, dim=1)

    h0_deaths = barcode[0]["death"]
    assert np.any(h0_deaths == np.inf), (
        "H0 should contain at least one essential bar (death=inf) "
        f"for a connected point set; got deaths={h0_deaths}"
    )


def test_run_ripser_on_matrix_returns_structured_barcodes():
    from ripser_count_compat import run_ripser_on_matrix

    m = _make_distance_matrix(8, seed=1)
    barcode = run_ripser_on_matrix(m, dim=1)

    assert isinstance(barcode, dict), "barcode must be a dict"
    assert 0 in barcode, "barcode must have dim 0"
    assert 1 in barcode, "barcode must have dim 1"
    for d in (0, 1):
        arr = barcode[d]
        assert arr.dtype.names is not None, f"dim {d} array must be structured"
        assert "birth" in arr.dtype.names, f"dim {d} must have 'birth' field"
        assert "death" in arr.dtype.names, f"dim {d} must have 'death' field"
        # Pin the dtype exactly — this is the whole point of the shim.
        assert arr.dtype == np.dtype([("birth", "<f4"), ("death", "<f4")]), (
            f"dim {d} structured array dtype must be [('birth','<f4'),('death','<f4')], "
            f"got {arr.dtype}"
        )


# ---------------------------------------------------------------------------
# count_ripser_features — h0_e and h0_t dispatch branches (item D)
# ---------------------------------------------------------------------------

def test_count_ripser_features_entropy_dispatch():
    """'e' (entropy) dispatch branch produces non-zero entropy for unequal bars."""
    from ripser_count_compat import count_ripser_features, barcode_entropy

    barcode = {
        0: np.array(
            [(0.0, 1.0), (0.0, 0.5)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        ),
    }
    features = count_ripser_features([barcode], ["h0_e"])
    expected = barcode_entropy(barcode, dim=0)
    assert features.shape == (1, 1)
    np.testing.assert_allclose(features[0, 0], expected, rtol=1e-4,
                               err_msg="h0_e dispatch should call barcode_entropy")
    assert features[0, 0] > 0, "Entropy of non-trivial barcode must be positive"


def test_count_ripser_features_time_dispatch():
    """'t' (time) dispatch branch: picks birth/death of longest barcode."""
    from ripser_count_compat import count_ripser_features

    barcode = {
        0: np.array(
            [(0.0, 1.0), (0.0, 0.5), (0.1, 0.4)],
            dtype=[("birth", "<f4"), ("death", "<f4")],
        ),
    }
    # Longest bar is (0.0, 1.0) — length 1.0.  death = 1.0, birth = 0.0
    features_death = count_ripser_features([barcode], ["h0_t_d"])
    features_birth = count_ripser_features([barcode], ["h0_t_b"])

    assert features_death.shape == (1, 1)
    assert features_birth.shape == (1, 1)
    np.testing.assert_allclose(features_death[0, 0], 1.0, rtol=1e-5,
                               err_msg="h0_t_d should return death of longest bar")
    np.testing.assert_allclose(features_birth[0, 0], 0.0, atol=1e-5,
                               err_msg="h0_t_b should return birth of longest bar")


# ---------------------------------------------------------------------------
# barcode_mean — empty case symmetry (item E)
# ---------------------------------------------------------------------------

def test_barcode_mean_empty_returns_zero():
    from ripser_count_compat import barcode_mean

    barcode = {0: np.array([], dtype=[("birth", "<f4"), ("death", "<f4")])}
    assert barcode_mean(barcode, dim=0) == 0.0


# ---------------------------------------------------------------------------
# count_ripser_features — narrow ValueError for unknown type (item F)
# ---------------------------------------------------------------------------

def test_count_ripser_features_unknown_type_raises_value_error():
    """Unknown feature type raises ValueError specifically (not just any Exception)."""
    from ripser_count_compat import count_ripser_features

    barcode = {0: np.array([(0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    with pytest.raises(ValueError):
        count_ripser_features([barcode], ["h0_z"])


# ---------------------------------------------------------------------------
# count_ripser_features — malformed bd character raises ValueError (item H)
# ---------------------------------------------------------------------------

def test_count_ripser_features_malformed_bd_raises_value_error():
    """A malformed bd character (not 'b' or 'd') in 'n' feature string raises ValueError."""
    from ripser_count_compat import count_ripser_features

    barcode = {0: np.array([(0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    # Feature string: h0_n_x_m_t0.5 — 'x' is not a valid bd character in 'n' branch
    with pytest.raises(ValueError):
        count_ripser_features([barcode], ["h0_n_x_m_t0.5"])


def test_count_ripser_features_malformed_bd_in_t_branch_raises_value_error():
    """A malformed bd character (not 'b' or 'd') in 't' feature string raises ValueError."""
    from ripser_count_compat import count_ripser_features

    barcode = {0: np.array([(0.0, 0.5)], dtype=[("birth", "<f4"), ("death", "<f4")])}
    # Feature string: h0_t_x — 'x' is not a valid bd character in 't' branch
    with pytest.raises(ValueError):
        count_ripser_features([barcode], ["h0_t_x"])
