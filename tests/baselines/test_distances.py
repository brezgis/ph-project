"""Tests for baselines/distances.py — extract_term_vectors.

Covers:
  * head strategy per language (EN, RU, ES)
  * single-word passthrough
  * mean strategy
  * skip strategy
  * OOV handling (MUSE-style and FastText CC-style)
  * head_position override argument
  * output shape invariant: matrix.shape[0] == found_mask.sum()
"""
import pathlib

import numpy as np
import pytest
from gensim.models import KeyedVectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kv(words: list[str], dim: int = 8) -> KeyedVectors:
    """Build a tiny in-memory KeyedVectors fixture with deterministic vectors."""
    kv = KeyedVectors(vector_size=dim)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((len(words), dim)).astype(np.float32)
    kv.add_vectors(words, vectors)
    return kv


# ---------------------------------------------------------------------------
# Import smoke (kept from ssa.2)
# ---------------------------------------------------------------------------

def test_distances_module_importable():
    """Smoke: module tree is valid and the public API is present."""
    from baselines.distances import cosine_distance_matrix, extract_term_vectors
    assert callable(cosine_distance_matrix)
    assert callable(extract_term_vectors)


# ---------------------------------------------------------------------------
# extract_term_vectors — full tests
# ---------------------------------------------------------------------------

class TestExtractTermVectors:
    """Unit tests for extract_term_vectors using synthetic KeyedVectors."""

    # ------------------------------------------------------------------
    # Test 1: head strategy, EN — right-headed
    # ------------------------------------------------------------------

    def test_head_en_picks_right_word(self):
        """EN head strategy: 'maternal uncle' → looks up 'uncle' (right word)."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["uncle"])  # 'maternal' intentionally absent
        matrix, mask = extract_term_vectors(
            ["maternal uncle"], kv, strategy="head", lang="en"
        )
        assert mask.shape == (1,)
        assert mask[0] is np.bool_(True) or bool(mask[0]) is True, (
            "EN head should find 'uncle' and mark mask[0]=True"
        )
        assert matrix.shape == (1, kv.vector_size)
        assert np.allclose(matrix[0], kv["uncle"])

    # ------------------------------------------------------------------
    # Test 2: head strategy, RU — right-headed
    # ------------------------------------------------------------------

    def test_head_ru_picks_right_word(self):
        """RU head strategy: 'двоюродный брат' → looks up 'брат' (right word)."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["брат"])  # 'двоюродный' absent
        matrix, mask = extract_term_vectors(
            ["двоюродный брат"], kv, strategy="head", lang="ru"
        )
        assert bool(mask[0]) is True
        assert matrix.shape == (1, kv.vector_size)
        assert np.allclose(matrix[0], kv["брат"])

    # ------------------------------------------------------------------
    # Test 3: head strategy, ES — left-headed (CRITICAL)
    # ------------------------------------------------------------------

    def test_head_es_picks_left_word(self):
        """ES head strategy: 'tío materno' → looks up 'tío' (LEFT word).

        CRITICAL: confirms the left-head rule for Spanish. If this test
        fails, the head-position lookup is inverted.
        """
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["tío"])  # 'materno' intentionally absent
        matrix, mask = extract_term_vectors(
            ["tío materno"], kv, strategy="head", lang="es"
        )
        assert bool(mask[0]) is True, (
            "ES head should find 'tío' (left word) and mark mask[0]=True. "
            "If this fails, head-position is inverted."
        )
        assert matrix.shape == (1, kv.vector_size)
        assert np.allclose(matrix[0], kv["tío"])

    # ------------------------------------------------------------------
    # Test 4: single-word terms — returned as-is regardless of language
    # ------------------------------------------------------------------

    def test_single_word_passthrough_any_lang(self):
        """Single-word terms are returned as-is; language has no effect."""
        from baselines.distances import extract_term_vectors

        for lang in ("en", "ru", "es"):
            kv = _make_kv(["joy"])
            matrix, mask = extract_term_vectors(["joy"], kv, strategy="head", lang=lang)
            assert bool(mask[0]) is True, f"Single word 'joy' should be found for lang={lang}"
            assert matrix.shape == (1, kv.vector_size)
            assert np.allclose(matrix[0], kv["joy"])

    # ------------------------------------------------------------------
    # Test 5: mean strategy averages component vectors
    # ------------------------------------------------------------------

    def test_mean_strategy_averages_components(self):
        """'mean' strategy averages component vectors; result ≈ numpy mean."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["maternal", "uncle"])
        expected = np.mean(
            np.stack([kv["maternal"], kv["uncle"]]), axis=0
        )

        matrix, mask = extract_term_vectors(
            ["maternal uncle"], kv, strategy="mean", lang="en"
        )
        assert bool(mask[0]) is True
        assert matrix.shape == (1, kv.vector_size)
        assert np.allclose(matrix[0], expected, atol=1e-5)

    # ------------------------------------------------------------------
    # Test 6: skip strategy drops multi-word terms
    # ------------------------------------------------------------------

    def test_skip_strategy_drops_multiword(self):
        """'skip' strategy: multi-word terms get mask=False and are excluded."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["joy", "maternal", "uncle"])
        terms = ["joy", "maternal uncle"]

        matrix, mask = extract_term_vectors(terms, kv, strategy="skip", lang="en")

        assert mask.shape == (2,)
        assert bool(mask[0]) is True,  "Single word 'joy' should be kept by skip strategy"
        assert bool(mask[1]) is False, "Multi-word 'maternal uncle' should be dropped by skip"
        assert matrix.shape == (1, kv.vector_size)
        assert matrix.shape[0] == int(mask.sum())

    # ------------------------------------------------------------------
    # Test 7: OOV handling — MUSE-style (no subword), mask=False, excluded
    # ------------------------------------------------------------------

    def test_oov_muse_style_excluded_from_matrix(self):
        """Terms missing from MUSE vocab get mask=False and no row in matrix."""
        from baselines.distances import extract_term_vectors

        # Only 'joy' is in vocab; 'grief' is OOV
        kv = _make_kv(["joy"])
        terms = ["joy", "grief"]

        matrix, mask = extract_term_vectors(terms, kv, strategy="head", lang="en")

        assert mask.shape == (2,)
        assert bool(mask[0]) is True,  "'joy' is in vocab → mask[0]=True"
        assert bool(mask[1]) is False, "'grief' is OOV  → mask[1]=False"
        # Shape invariant
        assert matrix.shape[0] == int(mask.sum()), (
            "matrix.shape[0] must equal mask.sum()"
        )
        assert matrix.shape == (1, kv.vector_size)

    # ------------------------------------------------------------------
    # Test 8: OOV handling — FastText CC-style subword composition
    #   Uses real .bin model; skips gracefully if file is absent.
    # ------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def cc_en_kv(self):
        """Load cc.en.300.bin if present, else None."""
        p = pathlib.Path("/home/anna/ph-project/data/fasttext/cc/cc.en.300.bin")
        if not p.exists():
            return None
        from gensim.models.fasttext import load_facebook_model
        return load_facebook_model(str(p)).wv

    def test_fasttext_subword_returns_finite_vector_for_unknown(self, cc_en_kv):
        """FastText CC .bin returns a finite vector for an unknown word via subword."""
        if cc_en_kv is None:
            pytest.skip("CC-300 .bin not downloaded — skipping subword test")

        from baselines.distances import extract_term_vectors

        # "xylophonist" is almost certainly not an exact vocab entry
        unknown_word = "xylophonist"
        terms = [unknown_word]
        matrix, mask = extract_term_vectors(terms, cc_en_kv, strategy="head", lang="en")

        assert bool(mask[0]) is True, (
            f"FastText subword should compose a vector for '{unknown_word}'"
        )
        assert matrix.shape == (1, cc_en_kv.vector_size)
        assert np.all(np.isfinite(matrix[0])), "Subword vector must be finite"

    # ------------------------------------------------------------------
    # Test 9: head_position override argument
    # ------------------------------------------------------------------

    def test_head_position_override_overrides_lang_default(self):
        """head_position='right' on lang='es' overrides the left-default.

        'tío materno' with lang='es' normally → 'tío' (left).
        Overriding head_position='right' should yield 'materno' (right).
        """
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["materno"])  # 'tío' absent; only right word in vocab
        matrix, mask = extract_term_vectors(
            ["tío materno"],
            kv,
            strategy="head",
            lang="es",
            head_position="right",
        )
        assert bool(mask[0]) is True, (
            "With head_position='right' override, 'materno' should be found"
        )
        assert np.allclose(matrix[0], kv["materno"])


# ------------------------------------------------------------------
# Shape invariant: matrix.shape[0] == mask.sum()  (always)
# ------------------------------------------------------------------

class TestShapeInvariant:
    """Ensure shape invariant holds across strategies and OOV mixes."""

    @pytest.mark.parametrize("strategy", ["head", "mean", "skip"])
    def test_shape_invariant(self, strategy):
        from baselines.distances import extract_term_vectors

        # vocab: 'joy', 'grief'; 'maternal' absent (tests OOV in mean/head),
        # and 'maternal uncle' is multi-word (tests skip).
        kv = _make_kv(["joy", "grief"])
        terms = ["joy", "grief", "maternal uncle", "unknown_word"]

        matrix, mask = extract_term_vectors(terms, kv, strategy=strategy, lang="en")

        assert mask.shape == (4,)
        assert matrix.shape[0] == int(mask.sum()), (
            f"Shape invariant violated for strategy={strategy!r}: "
            f"matrix.shape[0]={matrix.shape[0]} != mask.sum()={mask.sum()}"
        )


# ------------------------------------------------------------------
# Error paths
# ------------------------------------------------------------------

class TestErrorPaths:
    def test_unknown_strategy_raises_value_error(self):
        """Passing an unrecognized strategy raises ValueError."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["joy"])
        with pytest.raises(ValueError, match="strategy"):
            extract_term_vectors(["maternal uncle"], kv, strategy="bogus", lang="en")


# ------------------------------------------------------------------
# Boundary cases: empty input, all-OOV
# ------------------------------------------------------------------

class TestBoundaryCases:
    def test_empty_input_returns_empty_matrix_and_mask(self):
        """Empty terms list → matrix shape (0, dim), mask shape (0,)."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["joy"], dim=8)
        matrix, mask = extract_term_vectors([], kv, strategy="head", lang="en")

        assert matrix.shape == (0, 8)
        assert mask.shape == (0,)
        assert mask.dtype == bool

    def test_all_oov_returns_empty_matrix_and_all_false_mask(self):
        """Every term OOV in MUSE-style vectors → (0, dim) matrix, all-False mask."""
        from baselines.distances import extract_term_vectors

        kv = _make_kv(["joy"], dim=8)  # nothing else in vocab
        terms = ["unknown_a", "unknown_b", "unknown_c"]
        matrix, mask = extract_term_vectors(terms, kv, strategy="head", lang="en")

        assert matrix.shape == (0, 8)
        assert mask.shape == (3,)
        assert not mask.any()


# ------------------------------------------------------------------
# cosine_distance_matrix  (ssa.4)
# ------------------------------------------------------------------

class TestCosineDistanceMatrix:
    """Unit tests for cosine_distance_matrix."""

    def _small_matrix(self) -> np.ndarray:
        """Three 4-dim unit vectors with known cosine relationships."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 4)).astype(np.float32)
        return X

    def test_shape_n_by_n(self):
        """Output shape is (n, n) for input (n, dim)."""
        from baselines.distances import cosine_distance_matrix

        X = self._small_matrix()
        D = cosine_distance_matrix(X)
        assert D.shape == (X.shape[0], X.shape[0])

    def test_zero_diagonal(self):
        """Diagonal entries are zero (distance from a vector to itself)."""
        from baselines.distances import cosine_distance_matrix

        X = self._small_matrix()
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)

    def test_symmetry(self):
        """Distance matrix is symmetric: D == D.T."""
        from baselines.distances import cosine_distance_matrix

        X = self._small_matrix()
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(D, D.T, atol=1e-6)

    def test_bounded_0_to_2(self):
        """All values are in [0, 2] (cosine distance range)."""
        from baselines.distances import cosine_distance_matrix

        # Use an adversarial matrix with negative entries to stress [0, 2]
        rng = np.random.default_rng(7)
        X = rng.standard_normal((6, 8)).astype(np.float32)
        D = cosine_distance_matrix(X)
        assert np.all(D >= -1e-6), f"Negative values found: min={D.min()}"
        assert np.all(D <= 2.0 + 1e-6), f"Values > 2 found: max={D.max()}"

    def test_matches_scipy_cosine_on_handpicked_pairs(self):
        """Values match scipy.spatial.distance.cosine on at least 3 pairs."""
        from baselines.distances import cosine_distance_matrix
        from scipy.spatial.distance import cosine as sp_cosine

        # Three handpicked small vectors
        X = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],  # 45° from both axis vectors
        ], dtype=np.float32)

        D = cosine_distance_matrix(X)

        # Pair (0,1): orthogonal → cosine distance = 1.0
        np.testing.assert_allclose(D[0, 1], sp_cosine(X[0], X[1]), atol=1e-6)
        # Pair (0,2)
        np.testing.assert_allclose(D[0, 2], sp_cosine(X[0], X[2]), atol=1e-6)
        # Pair (1,2)
        np.testing.assert_allclose(D[1, 2], sp_cosine(X[1], X[2]), atol=1e-6)

        # Verify orthogonal pair explicitly
        np.testing.assert_allclose(D[0, 1], 1.0, atol=1e-6)

    def test_matches_scipy_cosine_random_vectors(self):
        """Values match scipy on a random (4, 16) matrix for all pairs."""
        from baselines.distances import cosine_distance_matrix
        from scipy.spatial.distance import cosine as sp_cosine

        rng = np.random.default_rng(99)
        X = rng.standard_normal((4, 16)).astype(np.float32)
        D = cosine_distance_matrix(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                expected = sp_cosine(X[i], X[j]) if i != j else 0.0
                np.testing.assert_allclose(D[i, j], expected, atol=1e-5,
                    err_msg=f"Mismatch at ({i},{j})")

    def test_opposite_vectors_distance_two(self):
        """Anti-parallel vectors have cosine distance = 2.0."""
        from baselines.distances import cosine_distance_matrix

        X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        D = cosine_distance_matrix(X)
        np.testing.assert_allclose(D[0, 1], 2.0, atol=1e-6)
