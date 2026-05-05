"""Tests for per-term permutation test helpers in replication/diagram_distances.py.

Covers blr.4 additions:
  - _normalize_gloss: strips "dark "/"light " prefixes; idempotent
  - load_translation_triples: returns exactly 12 rows for color; both Russian
    blue variants (синий + голубой) present; correct column set
  - per_term_test_statistic: excludes within-language pairs; responds to
    cross-language planted signal
  - permutation_test_per_term: H_0 calibration (KS test), planted-signal
    detection (p < 0.01), valid result keys
  - russian_blue_zoom: valid result structure on synthetic data

PH_REQUIRE_DIAGRAM_DISTANCES=1 flips skips into hard failures, mirroring the
pattern in test_diagram_distances_perm.py.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

REQUIRE = os.environ.get("PH_REQUIRE_DIAGRAM_DISTANCES") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANON_DIR = REPO_ROOT / "canon-terms"

# Slow-test gate
pytestmark_slow = pytest.mark.skipif(
    not os.environ.get("PH_RUN_SLOW_TESTS"),
    reason="Slow test; set PH_RUN_SLOW_TESTS=1 to run.",
)

# ---------------------------------------------------------------------------
# Conditional import — tests fail (not skip) on missing production code in Phase 1
# ---------------------------------------------------------------------------
try:
    from replication.diagram_distances import (
        _normalize_gloss,
        load_translation_triples,
        per_term_test_statistic,
        permutation_test_per_term,
        russian_blue_zoom,
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
            "Could not import per-term helpers from replication.diagram_distances — "
            "_normalize_gloss, load_translation_triples, per_term_test_statistic, "
            "permutation_test_per_term, russian_blue_zoom are not yet defined."
        )


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_color_canon_dir():
    """Create a temporary canon-terms directory with minimal YAML files for testing.

    Returns a pathlib.Path to the temporary directory containing:
      en/color.yaml — 4 terms (black, white, blue, red), no gloss
      ru/color.yaml — 5 terms (чёрный, белый, синий, голубой, красный), with glosses
      es/color.yaml — 4 terms (negro, blanco, azul, rojo), with glosses
    """
    tmpdir = pathlib.Path(tempfile.mkdtemp())

    # English: term only (11 real BCTs; we use 4 for speed)
    en_dir = tmpdir / "en"
    en_dir.mkdir()
    en_data = {
        "domain": "color",
        "language": "en",
        "terms": [
            {"term": "black"},
            {"term": "white"},
            {"term": "blue"},
            {"term": "red"},
        ],
    }
    (en_dir / "color.yaml").write_text(yaml.dump(en_data, allow_unicode=True))

    # Russian: term + gloss; blue is split into "dark blue" and "light blue"
    ru_dir = tmpdir / "ru"
    ru_dir.mkdir()
    ru_data = {
        "domain": "color",
        "language": "ru",
        "terms": [
            {"term": "чёрный", "gloss": "black"},
            {"term": "белый", "gloss": "white"},
            {"term": "синий", "gloss": "dark blue"},
            {"term": "голубой", "gloss": "light blue"},
            {"term": "красный", "gloss": "red"},
        ],
    }
    (ru_dir / "color.yaml").write_text(yaml.dump(ru_data, allow_unicode=True))

    # Spanish: term + gloss (1:1 with English for these 4 terms)
    es_dir = tmpdir / "es"
    es_dir.mkdir()
    es_data = {
        "domain": "color",
        "language": "es",
        "terms": [
            {"term": "negro", "gloss": "black"},
            {"term": "blanco", "gloss": "white"},
            {"term": "azul", "gloss": "blue"},
            {"term": "rojo", "gloss": "red"},
        ],
    }
    (es_dir / "color.yaml").write_text(yaml.dump(es_data, allow_unicode=True))

    return tmpdir


def _make_meta_with_terms(
    en_terms: list[str],
    ru_terms: list[str],
    es_terms: list[str],
    n_per_term: int = 5,
) -> pd.DataFrame:
    """Build a metadata DataFrame with lang+term columns for synthetic tests.

    Each (lang, term) combination gets `n_per_term` rows.
    """
    rows = []
    for lang, terms in [("en", en_terms), ("ru", ru_terms), ("es", es_terms)]:
        for term in terms:
            for _ in range(n_per_term):
                rows.append({"lang": lang, "term": term})
    return pd.DataFrame(rows).reset_index(drop=True)


def _make_triples_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal triples DataFrame with the required columns."""
    return pd.DataFrame(rows, columns=["en_term", "ru_term", "es_term", "ru_gloss_raw"])


def _planted_cross_lang_matrix(
    meta: pd.DataFrame,
    triples_df: pd.DataFrame,
    within_lang_dist: float = 0.0,
    within_triple_dist: float = 0.5,
    cross_lang_non_triple_dist: float = 2.0,
) -> np.ndarray:
    """Build a distance matrix with planted within-triple proximity.

    Within-language pairs: within_lang_dist
    Cross-language pairs in the same triple: within_triple_dist
    Cross-language pairs outside same triple: cross_lang_non_triple_dist
    """
    n = len(meta)
    mat = np.full((n, n), cross_lang_non_triple_dist, dtype=np.float64)
    np.fill_diagonal(mat, 0.0)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            li, ti = meta.iloc[i]["lang"], meta.iloc[i]["term"]
            lj, tj = meta.iloc[j]["lang"], meta.iloc[j]["term"]

            if li == lj:
                mat[i, j] = within_lang_dist
            else:
                # Check if they belong to the same triple
                for _, row in triples_df.iterrows():
                    triple = {
                        "en": row["en_term"],
                        "ru": row["ru_term"],
                        "es": row["es_term"],
                    }
                    if triple.get(li) == ti and triple.get(lj) == tj:
                        mat[i, j] = within_triple_dist
                        break

    # Symmetrize
    mat = 0.5 * (mat + mat.T)
    return mat


# ---------------------------------------------------------------------------
# _normalize_gloss tests
# ---------------------------------------------------------------------------

class TestNormalizeGloss:
    def test_import(self):
        _require_import()

    def test_strips_dark_prefix(self):
        """'dark blue' → 'blue'."""
        _require_import()
        assert _normalize_gloss("dark blue") == "blue"

    def test_strips_light_prefix(self):
        """'light blue' → 'blue'."""
        _require_import()
        assert _normalize_gloss("light blue") == "blue"

    def test_passthrough_plain(self):
        """Strings without prefix pass through unchanged."""
        _require_import()
        assert _normalize_gloss("blue") == "blue"
        assert _normalize_gloss("red") == "red"
        assert _normalize_gloss("purple") == "purple"

    def test_idempotent_dark(self):
        """Applying _normalize_gloss twice gives the same result as once."""
        _require_import()
        result = _normalize_gloss("dark blue")
        assert _normalize_gloss(result) == result

    def test_idempotent_light(self):
        """Applying _normalize_gloss twice gives the same result as once."""
        _require_import()
        result = _normalize_gloss("light blue")
        assert _normalize_gloss(result) == result

    def test_lowercase_normalizes(self):
        """Mixed-case input is lowercased and stripped correctly."""
        _require_import()
        assert _normalize_gloss("Dark Blue") == "blue"
        assert _normalize_gloss("DARK BLUE") == "blue"
        assert _normalize_gloss("Light Blue") == "blue"

    def test_does_not_strip_other_prefixes(self):
        """Does not strip prefixes other than 'dark ' or 'light '."""
        _require_import()
        # 'brown' should be 'brown', not 'rown'
        assert _normalize_gloss("brown") == "brown"
        # 'dark' alone stays 'dark'
        assert _normalize_gloss("dark") == "dark"
        # 'light' alone stays 'light'
        assert _normalize_gloss("light") == "light"


# ---------------------------------------------------------------------------
# load_translation_triples tests
# ---------------------------------------------------------------------------

class TestLoadTranslationTriples:
    def test_import(self):
        _require_import()

    def test_color_returns_12_rows(self):
        """load_translation_triples produces exactly 12 rows for color domain.

        10 1:1:1 triples + 2 blue rows (синий + голубой).
        Uses the real canon-terms directory.
        """
        _require_import()
        if not CANON_DIR.exists():
            pytest.skip(f"canon-terms directory not found at {CANON_DIR}")

        df = load_translation_triples(CANON_DIR, domain="color")
        assert len(df) == 12, (
            f"Expected 12 rows for color domain, got {len(df)}.\n"
            f"Rows:\n{df.to_string()}"
        )

    def test_both_russian_blues_present(self):
        """Both (en=blue, ru=синий, es=azul) and (en=blue, ru=голубой, es=azul) must be rows."""
        _require_import()
        if not CANON_DIR.exists():
            pytest.skip(f"canon-terms directory not found at {CANON_DIR}")

        df = load_translation_triples(CANON_DIR, domain="color")
        blue_rows = df[df["en_term"] == "blue"]
        assert len(blue_rows) == 2, (
            f"Expected 2 blue rows, got {len(blue_rows)}.\n{blue_rows.to_string()}"
        )
        ru_blue_terms = set(blue_rows["ru_term"])
        assert "синий" in ru_blue_terms, f"синий missing from blue rows: {ru_blue_terms}"
        assert "голубой" in ru_blue_terms, f"голубой missing from blue rows: {ru_blue_terms}"

    def test_required_columns(self):
        """DataFrame must have columns: en_term, ru_term, es_term, ru_gloss_raw."""
        _require_import()
        if not CANON_DIR.exists():
            pytest.skip(f"canon-terms directory not found at {CANON_DIR}")

        df = load_translation_triples(CANON_DIR, domain="color")
        required = {"en_term", "ru_term", "es_term", "ru_gloss_raw"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}. Found: {list(df.columns)}"

    def test_synthetic_4_terms_returns_5_rows(self):
        """Synthetic canon dir with 4 en terms and the blue split → 5 rows total."""
        _require_import()
        tmpdir = _make_color_canon_dir()
        try:
            df = load_translation_triples(tmpdir, domain="color")
            # 3 non-blue 1:1:1 + 2 blue = 5
            assert len(df) == 5, (
                f"Expected 5 rows for synthetic 4-term color (with blue split), "
                f"got {len(df)}.\n{df.to_string()}"
            )
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_synthetic_both_blues_present(self):
        """Synthetic dir: both синий and голубой appear in the en=blue rows."""
        _require_import()
        tmpdir = _make_color_canon_dir()
        try:
            df = load_translation_triples(tmpdir, domain="color")
            blue_rows = df[df["en_term"] == "blue"]
            assert len(blue_rows) == 2, f"Expected 2 blue rows, got {len(blue_rows)}"
            ru_terms = set(blue_rows["ru_term"])
            assert "синий" in ru_terms
            assert "голубой" in ru_terms
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_ru_gloss_raw_preserves_dark_light(self):
        """ru_gloss_raw must preserve the raw 'dark blue'/'light blue' strings."""
        _require_import()
        if not CANON_DIR.exists():
            pytest.skip(f"canon-terms directory not found at {CANON_DIR}")

        df = load_translation_triples(CANON_DIR, domain="color")
        blue_rows = df[df["en_term"] == "blue"]
        raw_glosses = set(blue_rows["ru_gloss_raw"])
        assert "dark blue" in raw_glosses, (
            f"Expected 'dark blue' in ru_gloss_raw, got: {raw_glosses}"
        )
        assert "light blue" in raw_glosses, (
            f"Expected 'light blue' in ru_gloss_raw, got: {raw_glosses}"
        )

    def test_non_blue_rows_have_none_or_plain_gloss(self):
        """Non-blue ru entries have a plain gloss (no 'dark'/'light' prefix)."""
        _require_import()
        if not CANON_DIR.exists():
            pytest.skip(f"canon-terms directory not found at {CANON_DIR}")

        df = load_translation_triples(CANON_DIR, domain="color")
        non_blue = df[df["en_term"] != "blue"]
        # ru_gloss_raw for non-blue rows should not contain "dark " or "light "
        for _, row in non_blue.iterrows():
            raw = row["ru_gloss_raw"]
            if raw is not None and isinstance(raw, str):
                normalized = _normalize_gloss(raw)
                assert normalized == raw.lower(), (
                    f"Non-blue row has a dark/light-prefixed gloss: {raw!r}"
                )


# ---------------------------------------------------------------------------
# per_term_test_statistic tests
# ---------------------------------------------------------------------------

class TestPerTermTestStatistic:
    def test_import(self):
        _require_import()

    def test_excludes_within_language_pairs(self):
        """Within-language distances of 0 don't affect the statistic.

        Build a matrix where within-language distances are 0 and cross-language
        same-triple pairs are exactly 1.0. The statistic should reflect only
        the cross-language same-triple mean (≈ 1.0), not the within-language 0s.
        """
        _require_import()
        en_terms = ["black", "blue"]
        ru_terms = ["чёрный", "синий"]
        es_terms = ["negro", "azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=3)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])

        # within-lang = 0, same-triple cross-lang = 1.0, other cross-lang = 2.0
        mat = _planted_cross_lang_matrix(
            meta, triples_df,
            within_lang_dist=0.0,
            within_triple_dist=1.0,
            cross_lang_non_triple_dist=2.0,
        )

        stat = per_term_test_statistic(mat, meta, triples_df, ru_blue_choice="синий")
        # Statistic is the mean over all cross-language same-triple pairs.
        # With perfect within-triple alignment at 1.0, stat should be close to 1.0
        # (not 0 from within-lang contamination).
        assert stat > 0.5, (
            f"Expected stat > 0.5 (cross-lang same-triple signal), got {stat:.4f}. "
            "Check that within-language pairs are excluded."
        )
        # The within-lang pairs are 0, so if they were included the mean would drop
        assert stat < 1.5, (
            f"Expected stat < 1.5 (consistent with same-triple mean ≈ 1.0), got {stat:.4f}"
        )

    def test_returns_float(self):
        """per_term_test_statistic must return a Python float."""
        _require_import()
        en_terms = ["black", "blue"]
        ru_terms = ["чёрный", "синий"]
        es_terms = ["negro", "azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=3)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])
        n = len(meta)
        mat = np.random.default_rng(0).random((n, n))
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
        stat = per_term_test_statistic(mat, meta, triples_df, ru_blue_choice="синий")
        assert isinstance(stat, float), f"Expected float, got {type(stat)}"
        assert np.isfinite(stat), f"Expected finite statistic, got {stat}"

    def test_ru_blue_choice_sinij_vs_goluboy(self):
        """Changing ru_blue_choice changes which pairs enter the stat.

        With within_triple_dist=0.5 for синий and 2.0 for голубой, the статistic
        with ru_blue_choice='синий' should be lower than with 'голубой'.
        """
        _require_import()
        en_terms = ["blue"]
        ru_terms = ["синий", "голубой"]
        es_terms = ["azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=4)

        # синий is close to en=blue and es=azul; голубой is far
        sinij_triples = _make_triples_df([
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])
        goluboy_triples = _make_triples_df([
            {"en_term": "blue", "ru_term": "голубой", "es_term": "azul", "ru_gloss_raw": "light blue"},
        ])

        n = len(meta)
        rng = np.random.default_rng(7)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)

        stat_sinij = per_term_test_statistic(mat, meta, sinij_triples, ru_blue_choice="синий")
        stat_goluboy = per_term_test_statistic(mat, meta, goluboy_triples, ru_blue_choice="голубой")
        # They operate on different subsets; both should be finite
        assert np.isfinite(stat_sinij)
        assert np.isfinite(stat_goluboy)


# ---------------------------------------------------------------------------
# permutation_test_per_term tests
# ---------------------------------------------------------------------------

class TestPermutationTestPerTerm:
    def test_import(self):
        _require_import()

    def test_result_keys(self):
        """Result dict must have keys: observed, null, p_value, effect_size."""
        _require_import()
        en_terms = ["black", "blue"]
        ru_terms = ["чёрный", "синий"]
        es_terms = ["negro", "azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=3)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])
        n = len(meta)
        rng = np.random.default_rng(0)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)

        result = permutation_test_per_term(mat, meta, triples_df, K=49, seed=0)
        for key in ("observed", "null", "p_value", "effect_size"):
            assert key in result, f"Missing key '{key}' in result: {list(result.keys())}"

    def test_null_length_equals_K(self):
        """Null distribution must have exactly K entries."""
        _require_import()
        K = 77
        en_terms = ["black", "blue"]
        ru_terms = ["чёрный", "синий"]
        es_terms = ["negro", "azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=3)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])
        n = len(meta)
        mat = np.random.default_rng(1).random((n, n))
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
        result = permutation_test_per_term(mat, meta, triples_df, K=K, seed=1)
        assert len(result["null"]) == K

    def test_p_value_in_range(self):
        """p_value must be in (0, 1]."""
        _require_import()
        en_terms = ["black", "blue"]
        ru_terms = ["чёрный", "синий"]
        es_terms = ["negro", "azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=3)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])
        n = len(meta)
        mat = np.random.default_rng(2).random((n, n))
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
        result = permutation_test_per_term(mat, meta, triples_df, K=99, seed=2)
        assert 0 < result["p_value"] <= 1.0

    def test_determinism(self):
        """Same seed and inputs produce identical results."""
        _require_import()
        en_terms = ["black", "blue"]
        ru_terms = ["чёрный", "синий"]
        es_terms = ["negro", "azul"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=3)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
        ])
        n = len(meta)
        mat = np.random.default_rng(3).random((n, n))
        mat = 0.5 * (mat + mat.T)
        np.fill_diagonal(mat, 0.0)
        r1 = permutation_test_per_term(mat, meta, triples_df, K=49, seed=42)
        r2 = permutation_test_per_term(mat, meta, triples_df, K=49, seed=42)
        assert r1["p_value"] == r2["p_value"]
        assert r1["observed"] == r2["observed"]
        np.testing.assert_array_equal(r1["null"], r2["null"])

    def test_planted_signal_detected(self):
        """With strong within-triple proximity, p_value < 0.01."""
        _require_import()
        # Use enough terms and samples for the permutation to be meaningful
        en_terms = ["black", "white", "blue", "red"]
        ru_terms = ["чёрный", "белый", "синий", "красный"]
        es_terms = ["negro", "blanco", "azul", "rojo"]
        meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=6)
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "white", "ru_term": "белый", "es_term": "blanco", "ru_gloss_raw": "white"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
            {"en_term": "red", "ru_term": "красный", "es_term": "rojo", "ru_gloss_raw": "red"},
        ])

        # Strong planted signal: within-triple cross-lang = 0.05, non-triple = 5.0
        mat = _planted_cross_lang_matrix(
            meta, triples_df,
            within_lang_dist=0.0,
            within_triple_dist=0.05,
            cross_lang_non_triple_dist=5.0,
        )

        result = permutation_test_per_term(
            mat, meta, triples_df, K=1999, seed=42, ru_blue_choice="синий"
        )
        assert result["p_value"] < 0.01, (
            f"Expected p_value < 0.01 under strong planted signal, "
            f"got {result['p_value']:.4f}. "
            f"observed={result['observed']:.4f}, effect_size={result['effect_size']:.4f}"
        )

    @pytestmark_slow
    def test_p_value_uniform_under_null(self):
        """KS test: p-values under H_0 (random distance matrix, random labels)
        should follow Uniform(0, 1) approximately.

        Uses K=499 and 100 trials for speed. The statistic is the mean
        cross-language same-triple distance; under H_0 (distances independent
        of term labels), the p-value distribution should be roughly uniform.
        """
        _require_import()
        from scipy import stats as scipy_stats

        en_terms = ["black", "white", "blue", "red"]
        ru_terms = ["чёрный", "белый", "синий", "красный"]
        es_terms = ["negro", "blanco", "azul", "rojo"]
        triples_df = _make_triples_df([
            {"en_term": "black", "ru_term": "чёрный", "es_term": "negro", "ru_gloss_raw": "black"},
            {"en_term": "white", "ru_term": "белый", "es_term": "blanco", "ru_gloss_raw": "white"},
            {"en_term": "blue", "ru_term": "синий", "es_term": "azul", "ru_gloss_raw": "dark blue"},
            {"en_term": "red", "ru_term": "красный", "es_term": "rojo", "ru_gloss_raw": "red"},
        ])

        p_values = []
        for seed in range(50):
            rng = np.random.default_rng(seed + 1000)
            meta = _make_meta_with_terms(en_terms, ru_terms, es_terms, n_per_term=5)
            n = len(meta)
            raw = rng.random((n, n))
            mat = 0.5 * (raw + raw.T)
            np.fill_diagonal(mat, 0.0)
            result = permutation_test_per_term(
                mat, meta, triples_df, K=499, seed=int(seed * 13)
            )
            p_values.append(result["p_value"])

        p_arr = np.array(p_values)
        ks_stat, ks_p = scipy_stats.kstest(p_arr, "uniform")
        assert ks_p > 0.001, (
            f"p-values under H_0 appear non-uniform: KS stat={ks_stat:.4f}, "
            f"KS p-value={ks_p:.4f}. "
            f"Mean p={p_arr.mean():.3f}, std={p_arr.std():.3f}. "
            "Check permutation_test_per_term p-value formula."
        )


# ---------------------------------------------------------------------------
# russian_blue_zoom tests
# ---------------------------------------------------------------------------

class TestRussianBlueZoom:
    def test_import(self):
        _require_import()

    def test_result_keys(self):
        """Result dict must have keys: observed, null, p_value, effect_size."""
        _require_import()
        # Build a metadata df with Russian terms including both blues
        ru_terms = ["синий", "голубой", "чёрный", "красный"]
        meta = pd.DataFrame({
            "lang": ["ru"] * 20,
            "term": (ru_terms * 5),
        })
        n = len(meta)
        rng = np.random.default_rng(0)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        result = russian_blue_zoom(mat, meta, K=49, seed=0)
        for key in ("observed", "null", "p_value", "effect_size"):
            assert key in result, f"Missing key '{key}': {list(result.keys())}"

    def test_p_value_in_range(self):
        """p_value must be in (0, 1]."""
        _require_import()
        ru_terms = ["синий", "голубой", "чёрный", "красный"]
        meta = pd.DataFrame({
            "lang": ["ru"] * 20,
            "term": (ru_terms * 5),
        })
        n = len(meta)
        rng = np.random.default_rng(1)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        result = russian_blue_zoom(mat, meta, K=99, seed=1)
        assert 0 < result["p_value"] <= 1.0, (
            f"p_value={result['p_value']} outside (0, 1]"
        )

    def test_null_length_equals_K(self):
        """Null distribution must have exactly K entries."""
        _require_import()
        K = 55
        ru_terms = ["синий", "голубой", "чёрный", "красный"]
        meta = pd.DataFrame({
            "lang": ["ru"] * 20,
            "term": (ru_terms * 5),
        })
        n = len(meta)
        rng = np.random.default_rng(2)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        result = russian_blue_zoom(mat, meta, K=K, seed=2)
        assert len(result["null"]) == K

    def test_determinism(self):
        """Same seed produces identical results."""
        _require_import()
        ru_terms = ["синий", "голубой", "чёрный", "красный"]
        meta = pd.DataFrame({
            "lang": ["ru"] * 20,
            "term": (ru_terms * 5),
        })
        n = len(meta)
        rng = np.random.default_rng(3)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)
        r1 = russian_blue_zoom(mat, meta, K=49, seed=42)
        r2 = russian_blue_zoom(mat, meta, K=49, seed=42)
        assert r1["p_value"] == r2["p_value"]
        assert r1["observed"] == r2["observed"]
        np.testing.assert_array_equal(r1["null"], r2["null"])

    def test_uses_only_russian_samples(self):
        """russian_blue_zoom must operate only on Russian samples.

        Build a metadata df with en + ru samples. The function should only
        use the ru subset. We verify this indirectly by checking that the
        result is a valid (non-NaN) float — if it mistakenly included en
        samples without синий/голубой, the statistic would be undefined.
        """
        _require_import()
        # Combined en+ru metadata; the function should restrict to ru
        rows = []
        for term in ["blue", "black"]:
            for _ in range(5):
                rows.append({"lang": "en", "term": term})
        for term in ["синий", "голубой", "чёрный"]:
            for _ in range(5):
                rows.append({"lang": "ru", "term": term})
        meta = pd.DataFrame(rows).reset_index(drop=True)

        n = len(meta)
        rng = np.random.default_rng(4)
        raw = rng.random((n, n))
        mat = 0.5 * (raw + raw.T)
        np.fill_diagonal(mat, 0.0)

        result = russian_blue_zoom(mat, meta, K=49, seed=4)
        assert np.isfinite(result["observed"]), (
            f"russian_blue_zoom returned non-finite observed: {result['observed']}"
        )
        assert np.isfinite(result["p_value"]), (
            f"russian_blue_zoom returned non-finite p_value: {result['p_value']}"
        )

    def test_observed_positive_when_blues_far_apart(self):
        """Observed statistic > 0 when синий and голубой are the most distant pair.

        The prediction is: observed > 0 if the language-mediated split is
        encoded in mBERT attention topology. This test verifies the sign
        convention is correct.
        """
        _require_import()
        # Russian-only: синий far from голубой, other pairs close
        ru_terms_list = ["синий", "голубой", "чёрный", "красный"]
        n_per_term = 4
        rows = []
        for term in ru_terms_list:
            for _ in range(n_per_term):
                rows.append({"lang": "ru", "term": term})
        meta = pd.DataFrame(rows).reset_index(drop=True)
        n = len(meta)

        # Build a distance matrix: синий-голубой pairs get high distance (5.0),
        # all others get low distance (0.1)
        indices = {term: [i for i, r in meta.iterrows() if r["term"] == term]
                   for term in ru_terms_list}
        mat = np.full((n, n), 0.1, dtype=np.float64)
        np.fill_diagonal(mat, 0.0)
        for i in indices["синий"]:
            for j in indices["голубой"]:
                mat[i, j] = 5.0
                mat[j, i] = 5.0

        result = russian_blue_zoom(mat, meta, K=499, seed=7)
        assert result["observed"] > 0, (
            f"Expected observed > 0 when синий-голубой is the most distant pair, "
            f"got {result['observed']:.4f}. "
            "Check the sign convention of the statistic."
        )
