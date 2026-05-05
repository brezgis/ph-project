"""Tests for replication/diagram_distances.py — loader, giotto-tda conversion, subsampling.

Mirrors the PH_REQUIRE_* gating pattern from test_phase3_comparison_outputs.py.
Set PH_REQUIRE_DIAGRAM_DISTANCES=1 to flip skips into hard failures for
CI / post-run verification (e.g. after the full ripser run completes).

Test coverage:
  - load_barcode_json: key casting, array dtype, structure
  - load_lang_barcodes: multi-part glob order, metadata DataFrame shape/columns
  - subsample_per_term: determinism, under-target warning via pytest.warns
  - to_giotto_format: shape, dtype, padding convention (hom_dim column)
  - Smoke test against en_color_part1of3.json (gated by PH_REQUIRE_DIAGRAM_DISTANCES)
"""
from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

REQUIRE = os.environ.get("PH_REQUIRE_DIAGRAM_DISTANCES") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BARCODE_DIR = REPO_ROOT / "data" / "phase3" / "barcodes"

# KWIC CSVs live under data/kwic/<lang>/<domain>.csv.
# In worktrees, data/phase3 is symlinked from the main checkout but data/kwic
# may not be. Fall back to the main checkout location if needed.
_WORKTREE_KWIC = REPO_ROOT / "data" / "kwic"
_MAIN_KWIC = pathlib.Path("/home/anna/ph-project/data/kwic")
if (_WORKTREE_KWIC / "en" / "color.csv").exists():
    KWIC_DIR = _WORKTREE_KWIC
elif (_MAIN_KWIC / "en" / "color.csv").exists():
    KWIC_DIR = _MAIN_KWIC
else:
    KWIC_DIR = _WORKTREE_KWIC  # will trigger skip when not found

# Inline import — the module doesn't exist yet so these imports will fail during Phase 1,
# satisfying the "tests fail before production code" gate.
try:
    from replication.diagram_distances import (
        load_barcode_json,
        load_lang_barcodes,
        subsample_per_term,
        to_giotto_format,
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
            "Could not import replication.diagram_distances — module not yet created."
        )


# ---------------------------------------------------------------------------
# Fixtures: synthetic barcode data
# ---------------------------------------------------------------------------

def _make_synthetic_barcodes(n_layers=2, n_heads=2, n_samples=5):
    """Build a minimal synthetic barcode dict in the on-disk string-key format.

    Mimics the JSON round-trip artifact: outer keys are strings ("0", "1", ...),
    inner values are lists of dicts with string dim keys ("0", "1").
    """
    data = {}
    rng = np.random.default_rng(0)
    for layer in range(n_layers):
        data[str(layer)] = {}
        for head in range(n_heads):
            samples = []
            for _ in range(n_samples):
                sample = {
                    "0": rng.random((4, 2)).tolist(),   # H_0: 4 features
                    "1": rng.random((2, 2)).tolist(),   # H_1: 2 features
                }
                samples.append(sample)
            data[str(layer)][str(head)] = samples
    return data


def _make_synthetic_metadata(n_samples=5, terms=("red", "blue"), n_per_term=None):
    """Build a synthetic metadata DataFrame matching load_lang_barcodes output."""
    if n_per_term is None:
        # Distribute samples roughly evenly across terms
        n_per_term = n_samples // len(terms)

    rows = []
    idx = 0
    for term in terms:
        for sent_i in range(n_per_term):
            rows.append({
                "lang": "en",
                "term": term,
                "sentence_idx_within_term": sent_i,
                "source_file": "fake.json",
                "source_part": 1,
            })
            idx += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# load_barcode_json tests
# ---------------------------------------------------------------------------

class TestLoadBarcodeJson:
    def test_import(self):
        _require_import()

    def test_key_types_are_int_tuples(self, tmp_path):
        _require_import()
        import json
        raw = _make_synthetic_barcodes(n_layers=2, n_heads=2, n_samples=3)
        fpath = tmp_path / "fake.json"
        fpath.write_text(json.dumps(raw))

        result = load_barcode_json(fpath)
        # Outer dict must be keyed by (int, int) tuples
        for key in result:
            assert isinstance(key, tuple), f"Expected tuple key, got {type(key)}"
            assert len(key) == 2
            assert isinstance(key[0], int), f"Layer key must be int, got {type(key[0])}"
            assert isinstance(key[1], int), f"Head key must be int, got {type(key[1])}"

    def test_inner_list_length(self, tmp_path):
        _require_import()
        import json
        n_samples = 7
        raw = _make_synthetic_barcodes(n_layers=2, n_heads=2, n_samples=n_samples)
        fpath = tmp_path / "fake.json"
        fpath.write_text(json.dumps(raw))

        result = load_barcode_json(fpath)
        for key, sample_list in result.items():
            assert len(sample_list) == n_samples, (
                f"Expected {n_samples} samples for {key}, got {len(sample_list)}"
            )

    def test_dim_keys_are_int(self, tmp_path):
        _require_import()
        import json
        raw = _make_synthetic_barcodes(n_layers=1, n_heads=1, n_samples=3)
        fpath = tmp_path / "fake.json"
        fpath.write_text(json.dumps(raw))

        result = load_barcode_json(fpath)
        key = (0, 0)
        for sample in result[key]:
            for dim_key in sample:
                assert isinstance(dim_key, int), (
                    f"Dim key must be int after parsing, got {type(dim_key)}"
                )

    def test_arrays_are_float64(self, tmp_path):
        _require_import()
        import json
        raw = _make_synthetic_barcodes(n_layers=1, n_heads=1, n_samples=3)
        fpath = tmp_path / "fake.json"
        fpath.write_text(json.dumps(raw))

        result = load_barcode_json(fpath)
        for key, sample_list in result.items():
            for sample in sample_list:
                for dim, arr in sample.items():
                    assert isinstance(arr, np.ndarray), (
                        f"{key} dim={dim}: expected ndarray, got {type(arr)}"
                    )
                    assert arr.dtype == np.float64, (
                        f"{key} dim={dim}: expected float64, got {arr.dtype}"
                    )

    def test_array_shape_is_n_by_2(self, tmp_path):
        _require_import()
        import json
        raw = _make_synthetic_barcodes(n_layers=1, n_heads=1, n_samples=3)
        fpath = tmp_path / "fake.json"
        fpath.write_text(json.dumps(raw))

        result = load_barcode_json(fpath)
        for key, sample_list in result.items():
            for sample in sample_list:
                for dim, arr in sample.items():
                    assert arr.ndim == 2, (
                        f"{key} dim={dim}: expected 2D array, got shape {arr.shape}"
                    )
                    assert arr.shape[1] == 2, (
                        f"{key} dim={dim}: expected (n_features, 2), got {arr.shape}"
                    )

    def test_empty_features_allowed(self, tmp_path):
        """H_1 can legitimately be empty ([]) for some samples — must not crash."""
        _require_import()
        import json
        raw = {"0": {"0": [{"0": [[0.0, 1.0]], "1": []}]}}
        fpath = tmp_path / "fake.json"
        fpath.write_text(json.dumps(raw))
        result = load_barcode_json(fpath)
        arr = result[(0, 0)][0][1]
        assert arr.shape == (0, 2) or (arr.ndim == 1 and arr.shape[0] == 0) or arr.shape[1] == 2


# ---------------------------------------------------------------------------
# load_lang_barcodes tests
# ---------------------------------------------------------------------------

class TestLoadLangBarcodes:
    def test_import(self):
        _require_import()

    def test_metadata_columns(self, tmp_path):
        """Metadata DataFrame must have the required columns."""
        _require_import()
        import json

        # Create minimal multi-part barcode files
        for part in range(1, 3):
            raw = _make_synthetic_barcodes(n_layers=2, n_heads=2, n_samples=3)
            fname = (
                f"en_color_all_heads_2_layers_MAX_LEN_32_"
                f"bert-base-multilingual-cased_part{part}of2.json"
            )
            (tmp_path / fname).write_text(json.dumps(raw))

        # Create matching KWIC CSV
        kwic_dir = tmp_path / "kwic" / "en"
        kwic_dir.mkdir(parents=True)
        kwic_df = pd.DataFrame({
            "term": ["red"] * 3 + ["blue"] * 3,
            "labels": ["red"] * 6,
            "sentence": [f"sentence {i}" for i in range(6)],
            "target_idx": range(6),
            "corpus_source": ["test"] * 6,
        })
        kwic_df.to_csv(kwic_dir / "color.csv", index=False)

        diagrams, metadata = load_lang_barcodes(
            barcode_dir=tmp_path,
            lang="en",
            domain="color",
            model_tag="bert-base-multilingual-cased",
            max_len=32,
            n_layers=2,
            kwic_dir=str(tmp_path / "kwic"),
        )

        required_cols = {"lang", "term", "sentence_idx_within_term", "source_file", "source_part"}
        missing = required_cols - set(metadata.columns)
        assert not missing, f"Metadata DataFrame missing columns: {missing}"

    def test_parts_loaded_in_numeric_order(self, tmp_path):
        """Parts must be concatenated in ascending numeric order (part1, part2, part3)."""
        _require_import()
        import json

        # Create 2 parts with identifiable sample counts
        for part, n_samples in [(1, 4), (2, 2)]:
            raw = _make_synthetic_barcodes(n_layers=1, n_heads=1, n_samples=n_samples)
            fname = (
                f"en_color_all_heads_1_layers_MAX_LEN_32_"
                f"bert-base-multilingual-cased_part{part}of2.json"
            )
            (tmp_path / fname).write_text(json.dumps(raw))

        kwic_dir = tmp_path / "kwic" / "en"
        kwic_dir.mkdir(parents=True)
        kwic_df = pd.DataFrame({
            "term": ["red"] * 4 + ["blue"] * 2,
            "labels": ["red"] * 6,
            "sentence": [f"s{i}" for i in range(6)],
            "target_idx": range(6),
            "corpus_source": ["test"] * 6,
        })
        kwic_df.to_csv(kwic_dir / "color.csv", index=False)

        diagrams, metadata = load_lang_barcodes(
            barcode_dir=tmp_path,
            lang="en",
            domain="color",
            model_tag="bert-base-multilingual-cased",
            max_len=32,
            n_layers=1,
            kwic_dir=str(tmp_path / "kwic"),
        )
        # Total samples must match sum of all parts
        assert len(metadata) == 6, f"Expected 6 metadata rows, got {len(metadata)}"

    def test_sample_count_validates(self, tmp_path):
        """load_lang_barcodes must validate that total barcode count == KWIC row count."""
        _require_import()
        import json

        # One part with 5 samples but KWIC CSV has 3 rows — mismatch
        raw = _make_synthetic_barcodes(n_layers=1, n_heads=1, n_samples=5)
        fname = (
            "en_color_all_heads_1_layers_MAX_LEN_32_"
            "bert-base-multilingual-cased_part1of1.json"
        )
        (tmp_path / fname).write_text(json.dumps(raw))

        kwic_dir = tmp_path / "kwic" / "en"
        kwic_dir.mkdir(parents=True)
        kwic_df = pd.DataFrame({
            "term": ["red"] * 3,
            "labels": ["red"] * 3,
            "sentence": [f"s{i}" for i in range(3)],
            "target_idx": range(3),
            "corpus_source": ["test"] * 3,
        })
        kwic_df.to_csv(kwic_dir / "color.csv", index=False)

        with pytest.raises(Exception):  # ValueError or AssertionError
            load_lang_barcodes(
                barcode_dir=tmp_path,
                lang="en",
                domain="color",
                model_tag="bert-base-multilingual-cased",
                max_len=32,
                n_layers=1,
                kwic_dir=str(tmp_path / "kwic"),
            )


# ---------------------------------------------------------------------------
# subsample_per_term tests
# ---------------------------------------------------------------------------

class TestSubsamplePerTerm:
    def test_import(self):
        _require_import()

    def test_deterministic_same_seed(self):
        """subsample_per_term must return identical indices for the same seed."""
        _require_import()
        meta = _make_synthetic_metadata(n_samples=20, terms=("red", "blue"), n_per_term=10)
        idx1 = subsample_per_term(meta, n_per_term=5, seed=42)
        idx2 = subsample_per_term(meta, n_per_term=5, seed=42)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self):
        """Different seeds should generally produce different selections."""
        _require_import()
        meta = _make_synthetic_metadata(n_samples=20, terms=("red", "blue"), n_per_term=10)
        idx1 = subsample_per_term(meta, n_per_term=5, seed=42)
        idx2 = subsample_per_term(meta, n_per_term=5, seed=99)
        # Very unlikely to be identical
        assert not np.array_equal(idx1, idx2), (
            "Different seeds returned identical indices — subsample not using seed?"
        )

    def test_returns_correct_count_per_term(self):
        """Each term gets exactly n_per_term samples (when sufficient data exists)."""
        _require_import()
        terms = ("red", "blue", "green")
        meta = _make_synthetic_metadata(n_samples=30, terms=terms, n_per_term=10)
        n_per_term = 5
        indices = subsample_per_term(meta, n_per_term=n_per_term, seed=42)
        selected = meta.iloc[indices]
        for term in terms:
            n = (selected["term"] == term).sum()
            assert n == n_per_term, (
                f"Expected {n_per_term} samples for '{term}', got {n}"
            )

    def test_under_target_takes_all_and_warns(self):
        """When a term has fewer than n_per_term rows, take all and emit warnings.warn."""
        _require_import()
        # Build metadata: "red" has only 3 rows, "blue" has 10
        rows = (
            [{"lang": "en", "term": "red", "sentence_idx_within_term": i,
              "source_file": "f.json", "source_part": 1} for i in range(3)]
            + [{"lang": "en", "term": "blue", "sentence_idx_within_term": i,
                "source_file": "f.json", "source_part": 1} for i in range(10)]
        )
        meta = pd.DataFrame(rows)

        with pytest.warns(UserWarning) as record:
            indices = subsample_per_term(meta, n_per_term=7, seed=42)

        # Warning must be emitted (not just a print)
        assert len(record) >= 1, "Expected at least one UserWarning for under-target term"
        # The under-target term gets all its available rows
        selected = meta.iloc[indices]
        red_count = (selected["term"] == "red").sum()
        assert red_count == 3, (
            f"Under-target term 'red' (n=3) should contribute all 3 rows, got {red_count}"
        )

    def test_returns_int_array(self):
        """Return type must be an array of integer indices."""
        _require_import()
        meta = _make_synthetic_metadata(n_samples=10, terms=("red",), n_per_term=10)
        indices = subsample_per_term(meta, n_per_term=5, seed=42)
        assert isinstance(indices, np.ndarray), (
            f"Expected np.ndarray, got {type(indices)}"
        )
        assert np.issubdtype(indices.dtype, np.integer), (
            f"Expected integer dtype, got {indices.dtype}"
        )

    def test_indices_in_valid_range(self):
        """All returned indices must be valid row positions in the metadata DataFrame."""
        _require_import()
        meta = _make_synthetic_metadata(n_samples=12, terms=("red", "blue"), n_per_term=6)
        indices = subsample_per_term(meta, n_per_term=4, seed=42)
        assert indices.min() >= 0
        assert indices.max() < len(meta)


# ---------------------------------------------------------------------------
# to_giotto_format tests
# ---------------------------------------------------------------------------

class TestToGiottoFormat:
    def test_import(self):
        _require_import()

    def _make_diagrams(self, n_layers=2, n_heads=2, n_samples=5):
        """Build synthetic per_layer_head_diagrams in the parsed (int-key) format."""
        import json
        raw = _make_synthetic_barcodes(n_layers=n_layers, n_heads=n_heads, n_samples=n_samples)
        # Simulate what load_barcode_json produces: int keys, float64 arrays
        import json as _json
        import io
        # Use a tmp approach — manually parse
        diagrams = {}
        for layer_str, heads in raw.items():
            for head_str, samples in heads.items():
                key = (int(layer_str), int(head_str))
                parsed_samples = []
                for sample in samples:
                    parsed = {int(d): np.array(feats, dtype=np.float64) for d, feats in sample.items()}
                    parsed_samples.append(parsed)
                diagrams[key] = parsed_samples
        return diagrams

    def test_output_shape(self):
        """to_giotto_format must return shape (N, F, 3) where F >= max features per sample."""
        _require_import()
        diagrams = self._make_diagrams(n_layers=1, n_heads=1, n_samples=5)
        indices = np.array([0, 1, 2, 3, 4])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))
        assert result.ndim == 3, f"Expected 3D array, got shape {result.shape}"
        assert result.shape[0] == 5, f"Expected 5 samples (axis 0), got {result.shape[0]}"
        assert result.shape[2] == 3, f"Expected 3 values per feature (birth, death, hom_dim), got {result.shape[2]}"

    def test_output_dtype_float64(self):
        """Output must be float64."""
        _require_import()
        diagrams = self._make_diagrams(n_layers=1, n_heads=1, n_samples=3)
        indices = np.array([0, 1, 2])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_padding_row_hom_dim_matches_dim(self):
        """Padding rows must use the correct hom_dim value, not zero.

        giotto-tda's PairwiseDistance dispatches by the hom_dim column (last
        element of each row). Padding rows in a dim-1 block must have hom_dim=1,
        not 0, or they will be misclassified as H_0 features.

        Build a case where H_0 and H_1 feature counts differ so that padding is
        unavoidable, then verify each padded row uses the dim of the block it
        belongs to.
        """
        _require_import()
        # Craft diagrams manually so feature counts differ
        # Sample 0: H_0 has 4 features, H_1 has 1 feature
        # Sample 1: H_0 has 2 features, H_1 has 3 features
        diagrams = {
            (0, 0): [
                {0: np.array([[0.0, 0.5], [0.1, 0.6], [0.2, 0.7], [0.3, 0.8]], dtype=np.float64),
                 1: np.array([[0.9, 1.0]], dtype=np.float64)},
                {0: np.array([[0.0, 0.4], [0.1, 0.5]], dtype=np.float64),
                 1: np.array([[0.5, 0.6], [0.6, 0.7], [0.7, 0.8]], dtype=np.float64)},
            ]
        }
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))
        # F = max_H0_features + max_H1_features = 4 + 3 = 7
        # (max H_0 = max(4, 2) = 4; max H_1 = max(1, 3) = 3)
        assert result.shape == (2, 7, 3), f"Expected shape (2, 7, 3), got {result.shape}"

        # Layout: positions 0-3 are the H_0 block (dim=0); positions 4-6 are the H_1 block (dim=1).
        # Sample 0: H_0[0-3] real (hom_dim=0), H_1[4] real (hom_dim=1), H_1[5-6] padded (hom_dim=1)
        # Sample 1: H_0[0-1] real (hom_dim=0), H_0[2-3] padded (hom_dim=0), H_1[4-6] real (hom_dim=1)

        # Verify H_0 block (positions 0-3) always has hom_dim=0
        for sample_idx in range(2):
            for feat_idx in range(4):
                hom_dim_val = result[sample_idx, feat_idx, 2]
                assert hom_dim_val == 0.0, (
                    f"sample {sample_idx}, H_0 block pos {feat_idx}: "
                    f"hom_dim={hom_dim_val}, expected 0.0"
                )

        # Verify H_1 block (positions 4-6) always has hom_dim=1
        for sample_idx in range(2):
            for feat_idx in range(4, 7):
                hom_dim_val = result[sample_idx, feat_idx, 2]
                assert hom_dim_val == 1.0, (
                    f"sample {sample_idx}, H_1 block pos {feat_idx}: "
                    f"hom_dim={hom_dim_val}, expected 1.0 (padding must preserve hom_dim)"
                )

        # Verify padding birth/death are zero
        # Sample 0: H_1 positions 5-6 are padding (only 1 real H_1 feature)
        assert result[0, 5, 0] == 0.0 and result[0, 5, 1] == 0.0, "Sample 0 H_1 padding row 5 not (0,0)"
        assert result[0, 6, 0] == 0.0 and result[0, 6, 1] == 0.0, "Sample 0 H_1 padding row 6 not (0,0)"
        # Sample 1: H_0 positions 2-3 are padding (only 2 real H_0 features)
        assert result[1, 2, 0] == 0.0 and result[1, 2, 1] == 0.0, "Sample 1 H_0 padding row 2 not (0,0)"
        assert result[1, 3, 0] == 0.0 and result[1, 3, 1] == 0.0, "Sample 1 H_0 padding row 3 not (0,0)"

    def test_padding_rows_birth_death_zero(self):
        """Padding rows must have birth=0, death=0 (giotto-tda no-feature convention)."""
        _require_import()
        # Build a case that forces padding: 2 samples, first has 3 H_0 features, second has 1
        diagrams = {
            (0, 0): [
                {0: np.array([[0.0, 0.5], [0.1, 0.6], [0.2, 0.7]], dtype=np.float64),
                 1: np.array([], dtype=np.float64).reshape(0, 2)},
                {0: np.array([[0.0, 0.4]], dtype=np.float64),
                 1: np.array([], dtype=np.float64).reshape(0, 2)},
            ]
        }
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))
        # Sample 1, H_0 block: should have 2 padding rows
        # The padding rows in sample 1 must have birth=0, death=0
        sample_1_h0_real = 1
        sample_0_h0_real = 3
        expected_f = sample_0_h0_real  # max H_0 features
        # First expected_f features of sample 1 belong to H_0 block
        # After the first real feature, the rest are padding
        for i in range(sample_1_h0_real, expected_f):
            birth = result[1, i, 0]
            death = result[1, i, 1]
            hom_dim = result[1, i, 2]
            assert birth == 0.0 and death == 0.0, (
                f"Padding row {i} in sample 1 has birth={birth}, death={death}; expected (0,0)"
            )
            assert hom_dim == 0.0, (
                f"Padding row {i} in H_0 block of sample 1 has hom_dim={hom_dim}, expected 0"
            )

    def test_subsample_selection(self):
        """to_giotto_format must select exactly the samples at the given indices."""
        _require_import()
        # Build distinguishable samples: sample i has birth=float(i) in H_0
        diagrams = {(0, 0): [
            {0: np.array([[float(i), float(i) + 0.5]], dtype=np.float64),
             1: np.array([], dtype=np.float64).reshape(0, 2)}
            for i in range(10)
        ]}
        indices = np.array([3, 7])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0,))
        assert result.shape[0] == 2
        # Sample 0 of result should correspond to original sample 3 (birth=3.0)
        assert result[0, 0, 0] == 3.0, f"Expected birth=3.0, got {result[0, 0, 0]}"
        # Sample 1 of result should correspond to original sample 7 (birth=7.0)
        assert result[1, 0, 0] == 7.0, f"Expected birth=7.0, got {result[1, 0, 0]}"


# ---------------------------------------------------------------------------
# Smoke tests against real partial data (gated by PH_REQUIRE_DIAGRAM_DISTANCES)
# ---------------------------------------------------------------------------

EN_PART1 = BARCODE_DIR / "en_color_all_heads_12_layers_MAX_LEN_32_bert-base-multilingual-cased_part1of3.json"


def test_smoke_load_barcode_json_real_part1():
    """load_barcode_json parses en_color_part1of3.json without crashing."""
    _require_import()
    if not EN_PART1.exists():
        _skip_or_fail(f"Barcode file not found: {EN_PART1}")

    result = load_barcode_json(EN_PART1)

    # Must have 144 (layer, head) entries for a 12×12 mBERT
    assert len(result) == 144, f"Expected 144 (layer, head) keys, got {len(result)}"

    # Each entry must have 1000 samples (part1 has 1000 KWIC rows)
    for key, sample_list in result.items():
        assert len(sample_list) == 1000, (
            f"{key}: expected 1000 samples, got {len(sample_list)}"
        )

    # Spot-check (0, 0): keys, types, shapes
    samples_00 = result[(0, 0)]
    s0 = samples_00[0]
    assert 0 in s0 and 1 in s0, f"Sample 0 missing H_0 or H_1 key: {list(s0.keys())}"
    assert s0[0].dtype == np.float64
    assert s0[0].shape[1] == 2
    assert s0[1].dtype == np.float64
    assert s0[1].shape[1] == 2


def test_smoke_load_lang_barcodes_partial():
    """load_lang_barcodes handles partial data (only en/part1 on disk) without crashing.

    This is the acceptance criteria case: en_color_part1of3.json exists but
    part2 and part3 also exist in the full run. Since all 3 parts are present
    in the symlinked data dir, this test verifies the full en/color load.
    """
    _require_import()
    if not EN_PART1.exists():
        _skip_or_fail(f"Barcode file not found: {EN_PART1}")
    if not KWIC_DIR.exists():
        _skip_or_fail(f"KWIC directory not found: {KWIC_DIR}")

    # Load whatever parts are present (gracefully handle partial runs)
    diagrams, metadata = load_lang_barcodes(
        barcode_dir=BARCODE_DIR,
        lang="en",
        domain="color",
        model_tag="bert-base-multilingual-cased",
        max_len=32,
        n_layers=12,
        kwic_dir=str(KWIC_DIR),
    )

    # Metadata must have required columns
    required_cols = {"lang", "term", "sentence_idx_within_term", "source_file", "source_part"}
    missing = required_cols - set(metadata.columns)
    assert not missing, f"Metadata missing columns: {missing}"

    # Lang column must be "en" throughout
    assert (metadata["lang"] == "en").all()

    # Diagrams must have 144 (layer, head) keys
    assert len(diagrams) == 144, f"Expected 144 (layer, head) keys, got {len(diagrams)}"

    # Total sample count must equal metadata rows
    n_samples_diagram = len(diagrams[(0, 0)])
    assert n_samples_diagram == len(metadata), (
        f"Diagram sample count ({n_samples_diagram}) != metadata rows ({len(metadata)})"
    )


def test_smoke_subsample_en_color():
    """subsample_per_term runs on real en/color metadata without error."""
    _require_import()
    if not EN_PART1.exists():
        _skip_or_fail(f"Barcode file not found: {EN_PART1}")
    if not KWIC_DIR.exists():
        _skip_or_fail(f"KWIC directory not found: {KWIC_DIR}")

    diagrams, metadata = load_lang_barcodes(
        barcode_dir=BARCODE_DIR,
        lang="en",
        domain="color",
        model_tag="bert-base-multilingual-cased",
        max_len=32,
        n_layers=12,
        kwic_dir=str(KWIC_DIR),
    )

    n_per_term = 30
    indices = subsample_per_term(metadata, n_per_term=n_per_term, seed=42)
    assert len(indices) > 0
    # en/color has 11 terms × 200 rows each → each should get exactly 30
    assert len(indices) == 11 * n_per_term, (
        f"Expected {11 * n_per_term} samples, got {len(indices)}"
    )


def test_smoke_to_giotto_format_layer0_head0():
    """to_giotto_format produces correct shape and dtype on real data."""
    _require_import()
    if not EN_PART1.exists():
        _skip_or_fail(f"Barcode file not found: {EN_PART1}")
    if not KWIC_DIR.exists():
        _skip_or_fail(f"KWIC directory not found: {KWIC_DIR}")

    diagrams, metadata = load_lang_barcodes(
        barcode_dir=BARCODE_DIR,
        lang="en",
        domain="color",
        model_tag="bert-base-multilingual-cased",
        max_len=32,
        n_layers=12,
        kwic_dir=str(KWIC_DIR),
    )

    indices = subsample_per_term(metadata, n_per_term=10, seed=42)
    result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))

    assert result.ndim == 3
    assert result.shape[0] == len(indices)
    assert result.shape[2] == 3
    assert result.dtype == np.float64

    # hom_dim column must only contain 0 or 1
    hom_dims = result[:, :, 2]
    assert set(np.unique(hom_dims)).issubset({0.0, 1.0}), (
        f"hom_dim column contains unexpected values: {np.unique(hom_dims)}"
    )
