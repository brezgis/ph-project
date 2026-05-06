"""Tests for replication/giotto_format.py — to_giotto_format helper.

Covers the three acceptance criteria from ph-project-inu.5:
  - test_padding_uses_hom_dim: H_1 padding rows must have hom_dim=1, not 0.
  - test_shape: output shape is (N, max_H0 + max_H1, 3).
  - test_keyerror_on_missing_layer_head: unknown (layer, head) raises KeyError.

These tests are intentionally strict about the padding contract because the
giotto-tda bug they protect against is silent — wrong hom_dim in padding rows
causes PairwiseDistance to silently corrupt distance matrices.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the helper under test
# ---------------------------------------------------------------------------

try:
    from replication.giotto_format import to_giotto_format
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _require_import():
    if not _IMPORT_OK:
        pytest.fail(
            "Could not import replication.giotto_format — module not yet created."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unequal_diagrams():
    """Build diagrams where H_0 and H_1 feature counts differ between samples.

    Sample 0: H_0 has 3 features, H_1 has 1 feature.
    Sample 1: H_0 has 1 feature, H_1 has 3 features.

    This forces padding in both dim blocks, allowing us to verify that every
    padded row carries the correct hom_dim value for its block.
    """
    return {
        (0, 0): [
            {
                0: np.array([[0.0, 0.5], [0.1, 0.6], [0.2, 0.7]], dtype=np.float64),
                1: np.array([[0.9, 1.0]], dtype=np.float64),
            },
            {
                0: np.array([[0.0, 0.4]], dtype=np.float64),
                1: np.array([[0.5, 0.6], [0.6, 0.7], [0.7, 0.8]], dtype=np.float64),
            },
        ]
    }


# ---------------------------------------------------------------------------
# test_padding_uses_hom_dim
# ---------------------------------------------------------------------------

class TestPaddingUsesHomDim:
    """Padding rows must carry the hom_dim of their block, not 0."""

    def test_padding_uses_hom_dim(self):
        """H_1 padding rows must have arr[..., 2] == 1.0, not 0.0.

        This is the critical correctness property: if anyone reverts the
        padding to (0, 0, 0) instead of (0, 0, hom_dim), PairwiseDistance
        silently misclassifies H_1 padding rows as H_0 features and corrupts
        the resulting distance matrices.
        """
        _require_import()
        diagrams = _make_unequal_diagrams()
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))

        # Layout:
        #   positions 0-2: H_0 block (max H_0 = max(3, 1) = 3)
        #   positions 3-5: H_1 block (max H_1 = max(1, 3) = 3)
        assert result.shape == (2, 6, 3), (
            f"Expected shape (2, 6, 3), got {result.shape}. "
            "max_H0=3, max_H1=3, so F=6."
        )

        # Every cell in the H_1 block (positions 3-5) must have hom_dim == 1.0
        # regardless of whether it is a real feature or a padding row.
        h1_block = result[:, 3:6, 2]  # shape (2, 3), hom_dim column for H_1 block
        for sample_idx in range(2):
            for feat_idx in range(3):
                hom_dim_val = h1_block[sample_idx, feat_idx]
                assert hom_dim_val == 1.0, (
                    f"sample {sample_idx}, H_1 block position {feat_idx + 3}: "
                    f"hom_dim={hom_dim_val}, expected 1.0. "
                    "Padding rows in H_1 block MUST use hom_dim=1, not 0."
                )

        # Every cell in the H_0 block (positions 0-2) must have hom_dim == 0.0.
        h0_block = result[:, 0:3, 2]  # shape (2, 3), hom_dim column for H_0 block
        for sample_idx in range(2):
            for feat_idx in range(3):
                hom_dim_val = h0_block[sample_idx, feat_idx]
                assert hom_dim_val == 0.0, (
                    f"sample {sample_idx}, H_0 block position {feat_idx}: "
                    f"hom_dim={hom_dim_val}, expected 0.0."
                )

    def test_padding_birth_death_are_zero(self):
        """Padding rows must have birth=0 and death=0 (no-feature convention)."""
        _require_import()
        diagrams = _make_unequal_diagrams()
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))

        # Sample 0: H_0 has 3 real features → no H_0 padding; H_1 has 1 real → 2 padding rows
        # H_1 padding rows for sample 0 are at positions 4 and 5 (0-indexed from start of array)
        assert result[0, 4, 0] == 0.0 and result[0, 4, 1] == 0.0, (
            f"Sample 0, H_1 padding row at pos 4: birth={result[0, 4, 0]}, death={result[0, 4, 1]}"
        )
        assert result[0, 5, 0] == 0.0 and result[0, 5, 1] == 0.0, (
            f"Sample 0, H_1 padding row at pos 5: birth={result[0, 5, 0]}, death={result[0, 5, 1]}"
        )

        # Sample 1: H_0 has 1 real feature → 2 padding rows at positions 1 and 2
        assert result[1, 1, 0] == 0.0 and result[1, 1, 1] == 0.0, (
            f"Sample 1, H_0 padding row at pos 1: birth={result[1, 1, 0]}, death={result[1, 1, 1]}"
        )
        assert result[1, 2, 0] == 0.0 and result[1, 2, 1] == 0.0, (
            f"Sample 1, H_0 padding row at pos 2: birth={result[1, 2, 0]}, death={result[1, 2, 1]}"
        )


# ---------------------------------------------------------------------------
# test_shape
# ---------------------------------------------------------------------------

class TestShape:
    """Output shape is (N, total_features_padded, 3)."""

    def test_shape_basic(self):
        """Shape is (N, max_H0 + max_H1, 3) for two-dim diagrams."""
        _require_import()
        diagrams = _make_unequal_diagrams()
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))

        n = 2
        max_h0 = 3  # max(3, 1)
        max_h1 = 3  # max(1, 3)
        expected_shape = (n, max_h0 + max_h1, 3)
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result.shape}. "
            f"F = max_H0({max_h0}) + max_H1({max_h1}) = {max_h0 + max_h1}."
        )

    def test_shape_single_dim(self):
        """When dims=(0,), F = max_H0 features only."""
        _require_import()
        diagrams = {
            (0, 0): [
                {0: np.array([[0.0, 0.5], [0.1, 0.6]], dtype=np.float64)},
                {0: np.array([[0.0, 0.4], [0.1, 0.5], [0.2, 0.6]], dtype=np.float64)},
            ]
        }
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0,))

        n = 2
        max_h0 = 3  # max(2, 3)
        expected_shape = (n, max_h0, 3)
        assert result.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {result.shape}."
        )

    def test_shape_matches_subsample(self):
        """N in output shape equals len(sample_indices), not total samples."""
        _require_import()
        diagrams = {
            (0, 0): [
                {0: np.array([[float(i), float(i) + 0.5]], dtype=np.float64), 1: np.empty((0, 2))}
                for i in range(10)
            ]
        }
        indices = np.array([2, 5, 8])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))

        assert result.shape[0] == 3, (
            f"Expected N=3 (len of indices), got {result.shape[0]}."
        )
        assert result.shape[2] == 3, (
            f"Expected 3 columns (birth, death, hom_dim), got {result.shape[2]}."
        )

    def test_dtype_float64(self):
        """Output dtype must be float64."""
        _require_import()
        diagrams = _make_unequal_diagrams()
        indices = np.array([0, 1])
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))
        assert result.dtype == np.float64, (
            f"Expected dtype float64, got {result.dtype}."
        )


# ---------------------------------------------------------------------------
# test_keyerror_on_missing_layer_head
# ---------------------------------------------------------------------------

class TestKeyErrorOnMissingLayerHead:
    """Unknown (layer, head) must raise KeyError."""

    def test_keyerror_on_missing_layer_head(self):
        """Passing an unknown (layer, head) raises KeyError immediately."""
        _require_import()
        diagrams = _make_unequal_diagrams()  # only has key (0, 0)
        indices = np.array([0, 1])

        with pytest.raises(KeyError):
            to_giotto_format(diagrams, indices, layer=5, head=11, dims=(0, 1))

    def test_keyerror_message_contains_layer_head(self):
        """KeyError message should mention the missing layer and head."""
        _require_import()
        diagrams = _make_unequal_diagrams()
        indices = np.array([0, 1])

        with pytest.raises(KeyError, match="layer=3"):
            to_giotto_format(diagrams, indices, layer=3, head=7, dims=(0, 1))

    def test_keyerror_does_not_fire_on_valid_key(self):
        """Valid (layer, head) must NOT raise KeyError."""
        _require_import()
        diagrams = _make_unequal_diagrams()
        indices = np.array([0, 1])

        # Should not raise
        result = to_giotto_format(diagrams, indices, layer=0, head=0, dims=(0, 1))
        assert result is not None
