"""Tests for the length-matching used by scripts/language_effect.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from language_effect import length_matched_indices  # noqa: E402


def test_matched_subset_has_identical_histograms_per_language():
    rng = np.random.default_rng(0)
    langs = np.repeat(["en", "es", "ru"], 200)
    counts = np.concatenate([rng.integers(20, 30, 200), rng.integers(22, 34, 200), rng.integers(26, 45, 200)])
    idx = length_matched_indices(langs, counts, seed=1)
    hists = {
        lang: np.bincount(counts[idx][langs[idx] == lang], minlength=50) for lang in ("en", "es", "ru")
    }
    assert np.array_equal(hists["en"], hists["es"]) and np.array_equal(hists["en"], hists["ru"])
    assert len(idx) == len(set(idx.tolist())) and len(idx) > 0
    # token counts outside the common range are dropped entirely
    assert counts[idx].min() >= 26 and counts[idx].max() <= 29


def test_matching_is_deterministic_for_a_seed():
    langs = np.repeat(["en", "ru"], 50)
    counts = np.tile(np.arange(10), 10)
    assert np.array_equal(length_matched_indices(langs, counts, 3), length_matched_indices(langs, counts, 3))
