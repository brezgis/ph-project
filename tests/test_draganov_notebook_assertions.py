"""Tests for logic that lives in draganov_color_per_term.ipynb.

Covers the same-language exclusion assertion added in ph-project-fjq, plus
the annotation-coordinate helper that determines label placement in Figure 1.

These are unit tests of the logic extracted from the notebook cells — they
do not execute the full notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fix 2: same-language exclusion assertion
# ---------------------------------------------------------------------------

def _make_ranking_df(rows: list[dict]) -> pd.DataFrame:
    cols = ['lang', 'term', 'distance', 'dim', 'rank',
            'target_lang', 'target_term', 'distance_value']
    return pd.DataFrame(rows, columns=cols)


def test_same_lang_assertion_passes_for_cross_language_rows():
    """ranking_df rows that all differ in lang vs target_lang satisfy the assertion."""
    df = _make_ranking_df([
        dict(lang='en', term='blue', distance='bottleneck', dim=0,
             rank=1, target_lang='ru', target_term='синий', distance_value=0.1),
        dict(lang='en', term='blue', distance='bottleneck', dim=0,
             rank=2, target_lang='es', target_term='azul', distance_value=0.2),
        dict(lang='ru', term='синий', distance='bottleneck', dim=0,
             rank=1, target_lang='en', target_term='blue', distance_value=0.1),
    ])
    # The assertion as written in the notebook must NOT raise
    assert (df['lang'] != df['target_lang']).all(), \
        "ranking must exclude same-language cells"


def test_same_lang_assertion_fails_when_same_language_row_present():
    """If a same-language row slips in, the assertion must fail (catch the bug)."""
    df = _make_ranking_df([
        # Cross-language row — fine
        dict(lang='en', term='blue', distance='bottleneck', dim=0,
             rank=1, target_lang='ru', target_term='синий', distance_value=0.1),
        # Same-language row — this is the bug the assertion catches
        dict(lang='en', term='blue', distance='bottleneck', dim=0,
             rank=2, target_lang='en', target_term='green', distance_value=0.05),
    ])
    # The assertion must evaluate to False (i.e., the bug is detectable)
    assert not (df['lang'] != df['target_lang']).all(), \
        "Expected assertion to be False when same-language row is present"


# ---------------------------------------------------------------------------
# Fix 1: heatmap annotation y-coordinate
# ---------------------------------------------------------------------------

def _compute_mid(cumpos: int, count: int) -> float:
    """Return the data-coordinate midpoint for a language block."""
    return cumpos + count / 2 - 0.5


def test_annotation_mid_is_raw_data_coordinate_not_normalized():
    """mid must be returned as-is, NOT divided by (n_cells - 1).

    n_cells=34, lang block sizes: en=11, es=11, ru=12 (sorted: en, es, ru).
    Expected midpoints in data coords:
      en: 0  + 11/2 - 0.5 =  5.0
      es: 11 + 11/2 - 0.5 = 16.0
      ru: 22 + 12/2 - 0.5 = 27.5
    """
    lang_order = ['en', 'es', 'ru']
    lang_counts = {'en': 11, 'es': 11, 'ru': 12}
    n_cells = 34

    cumpos = 0
    mids = []
    for lang in lang_order:
        cnt = lang_counts[lang]
        mid = _compute_mid(cumpos, cnt)
        mids.append(mid)
        cumpos += cnt

    expected_mids = [5.0, 16.0, 27.5]
    for lang, got, want in zip(lang_order, mids, expected_mids):
        assert got == pytest.approx(want), \
            f"mid for {lang}: expected {want}, got {got}"

    # The buggy code divided by (n_cells - 1) = 33 — verify that would be wrong.
    buggy_mids = [m / (n_cells - 1) for m in mids]
    # All buggy mids are in [0, 1), not in the data range [0, 33]
    for lang, bm in zip(lang_order, buggy_mids):
        assert bm < 1.0, \
            f"Buggy mid for {lang} should be < 1.0, got {bm}"
    # The buggy code stacks all labels at the top of a 34-row heatmap
    # (the data y range is [0, 33], so values in [0.15, 0.84] cluster near top)
    for bm in buggy_mids:
        assert 0.0 < bm < 1.0
