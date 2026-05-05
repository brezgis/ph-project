"""Tests for phase3_thresholds.ipynb feature outputs — cross-linguistic.

These tests verify that the adapted threshold notebook produces correctly-shaped
feature tensors for each (lang, domain) combination in scope. The notebook
iterates over (lang in {en, ru, es}) × (domain in {color}) — the May 2026
rescoping restricted analysis to color only; see CLAUDE.md and bd show
ph-project-mwk for the rescoping note.

Expected shape: (12, 12, 6, N_kwic, 6)
  - 12 layers × 12 heads × 6 features × N_kwic sentences × 6 thresholds
  - N_kwic varies per (lang, domain): en/color ~2200, ru/color ~2267, es/color ~2161

We do NOT assert the exact middle dimension (N_kwic) because the KWIC CSV may
be regenerated with a different sample count. Instead we check that the shape
prefix (12, 12, 6) and suffix (6,) are correct, and that N_kwic > 0. The exact
row count is read dynamically from the CSV so the assertion is always in sync
with the data on disk.

By default, per-(lang, domain) checks skip when the .npy file is absent so
local development stays green before the notebook has been run. Set
PH_REQUIRE_FEATURES=1 to flip absent files into hard failures — useful for CI
or post-run verification where the files MUST exist.
"""
import os
import numpy as np
import pandas as pd
import pytest

REQUIRE_FEATURES = os.environ.get("PH_REQUIRE_FEATURES") == "1"

# ---------------------------------------------------------------------------
# Scope — color only per 2026-05-04 rescoping decision.
# To re-enable emotion/kinship after May 6, add them to DOMAINS below.
# See CLAUDE.md ("Current scope") and `bd show ph-project-mwk` for context.
# ---------------------------------------------------------------------------
LANGS = ["en", "ru", "es"]
DOMAINS = ["color"]  # one-line re-enable: ["color", "emotion", "kinship"]

FEATURES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "phase3_thresholds",
)

KWIC_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "kwic",
)

EXPECTED_SHAPE_PREFIX = (12, 12, 6)   # layers, heads, features
EXPECTED_SHAPE_SUFFIX = (6,)           # thresholds


def _kwic_row_count(lang: str, domain: str) -> int:
    """Return number of rows in the KWIC CSV for (lang, domain)."""
    csv_path = os.path.join(KWIC_DIR, lang, f"{domain}.csv")
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path)
    return len(df)


def _find_features_file(lang: str, domain: str) -> str | None:
    """Return path to the features .npy for (lang, domain), or None if absent."""
    if not os.path.isdir(FEATURES_DIR):
        return None
    prefix = f"{lang}_{domain}_all_heads"
    for fname in os.listdir(FEATURES_DIR):
        if fname.startswith(prefix) and fname.endswith(".npy"):
            return os.path.join(FEATURES_DIR, fname)
    return None


def _missing(lang: str, domain: str) -> None:
    """Either skip or fail when a feature file is absent, per env var."""
    msg = f"Features file for ({lang!r}, {domain!r}) not yet produced."
    if REQUIRE_FEATURES:
        pytest.fail(msg + " (PH_REQUIRE_FEATURES=1)")
    pytest.skip(msg)


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_features_shape(lang, domain):
    """Feature tensor must have shape (12, 12, 6, N_kwic, 6).

    N_kwic is read dynamically from the KWIC CSV so this assertion stays in
    sync even if the CSV is regenerated. We check the prefix (12, 12, 6), the
    suffix (6,), and that N_kwic matches the CSV row count exactly.

    Skips when the feature file does not yet exist (set PH_REQUIRE_FEATURES=1
    to turn skips into failures).
    """
    path = _find_features_file(lang, domain)
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True)
    n_kwic = _kwic_row_count(lang, domain)
    expected_shape = EXPECTED_SHAPE_PREFIX + (n_kwic,) + EXPECTED_SHAPE_SUFFIX
    assert arr.shape == expected_shape, (
        f"({lang!r}, {domain!r}): expected shape {expected_shape}, got {arr.shape}"
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_features_no_inf(lang, domain):
    """Feature tensor must not contain +/-inf values."""
    path = _find_features_file(lang, domain)
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True).astype(float)
    n_inf = np.sum(np.isinf(arr))
    assert n_inf == 0, (
        f"({lang!r}, {domain!r}): found {n_inf} +/-inf values in feature tensor."
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_features_not_all_zero(lang, domain):
    """No (layer, head) slice should be entirely zero (would indicate a ripser bug)."""
    path = _find_features_file(lang, domain)
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True)
    # Shape: (12, 12, 6, N_kwic, 6) — check that no (layer, head) slice is all-zero
    for layer in range(arr.shape[0]):
        for head in range(arr.shape[1]):
            slice_ = arr[layer, head]
            assert np.any(slice_ != 0), (
                f"({lang!r}, {domain!r}), layer={layer}, head={head}: "
                f"feature slice is entirely zero."
            )
