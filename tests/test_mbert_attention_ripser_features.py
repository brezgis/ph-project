"""Tests for mbert_attention_ripser.ipynb feature outputs — cross-linguistic.

These tests verify that the adapted ripser+templates notebook produces correctly-shaped
feature tensors for each (lang, domain) combination in scope. The notebook
iterates over (lang in {en, ru, es}) × (domain in {color}) — the May 2026
rescoping restricted analysis to color only; see CLAUDE.md and bd show
ph-project-mwk for the rescoping note.

Ripser feature expected shape: (12, 12, 14, N_kwic)
  - 12 layers × 12 heads × 14 ripser features × N_kwic sentences
  - 14 = len(ripser_feature_names) in cell 31 of mbert_attention_ripser.ipynb
  - N_kwic varies per (lang, domain): en/color ~2200, ru/color ~2267, es/color ~2161

Template feature expected shape: (12, 12, 6, N_kwic)
  - 12 layers × 12 heads × 6 template features × N_kwic sentences
  - 6 = len(['self', 'beginning', 'prev', 'next', 'comma', 'dot'])

We do NOT assert the exact N_kwic dimension because the KWIC CSV may be
regenerated with a different sample count. Instead we check that the shape
prefix (12, 12, 14) or (12, 12, 6) is correct and that N_kwic matches the
CSV row count exactly. The exact row count is read dynamically from the CSV
so the assertion is always in sync with the data on disk.

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

RIPSER_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "phase3_ripser",
)

TEMPLATE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "phase3_templates",
)

KWIC_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "kwic",
)

# Ripser feature dimension — must match cell 31 of mbert_attention_ripser.ipynb
RIPSER_N_FEATURES = 14  # must match cell 31 of mbert_attention_ripser.ipynb

# Template feature dimension — len(['self', 'beginning', 'prev', 'next', 'comma', 'dot'])
TEMPLATE_N_FEATURES = 6


def _kwic_row_count(lang: str, domain: str) -> int:
    """Return number of rows in the KWIC CSV for (lang, domain).

    Raises FileNotFoundError if the CSV is missing — surfaces config errors
    clearly instead of letting a 0 propagate into a confusing shape mismatch.
    Also asserts SCHEMA.md guarantees: no NaN or empty `sentence` rows. A
    SCHEMA violation here would otherwise show up as a phantom shape diff.
    """
    csv_path = os.path.join(KWIC_DIR, lang, f"{domain}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KWIC CSV missing: {csv_path}")
    df = pd.read_csv(csv_path)
    assert df["sentence"].notna().all(), (
        f"KWIC CSV {csv_path} has NaN `sentence` rows — violates data/kwic/SCHEMA.md."
    )
    assert (df["sentence"].astype(str).str.len() > 0).all(), (
        f"KWIC CSV {csv_path} has empty `sentence` rows — violates data/kwic/SCHEMA.md."
    )
    return len(df)


def _find_features_file(search_dir: str, lang: str, domain: str, suffix: str) -> "str | None":
    """Return path to a feature .npy for (lang, domain) matching suffix, or None if absent."""
    if not os.path.isdir(search_dir):
        return None
    prefix = f"{lang}_{domain}_all_heads"
    for fname in os.listdir(search_dir):
        if fname.startswith(prefix) and fname.endswith(suffix + ".npy"):
            return os.path.join(search_dir, fname)
    return None


def _missing(lang: str, domain: str) -> None:
    """Either skip or fail when a feature file is absent, per env var."""
    msg = f"Features file for ({lang!r}, {domain!r}) not yet produced."
    if REQUIRE_FEATURES:
        pytest.fail(msg + " (PH_REQUIRE_FEATURES=1)")
    pytest.skip(msg)


# ---------------------------------------------------------------------------
# Ripser feature tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_ripser_features_shape(lang, domain):
    """Ripser feature tensor must have shape (12, 12, 14, N_kwic).

    N_kwic is read dynamically from the KWIC CSV so this assertion stays in
    sync even if the CSV is regenerated. We check the prefix (12, 12, 14) and
    that N_kwic matches the CSV row count exactly.

    Skips when the feature file does not yet exist (set PH_REQUIRE_FEATURES=1
    to turn skips into failures).
    """
    path = _find_features_file(RIPSER_DIR, lang, domain, "_ripser")
    if path is None:
        _missing(lang, domain)
    # Pin the model identity in the resolved filename. Catches the case
    # where a stale .npy from an earlier model (e.g. bert-base-uncased
    # during development) lingers and silently passes shape checks.
    assert "bert-base-multilingual-cased" in os.path.basename(path), (
        f"({lang!r}, {domain!r}): resolved ripser file {path!r} does not encode "
        f"`bert-base-multilingual-cased` in its name — wrong model checkpoint?"
    )
    arr = np.load(path, allow_pickle=True)
    n_kwic = _kwic_row_count(lang, domain)
    expected_shape = (12, 12, RIPSER_N_FEATURES, n_kwic)
    assert arr.shape == expected_shape, (
        f"({lang!r}, {domain!r}): expected ripser shape {expected_shape}, got {arr.shape}"
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_ripser_features_no_inf(lang, domain):
    """Ripser feature tensor must not contain +/-inf values."""
    path = _find_features_file(RIPSER_DIR, lang, domain, "_ripser")
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True).astype(float)
    n_inf = np.sum(np.isinf(arr))
    assert n_inf == 0, (
        f"({lang!r}, {domain!r}): found {n_inf} +/-inf values in ripser feature tensor."
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_ripser_features_not_all_zero(lang, domain):
    """No (layer, head) slice should be entirely zero (would indicate a ripser bug)."""
    path = _find_features_file(RIPSER_DIR, lang, domain, "_ripser")
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True)
    # Shape: (12, 12, 14, N_kwic) — check that no (layer, head) slice is all-zero
    for layer in range(arr.shape[0]):
        for head in range(arr.shape[1]):
            slice_ = arr[layer, head]
            assert np.any(slice_ != 0), (
                f"({lang!r}, {domain!r}), layer={layer}, head={head}: "
                f"ripser feature slice is entirely zero."
            )


# ---------------------------------------------------------------------------
# Template feature tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_template_features_shape(lang, domain):
    """Template feature tensor must have shape (12, 12, 6, N_kwic).

    N_kwic is read dynamically from the KWIC CSV so this assertion stays in
    sync even if the CSV is regenerated. We check the prefix (12, 12, 6) and
    that N_kwic matches the CSV row count exactly.

    Skips when the feature file does not yet exist (set PH_REQUIRE_FEATURES=1
    to turn skips into failures).
    """
    path = _find_features_file(TEMPLATE_DIR, lang, domain, "_template")
    if path is None:
        _missing(lang, domain)
    # Pin the model identity in the resolved filename.
    assert "bert-base-multilingual-cased" in os.path.basename(path), (
        f"({lang!r}, {domain!r}): resolved template file {path!r} does not encode "
        f"`bert-base-multilingual-cased` in its name — wrong model checkpoint?"
    )
    arr = np.load(path, allow_pickle=True)
    n_kwic = _kwic_row_count(lang, domain)
    expected_shape = (12, 12, TEMPLATE_N_FEATURES, n_kwic)
    assert arr.shape == expected_shape, (
        f"({lang!r}, {domain!r}): expected template shape {expected_shape}, got {arr.shape}"
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_template_features_no_inf(lang, domain):
    """Template feature tensor must not contain +/-inf values."""
    path = _find_features_file(TEMPLATE_DIR, lang, domain, "_template")
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True).astype(float)
    n_inf = np.sum(np.isinf(arr))
    assert n_inf == 0, (
        f"({lang!r}, {domain!r}): found {n_inf} +/-inf values in template feature tensor."
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_template_features_not_all_zero(lang, domain):
    """No (layer, head) slice should be entirely zero (would indicate a template feature bug)."""
    path = _find_features_file(TEMPLATE_DIR, lang, domain, "_template")
    if path is None:
        _missing(lang, domain)
    arr = np.load(path, allow_pickle=True)
    # Shape: (12, 12, 6, N_kwic) — check that no (layer, head) slice is all-zero
    for layer in range(arr.shape[0]):
        for head in range(arr.shape[1]):
            slice_ = arr[layer, head]
            assert np.any(slice_ != 0), (
                f"({lang!r}, {domain!r}), layer={layer}, head={head}: "
                f"template feature slice is entirely zero."
            )
