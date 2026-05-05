"""Tests for phase3_thresholds.ipynb embedding + manifest outputs (16v.2).

These tests verify that the adapted threshold notebook produces:
  1. Embedding .npy parts under data/phase3/embeddings/ with shape
     (batch_per_part, 32, 768) and dtype float16.
  2. A manifest parquet at data/phase3/embeddings/{lang}_{domain}_manifest.parquet
     with one row per KWIC sentence, `target_wp_start` populated for the vast
     majority of rows, and correct `kwic_row_id` values.

By default, checks skip when the embedding files are absent (notebook hasn't
been run yet). Set PH_REQUIRE_EMBEDDINGS=1 to turn skips into failures — useful
for CI or post-run acceptance verification.
"""
import os
import glob

import numpy as np
import pandas as pd
import pytest

REQUIRE_EMBEDDINGS = os.environ.get("PH_REQUIRE_EMBEDDINGS") == "1"

LANGS = ["en"]       # acceptance gate only requires en/color per task spec
DOMAINS = ["color"]

EMB_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "phase3",
    "embeddings",
)

KWIC_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "kwic",
)

MAX_LEN = 32
HIDDEN_SIZE = 768


def _kwic_row_count(lang: str, domain: str) -> int:
    csv_path = os.path.join(KWIC_DIR, lang, f"{domain}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"KWIC CSV missing: {csv_path}")
    return len(pd.read_csv(csv_path))


def _find_embedding_parts(lang: str, domain: str):
    """Return sorted list of embedding .npy part paths for (lang, domain)."""
    if not os.path.isdir(EMB_DIR):
        return []
    pattern = os.path.join(EMB_DIR, f"{lang}_{domain}_final_layer_MAX_LEN_*_bert-base-multilingual-cased_part*.npy")
    return sorted(glob.glob(pattern))


def _manifest_path(lang: str, domain: str) -> str:
    return os.path.join(EMB_DIR, f"{lang}_{domain}_manifest.parquet")


def _skip_or_fail(msg: str) -> None:
    if REQUIRE_EMBEDDINGS:
        pytest.fail(msg + " (PH_REQUIRE_EMBEDDINGS=1)")
    pytest.skip(msg)


# ---------------------------------------------------------------------------
# Embedding .npy part tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_embedding_parts_exist(lang, domain):
    """At least one embedding .npy part must exist for (lang, domain)."""
    parts = _find_embedding_parts(lang, domain)
    if not parts:
        _skip_or_fail(f"No embedding parts found for ({lang!r}, {domain!r}) under {EMB_DIR}")
    assert len(parts) >= 1, f"Expected at least 1 embedding part, found {len(parts)}"


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_embedding_parts_shape(lang, domain):
    """Each embedding part must have shape (N, 32, 768) for some N > 0."""
    parts = _find_embedding_parts(lang, domain)
    if not parts:
        _skip_or_fail(f"No embedding parts found for ({lang!r}, {domain!r})")
    for part_path in parts:
        arr = np.load(part_path)
        assert arr.ndim == 3, (
            f"{os.path.basename(part_path)}: expected 3D array, got ndim={arr.ndim}"
        )
        assert arr.shape[1] == MAX_LEN, (
            f"{os.path.basename(part_path)}: expected dim[1]={MAX_LEN}, got {arr.shape[1]}"
        )
        assert arr.shape[2] == HIDDEN_SIZE, (
            f"{os.path.basename(part_path)}: expected dim[2]={HIDDEN_SIZE}, got {arr.shape[2]}"
        )
        assert arr.shape[0] > 0, (
            f"{os.path.basename(part_path)}: batch dimension is 0"
        )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_embedding_parts_dtype(lang, domain):
    """Each embedding part must have dtype float16."""
    parts = _find_embedding_parts(lang, domain)
    if not parts:
        _skip_or_fail(f"No embedding parts found for ({lang!r}, {domain!r})")
    for part_path in parts:
        arr = np.load(part_path)
        assert arr.dtype == np.float16, (
            f"{os.path.basename(part_path)}: expected float16, got {arr.dtype}"
        )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_embedding_parts_total_rows_match_kwic(lang, domain):
    """Total rows across all embedding parts must equal the KWIC CSV row count."""
    parts = _find_embedding_parts(lang, domain)
    if not parts:
        _skip_or_fail(f"No embedding parts found for ({lang!r}, {domain!r})")
    n_kwic = _kwic_row_count(lang, domain)
    total = sum(np.load(p).shape[0] for p in parts)
    assert total == n_kwic, (
        f"({lang!r}, {domain!r}): embedding parts total {total} rows, "
        f"but KWIC CSV has {n_kwic} rows"
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_embedding_parts_finite(lang, domain):
    """All embedding values must be finite (no NaN or Inf)."""
    parts = _find_embedding_parts(lang, domain)
    if not parts:
        _skip_or_fail(f"No embedding parts found for ({lang!r}, {domain!r})")
    for part_path in parts:
        arr = np.load(part_path).astype(np.float32)
        n_bad = (~np.isfinite(arr)).sum()
        assert n_bad == 0, (
            f"{os.path.basename(part_path)}: found {n_bad} non-finite values"
        )


# ---------------------------------------------------------------------------
# Manifest parquet tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_exists(lang, domain):
    """Manifest parquet must exist for (lang, domain)."""
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    assert os.path.exists(path)


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_row_count_matches_kwic(lang, domain):
    """Manifest row count must equal the KWIC CSV row count."""
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    n_kwic = _kwic_row_count(lang, domain)
    df = pd.read_parquet(path)
    assert len(df) == n_kwic, (
        f"({lang!r}, {domain!r}): manifest has {len(df)} rows, "
        f"but KWIC CSV has {n_kwic} rows"
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_required_columns(lang, domain):
    """Manifest must contain all required columns."""
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    df = pd.read_parquet(path)
    required = {
        "kwic_row_id", "lang", "domain", "term",
        "target_idx", "target_wp_start", "target_wp_end",
        "embedding_part", "embedding_offset",
    }
    missing = required - set(df.columns)
    assert not missing, f"Manifest missing columns: {missing}"


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_kwic_row_ids_contiguous(lang, domain):
    """kwic_row_id values must be 0-indexed and contiguous (0..N-1)."""
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    df = pd.read_parquet(path)
    n = len(df)
    expected = list(range(n))
    actual = sorted(df["kwic_row_id"].tolist())
    assert actual == expected, (
        f"kwic_row_id values are not 0..{n-1}: got {actual[:10]}..."
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_target_wp_start_mostly_populated(lang, domain):
    """target_wp_start must be >= 0 (valid) for at least 95% of rows.

    A -1 means the target whitespace token wasn't found in the wordpiece sequence
    (truncation or punctuation stickiness). Up to 5% missing is acceptable.
    """
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    df = pd.read_parquet(path)
    n_valid = (df["target_wp_start"] >= 0).sum()
    pct_valid = n_valid / len(df)
    assert pct_valid >= 0.95, (
        f"({lang!r}, {domain!r}): only {pct_valid:.1%} of rows have valid "
        f"target_wp_start (expected >= 95%); {len(df) - n_valid} rows have -1"
    )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_embedding_offsets_consistent(lang, domain):
    """embedding_offset for each part must start at 0 and be contiguous within the part."""
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    df = pd.read_parquet(path)
    for part_num, group in df.groupby("embedding_part"):
        offsets = group["embedding_offset"].tolist()
        expected = list(range(len(offsets)))
        assert offsets == expected, (
            f"Part {part_num}: embedding_offset values are not contiguous 0..{len(offsets)-1}: "
            f"got {offsets[:10]}..."
        )


@pytest.mark.parametrize("lang,domain", [(l, d) for l in LANGS for d in DOMAINS])
def test_manifest_embedding_offset_indexes_correct_part(lang, domain):
    """Spot-check: embedding_offset for a few rows must fall within the part array bounds."""
    path = _manifest_path(lang, domain)
    if not os.path.exists(path):
        _skip_or_fail(f"Manifest not found: {path}")
    parts = _find_embedding_parts(lang, domain)
    if not parts:
        _skip_or_fail(f"No embedding parts found for ({lang!r}, {domain!r})")
    df = pd.read_parquet(path)
    # Build part_num → array shape map
    part_shapes = {}
    for p in parts:
        # Extract part number from filename e.g. _part2of3.npy
        basename = os.path.basename(p)
        # e.g. "...part2of3.npy" → part_num = 2
        part_str = [seg for seg in basename.split("_part") if "of" in seg]
        if not part_str:
            continue
        part_num = int(part_str[0].split("of")[0])
        part_shapes[part_num] = np.load(p).shape[0]

    # Check every row: embedding_offset must be < part's row count
    for _, row in df.iterrows():
        pnum = int(row["embedding_part"])
        offset = int(row["embedding_offset"])
        if pnum not in part_shapes:
            continue
        assert offset < part_shapes[pnum], (
            f"kwic_row_id={row['kwic_row_id']}: embedding_offset={offset} "
            f"is out of bounds for part {pnum} (size {part_shapes[pnum]})"
        )
