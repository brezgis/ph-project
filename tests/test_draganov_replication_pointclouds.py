"""Tests for draganov_replication/pointclouds.py.

Five tests per issue spec:
  1. test_canon_term_coverage     — gated by PH_REQUIRE_DRAGANOV_POINTCLOUDS=1
  2. test_pointcloud_shape        — gated
  3. test_fioletovyi_kept         — gated
  4. test_russian_blues_present   — gated
  5. test_pooling_correctness     — NOT gated (pure synthetic unit test)

Set PH_REQUIRE_DRAGANOV_POINTCLOUDS=1 to turn skips into failures after
`build_pointclouds()` has been run to completion.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Gate setup
# ---------------------------------------------------------------------------

REQUIRE = os.environ.get("PH_REQUIRE_DRAGANOV_POINTCLOUDS") == "1"

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
POINTCLOUD_DIR = REPO_ROOT / "data" / "phase3" / "draganov_pointclouds"
CANON_DIR = REPO_ROOT / "canon-terms"

try:
    from draganov_replication.pointclouds import build_pointclouds
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _require_import() -> None:
    if not _IMPORT_OK:
        pytest.fail(
            "Could not import draganov_replication.pointclouds — module not yet created."
        )


def _skip_or_fail(reason: str) -> None:
    if REQUIRE:
        pytest.fail(reason + " (PH_REQUIRE_DRAGANOV_POINTCLOUDS=1)")
    pytest.skip(reason)


def _pointclouds_ready() -> bool:
    """Return True when draganov_pointclouds/ is non-empty (manifest.csv present)."""
    return (POINTCLOUD_DIR / "manifest.csv").exists()


def _load_manifest() -> pd.DataFrame:
    return pd.read_csv(POINTCLOUD_DIR / "manifest.csv")


def _load_canon_terms(lang: str, domain: str = "color") -> list[str]:
    path = CANON_DIR / lang / f"{domain}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return [t["term"] for t in data["terms"]]


# ---------------------------------------------------------------------------
# Test 1: every canon term produces a .npy file
# ---------------------------------------------------------------------------

def test_canon_term_coverage():
    """Every (lang, term) in the color canon-terms YAML has a matching .npy."""
    _require_import()
    if not _pointclouds_ready():
        _skip_or_fail(f"draganov_pointclouds not built yet: {POINTCLOUD_DIR}")

    for lang in ("en", "ru", "es"):
        terms = _load_canon_terms(lang)
        for term in terms:
            expected_npy = POINTCLOUD_DIR / f"{lang}_{term}.npy"
            assert expected_npy.exists(), (
                f"Missing point-cloud file for ({lang}, {term}): {expected_npy}"
            )


# ---------------------------------------------------------------------------
# Test 2: shape, dtype, n_samples consistency
# ---------------------------------------------------------------------------

def test_pointcloud_shape():
    """Each .npy has shape (n_samples, 768), dtype float32, and n_samples matches manifest.csv."""
    _require_import()
    if not _pointclouds_ready():
        _skip_or_fail(f"draganov_pointclouds not built yet: {POINTCLOUD_DIR}")

    manifest = _load_manifest()
    assert len(manifest) > 0, "manifest.csv is empty"

    for _, row in manifest.iterrows():
        npy_path = POINTCLOUD_DIR / pathlib.Path(row["file"]).name
        assert npy_path.exists(), f"manifest references missing file: {npy_path}"
        arr = np.load(npy_path)
        assert arr.ndim == 2, (
            f"({row['lang']}, {row['term']}): expected 2D array, got shape {arr.shape}"
        )
        assert arr.shape[1] == 768, (
            f"({row['lang']}, {row['term']}): expected 768 dims, got {arr.shape[1]}"
        )
        assert arr.dtype == np.float32, (
            f"({row['lang']}, {row['term']}): expected float32, got {arr.dtype}"
        )
        assert arr.shape[0] == row["n_samples"], (
            f"({row['lang']}, {row['term']}): manifest says n_samples={row['n_samples']}, "
            f"but .npy has {arr.shape[0]} rows"
        )
        assert row["n_samples"] > 0, (
            f"({row['lang']}, {row['term']}): n_samples must be > 0"
        )


# ---------------------------------------------------------------------------
# Test 3: фиолетовый kept at n >= 100
# ---------------------------------------------------------------------------

def test_fioletovyi_kept():
    """ru_фиолетовый.npy exists with n >= 100 (documented under-target, kept per CLAUDE.md)."""
    _require_import()
    if not _pointclouds_ready():
        _skip_or_fail(f"draganov_pointclouds not built yet: {POINTCLOUD_DIR}")

    npy_path = POINTCLOUD_DIR / "ru_фиолетовый.npy"
    assert npy_path.exists(), (
        "ru_фиолетовый.npy is missing — the under-target term must be kept, not dropped"
    )
    arr = np.load(npy_path)
    assert arr.shape[0] >= 100, (
        f"ru_фиолетовый has n={arr.shape[0]}, expected >= 100 "
        "(n=104 documented in CLAUDE.md; n must not have dropped further)"
    )


# ---------------------------------------------------------------------------
# Test 4: Russian-blues centerpiece cells at full n >= 150
# ---------------------------------------------------------------------------

def test_russian_blues_present():
    """All four Russian-blues centerpiece cells exist with n >= 150."""
    _require_import()
    if not _pointclouds_ready():
        _skip_or_fail(f"draganov_pointclouds not built yet: {POINTCLOUD_DIR}")

    cells = [
        ("en", "blue"),
        ("es", "azul"),
        ("ru", "синий"),
        ("ru", "голубой"),
    ]
    for lang, term in cells:
        npy_path = POINTCLOUD_DIR / f"{lang}_{term}.npy"
        assert npy_path.exists(), (
            f"Russian-blues centerpiece cell missing: {lang}_{term}.npy"
        )
        arr = np.load(npy_path)
        assert arr.shape[0] >= 150, (
            f"Centerpiece cell ({lang}, {term}) has n={arr.shape[0]}, expected >= 150"
        )


# ---------------------------------------------------------------------------
# Test 5: pooling correctness — synthetic, NOT gated
# ---------------------------------------------------------------------------

def test_pooling_correctness(tmp_path: pathlib.Path):
    """Mean-pooling over WP span produces exactly np.mean(emb[i, ws:we+1], axis=0).

    Constructs a small synthetic 1-row manifest + 1-part embedding npy with
    known values, calls build_pointclouds on a temp dir, and checks the saved
    vector matches the hand-computed mean.
    """
    _require_import()

    # Build a tiny fake embedding: 1 sentence, MAX_LEN=8, hidden=768
    MAX_LEN = 8
    HIDDEN = 768
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((1, MAX_LEN, HIDDEN)).astype(np.float16)

    # Target span: WP positions 2 through 4 (inclusive) = 3 WordPieces
    ws, we = 2, 4
    expected_vector = np.mean(emb[0, ws : we + 1, :].astype(np.float32), axis=0)

    # Write the embedding npy
    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    npy_name = "en_color_final_layer_MAX_LEN_8_bert-base-multilingual-cased_part1of1.npy"
    np.save(emb_dir / npy_name, emb)

    # Write the manifest parquet
    manifest_df = pd.DataFrame(
        [
            {
                "kwic_row_id": 0,
                "lang": "en",
                "domain": "color",
                "term": "blue",
                "target_idx": 3,
                "target_wp_start": ws,
                "target_wp_end": we,
                "embedding_part": 1,
                "embedding_offset": 0,
            }
        ]
    )
    manifest_df.to_parquet(emb_dir / "en_color_manifest.parquet", index=False)

    # Build a minimal canon-terms dir with only en/color.yaml
    canon_dir = tmp_path / "canon-terms" / "en"
    canon_dir.mkdir(parents=True)
    canon_yaml = {"domain": "color", "language": "en", "terms": [{"term": "blue"}]}
    import yaml as _yaml
    with open(canon_dir / "color.yaml", "w") as f:
        _yaml.dump(canon_yaml, f)

    out_dir = tmp_path / "draganov_pointclouds"

    result_manifest = build_pointclouds(
        emb_dir=emb_dir,
        canon_dir=tmp_path / "canon-terms",
        out_dir=out_dir,
        langs=("en",),
        domain="color",
        overwrite=False,
    )

    saved_npy = out_dir / "en_blue.npy"
    assert saved_npy.exists(), "build_pointclouds did not write en_blue.npy"

    arr = np.load(saved_npy)
    assert arr.shape == (1, HIDDEN), f"Expected shape (1, {HIDDEN}), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"

    np.testing.assert_allclose(
        arr[0],
        expected_vector,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            "Saved point-cloud vector does not match "
            "np.mean(emb[offset, ws:we+1], axis=0).astype(float32)"
        ),
    )


# ---------------------------------------------------------------------------
# Synthetic fixture helper for skip-path tests
# ---------------------------------------------------------------------------

def _write_synthetic_fixture(
    tmp_path: pathlib.Path,
    rows: list[dict],
    max_len: int = 8,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Write a minimal embeddings/manifest/canon fixture for a single en/blue cell.

    Returns (emb_dir, canon_dir, out_dir).  Each row in *rows* must contain
    target_wp_start, target_wp_end, embedding_part, embedding_offset.
    The embedding npy has shape (n_offsets, max_len, 768) populated with
    deterministic random values.
    """
    HIDDEN = 768
    n_offsets = max((r["embedding_offset"] for r in rows), default=0) + 1
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n_offsets, max_len, HIDDEN)).astype(np.float16)

    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    npy_name = (
        f"en_color_final_layer_MAX_LEN_{max_len}"
        "_bert-base-multilingual-cased_part1of1.npy"
    )
    np.save(emb_dir / npy_name, emb)

    manifest_df = pd.DataFrame(
        [
            {
                "kwic_row_id": i,
                "lang": "en",
                "domain": "color",
                "term": "blue",
                "target_idx": 3,
                **r,
            }
            for i, r in enumerate(rows)
        ]
    )
    manifest_df.to_parquet(emb_dir / "en_color_manifest.parquet", index=False)

    canon_dir = tmp_path / "canon-terms"
    (canon_dir / "en").mkdir(parents=True)
    canon_yaml = {"domain": "color", "language": "en", "terms": [{"term": "blue"}]}
    with open(canon_dir / "en" / "color.yaml", "w") as f:
        yaml.dump(canon_yaml, f)

    return emb_dir, canon_dir, tmp_path / "draganov_pointclouds"


def test_negative_wp_skipped_with_warning(tmp_path: pathlib.Path):
    """Rows with negative target_wp_start/end are skipped with UserWarning."""
    _require_import()
    rows = [
        {"target_wp_start": -1, "target_wp_end": -1,
         "embedding_part": 1, "embedding_offset": 0},
        {"target_wp_start": 2, "target_wp_end": 4,
         "embedding_part": 1, "embedding_offset": 1},
    ]
    emb_dir, canon_dir, out_dir = _write_synthetic_fixture(tmp_path, rows)

    with pytest.warns(UserWarning, match="negative"):
        result = build_pointclouds(
            emb_dir=emb_dir, canon_dir=canon_dir, out_dir=out_dir,
            langs=("en",), domain="color", overwrite=False,
        )

    assert result["n_samples"].iloc[0] == 1, (
        "Expected the negative-WP row to be skipped, leaving 1 sample."
    )


def test_empty_span_skipped_with_warning(tmp_path: pathlib.Path):
    """Rows with we < ws (empty inclusive span) are skipped with UserWarning."""
    _require_import()
    rows = [
        # ws=4, we=2 → emb[0, 4:3, :] has shape (0, 768) → empty span branch
        {"target_wp_start": 4, "target_wp_end": 2,
         "embedding_part": 1, "embedding_offset": 0},
        {"target_wp_start": 2, "target_wp_end": 4,
         "embedding_part": 1, "embedding_offset": 1},
    ]
    emb_dir, canon_dir, out_dir = _write_synthetic_fixture(tmp_path, rows)

    with pytest.warns(UserWarning, match="empty WP span"):
        result = build_pointclouds(
            emb_dir=emb_dir, canon_dir=canon_dir, out_dir=out_dir,
            langs=("en",), domain="color", overwrite=False,
        )

    assert result["n_samples"].iloc[0] == 1, (
        "Expected the empty-span row to be skipped, leaving 1 sample."
    )


def test_all_rows_skipped_raises(tmp_path: pathlib.Path):
    """If every row in a (lang, term) cell has invalid spans, build_pointclouds raises."""
    _require_import()
    rows = [
        {"target_wp_start": -1, "target_wp_end": -1,
         "embedding_part": 1, "embedding_offset": 0},
        {"target_wp_start": -1, "target_wp_end": -1,
         "embedding_part": 1, "embedding_offset": 1},
    ]
    emb_dir, canon_dir, out_dir = _write_synthetic_fixture(tmp_path, rows)

    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="were skipped"):
            build_pointclouds(
                emb_dir=emb_dir, canon_dir=canon_dir, out_dir=out_dir,
                langs=("en",), domain="color", overwrite=False,
            )
