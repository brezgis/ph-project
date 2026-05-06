"""Build per-(lang, term) contextual point clouds from mBERT embeddings.

Each point cloud is a float32 array of shape (n_samples, 768) where each row
is the mean-pooled embedding of the target color-term WordPiece span in one
sentence.  Arrays are written as ``{lang}_{term}.npy`` under *out_dir*.

Public API
----------
build_pointclouds(...)
    Build all 34 per-(lang, term) point clouds and return a manifest DataFrame.
"""
from __future__ import annotations

import glob
import logging
import pathlib
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Expected embedding hidden size.  Hard-coded: we always use mBERT (768-d).
_HIDDEN = 768


def _load_canon_terms(canon_dir: pathlib.Path, lang: str, domain: str) -> list[str]:
    """Load the ordered list of canonical term strings from ``canon_dir/lang/domain.yaml``."""
    path = canon_dir / lang / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Canon-terms file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return [entry["term"] for entry in data["terms"]]


def _find_npy_part(emb_dir: pathlib.Path, lang: str, domain: str, part: int) -> pathlib.Path:
    """Return the path of the embedding .npy for *lang*, *domain*, *part*.

    Glob pattern: ``{lang}_{domain}_final_layer_*_part{P}of*.npy``.
    Raises ``FileNotFoundError`` if no match or more than one match.
    """
    pattern = str(emb_dir / f"{lang}_{domain}_final_layer_*_part{part}of*.npy")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No embedding file found matching: {pattern}"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous embedding files for part {part}: {matches}"
        )
    return pathlib.Path(matches[0])


def _pool_term(
    manifest: pd.DataFrame,
    term: str,
    emb_dir: pathlib.Path,
    lang: str,
    domain: str,
) -> np.ndarray:
    """Compute mean-pooled embeddings for all sentences containing *term*.

    Parameters
    ----------
    manifest:
        Full per-language manifest DataFrame loaded from
        ``{emb_dir}/{lang}_{domain}_manifest.parquet``.
    term:
        The surface-form canon term to filter for.
    emb_dir:
        Directory containing the embedding ``.npy`` parts.
    lang, domain:
        Language and domain strings (needed for filename patterns).

    Returns
    -------
    np.ndarray of shape (n_rows, 768), dtype float32, in manifest row order.
    """
    term_rows = manifest[manifest["term"] == term].copy()
    if len(term_rows) == 0:
        raise ValueError(f"No manifest rows found for ({lang}, {term!r})")

    # Open all required parts (mmap_mode='r' — never load into RAM).
    parts_needed = term_rows["embedding_part"].unique()
    mmap_cache: dict[int, np.ndarray] = {}
    for part in parts_needed:
        path = _find_npy_part(emb_dir, lang, domain, int(part))
        mmap_cache[int(part)] = np.load(path, mmap_mode="r")

    vectors: list[np.ndarray] = []
    for _, row in term_rows.iterrows():
        part = int(row["embedding_part"])
        offset = int(row["embedding_offset"])
        ws = int(row["target_wp_start"])
        we = int(row["target_wp_end"])

        emb_part = mmap_cache[part]  # shape (N_part, MAX_LEN, 768)
        sentence_emb = emb_part[offset]  # shape (MAX_LEN, 768), float16

        if ws < 0 or we < 0:
            # Tokeniser failed to locate the target word — skip with a warning.
            warnings.warn(
                f"({lang}, {term!r}) row {row.name}: "
                f"target_wp_start/end are negative ({ws}, {we}); skipping row.",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Inclusive slice [ws : we+1], then mean-pool.
        span = sentence_emb[ws : we + 1, :].astype(np.float32)  # (n_wps, 768) float32
        if span.shape[0] == 0:
            warnings.warn(
                f"({lang}, {term!r}) row {row.name}: "
                f"empty WP span at positions {ws}:{we+1}; skipping row.",
                UserWarning,
                stacklevel=2,
            )
            continue
        pooled = np.mean(span, axis=0)  # (768,) float32
        vectors.append(pooled)

    if not vectors:
        raise ValueError(
            f"All manifest rows for ({lang}, {term!r}) were skipped "
            "(negative or empty WP spans)."
        )

    return np.stack(vectors, axis=0)  # (n_samples, 768) float32


def build_pointclouds(
    emb_dir: pathlib.Path = pathlib.Path("data/phase3/embeddings"),
    canon_dir: pathlib.Path = pathlib.Path("canon-terms"),
    out_dir: pathlib.Path = pathlib.Path("data/phase3/draganov_pointclouds"),
    langs: tuple[str, ...] = ("en", "ru", "es"),
    domain: str = "color",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Build per-(lang, term) contextual point clouds from existing mBERT embeddings.

    For each (lang, term) cell, loads the relevant rows from the manifest
    parquet, slices the target WordPiece span from the stored embedding ``.npy``
    parts, mean-pools to a 768-d vector, casts to float32, and saves as
    ``out_dir/{lang}_{term}.npy``.

    Parameters
    ----------
    emb_dir:
        Directory with ``{lang}_{domain}_manifest.parquet`` and
        ``{lang}_{domain}_final_layer_*_part{P}of*.npy`` files.
    canon_dir:
        Root of the ``canon-terms/`` tree.  Per-language files are read from
        ``canon_dir/{lang}/{domain}.yaml``.
    out_dir:
        Output directory.  Created if it does not exist.
    langs:
        Languages to process (default: all three study languages).
    domain:
        Domain string (default: ``"color"``).
    overwrite:
        When *False* (default), skip cells whose ``.npy`` already exists.
        Pass *True* to force regeneration.

    Returns
    -------
    pd.DataFrame
        Manifest with columns ``[lang, term, n_samples, file]``, sorted by
        ``(lang, term)``.  Also written to ``out_dir/manifest.csv``.
    """
    emb_dir = pathlib.Path(emb_dir)
    canon_dir = pathlib.Path(canon_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []

    for lang in langs:
        # --- Load manifest parquet ---
        parquet_path = emb_dir / f"{lang}_{domain}_manifest.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Embedding manifest not found: {parquet_path}"
            )
        lang_manifest = pd.read_parquet(parquet_path)

        # --- Load canon terms ---
        canon_terms = _load_canon_terms(canon_dir, lang, domain)

        # --- Validate: every canon term appears in the manifest ---
        manifest_terms = set(lang_manifest["term"].unique())
        for term in canon_terms:
            if term not in manifest_terms:
                raise ValueError(
                    f"Canon term ({lang!r}, {term!r}) not found in manifest "
                    f"{parquet_path}.  Available terms: {sorted(manifest_terms)}"
                )

        # --- Warn about manifest terms not in canon ---
        extra = manifest_terms - set(canon_terms)
        if extra:
            warnings.warn(
                f"({lang}) manifest contains terms not in canon: {sorted(extra)}. "
                "These are included in the manifest parquet but no point cloud "
                "will be built for them (they are not in the canon term list).",
                UserWarning,
                stacklevel=2,
            )

        # --- Build point cloud per canon term ---
        for term in canon_terms:
            out_npy = out_dir / f"{lang}_{term}.npy"

            if out_npy.exists() and not overwrite:
                # Read shape from existing file to populate manifest row.
                arr = np.load(out_npy)
                n_samples = arr.shape[0]
                logger.debug("Skipping existing: %s (n=%d)", out_npy, n_samples)
            else:
                logger.info("Building point cloud: (%s, %s)", lang, term)
                arr = _pool_term(lang_manifest, term, emb_dir, lang, domain)
                np.save(out_npy, arr)
                n_samples = arr.shape[0]
                logger.info("  Saved %s: shape=%s", out_npy.name, arr.shape)

            manifest_rows.append(
                {
                    "lang": lang,
                    "term": term,
                    "n_samples": n_samples,
                    "file": str(out_npy),
                }
            )

    result = pd.DataFrame(manifest_rows, columns=["lang", "term", "n_samples", "file"])
    result = result.sort_values(["lang", "term"]).reset_index(drop=True)

    manifest_csv = out_dir / "manifest.csv"
    result.to_csv(manifest_csv, index=False)
    logger.info("Manifest written: %s (%d rows)", manifest_csv, len(result))

    return result
