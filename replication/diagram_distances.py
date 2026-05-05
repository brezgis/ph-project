"""Diagram loader and giotto-tda format conversion for Wasserstein/bottleneck eval.

This module is the data ingestion layer for the cross-linguistic color topology
evaluation described in the blr epic (bd show ph-project-blr). It:

  1. Loads per-language Kushnareva-format barcode JSONs from disk.
  2. Attaches metadata (lang, term, sentence_idx) to each sample.
  3. Subsamples deterministically per term.
  4. Converts persistence diagrams to the giotto-tda format expected by
     ``gtda.diagrams.PairwiseDistance``.

## On-disk JSON structure

Barcode JSONs are produced by ``mbert_attention_ripser.ipynb``
(``save_barcodes`` function, cell 28). They use **string keys throughout**
as a JSON round-trip artifact:

    {
        "<layer_str>": {         # "0".."11"
            "<head_str>": [      # "0".."11"
                {                # one dict per sample
                    "0": [[birth, death], ...],   # H_0 features
                    "1": [[birth, death], ...]    # H_1 features
                },
                ...
            ]
        }
    }

Each ``partNofM`` JSON contains up to 1000 samples. Three parts per
(lang, domain) → up to 3000 total, but actual count equals the KWIC CSV
row count (samples are 1-to-1 with KWIC sentences).

## KWIC lookup contract

``load_lang_barcodes`` maps each barcode sample to its originating KWIC
sentence by replaying the same ordering the ripser notebook used:

  - The ripser notebook reads ``data/kwic/<lang>/<domain>.csv`` via
    ``pd.read_csv(...).reset_index(drop=True)``.
  - Samples are written to barcode JSONs in the same row order.
  - Parts are sorted ascending by part number (part1, part2, part3).
  - Therefore: barcode sample ``i`` across all parts corresponds to
    ``kwic_df.iloc[i]``, where ``kwic_df`` is the KWIC CSV loaded with
    ``reset_index(drop=True)``.

This 1-to-1 correspondence is validated by asserting that total barcode
sample count equals KWIC CSV row count. A mismatch raises ``ValueError``.

## giotto-tda format

``gtda.diagrams.PairwiseDistance`` expects diagrams as a numpy array of
shape ``(N, F, 3)`` where:

  - ``N`` = number of samples
  - ``F`` = maximum feature count across all samples (padded to constant size)
  - Last axis = ``[birth, death, hom_dim]``

Padding rows use ``(0, 0, hom_dim)`` where ``hom_dim`` matches the
homology dimension of the real features in that block. Using ``hom_dim=0``
for H_1 padding rows would cause ``PairwiseDistance`` to misclassify them
as H_0 features and corrupt distance computations.

## References

- Kushnareva et al. (2021) EMNLP — substrate methodology.
- Berlin & Kay (1969) — color BCT anchor.
- Paramei (2005), Winawer et al. (2007) — Russian sinij/голубой split.
- bd ph-project-bcy — фиолетовый under-target casualty (n=104, kept).
"""
from __future__ import annotations

import json
import pathlib
import re
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Type aliases (for readability in docstrings; not runtime-enforced)
# ---------------------------------------------------------------------------
#   PerLayerHeadDiagrams = dict[tuple[int, int], list[dict[int, np.ndarray]]]
#   Each outer key is (layer, head), each inner list is indexed by sample_idx,
#   each sample dict maps hom_dim (int) → np.ndarray of shape (n_features, 2).


def load_barcode_json(
    path: Union[str, pathlib.Path],
) -> dict:
    """Parse a Kushnareva-format barcode JSON file.

    Converts the string-keyed JSON round-trip artifact to a dict keyed by
    ``(layer: int, head: int)`` tuples. Each value is a list of sample dicts,
    where each sample dict maps ``hom_dim: int`` to a ``np.ndarray`` of shape
    ``(n_features, 2)`` and dtype ``float64``.

    Parameters
    ----------
    path:
        Path to a ``*partNofM.json`` barcode file produced by
        ``mbert_attention_ripser.ipynb``.

    Returns
    -------
    dict[tuple[int, int], list[dict[int, np.ndarray]]]
        Outer dict keyed by ``(layer, head)`` tuples.
        Inner list indexed by sample index.
        Each sample dict keyed by homology dimension (int: 0 or 1).
        Arrays have dtype ``float64`` and shape ``(n_features, 2)`` where
        axis 1 is ``[birth, death]``. Empty feature sets have shape ``(0, 2)``.

    Notes
    -----
    The on-disk format uses string keys because Python's ``json`` module does
    not support non-string dict keys. Callers should never rely on string keys;
    always use integer (layer, head) tuples and integer hom_dim keys.
    """
    path = pathlib.Path(path)
    with path.open("r") as fh:
        raw = json.load(fh)

    result: dict = {}
    for layer_str, heads_dict in raw.items():
        layer = int(layer_str)
        for head_str, samples_list in heads_dict.items():
            head = int(head_str)
            parsed_samples = []
            for sample_raw in samples_list:
                sample: dict[int, np.ndarray] = {}
                for dim_str, features_list in sample_raw.items():
                    dim = int(dim_str)
                    if features_list:
                        arr = np.array(features_list, dtype=np.float64)
                        # Ensure shape is (n_features, 2)
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 2)
                    else:
                        arr = np.empty((0, 2), dtype=np.float64)
                    sample[dim] = arr
                parsed_samples.append(sample)
            result[(layer, head)] = parsed_samples

    return result


def load_lang_barcodes(
    barcode_dir: Union[str, pathlib.Path],
    lang: str,
    domain: str,
    model_tag: str = "bert-base-multilingual-cased",
    max_len: int = 32,
    n_layers: int = 12,
    kwic_dir: Optional[str] = None,
) -> tuple:
    """Load and concatenate all part files for a (lang, domain) pair.

    Globs all ``partNofM`` JSON files matching ``<lang>_<domain>_all_heads_
    <n_layers>_layers_MAX_LEN_<max_len>_<model_tag>_part*of*.json``, sorts
    them in ascending numeric part order, and concatenates.

    Builds a metadata DataFrame indexed by global sample index (0-based)
    with the per-KWIC-sentence information needed by downstream permutation
    tests (term identity, sentence position within term).

    ## KWIC lookup contract

    The ripser notebook processes KWIC rows in ``pd.read_csv`` order
    (``reset_index(drop=True)``). Parts are sorted ascending by part number
    and written sequentially. Therefore barcode sample ``i`` corresponds to
    ``kwic_df.iloc[i]``. This function validates this correspondence by
    asserting ``total_barcode_samples == len(kwic_df)``.

    Parameters
    ----------
    barcode_dir:
        Directory containing the barcode JSON files.
    lang:
        Language code (``"en"``, ``"ru"``, ``"es"``).
    domain:
        Domain (``"color"`` for the May 2026 scope).
    model_tag:
        Model identifier as it appears in the filename.
    max_len:
        MAX_LEN value as it appears in the filename.
    n_layers:
        Number of layers as it appears in the filename.
    kwic_dir:
        Root of the KWIC CSV directory tree. Expected layout:
        ``<kwic_dir>/<lang>/<domain>.csv``. If ``None``, defaults to
        ``<barcode_dir>/../../data/kwic`` (two levels up from barcodes/).

    Returns
    -------
    tuple[dict, pd.DataFrame]
        ``(diagrams, metadata)`` where:
        - ``diagrams``: ``dict[tuple[int, int], list[dict[int, np.ndarray]]]``
          with all parts concatenated. Outer key is ``(layer, head)``.
        - ``metadata``: ``pd.DataFrame`` with columns
          ``[lang, term, sentence_idx_within_term, source_file, source_part]``,
          one row per global sample index.

    Raises
    ------
    FileNotFoundError
        If no barcode files matching the pattern are found.
    ValueError
        If total barcode sample count does not match KWIC CSV row count.
    """
    barcode_dir = pathlib.Path(barcode_dir)

    # Resolve KWIC dir
    if kwic_dir is None:
        # Default: barcode_dir is data/phase3/barcodes/ → go up to data/kwic/
        kwic_dir_path = barcode_dir.parent.parent / "kwic"
    else:
        kwic_dir_path = pathlib.Path(kwic_dir)

    # Glob part files
    pattern = (
        f"{lang}_{domain}_all_heads_{n_layers}_layers_"
        f"MAX_LEN_{max_len}_{model_tag}_part*of*.json"
    )
    part_files = sorted(
        barcode_dir.glob(pattern),
        key=lambda p: int(re.search(r"_part(\d+)of\d+\.json$", p.name).group(1)),
    )

    if not part_files:
        raise FileNotFoundError(
            f"No barcode files found in {barcode_dir} matching pattern: {pattern}"
        )

    # Load KWIC CSV for metadata
    kwic_csv = kwic_dir_path / lang / f"{domain}.csv"
    if not kwic_csv.exists():
        raise FileNotFoundError(f"KWIC CSV not found: {kwic_csv}")
    kwic_df = pd.read_csv(kwic_csv).reset_index(drop=True)

    # Load and concatenate all parts
    all_diagrams: dict = {}
    metadata_rows = []
    global_sample_idx = 0

    # First pass: load all parts and validate total sample count against KWIC before
    # building metadata rows.  Without this pre-check, the metadata loop would raise
    # IndexError (out-of-bounds iloc) instead of ValueError on barcode > KWIC mismatch.
    loaded_parts: list[tuple[pathlib.Path, int, dict]] = []
    for part_file in part_files:
        part_num = int(re.search(r"_part(\d+)of\d+\.json$", part_file.name).group(1))
        part_diagrams = load_barcode_json(part_file)

        # Determine sample count for this part (all (layer, head) must agree)
        part_keys = list(part_diagrams.keys())
        part_n = len(part_diagrams[part_keys[0]])

        # Validate internal consistency of this part
        for key, samples in part_diagrams.items():
            if len(samples) != part_n:
                raise ValueError(
                    f"{part_file.name}: (layer, head) {key} has {len(samples)} samples; "
                    f"expected {part_n} (from key {part_keys[0]})"
                )

        loaded_parts.append((part_file, part_num, part_diagrams))

    # Validate total count matches KWIC before entering the metadata loop
    total_samples = sum(
        len(next(iter(part_diagrams.values()))) for _, _, part_diagrams in loaded_parts
    )
    if total_samples != len(kwic_df):
        raise ValueError(
            f"Barcode sample count ({total_samples}) does not match KWIC CSV row count "
            f"({len(kwic_df)}) for ({lang!r}, {domain!r}). "
            "This indicates a mismatch between the ripser run and the KWIC data. "
            "Check that the barcode files correspond to the current KWIC CSV."
        )

    # Second pass: concatenate diagrams and build metadata rows
    for part_file, part_num, part_diagrams in loaded_parts:
        part_keys = list(part_diagrams.keys())
        part_n = len(part_diagrams[part_keys[0]])

        # Concatenate into all_diagrams
        if not all_diagrams:
            all_diagrams = {k: list(v) for k, v in part_diagrams.items()}
        else:
            for key, samples in part_diagrams.items():
                all_diagrams[key].extend(samples)

        # Build metadata rows for this part's samples
        for local_idx in range(part_n):
            row = kwic_df.iloc[global_sample_idx]
            term = row["term"]
            # sentence_idx_within_term: count how many rows before this one share the same term
            sentence_idx_within_term = (
                kwic_df.iloc[:global_sample_idx]["term"] == term
            ).sum()
            metadata_rows.append({
                "lang": lang,
                "term": term,
                "sentence_idx_within_term": int(sentence_idx_within_term),
                "source_file": part_file.name,
                "source_part": part_num,
            })
            global_sample_idx += 1

    metadata = pd.DataFrame(metadata_rows)
    return all_diagrams, metadata


def subsample_per_term(
    metadata_df: pd.DataFrame,
    n_per_term: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """Return sample indices to keep, selecting n_per_term samples per term.

    Uses ``np.random.default_rng(seed)`` for determinism. Sampling is done
    independently per term (without-replacement within each term group).

    If a term has fewer than ``n_per_term`` samples, **all** its samples are
    kept and a ``warnings.warn(UserWarning)`` is emitted. This is the
    documented case for Russian ``фиолетовый`` (n=104, kept per CLAUDE.md
    "Current scope (May 2026)" and bd ph-project-bcy).

    Parameters
    ----------
    metadata_df:
        DataFrame produced by ``load_lang_barcodes``. Must have a ``"term"``
        column. The integer index of each row is used as the sample index.
    n_per_term:
        Number of samples to select per term. Default 30 (the "two-night
        budget" value; default for the overnight CLI is 20).
    seed:
        Random seed for ``np.random.default_rng``. Same seed → same output.

    Returns
    -------
    np.ndarray of int
        Sorted array of integer row positions (iloc indices) into
        ``metadata_df``. Length = ``sum(min(n_term, n_per_term) for each term)``.
    """
    rng = np.random.default_rng(seed)
    selected_indices = []

    for term, group in metadata_df.groupby("term", sort=True):
        group_indices = group.index.to_numpy()
        if len(group_indices) < n_per_term:
            warnings.warn(
                f"Term '{term}' has only {len(group_indices)} samples "
                f"(fewer than n_per_term={n_per_term}). "
                f"Taking all {len(group_indices)} available samples. "
                "This is expected for ru/фиолетовый (n=104; see CLAUDE.md and bd ph-project-bcy).",
                UserWarning,
                stacklevel=2,
            )
            chosen = group_indices
        else:
            # Convert group positions to positional iloc indices within the full DataFrame
            # group.index values are the actual integer row labels from metadata_df
            chosen = rng.choice(group_indices, size=n_per_term, replace=False)

        selected_indices.append(chosen)

    all_indices = np.concatenate(selected_indices)
    # Convert from index labels to positional indices
    # (metadata_df may have non-contiguous integer index if it was sliced)
    pos_map = {label: pos for pos, label in enumerate(metadata_df.index)}
    positional = np.array([pos_map[label] for label in all_indices], dtype=np.intp)
    return np.sort(positional)


def to_giotto_format(
    per_layer_head_diagrams: dict,
    sample_indices: np.ndarray,
    layer: int,
    head: int,
    dims: tuple = (0, 1),
) -> np.ndarray:
    """Convert persistence diagrams to giotto-tda ``PairwiseDistance`` format.

    Extracts diagrams for a single ``(layer, head)`` pair, selects the
    samples at ``sample_indices``, pads all samples to the same feature count
    per homology dimension, and stacks into the ``(N, F, 3)`` array expected
    by ``gtda.diagrams.PairwiseDistance``.

    ## Output format

    Shape: ``(N, F, 3)`` where:

    - ``N`` = ``len(sample_indices)``
    - ``F`` = ``sum(max_features_per_dim for dim in dims)`` — padded to the
      maximum feature count across all selected samples for each dim.
    - Last axis = ``[birth, death, hom_dim]``

    ## Padding convention

    Padding rows use ``(0.0, 0.0, hom_dim)`` where ``hom_dim`` is the
    homology dimension of the block being padded. **This is critical:**
    using ``hom_dim=0`` for H_1 padding rows would cause
    ``PairwiseDistance`` to misclassify those rows as H_0 features and
    corrupt distance computations.

    Parameters
    ----------
    per_layer_head_diagrams:
        ``dict[tuple[int, int], list[dict[int, np.ndarray]]]`` as returned
        by ``load_barcode_json`` or ``load_lang_barcodes``. The outer key
        ``(layer, head)`` must be present.
    sample_indices:
        Integer array of positions (0-based) into the sample list to include.
        Produced by ``subsample_per_term``.
    layer:
        Layer index (0-based).
    head:
        Head index (0-based).
    dims:
        Tuple of homology dimensions to include. Default ``(0, 1)``.

    Returns
    -------
    np.ndarray of float64
        Shape ``(N, F, 3)``. dtype ``float64``.

    Raises
    ------
    KeyError
        If ``(layer, head)`` is not in ``per_layer_head_diagrams``.
    """
    key = (layer, head)
    if key not in per_layer_head_diagrams:
        raise KeyError(
            f"(layer={layer}, head={head}) not found in diagrams. "
            f"Available keys: {list(per_layer_head_diagrams.keys())[:5]}..."
        )

    all_samples = per_layer_head_diagrams[key]
    selected = [all_samples[i] for i in sample_indices]
    n_samples = len(selected)

    # Determine max feature count per dim across selected samples
    max_features_per_dim: dict[int, int] = {}
    for dim in dims:
        max_f = 0
        for sample in selected:
            arr = sample.get(dim, np.empty((0, 2), dtype=np.float64))
            max_f = max(max_f, len(arr))
        max_features_per_dim[dim] = max_f

    total_features = sum(max_features_per_dim[d] for d in dims)

    # Build output array
    out = np.zeros((n_samples, total_features, 3), dtype=np.float64)

    for s_idx, sample in enumerate(selected):
        feat_offset = 0
        for dim in dims:
            arr = sample.get(dim, np.empty((0, 2), dtype=np.float64))
            max_f = max_features_per_dim[dim]
            n_real = len(arr)

            if n_real > 0:
                out[s_idx, feat_offset : feat_offset + n_real, 0] = arr[:, 0]  # birth
                out[s_idx, feat_offset : feat_offset + n_real, 1] = arr[:, 1]  # death
                out[s_idx, feat_offset : feat_offset + n_real, 2] = float(dim)  # hom_dim

            # Padding rows: (0, 0, hom_dim) — hom_dim must match the block's dim
            out[s_idx, feat_offset + n_real : feat_offset + max_f, 0] = 0.0   # birth
            out[s_idx, feat_offset + n_real : feat_offset + max_f, 1] = 0.0   # death
            out[s_idx, feat_offset + n_real : feat_offset + max_f, 2] = float(dim)  # hom_dim

            feat_offset += max_f

    return out
