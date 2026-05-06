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

import io
import json
import logging
import pathlib
import re
import time
import warnings
from typing import Optional, Union

import yaml

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Pairwise distance computation
# ---------------------------------------------------------------------------

def compute_per_head_distances(
    per_layer_head_diagrams: dict,
    sample_indices: np.ndarray,
    layer: int,
    head: int,
    metric: str,
    dims: tuple = (0, 1),
    n_jobs: int = -1,
) -> np.ndarray:
    """Compute pairwise persistence-diagram distances for a single (layer, head).

    Extracts diagrams for ``(layer, head)``, converts them to giotto-tda format
    via ``to_giotto_format``, then runs
    ``gtda.diagrams.PairwiseDistance(metric=metric, order=None)`` to get the
    per-homology-dimension distance matrix.

    ``order=None`` is the giotto-tda parameter that returns shape
    ``(N, N, n_dims)`` rather than aggregating across dims. The ``n_dims``
    output axis is determined by the unique homology dimensions present in the
    giotto-tda diagram array — matching ``dims`` exactly when ``to_giotto_format``
    is called with the same ``dims``.

    Parameters
    ----------
    per_layer_head_diagrams:
        ``dict[tuple[int, int], list[dict[int, np.ndarray]]]`` as returned by
        ``load_barcode_json`` or ``load_lang_barcodes``.
    sample_indices:
        Integer array of sample positions (0-based) to include.
    layer:
        Layer index (0-based).
    head:
        Head index (0-based).
    metric:
        Distance metric: ``'wasserstein'`` (W_2) or ``'bottleneck'``.
    dims:
        Tuple of homology dimensions. Default ``(0, 1)``.
    n_jobs:
        Number of parallel jobs for giotto-tda. ``-1`` = use all cores.

    Returns
    -------
    np.ndarray of float32
        Shape ``(N, N, len(dims))`` where ``N = len(sample_indices)``.
        The last axis indexes homology dimensions in the same order as ``dims``.
        The matrix is symmetrized in float64 before downcasting (giotto-tda's
        parallel implementation can introduce ~1e-4 asymmetry in the output;
        downstream consumers expect ``D[i, j] == D[j, i]`` exactly).
    """
    from gtda.diagrams import PairwiseDistance  # deferred to avoid top-level import cost

    giotto_diagrams = to_giotto_format(per_layer_head_diagrams, sample_indices, layer, head, dims=dims)

    metric_params = {"p": 2} if metric == "wasserstein" else None

    pd_obj = PairwiseDistance(
        metric=metric,
        metric_params=metric_params,
        order=None,
        n_jobs=n_jobs,
    )

    result = pd_obj.fit_transform(giotto_diagrams)  # shape (N, N, n_dims), float64

    # Enforce mathematical symmetry. giotto-tda's parallel computation produces
    # D[i,j] != D[j,i] up to ~1e-4 in float32; symmetrize in float64 first.
    result = 0.5 * (result + np.swapaxes(result, 0, 1))

    return result.astype(np.float32)


def compute_full_distance_tensor(
    per_layer_head_diagrams: dict,
    sample_indices: np.ndarray,
    metric: str,
    dims: tuple = (0, 1),
    layers: range = range(12),
    heads: range = range(12),
    progress: bool = True,
    n_jobs: int = -1,
) -> np.ndarray:
    """Compute the full ``(n_layers, n_heads, N, N, len(dims))`` distance tensor.

    Calls ``compute_per_head_distances`` for every ``(layer, head)`` pair,
    logging per-pair wall time so the overnight sweep can fail fast if
    calibration drifts significantly.

    Parameters
    ----------
    per_layer_head_diagrams:
        ``dict[tuple[int, int], list[dict[int, np.ndarray]]]`` from the loader.
    sample_indices:
        Integer array of sample positions (0-based) to include.
    metric:
        Distance metric: ``'wasserstein'`` or ``'bottleneck'``.
    dims:
        Homology dimensions. Default ``(0, 1)``.
    layers:
        Which layer indices to compute. Default ``range(12)`` (all layers).
    heads:
        Which head indices to compute. Default ``range(12)`` (all heads).
    progress:
        If ``True``, display a tqdm progress bar over ``(layer, head)`` pairs.
    n_jobs:
        Parallelism per ``PairwiseDistance`` call. ``-1`` = all cores.

    Returns
    -------
    np.ndarray of float32
        Shape ``(len(layers), len(heads), N, N, len(dims))``.

    Notes
    -----
    At N=600 the tensor is approximately 166 MB per metric (float32 × 12 × 12 × 600 × 600 × 2).
    """
    try:
        from tqdm import tqdm as _tqdm
        _HAS_TQDM = True
    except ImportError:
        _HAS_TQDM = False

    layers_list = list(layers)
    heads_list = list(heads)

    n_l = len(layers_list)
    n_h = len(heads_list)
    n_samples = len(sample_indices)
    n_dims = len(dims)

    tensor = np.empty((n_l, n_h, n_samples, n_samples, n_dims), dtype=np.float32)

    pairs = [(li, hi, layer, head) for li, layer in enumerate(layers_list) for hi, head in enumerate(heads_list)]

    iterator = _tqdm(pairs, desc=f"PairwiseDistance ({metric})") if (progress and _HAS_TQDM) else pairs

    for li, hi, layer, head in iterator:
        t0 = time.perf_counter()
        per_head = compute_per_head_distances(
            per_layer_head_diagrams, sample_indices,
            layer=layer, head=head, metric=metric, dims=dims, n_jobs=n_jobs,
        )
        elapsed = time.perf_counter() - t0
        logger.info(
            "compute_full_distance_tensor: layer=%d head=%d metric=%s elapsed=%.2fs",
            layer, head, metric, elapsed,
        )
        tensor[li, hi] = per_head

    return tensor


def cache_path(cache_dir: Union[str, pathlib.Path], metric: str) -> pathlib.Path:
    """Return the canonical cache file path for a given metric.

    Both homology dimensions (H_0 and H_1) are stored in a single ``.npz``
    file; there is no per-dim suffix.

    Parameters
    ----------
    cache_dir:
        Directory where distance tensors are cached.
    metric:
        Distance metric string (``'wasserstein'`` or ``'bottleneck'``).

    Returns
    -------
    pathlib.Path
        ``cache_dir / f'{metric}.npz'``
    """
    return pathlib.Path(cache_dir) / f"{metric}.npz"


def save_distance_tensor(
    tensor: np.ndarray,
    sample_metadata: pd.DataFrame,
    dims: tuple,
    metric: str,
    path: Union[str, pathlib.Path],
) -> None:
    """Save a distance tensor and its sample metadata to a ``.npz`` file.

    The metadata DataFrame is serialised to JSON (records orientation) and
    stored as a bytes scalar inside the archive.  This lets downstream subtasks
    load a self-contained tensor without re-deriving the ``(lang, term)``
    mapping.

    Parameters
    ----------
    tensor:
        Float32 array of shape ``(n_layers, n_heads, N, N, len(dims))``.
    sample_metadata:
        DataFrame with one row per sample; must have at least ``['lang', 'term',
        'sentence_idx_within_term', 'source_file', 'source_part']`` columns.
    dims:
        Tuple of homology dimensions stored in the tensor's last axis.
    metric:
        Distance metric string (``'wasserstein'`` or ``'bottleneck'``).
    path:
        Destination ``.npz`` path.  Parent directory must exist.

    Notes
    -----
    Saved keys:
      - ``tensor`` — the float32 distance array.
      - ``metadata_json`` — DataFrame serialised as a UTF-8 JSON byte array.
      - ``metric`` — 0-d string array.
      - ``homology_dimensions`` — int array of length ``len(dims)``.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_json = sample_metadata.to_json(orient="records").encode("utf-8")

    np.savez(
        path,
        tensor=tensor.astype(np.float32),
        metadata_json=np.frombuffer(metadata_json, dtype=np.uint8),
        metric=np.array(metric),
        homology_dimensions=np.array(list(dims), dtype=np.int32),
    )


def load_distance_tensor(
    path: Union[str, pathlib.Path],
) -> tuple:
    """Load a distance tensor saved by ``save_distance_tensor``.

    Parameters
    ----------
    path:
        Path to a ``.npz`` file written by ``save_distance_tensor``.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame, str, tuple]
        ``(tensor, metadata_df, metric, homology_dimensions)`` where:

        - ``tensor`` — float32 array, shape as saved.
        - ``metadata_df`` — DataFrame reconstructed from the saved JSON.
        - ``metric`` — distance metric string (``'wasserstein'`` or ``'bottleneck'``).
        - ``homology_dimensions`` — tuple of ints, e.g. ``(0, 1)``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If required keys are missing from the archive.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Distance tensor cache not found: {path}")

    archive = np.load(path, allow_pickle=False)

    tensor = archive["tensor"].astype(np.float32)

    metadata_bytes = archive["metadata_json"].tobytes()
    metadata_df = pd.read_json(io.BytesIO(metadata_bytes), orient="records")

    metric = str(archive["metric"])

    homology_dimensions = tuple(int(d) for d in archive["homology_dimensions"])

    return tensor, metadata_df, metric, homology_dimensions


# ---------------------------------------------------------------------------
# Permutation test helpers
# ---------------------------------------------------------------------------

def _per_domain_test_statistic_from_arrays(
    dists_upper: np.ndarray,
    between_mask: np.ndarray,
    total_sum: float,
    n_pairs: int,
) -> float:
    """Compute the per-domain test statistic from pre-extracted upper-triangle arrays.

    This is the inner-loop kernel used by both ``per_domain_test_statistic``
    (public API, delegates here after extracting arrays) and
    ``permutation_test_per_domain`` (computes the observed statistic once
    before running the vectorised null distribution).

    Computing from pre-extracted arrays avoids rebuilding the full N×N mask
    and re-indexing the distance matrix on every call.  All three per-cell
    invariants (``dists_upper``, ``total_sum``, ``n_pairs``) are computed
    once outside the loop; only ``between_mask`` changes per permutation.

    Parameters
    ----------
    dists_upper:
        1-D float array of upper-triangle distances, length ``n_pairs``.
        Produced by ``distance_matrix[triu_i, triu_j]``.
    between_mask:
        1-D bool array of length ``n_pairs``.  ``True`` where the two
        samples belong to different languages.
    total_sum:
        ``dists_upper.sum()`` — precomputed once per cell so that
        ``within_sum = total_sum - between_sum`` avoids a second dot product.
    n_pairs:
        ``len(dists_upper)`` — number of upper-triangle pairs.

    Returns
    -------
    float
        ``mean(between-language pair distances) − mean(within-language pair
        distances)``.  If either set is empty the missing mean defaults to
        ``0.0``.
    """
    between_sum = float(np.dot(between_mask, dists_upper))
    between_count = int(between_mask.sum())
    within_sum = total_sum - between_sum
    within_count = n_pairs - between_count

    mean_between = between_sum / between_count if between_count > 0 else 0.0
    mean_within = within_sum / within_count if within_count > 0 else 0.0

    return mean_between - mean_within


def per_domain_test_statistic(
    distance_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
) -> float:
    """Compute mean(between-language distances) − mean(within-language distances).

    The test statistic for the per-domain permutation test. A positive value
    means between-language pairs are on average farther apart in persistence-
    diagram space than within-language pairs — the direction predicted by the
    hypothesis that languages encode distinct attentional topology.

    Parameters
    ----------
    distance_matrix:
        Square float array of shape ``(N, N)``. The pairwise distance matrix
        for a single ``(layer, head)``.
    metadata_df:
        DataFrame with a ``'lang'`` column, one row per sample. Must have
        ``N`` rows corresponding to the ``N`` samples in ``distance_matrix``.

    Returns
    -------
    float
        mean(between-language pair distances) − mean(within-language pair
        distances). If either set of pairs is empty, the missing mean is
        treated as 0.0.

    Notes
    -----
    Uses the upper-triangle of the mask to avoid double-counting symmetric
    pairs, consistent with the permutation null distribution computation.
    The cross-language mask is built via broadcasting:
    ``lang_vec[:, None] != lang_vec[None, :]``.

    Delegates to ``_per_domain_test_statistic_from_arrays`` after extracting
    the upper-triangle arrays, so the inner arithmetic is shared with the
    vectorised permutation loop.
    """
    lang_vec = metadata_df["lang"].values
    n = len(lang_vec)
    triu_i, triu_j = np.triu_indices(n, k=1)

    dists_upper = distance_matrix[triu_i, triu_j]
    total_sum = float(dists_upper.sum())
    n_pairs = len(dists_upper)
    between_mask = lang_vec[triu_i] != lang_vec[triu_j]

    return _per_domain_test_statistic_from_arrays(
        dists_upper=dists_upper,
        between_mask=between_mask,
        total_sum=total_sum,
        n_pairs=n_pairs,
    )


# Target peak memory per chunk for the (chunk_size, n_pairs) float32 matrix.
# 500 MB / 4 bytes per float32 = 125 M elements.  With n_pairs ~ 230k for
# N=680, chunk_size = 125M / 230k ≈ 540.  We cap at 256 to be safe and to
# keep BLAS working set in L3 cache.
_CHUNK_SIZE_DEFAULT = 256


def permutation_test_per_domain(
    distance_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    K: int = 10000,
    seed: int = 42,
    _chunk_size: int = _CHUNK_SIZE_DEFAULT,
) -> dict:
    """Permutation test: are between-language distances larger than within-language?

    Shuffles the ``lang`` label vector ``K`` times and builds the null
    distribution via a BLAS-accelerated matrix multiplication over all
    permutations at once (chunked to stay under ~500 MB peak memory).
    Returns the observed statistic, null distribution, two-tailed p-value,
    and effect size (z-score under the null).

    Parameters
    ----------
    distance_matrix:
        Square float array of shape ``(N, N)``.
    metadata_df:
        DataFrame with a ``'lang'`` column, one row per sample. ``N`` rows.
    K:
        Number of permutations. Default 10000.
    seed:
        Random seed for ``np.random.default_rng``. Default 42.
    _chunk_size:
        Number of permutations to process per BLAS chunk.  Default 256.
        Exposed for testing; production code uses the default.

    Returns
    -------
    dict with keys:
        - ``'observed'`` (float): test statistic on original labels.
        - ``'null'`` (np.ndarray, shape ``(K,)``): null distribution.
        - ``'p_value'`` (float): two-tailed p-value with finite-K correction
          ``(sum(|null - mean(null)| >= |observed - mean(null)|) + 1) / (K + 1)``.
        - ``'effect_size'`` (float): z-score under null,
          ``(observed - mean(null)) / std(null)``.

    Notes
    -----
    The permutation shuffles the **label vector** (length N), not the rows of
    the distance matrix. This is the standard Mantel-style approach: the
    distances are fixed; only the sample assignment to languages changes.

    The two-tailed p-value formula uses the standard finite-K correction
    (Phipson & Smyth 2010): ``(B + 1) / (K + 1)`` where B is the count of
    null statistics at least as extreme as observed (in absolute deviation from
    the null mean).

    **Vectorisation strategy** (Optimization 1 + 2 from ph-project-chz):

    *Optimization 1 — cache per-cell invariants:*
    Upper-triangle indices, ``dists_upper``, ``total_sum``, and ``n_pairs``
    are computed once.  The per-permutation between-mask uses only the
    ~n_pairs upper-triangle entries (not the full N×N matrix).

    *Optimization 2 — BLAS matmul across K permutations:*
    All K shuffled label vectors are generated upfront as a (K, N) matrix.
    The between-masks for all permutations are stacked into a
    (chunk_size, n_pairs) float32 matrix and multiplied against the
    (n_pairs,) ``dists_upper`` vector in one BLAS call per chunk:
    ``between_sums = between_masks @ dists_upper``.  BLAS uses SIMD and
    multi-core threading, replacing K small Python-loop numpy calls with
    one cache-blocked operation per chunk.

    The null array VALUES differ from the old Python-loop implementation
    because batched RNG generation consumes the stream in a different order
    than K sequential ``rng.permutation()`` calls.  This is expected and
    statistically valid — p-value and effect_size match within Monte-Carlo
    noise.
    """
    rng = np.random.default_rng(seed)
    lang_vec = metadata_df["lang"].values.copy()
    n = len(lang_vec)

    # --- Per-cell invariants (computed once) ---
    triu_i, triu_j = np.triu_indices(n, k=1)
    dists_upper = distance_matrix[triu_i, triu_j]
    total_sum = float(dists_upper.sum())
    n_pairs = len(dists_upper)

    # Observed statistic on the unshuffled labels
    between_mask_obs = lang_vec[triu_i] != lang_vec[triu_j]
    observed = _per_domain_test_statistic_from_arrays(
        dists_upper=dists_upper,
        between_mask=between_mask_obs,
        total_sum=total_sum,
        n_pairs=n_pairs,
    )

    # --- Generate all K permutations upfront as a (K, n) integer-coded array ---
    # Use a compact integer encoding to keep the permutation matrix small before
    # expanding to between-masks.
    lang_int = np.unique(lang_vec, return_inverse=True)[1].astype(np.int16)
    shuffled_all = np.empty((K, n), dtype=np.int16)
    for k in range(K):
        shuffled_all[k] = rng.permutation(lang_int)

    # --- Chunked BLAS matmul to compute null distribution ---
    # between_masks chunk: (chunk_size, n_pairs) float32
    # dists_upper_f32: (n_pairs,) float32  (cast once)
    dists_upper_f32 = dists_upper.astype(np.float32)

    null = np.empty(K, dtype=np.float64)
    chunk = _chunk_size

    for start in range(0, K, chunk):
        end = min(start + chunk, K)
        batch = shuffled_all[start:end]          # (batch_size, n)

        # Build between-masks: (batch_size, n_pairs) bool → float32
        between_masks = (batch[:, triu_i] != batch[:, triu_j]).astype(np.float32)

        # BLAS dot: (batch_size, n_pairs) @ (n_pairs,) → (batch_size,)
        between_sums = between_masks @ dists_upper_f32  # float32 accumulation
        between_counts = between_masks.sum(axis=1)

        within_sums = total_sum - between_sums.astype(np.float64)
        within_counts = n_pairs - between_counts

        # Avoid division by zero (same guard as the scalar helper).
        # Use np.divide with the 'out' default and 'where' mask to suppress
        # the RuntimeWarning that np.where emits (it evaluates both branches).
        mean_between = np.zeros(len(between_counts), dtype=np.float64)
        np.divide(between_sums, between_counts, out=mean_between, where=between_counts > 0)
        mean_within = np.zeros(len(within_counts), dtype=np.float64)
        np.divide(within_sums, within_counts, out=mean_within, where=within_counts > 0)

        null[start:end] = mean_between - mean_within

    null_mean = null.mean()
    null_std = null.std()

    # Two-tailed p-value with finite-K correction (Phipson & Smyth 2010)
    extreme_count = int(np.sum(np.abs(null - null_mean) >= np.abs(observed - null_mean)))
    p_value = (extreme_count + 1) / (K + 1)

    # Effect size: z-score under null
    effect_size = float((observed - null_mean) / null_std) if null_std > 1e-10 else 0.0

    return {
        "observed": float(observed),
        "null": null,
        "p_value": float(p_value),
        "effect_size": float(effect_size),
    }


def _bh_correction(
    pvalues: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns a boolean mask of the same length: True where the hypothesis is
    rejected at the given false discovery rate (``alpha``).

    Implementation follows the standard BH step-up procedure:
        1. Sort p-values ascending.
        2. For rank k (1-indexed), the BH threshold is ``k/m * alpha``.
        3. Find the largest k where ``p_(k) <= k/m * alpha``.
        4. Reject all hypotheses with rank <= that k (step-up property).

    Parameters
    ----------
    pvalues:
        Array of p-values. Length ``m``.
    alpha:
        False discovery rate threshold. Default 0.05.

    Returns
    -------
    np.ndarray of bool
        Boolean mask, length ``m``. ``True`` = rejected (significant).

    Notes
    -----
    Verbatim re-implementation of the ``_bh_correction`` helper in
    ``notebooks/phase3_comparison.ipynb`` cell 11. Kept here so notebook
    code can import it rather than redefining it inline.
    """
    pv = np.asarray(pvalues, dtype=float)
    m = len(pv)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pv)
    thresholds = (np.arange(1, m + 1) / m) * alpha
    # find the largest rank k where p_{(k)} <= threshold_k
    sig_at_rank = pv[order] <= thresholds
    # if any rank k is significant, all ranks <= k are also significant (step-up)
    # cummax from the right enforces this monotonicity property
    cummax_right = np.maximum.accumulate(sig_at_rank[::-1])[::-1]
    reject_order = cummax_right
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_order
    return reject


def permutation_test_per_head(
    distance_tensor: np.ndarray,
    metadata_df: pd.DataFrame,
    K: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply the per-domain permutation test independently for each (layer, head).

    Runs ``permutation_test_per_domain`` for every ``(layer, head)`` cell in the
    distance tensor and applies Benjamini-Hochberg FDR correction across all
    ``n_layers * n_heads`` tests at q = 0.05.

    Parameters
    ----------
    distance_tensor:
        Float array of shape ``(n_layers, n_heads, N, N)``. The H_1 (or H_0)
        sub-tensor sliced from the cached distance tensor.
    metadata_df:
        DataFrame with a ``'lang'`` column, one row per sample. ``N`` rows.
    K:
        Number of permutations per test. Default 10000.
    seed:
        Base random seed. Each ``(layer, head)`` uses ``seed + layer * n_heads + head``
        to ensure independence across cells while remaining deterministic.

    Returns
    -------
    pd.DataFrame
        One row per ``(layer, head)``. Columns:
        ``[layer, head, observed, p_value, effect_size, passes_bh]``.
        ``passes_bh`` is a boolean column indicating BH rejection at q = 0.05.

    Notes
    -----
    The BH correction is applied across all ``n_layers * n_heads`` tests
    simultaneously (144 tests for the full 12×12 mBERT grid), not per-layer
    or per-head independently.
    """
    n_layers, n_heads = distance_tensor.shape[:2]
    rows = []

    for layer in range(n_layers):
        for head in range(n_heads):
            cell_seed = seed + layer * n_heads + head
            result = permutation_test_per_domain(
                distance_tensor[layer, head],
                metadata_df,
                K=K,
                seed=cell_seed,
            )
            rows.append({
                "layer": layer,
                "head": head,
                "observed": result["observed"],
                "p_value": result["p_value"],
                "effect_size": result["effect_size"],
            })

    df = pd.DataFrame(rows)

    # BH correction across all n_layers * n_heads tests simultaneously
    reject_mask = _bh_correction(df["p_value"].values, alpha=0.05)
    df["passes_bh"] = reject_mask.astype(bool)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-term permutation test helpers (blr.4)
# ---------------------------------------------------------------------------

def _normalize_gloss(g: str) -> str:
    """Strip ``"dark "`` or ``"light "`` prefix and lowercase a YAML gloss string.

    The prefixes are stripped **unconditionally** — not just before "blue".
    The primary motivating case is the Russian-blue split:

    ``_normalize_gloss("dark blue")  → "blue"``
    ``_normalize_gloss("light blue") → "blue"``

    but the implementation strips the prefix regardless of what follows it
    (e.g. ``"dark green"`` → ``"green"`` if that ever appeared in a YAML).

    Without this normalisation, a naive merge on ``gloss == en_term`` silently
    drops **both** Russian blue BCTs (синий and голубой) from the translation
    table, leaving 10 rows instead of the correct 12.

    The function is idempotent: applying it twice yields the same result as
    applying it once.

    Parameters
    ----------
    g:
        Raw gloss string from a YAML entry (e.g. ``"dark blue"``, ``"purple"``).

    Returns
    -------
    str
        Lowercased gloss with ``"dark "`` or ``"light "`` prefix removed.
    """
    g = g.lower()
    if g.startswith("dark "):
        g = g[len("dark "):]
    elif g.startswith("light "):
        g = g[len("light "):]
    return g


def load_translation_triples(
    canon_dir: Union[str, pathlib.Path],
    domain: str = "color",
) -> pd.DataFrame:
    """Load cross-linguistic translation triples from canon-term YAMLs.

    Reads ``<canon_dir>/en/<domain>.yaml``, ``<canon_dir>/ru/<domain>.yaml``,
    and ``<canon_dir>/es/<domain>.yaml`` and builds a cross-linguistic
    translation table.

    ## Column layout

    ``[en_term, ru_term, es_term, ru_gloss_raw]``

    - ``en_term`` — English BCT string (e.g. ``"blue"``).
    - ``ru_term`` — Russian BCT string (e.g. ``"синий"`` or ``"голубой"``).
    - ``es_term`` — Spanish BCT string (e.g. ``"azul"``).
    - ``ru_gloss_raw`` — the raw Russian gloss as written in the YAML (e.g.
      ``"dark blue"``); preserves ``"dark blue"``/``"light blue"`` for
      downstream interpretation without stripping.

    ## Russian-blue duplication

    For the color domain, both Russian blue BCTs share the same normalized
    gloss (``"blue"``), so the join produces **two rows** for ``en_term="blue"``:

        en_term=blue  ru_term=синий    es_term=azul  ru_gloss_raw="dark blue"
        en_term=blue  ru_term=голубой  es_term=azul  ru_gloss_raw="light blue"

    This duplication is intentional and documented.  The color domain therefore
    produces exactly **12 rows** (10 one-to-one triples + 2 for blue).

    ## Join logic

    For each English term ``t``:
      - Find all Russian entries where ``_normalize_gloss(entry["gloss"]) == t``.
        Produces one match for all non-blue terms; two matches for "blue".
      - Find the Spanish entry where ``_normalize_gloss(entry["gloss"]) == t``.
        Expected to be exactly one match.
      - Emit one row per (Russian match, Spanish match) combination.

    Parameters
    ----------
    canon_dir:
        Root of the canon-terms directory tree.  Expected layout:
        ``<canon_dir>/<lang>/<domain>.yaml``.
    domain:
        Domain slug (``"color"`` for the May 2026 scope).

    Returns
    -------
    pd.DataFrame
        Columns: ``[en_term, ru_term, es_term, ru_gloss_raw]``.
        For the color domain, exactly 12 rows.

    Raises
    ------
    FileNotFoundError
        If any of the three YAML files are missing.

    Notes
    -----
    If an English term has no Spanish counterpart (i.e., the normalized gloss
    does not appear in ``es_by_norm``), the ``es_term`` column will be ``None``
    for that row.  With the current complete canon files this does not occur,
    but no ``ValueError`` is raised — callers should validate if needed.
    """
    canon_dir = pathlib.Path(canon_dir)

    def _load_yaml(lang: str) -> list[dict]:
        path = canon_dir / lang / f"{domain}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Canon-term YAML not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("terms", [])

    en_terms_raw = _load_yaml("en")
    ru_terms_raw = _load_yaml("ru")
    es_terms_raw = _load_yaml("es")

    # English: term only (no gloss field)
    # The en_term itself IS the canonical gloss key.
    en_list = [entry["term"] for entry in en_terms_raw]

    # Russian: build a map from normalized gloss → list of (term, raw_gloss) tuples
    # Multiple entries can share the same normalized gloss (синий + голубой → "blue").
    ru_by_norm: dict[str, list[tuple[str, str]]] = {}
    for entry in ru_terms_raw:
        raw_gloss = entry.get("gloss", entry["term"])
        norm = _normalize_gloss(raw_gloss)
        ru_by_norm.setdefault(norm, []).append((entry["term"], raw_gloss))

    # Spanish: build a map from normalized gloss → term
    # Expected to be 1:1 with English for all color BCTs.
    es_by_norm: dict[str, str] = {}
    for entry in es_terms_raw:
        raw_gloss = entry.get("gloss", entry["term"])
        norm = _normalize_gloss(raw_gloss)
        es_by_norm[norm] = entry["term"]

    rows = []
    for en_term in en_list:
        # Russian matches (may be >1 for "blue")
        ru_matches = ru_by_norm.get(en_term, [])
        # Spanish match (expected exactly 1)
        es_term = es_by_norm.get(en_term)

        for ru_term, ru_gloss_raw in ru_matches:
            rows.append({
                "en_term": en_term,
                "ru_term": ru_term,
                "es_term": es_term,
                "ru_gloss_raw": ru_gloss_raw,
            })

    return pd.DataFrame(rows, columns=["en_term", "ru_term", "es_term", "ru_gloss_raw"])


def per_term_test_statistic(
    distance_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    ru_blue_choice: str = "синий",
) -> float:
    """Compute the mean cross-language same-triple distance over all translation triples.

    For each row in ``triples_df``, this function identifies all sample pairs
    ``(i, j)`` where sample ``i`` and sample ``j`` are from **different languages**
    and their ``(lang, term)`` matches the triple's languages and terms.  The
    test statistic is the mean distance over all such cross-language same-triple
    pairs.

    **Within-language pairs are excluded by design.**  A "translation cluster"
    is defined by cross-language proximity; within-language similarity is
    irrelevant to this test.

    Parameters
    ----------
    distance_matrix:
        Square float array of shape ``(N, N)``.  Pairwise distances.
    metadata_df:
        DataFrame with ``'lang'`` and ``'term'`` columns.  ``N`` rows.
    triples_df:
        DataFrame with columns ``[en_term, ru_term, es_term, ru_gloss_raw]``
        as returned by ``load_translation_triples``.  Each row defines one
        translation triple.
    ru_blue_choice:
        Which Russian blue to use for the ``en_term == "blue"`` row.
        Must be one of ``{'синий', 'голубой'}``.  Selects the matching row
        from ``triples_df`` when there are two blue rows.

    Returns
    -------
    float
        Mean distance over all cross-language same-triple sample pairs.

    Notes
    -----
    The ``ru_blue_choice`` parameter resolves the ambiguity introduced by the
    Russian-blue duplication in ``triples_df``.  When the caller supplies a
    triples_df that already has exactly one blue row (because they pre-filtered
    to sinij or goluboy), this parameter is still safe to pass and will be
    consistent with that row.
    """
    lang_vec = metadata_df["lang"].values
    term_vec = metadata_df["term"].values
    n = len(lang_vec)

    triu_i, triu_j = np.triu_indices(n, k=1)

    li = lang_vec[triu_i]
    lj = lang_vec[triu_j]
    ti = term_vec[triu_i]
    tj = term_vec[triu_j]

    # Pre-compute cross-language mask (upper-triangle, shape (P,))
    cross_lang = li != lj

    # Pre-filter blue rows once to avoid re-filtering inside the loop.
    blue_rows = triples_df[triples_df["en_term"] == "blue"]
    has_multiple_blues = len(blue_rows) > 1

    # Build a (N,) array: triple_id[k] = the index of the triple that sample k
    # belongs to under the chosen selection, or -1 if no triple matches.
    # We then vectorize the pair membership check using boolean comparison on
    # (N,)-shaped arrays — same style as per_domain_test_statistic.
    triple_id_vec = np.full(n, -1, dtype=np.intp)

    for triple_idx, (_, row) in enumerate(triples_df.iterrows()):
        # Skip the non-chosen Russian-blue variant when two blue rows are present.
        if row["en_term"] == "blue" and has_multiple_blues:
            if row["ru_term"] != ru_blue_choice:
                continue

        triple_lang_to_term: dict[str, str] = {
            "en": row["en_term"],
            "ru": row["ru_term"],
            "es": row["es_term"],
        }

        # Vectorized membership: sample k belongs to this triple iff its
        # (lang, term) pair matches the triple's entry for that language.
        member = np.zeros(n, dtype=bool)
        for lang, term in triple_lang_to_term.items():
            if term is not None:
                member |= (lang_vec == lang) & (term_vec == term)

        # Assign triple_id to all samples that belong to this triple.
        # Samples already assigned to an earlier triple keep their original id
        # (triples are disjoint by construction, so this never overwrites).
        triple_id_vec = np.where(member & (triple_id_vec < 0), triple_idx, triple_id_vec)

    # A pair (triu_i[k], triu_j[k]) is a same-triple cross-language pair iff:
    #   - cross_lang[k] is True (different languages)
    #   - both samples are assigned to the same valid triple (triple_id >= 0)
    #   - and that triple is the same triple for both samples
    id_i = triple_id_vec[triu_i]
    id_j = triple_id_vec[triu_j]
    same_triple_mask = cross_lang & (id_i >= 0) & (id_i == id_j)

    dists_upper = distance_matrix[triu_i, triu_j]
    same_triple_dists = dists_upper[same_triple_mask]

    if len(same_triple_dists) == 0:
        return 0.0

    return float(same_triple_dists.mean())


def permutation_test_per_term(
    distance_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    triples_df: pd.DataFrame,
    K: int = 10000,
    seed: int = 42,
    ru_blue_choice: str = "синий",
) -> dict:
    """Permutation test: do translation triples form proximity clusters?

    Shuffles term labels WITHIN each language ``K`` times and recomputes
    ``per_term_test_statistic`` to build the null distribution.  Language
    labels are never shuffled — only term labels within each language group
    change, so cross-language pair membership changes while within-language
    structure is preserved.

    Parameters
    ----------
    distance_matrix:
        Square float array of shape ``(N, N)``.
    metadata_df:
        DataFrame with ``'lang'`` and ``'term'`` columns.  ``N`` rows.
    triples_df:
        Translation triple table from ``load_translation_triples``.
    K:
        Number of permutations.  Default 10000.
    seed:
        Random seed for ``np.random.default_rng``.  Default 42.
    ru_blue_choice:
        Which Russian blue to use in ``per_term_test_statistic``.
        Default ``'синий'``.

    Returns
    -------
    dict with keys:
        - ``'observed'`` (float): test statistic on original labels.
        - ``'null'`` (np.ndarray, shape ``(K,)``): null distribution.
        - ``'p_value'`` (float): two-tailed Phipson–Smyth correction
          ``(|null − mean(null)| ≥ |observed − mean(null)| + 1) / (K + 1)``.
        - ``'effect_size'`` (float): z-score under null.

    Notes
    -----
    The permutation shuffles term labels within each language group.
    Concretely: for each language ``l``, the subset of ``metadata_df`` rows
    where ``lang == l`` has its ``term`` column permuted independently of
    other languages.  This preserves the number of samples per language but
    randomizes which sample belongs to which term — destroying the within-triple
    signal while keeping the language structure intact.

    The two-tailed p-value formula is consistent with ``permutation_test_per_domain``
    (Phipson & Smyth 2010).
    """
    # Defensive reset_index: callers that pass a metadata_df after .iloc[start:]
    # slicing get a non-zero-based or non-contiguous index.  group.index.to_numpy()
    # returns label values, not positions — which would be used as positional
    # numpy indices, causing IndexError or silent row-shuffling corruption.
    # Resetting to 0-based here is safe and matches russian_blue_zoom's pattern.
    metadata_df = metadata_df.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    observed = per_term_test_statistic(
        distance_matrix, metadata_df, triples_df, ru_blue_choice=ru_blue_choice
    )

    null = np.empty(K, dtype=np.float64)
    for k in range(K):
        # Shuffle term labels within each language independently
        perm_terms = metadata_df["term"].values.copy()
        for lang, group in metadata_df.groupby("lang"):
            idx = group.index.to_numpy()
            perm_terms[idx] = rng.permutation(perm_terms[idx])

        perm_meta = metadata_df.copy()
        perm_meta["term"] = perm_terms

        null[k] = per_term_test_statistic(
            distance_matrix, perm_meta, triples_df, ru_blue_choice=ru_blue_choice
        )

    null_mean = null.mean()
    null_std = null.std()

    extreme_count = int(np.sum(np.abs(null - null_mean) >= np.abs(observed - null_mean)))
    p_value = (extreme_count + 1) / (K + 1)

    effect_size = float((observed - null_mean) / null_std) if null_std > 1e-10 else 0.0

    return {
        "observed": float(observed),
        "null": null,
        "p_value": float(p_value),
        "effect_size": float(effect_size),
    }


# ---------------------------------------------------------------------------
# Per-head signal aggregation helpers (blr.5)
# ---------------------------------------------------------------------------

def rank_heads_by_effect(
    per_head_results_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """Rank (layer, head) cells by absolute effect size, descending.

    Consumes the DataFrame produced by ``permutation_test_per_head`` and
    returns the top-K rows sorted by ``|effect_size|`` descending.  A
    ``rank`` column (1-indexed) is prepended.

    Parameters
    ----------
    per_head_results_df:
        DataFrame with at least columns
        ``[layer, head, observed, p_value, effect_size, passes_bh]``
        as returned by ``permutation_test_per_head``.
    top_k:
        Number of rows to return.  If ``top_k`` exceeds the number of rows
        in ``per_head_results_df``, all rows are returned.

    Returns
    -------
    pd.DataFrame
        Columns: ``[rank, layer, head, observed, p_value, effect_size, passes_bh]``
        (same column set as input plus ``rank``).  Exactly
        ``min(top_k, len(per_head_results_df))`` rows, sorted by
        ``|effect_size|`` descending.  Index is reset (0-based).

    Notes
    -----
    The ``rank`` column is assigned after sorting: rank 1 = largest
    ``|effect_size|``, rank 2 = second largest, etc.  The column is
    prepended (first column) for readability in the notebook table.
    """
    df = per_head_results_df.copy()
    df = df.sort_values(
        by="effect_size",
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)

    n = min(top_k, len(df))
    df = df.iloc[:n].copy().reset_index(drop=True)
    df.insert(0, "rank", range(1, n + 1))

    return df


def effect_heatmap_data(
    per_head_results_df: pd.DataFrame,
) -> np.ndarray:
    """Pivot per-head results into a 12×12 layer×head matrix of effect sizes.

    Rows are layers (0-based), columns are heads (0-based).  Cells not
    present in ``per_head_results_df`` are filled with ``NaN`` — a defensive
    guard that should never be triggered by the normal pipeline but protects
    downstream callers from silent errors if the input DataFrame is a partial
    result.

    Parameters
    ----------
    per_head_results_df:
        DataFrame with at least columns ``[layer, head, effect_size]``
        as returned by ``permutation_test_per_head``.

    Returns
    -------
    np.ndarray of float64
        Shape ``(12, 12)``.  ``result[layer, head]`` is the effect size for
        that ``(layer, head)`` cell, or ``NaN`` if the cell is missing from
        the input.

    Notes
    -----
    The 12×12 dimensions are hard-coded to match the mBERT architecture
    (12 encoder layers, 12 attention heads per layer).  The matrix is
    pre-filled with ``NaN`` via ``np.full`` and then populated by iterating
    rows; cells absent from the input therefore surface as ``NaN`` rather
    than raising a ``KeyError``.  Layer/head indices outside ``[0, 12)`` will
    raise ``IndexError`` from numpy — a deliberate fail-loud guard, since
    such indices indicate a non-mBERT input.  Duplicate ``(layer, head)``
    rows fall through with last-write-wins semantics.
    """
    mat = np.full((12, 12), np.nan, dtype=np.float64)

    for _, row in per_head_results_df.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        mat[layer, head] = float(row["effect_size"])

    return mat


def russian_blue_zoom(
    distance_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    K: int = 10000,
    seed: int = 42,
) -> dict:
    """Within-Russian zoom test: is the синий/голубой split detectable?

    Asks whether mBERT attention topology *within Russian* encodes the
    obligatory sinij/голубой distinction that Russian grammar requires.
    This is a sharper test than the triple test: the triple test asks
    "do translations cluster?"; this asks "does the language-internal
    blue split show up as above-average distance in attention topology?"

    ## Statistic

    Computed on the **Russian-language subset** of ``metadata_df``
    (rows where ``lang == 'ru'``):

        stat = mean d(синий samples, голубой samples)
               − median over all (a, b) where {a, b} ≠ {синий, голубой}
                 of mean d(samples with term=a, samples with term=b)

    A positive statistic means синий and голубой are farther apart (in W_2
    attention topology) than the typical Russian color-pair distance.  The
    prediction from Paramei (2005) and Winawer et al. (2007) is that
    ``observed > 0``.

    ## Null distribution

    Permute term labels within the Russian subset ``K`` times, recomputing
    the statistic each time.

    Parameters
    ----------
    distance_matrix:
        Square float array of shape ``(N, N)``.  Covers all samples, all
        languages.  This function extracts the Russian subset by index.
    metadata_df:
        DataFrame with ``'lang'`` and ``'term'`` columns.  ``N`` rows.
    K:
        Number of permutations.  Default 10000.
    seed:
        Random seed.  Default 42.

    Returns
    -------
    dict with keys:
        - ``'observed'`` (float): statistic on original labels.
        - ``'null'`` (np.ndarray, shape ``(K,)``): null distribution.
        - ``'p_value'`` (float): two-tailed Phipson–Smyth p-value.
        - ``'effect_size'`` (float): z-score under null.

    Notes
    -----
    The function sub-selects the Russian rows by position (iloc) from
    ``metadata_df`` and slices the corresponding rows/cols from
    ``distance_matrix``.  Only term labels within Russian are permuted;
    the Russian subset size is unchanged.
    """
    rng = np.random.default_rng(seed)

    # Extract Russian subset
    ru_mask = metadata_df["lang"].values == "ru"
    ru_indices = np.where(ru_mask)[0]  # positional indices

    ru_meta = metadata_df.iloc[ru_indices].reset_index(drop=True)
    ru_dist = distance_matrix[np.ix_(ru_indices, ru_indices)]

    def _blue_stat(meta_sub: pd.DataFrame, dist_sub: np.ndarray) -> float:
        """Compute the Russian-blues statistic on a sub-matrix."""
        term_vec = meta_sub["term"].values
        n = len(term_vec)
        triu_i, triu_j = np.triu_indices(n, k=1)

        # Get all unique Russian terms
        all_terms = list(meta_sub["term"].unique())

        # Find синий and голубой indices
        sinij_idx = np.where(term_vec == "синий")[0]
        goluboy_idx = np.where(term_vec == "голубой")[0]

        if len(sinij_idx) == 0 or len(goluboy_idx) == 0:
            # Can't compute if one blue is missing (shouldn't happen on real data)
            return 0.0

        # mean d(синий samples, голубой samples) — all cross-pair combinations
        blue_pairs = np.array([
            [i, j] for i in sinij_idx for j in goluboy_idx
        ])
        mean_blue = float(dist_sub[blue_pairs[:, 0], blue_pairs[:, 1]].mean())

        # median over all (a, b) term pairs where {a, b} != {синий, голубой}
        # of mean d(samples with a, samples with b)
        pair_means = []
        for k_a in range(len(all_terms)):
            for k_b in range(k_a + 1, len(all_terms)):
                ta, tb = all_terms[k_a], all_terms[k_b]
                if {ta, tb} == {"синий", "голубой"}:
                    continue
                idx_a = np.where(term_vec == ta)[0]
                idx_b = np.where(term_vec == tb)[0]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    continue
                cross = np.array([
                    [i, j] for i in idx_a for j in idx_b
                ])
                pair_means.append(float(dist_sub[cross[:, 0], cross[:, 1]].mean()))

        if len(pair_means) == 0:
            return 0.0

        median_other = float(np.median(pair_means))
        return mean_blue - median_other

    observed = _blue_stat(ru_meta, ru_dist)

    null = np.empty(K, dtype=np.float64)
    for k in range(K):
        perm_terms = rng.permutation(ru_meta["term"].values)
        perm_meta = ru_meta.copy()
        perm_meta["term"] = perm_terms
        null[k] = _blue_stat(perm_meta, ru_dist)

    null_mean = null.mean()
    null_std = null.std()

    extreme_count = int(np.sum(np.abs(null - null_mean) >= np.abs(observed - null_mean)))
    p_value = (extreme_count + 1) / (K + 1)

    effect_size = float((observed - null_mean) / null_std) if null_std > 1e-10 else 0.0

    return {
        "observed": float(observed),
        "null": null,
        "p_value": float(p_value),
        "effect_size": float(effect_size),
    }
