"""Shared helper: persistence diagram → giotto-tda PairwiseDistance format.

Used by both:
  - ``replication/diagram_distances.py`` (Kushnareva cross-linguistic pipeline)
  - ``draganov_replication/pd_distances.py`` (Draganov replication pipeline)

Extracted from ``replication/diagram_distances.py`` (ph-project-inu.5) so the
padding contract lives in exactly one place. Both callers import from here.
"""
from __future__ import annotations

import numpy as np


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
