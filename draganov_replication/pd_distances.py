"""Compute pairwise PD-distance grids across 34 (lang, term) cells.

Direct port of Draganov's B-step:
``draganov/B-from_persistence_diagrams_to_language_distances/run_compute_pd_distances.py``

Draganov correspondence
-----------------------

+---------------------------------------------------------------------+------------------------------------+
| Draganov function                                                   | Our equivalent                     |
+=====================================================================+====================================+
| ``compare_pds(pd_a, pd_b, distance_name)``                          | ``compute_pd_distance_grid(...)``  |
| ``run_compute_pd_distances.py:111-115``                             | (vectorised over the 34×34 grid;   |
|                                                                     | per-pair semantics preserved)      |
+---------------------------------------------------------------------+------------------------------------+
| ``vectorise_persistence_image(pd, dim)``                            | ``vectorise_persistence_image``    |
| ``run_compute_pd_distances.py:59-69``                               | same name, same math,              |
|                                                                     | recalibrated birth/pers ranges     |
|                                                                     | for mBERT cosine (see docstring)   |
+---------------------------------------------------------------------+------------------------------------+
| ``vectorise_bars_statistics(pd, only_death)``                       | ``vectorise_bars_statistics``      |
| ``run_compute_pd_distances.py:80-108``                              | same name, identical 10-statistic  |
|                                                                     | vector; 10-d (only_death=True)     |
|                                                                     | or 40-d otherwise                  |
+---------------------------------------------------------------------+------------------------------------+
| Bottleneck via ``gtda.diagrams.PairwiseDistance``                   | same library; uses shared          |
|                                                                     | ``replication.giotto_format.       |
|                                                                     | to_giotto_format`` helper          |
+---------------------------------------------------------------------+------------------------------------+
| Sliced Wasserstein via ``persim.sliced_wasserstein(M=50)``          | same library + same M=50           |
+---------------------------------------------------------------------+------------------------------------+
| Output: per-(distance, dim) matrix on disk                          | ``np.savez(out_dir /               |
| ``merge_pd_distances_to_matrix.py``                                 | f'{distance}_d{dim}.npz', ...)``   |
+---------------------------------------------------------------------+------------------------------------+

Substrate-driven divergences
-----------------------------

1. **Persistence-image calibration ranges**: Draganov's cosine ranges
   (``run_compute_pd_distances.py:156-162``) are calibrated for fastText 300d
   cosine, where H_0 deaths are tiny (``birth_range=(0, 0.1)``). Our mBERT 768d
   cosine H_0 deaths span much of [0, 0.72] (empirically: 50th pct ≈ 0.29,
   99th pct ≈ 0.60, max ≈ 0.72). Using ``birth_range=(0, 1)`` and
   ``pers_range=(0, 1)`` for both dim 0 and dim 1.

2. **Distributed runner**: Draganov parallelises across N=81 language pairs
   (3240 pairs per matrix). At N=34 (561 pairs), single-process is <10 minutes
   total — no distribution needed.

3. **Bottleneck packing**: Draganov calls ``gudhi.bottleneck_distance`` per pair.
   We use ``gtda.diagrams.PairwiseDistance(metric='bottleneck')`` which is
   vectorised and uses the shared ``replication.giotto_format.to_giotto_format``
   helper for packing.  The packing wrapper uses a fake ``{(0, 0): pds_list}``
   dict shape so the existing API (designed for per-(layer, head) Kushnareva
   diagrams) receives the 34 cells as if they are samples for a single head.

Public API
----------
compute_pd_distance_grid(...)
    Produce 34×34 PD distance matrices for each (distance, dim) combination.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd
import persim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration constants (mBERT cosine; diverge from Draganov's fastText)
# ---------------------------------------------------------------------------

# Draganov cosine dim-0 (fastText):  birth_range=(0, 0.1), pers_range=(0, 1)
# Draganov cosine dim-1 (fastText):  birth_range=(0, 1),   pers_range=(0, 1)
#
# mBERT 768d cosine empirical check (data/phase3/draganov_diagrams/*.npz):
#   H0 births: all exactly 0.0 (Vietoris-Rips H_0 always starts at 0)
#   H0 deaths: 50th pct ≈ 0.29, 90th ≈ 0.50, 99th ≈ 0.60, max ≈ 0.72
#   H1 births: 50th pct ≈ 0.40, 90th ≈ 0.56, 99th ≈ 0.63, max ≈ 0.67
#   H1 deaths: 50th pct ≈ 0.42, 90th ≈ 0.58, 99th ≈ 0.65, max ≈ 0.69
#
# Draganov's birth_range=(0, 0.1) would capture <1% of our H_0 death range.
# We use (0, 1) for both dims to cover the full [0, 0.72] range with headroom.

_PI_BIRTH_RANGE = (0.0, 1.0)
_PI_PERS_RANGE = (0.0, 1.0)
_PI_PIXEL_SIZE = 0.1
_PI_SIGMA = 0.1


# ---------------------------------------------------------------------------
# Vectorisation helpers
# (ports of Draganov run_compute_pd_distances.py:59-108; names preserved)
# ---------------------------------------------------------------------------

def vectorise_persistence_image(diagram: np.ndarray, dim: int) -> np.ndarray:
    """Vectorise a persistence diagram via persim.PersistenceImager.

    Reproduces ``draganov/B run_compute_pd_distances.py:59-69``.
    Wraps ``persim.PersistenceImager`` and returns a flat 1-D vector.

    Parameters
    ----------
    diagram:
        Shape ``(n, 2)`` float array of ``[birth, death]`` pairs.  May be
        empty (returns a zero vector of the same shape as a non-empty PD).
    dim:
        Homology dimension (0 or 1).  The value itself is not used to change
        the computation — both dims use the same calibrated ranges — but it is
        retained in the signature to match Draganov's API.

    Returns
    -------
    np.ndarray of float64
        Flat 1-D vector (length = (range/pixel_size)^2 = 100 for our defaults).

    Notes on calibration (delta from Draganov defaults)
    ---------------------------------------------------
    Draganov's fastText cosine ranges (``run_compute_pd_distances.py:155–162``):
      - dim 0:  birth_range=(0, 0.1), pers_range=(0, 1),  pixel_size=0.1, sigma=0.1
      - dim 1:  birth_range=(0, 1),   pers_range=(0, 1),  pixel_size=0.1, sigma=0.1

    Our mBERT cosine ranges (both dims):
      - birth_range=(0, 1), pers_range=(0, 1), pixel_size=0.1, sigma=0.1

    Empirical justification: H_0 deaths in our data span [0, 0.72] with the
    50th percentile at 0.29.  Draganov's dim-0 birth_range=(0, 0.1) would miss
    >99% of our H_0 death values.  Using (0, 1) ensures the imager captures
    the full range of our mBERT cosine distances for both dims.
    """
    pi_transformer = persim.PersistenceImager(
        birth_range=_PI_BIRTH_RANGE,
        pers_range=_PI_PERS_RANGE,
        pixel_size=_PI_PIXEL_SIZE,
        weight="persistence",
        weight_params={},
        kernel_params={"sigma": [[_PI_SIGMA, 0], [0, _PI_SIGMA]]},
    )

    if len(diagram) == 0:
        # Return a zero vector matching the imager's output size.
        # Compute shape on a dummy diagram, then return zeros.
        dummy = np.array([[0.0, 0.1]])
        dummy_img = pi_transformer.transform([dummy])[0]
        return np.zeros(dummy_img.size, dtype=np.float64)

    img = pi_transformer.transform([diagram])[0]
    return img.flatten()


def _entropy(values: np.ndarray) -> float:
    """Normalised topological entropy.

    Reproduces ``draganov/B run_compute_pd_distances.py:72-77``.
    """
    total = float(np.sum(values))
    if total <= 0:
        return 0.0
    positive = values[values > 0]
    entropy_unnorm = float(np.sum(positive * np.log(positive)))
    return float(np.log(total) - entropy_unnorm / total)


def vectorise_bars_statistics(diagram: np.ndarray, only_death: bool = False) -> np.ndarray:
    """Compute a 10- or 40-dimensional statistics vector from a persistence diagram.

    Reproduces ``draganov/B run_compute_pd_distances.py:80-108``.

    Computes 10 statistics (mean, std, median, IQR, range, 10/25/75/90th
    percentile, entropy) over either just the death values (``only_death=True``,
    used for dim 0) or all four quantities (deaths, births, lifespans, midpoints;
    used for dim >= 1).

    Parameters
    ----------
    diagram:
        Shape ``(n, 2)`` float array of ``[birth, death]`` pairs.  Empty arrays
        are handled gracefully: each empty quantity falls back to ``np.zeros(1)``
        (matching Draganov's guard).
    only_death:
        If ``True``, restrict to death values only (10-d output).
        If ``False``, include all four quantities (40-d output).

    Returns
    -------
    np.ndarray of float64
        Length 10 (only_death=True) or 40 (only_death=False).
    """
    if only_death:
        quantities = {
            "deaths": lambda birth, death: death,
        }
    else:
        quantities = {
            "births": lambda birth, death: birth,
            "deaths": lambda birth, death: death,
            "lifespans": lambda birth, death: max(0.0, death - birth),
            "midpoints": lambda birth, death: (birth + death) / 2.0,
        }

    statistic_order = [
        "mean",
        "standard deviation",
        "median",
        "interquartile range",
        "full range",
        "10th percentile",
        "25th percentile",
        "75th percentile",
        "90th percentile",
        "entropy",
    ]

    vector: list[float] = []

    for quantity_label in sorted(quantities.keys()):
        quantity = quantities[quantity_label]
        if len(diagram) == 0:
            values = np.zeros(1, dtype=np.float64)
        else:
            values = np.fromiter(
                (quantity(b, d) for b, d in diagram),
                dtype=np.float64,
                count=len(diagram),
            )
            if len(values) == 0:
                values = np.zeros(1, dtype=np.float64)

        statistics = {
            "mean": float(np.mean(values)),
            "standard deviation": float(np.std(values)),
            "median": float(np.median(values)),
            "interquartile range": float(np.subtract(*np.percentile(values, [75, 25]))),
            "full range": float(np.ptp(values)),
            "10th percentile": float(np.percentile(values, 10)),
            "25th percentile": float(np.percentile(values, 25)),
            "75th percentile": float(np.percentile(values, 75)),
            "90th percentile": float(np.percentile(values, 90)),
            "entropy": _entropy(values),
        }

        for stat_label in statistic_order:
            vector.append(statistics[stat_label])

    return np.array(vector, dtype=np.float64)


# ---------------------------------------------------------------------------
# Distance computation helpers
# ---------------------------------------------------------------------------

def _load_diagrams(
    diagrams_dir: pathlib.Path,
    cells: list[tuple[str, str]],
    dim: int,
) -> list[np.ndarray]:
    """Load one PD per cell for a given homology dimension.

    Returns a list of (n, 2) float64 arrays in the same order as ``cells``.
    """
    pds = []
    for lang, term in cells:
        npz_path = diagrams_dir / f"{lang}_{term}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Diagram file not found for ({lang}, {term!r}): {npz_path}"
            )
        d = np.load(npz_path)
        key = f"h{dim}"
        arr = d[key]
        if arr.ndim == 1 and len(arr) == 0:
            arr = np.empty((0, 2), dtype=np.float64)
        pds.append(arr.astype(np.float64))
    return pds


def _bottleneck_matrix(
    pds: list[np.ndarray],
    dim: int,
) -> np.ndarray:
    """Compute pairwise bottleneck distance matrix via giotto-tda.

    Uses ``replication.giotto_format.to_giotto_format`` to pack the 34 PDs
    into the (N, F, 3) format expected by PairwiseDistance.

    The packing wrapper uses a fake ``{(0, 0): pds_list}`` dict so that the
    existing ``to_giotto_format`` API (designed for per-(layer, head) Kushnareva
    diagrams where samples are sentences) receives the 34 cells as if they are
    samples for a single (layer=0, head=0) pair.  We request all 34 cells via
    ``sample_indices=np.arange(34)``.

    Note: ``to_giotto_format`` requires each sample to be a dict mapping
    homology dimension → (n, 2) array.  We pack each PD as ``{dim: pd_arr}``
    and request only that dim, so ``dims=(dim,)`` in the call.
    """
    from gtda.diagrams import PairwiseDistance
    from replication.giotto_format import to_giotto_format

    n = len(pds)

    # If every PD is empty for this dim, to_giotto_format would produce a
    # zero-feature array and PairwiseDistance.fit_transform would raise on
    # the empty stack. Distances between empty diagrams are all 0.
    if all(len(pd_arr) == 0 for pd_arr in pds):
        return np.zeros((n, n), dtype=np.float64)

    # Wrap each PD as a dict[int, np.ndarray] so to_giotto_format is satisfied.
    # Each "sample" is {dim: (n_features, 2)} — only one dim per PD here.
    per_layer_head_diagrams = {
        (0, 0): [{dim: pd_arr} for pd_arr in pds]
    }
    sample_indices = np.arange(n)

    giotto_arr = to_giotto_format(
        per_layer_head_diagrams,
        sample_indices,
        layer=0,
        head=0,
        dims=(dim,),
    )
    # giotto_arr shape: (n, max_features, 3)

    pd_obj = PairwiseDistance(
        metric="bottleneck",
        order=None,
        n_jobs=-1,
    )
    result = pd_obj.fit_transform(giotto_arr)  # (n, n, 1) — one dim
    M = result[:, :, 0]  # (n, n) float64

    # Symmetrise (giotto-tda parallel computation can produce small asymmetry)
    M = 0.5 * (M + M.T)
    return M


def _sliced_wasserstein_matrix(
    pds: list[np.ndarray],
    n_directions: int = 50,
) -> np.ndarray:
    """Compute pairwise sliced Wasserstein distance matrix via persim.

    Uses ``persim.sliced_wasserstein(pd_i, pd_j, M=n_directions)`` in a
    nested loop (i < j), mirroring Draganov's per-pair loop.
    M=50 matches Draganov's default.
    """
    n = len(pds)
    M = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dist = persim.sliced_wasserstein(pds[i], pds[j], M=n_directions)
            M[i, j] = dist
            M[j, i] = dist

    return M


def _vectorised_l2_matrix(
    vectors: list[np.ndarray],
) -> np.ndarray:
    """Compute pairwise L2 distances between a list of 1-D vectors.

    Matches Draganov's ``compare_pds`` logic for the persistence_image and
    bars_statistics branches: ``np.linalg.norm(vec_i - vec_j)``.
    """
    mat = np.stack(vectors, axis=0)  # (n, d)
    # Pairwise squared L2 distances via BLAS inner products
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    sq_norms = np.sum(mat ** 2, axis=1, keepdims=True)  # (n, 1)
    gram = mat @ mat.T  # (n, n)
    sq_dists = sq_norms + sq_norms.T - 2 * gram
    # Clamp tiny negatives from floating-point arithmetic before sqrt
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_pd_distance_grid(
    diagrams_dir: pathlib.Path = pathlib.Path("data/phase3/draganov_diagrams"),
    out_dir: pathlib.Path = pathlib.Path("data/phase3/draganov_pd_distances"),
    distances: tuple[str, ...] = (
        "bottleneck",
        "sliced_wasserstein",
        "persistence_image",
        "bars_statistics",
    ),
    dims: tuple[int, ...] = (0, 1),
    overwrite: bool = False,
) -> dict[tuple[str, int], np.ndarray]:
    """Produce 34×34 PD distance matrices for each (distance, dim).

    Adapted from
    ``draganov/B-from_persistence_diagrams_to_language_distances/
    run_compute_pd_distances.py`` (function names and math preserved).

    Loads all diagram .npz files in ``(lang, term)``-sorted order from
    ``diagrams_dir/manifest.csv``, then for each ``(distance, dim)`` computes
    the pairwise distance matrix and writes it to
    ``out_dir/f'{distance}_d{dim}.npz'`` alongside the cell labels.

    If ``out_dir/f'{distance}_d{dim}.npz'`` exists and ``overwrite=False``,
    the cached matrix is loaded from disk and returned without recomputation.
    The returned dict is identical regardless of cache hit vs miss.

    Parameters
    ----------
    diagrams_dir:
        Directory containing per-(lang, term) ``.npz`` files and
        ``manifest.csv`` (output from ``inu.2`` / ``compute_diagrams``).
    out_dir:
        Output directory.  Created if it does not exist.  One
        ``{distance}_d{dim}.npz`` per (distance, dim) pair.
    distances:
        Tuple of distance names to compute.  Defaults to all four from
        Draganov's script: bottleneck, sliced_wasserstein, persistence_image,
        bars_statistics.
    dims:
        Tuple of homology dimensions to compute.  Defaults to (0, 1).
    overwrite:
        When ``False`` (default), load from cache if the .npz exists.
        Pass ``True`` to force recomputation.

    Returns
    -------
    dict[tuple[str, int], np.ndarray]
        Keys are ``(distance, dim)`` tuples; values are ``(34, 34)`` float32
        matrices.  The dict contains one entry per (distance, dim) combination.

    Raises
    ------
    FileNotFoundError
        If ``diagrams_dir/manifest.csv`` does not exist or a cell's ``.npz``
        is missing.
    ValueError
        If an unrecognised distance name is requested.
    """
    diagrams_dir = pathlib.Path(diagrams_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = diagrams_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Diagrams manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)
    # Sort by (lang, term) to ensure a deterministic cell ordering
    manifest = manifest.sort_values(["lang", "term"]).reset_index(drop=True)
    cells: list[tuple[str, str]] = [
        (str(row["lang"]), str(row["term"])) for _, row in manifest.iterrows()
    ]
    n_cells = len(cells)
    cell_labels = np.array([f"{lang}/{term}" for lang, term in cells], dtype=object)

    valid_distances = {
        "bottleneck", "sliced_wasserstein", "persistence_image", "bars_statistics"
    }
    for d in distances:
        if d not in valid_distances:
            raise ValueError(
                f"Unknown distance: {d!r}. Valid options: {sorted(valid_distances)}"
            )

    result: dict[tuple[str, int], np.ndarray] = {}

    for distance in distances:
        for dim in dims:
            cache_file = out_dir / f"{distance}_d{dim}.npz"

            if cache_file.exists() and not overwrite:
                logger.info("Loading from cache: %s", cache_file)
                data = np.load(cache_file, allow_pickle=True)
                result[(distance, dim)] = data["matrix"]
                continue

            logger.info("Computing %s distance for H_%d ...", distance, dim)

            # Load all PDs for this dim
            pds = _load_diagrams(diagrams_dir, cells, dim)

            if distance == "bottleneck":
                M = _bottleneck_matrix(pds, dim=dim)

            elif distance == "sliced_wasserstein":
                M = _sliced_wasserstein_matrix(pds, n_directions=50)

            elif distance == "persistence_image":
                # Vectorise each PD once, then compute pairwise L2
                vectors = [vectorise_persistence_image(pd_arr, dim=dim) for pd_arr in pds]
                M = _vectorised_l2_matrix(vectors)

            elif distance == "bars_statistics":
                # Use only_death=True for dim 0 (matching Draganov's dim-0 convention)
                only_death = (dim == 0)
                vectors = [
                    vectorise_bars_statistics(pd_arr, only_death=only_death)
                    for pd_arr in pds
                ]
                M = _vectorised_l2_matrix(vectors)

            else:
                raise ValueError(f"Unreachable: unknown distance {distance!r}")

            # Enforce zero diagonal (may have tiny floating-point residuals)
            np.fill_diagonal(M, 0.0)

            # Symmetrise to clean up any numerical asymmetry
            M = 0.5 * (M + M.T)

            # Cast to float32 for storage
            M_f32 = M.astype(np.float32)

            np.savez(
                cache_file,
                matrix=M_f32,
                cells=cell_labels,
                distance=distance,
                dim=dim,
            )
            logger.info(
                "  Saved %s (shape=%s, max=%.4f)",
                cache_file.name, M_f32.shape, float(M_f32.max()),
            )

            result[(distance, dim)] = M_f32

    return result
