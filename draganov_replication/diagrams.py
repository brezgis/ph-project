"""Compute per-(lang, term) persistence diagrams (H_0 + H_1) from mBERT point clouds.

Adapts the Draganov A-step pipeline to contextual mBERT embeddings using an
in-process numpy/ripser stack instead of a three-stage shell pipeline.

Draganov A-step correspondence
-------------------------------
The original pipeline in ``draganov/A-word_embeddings_to_persistence_diagrams/``
is split across three scripts.  We collapse them into a single ``compute_diagrams()``
call because everything is in-process numpy:

+------------------------------------------------------------+------------------------------------------+---------------------------------------------+
| Draganov script                                            | Their action                             | Our equivalent                              |
+============================================================+==========================================+=============================================+
| ``compute_point-cloud_distance_matrix.py --metric cosine`` | point cloud → lower-triangular cosine    | ``baselines.distances.cosine_distance_      |
|                                                            | distance SSV                             | matrix(X)`` — returns dense symmetric       |
|                                                            |                                          | matrix in memory                            |
+------------------------------------------------------------+------------------------------------------+---------------------------------------------+
| ``2-point_clouds_to_persistence_bars.sh``                  | distance matrix → ripser bars (stdout    | ``baselines.topology.rips_barcode(D,        |
| (calls ``./ripser/ripser``)                                | text)                                    | max_dim=1)`` — uses the ``ripser`` Python   |
|                                                            |                                          | package wrapping the same C++ ripser binary |
+------------------------------------------------------------+------------------------------------------+---------------------------------------------+
| ``ripser_output_to_bars.py``                               | parse ripser stdout → numpy arrays       | handled inside ``rips_barcode`` (returns    |
|                                                            |                                          | structured arrays directly)                 |
+------------------------------------------------------------+------------------------------------------+---------------------------------------------+

**Faithfulness**: The math is a direct port.  The only divergence is in-process
numpy vs three-stage shell pipeline — purely an ergonomics choice, not a
computational one.  Metric is locked to cosine (matching Draganov's
``--metric cosine`` for fastText), for the same reasons: 768-d mBERT embeddings
should be compared in cosine space.

Public API
----------
compute_diagrams(...)
    Compute persistence diagrams for all 34 per-(lang, term) point clouds and
    return a summary DataFrame.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from baselines.distances import cosine_distance_matrix
from baselines.topology import rips_barcode

logger = logging.getLogger(__name__)

_BARCODE_DTYPE = np.dtype([("birth", "f8"), ("death", "f8")])


def _to_plain(sa: np.ndarray) -> np.ndarray:
    """Structured array with 'birth'/'death' fields → (n, 2) float64."""
    if len(sa) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack(
        [sa["birth"].astype(np.float64), sa["death"].astype(np.float64)]
    )


def compute_diagrams(
    pointcloud_dir: pathlib.Path = pathlib.Path("data/phase3/draganov_pointclouds"),
    out_dir: pathlib.Path = pathlib.Path("data/phase3/draganov_diagrams"),
    max_dim: int = 1,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compute persistence diagrams for each (lang, term) cell.

    Reads ``{pointcloud_dir}/manifest.csv``, loads each ``.npy``, computes
    the cosine distance matrix via
    ``baselines.distances.cosine_distance_matrix``, runs
    ``baselines.topology.rips_barcode`` at *max_dim*, and saves an ``.npz``
    per cell with arrays ``h0`` and ``h1`` (each shape ``(n_features, 2)``,
    ``float64``).

    Returns a summary DataFrame with columns
    ``[lang, term, n_h0, n_h1, file]`` (also written to
    ``out_dir/manifest.csv``).  Skips cells whose ``.npz`` already exists
    unless *overwrite* is ``True``.

    Adapted from Draganov A-step (``compute_point-cloud_distance_matrix.py``
    + ``2-point_clouds_to_persistence_bars.sh`` +
    ``ripser_output_to_bars.py``).  See module docstring for the
    step-by-step correspondence table.

    Parameters
    ----------
    pointcloud_dir:
        Directory containing per-(lang, term) ``.npy`` files and
        ``manifest.csv`` (output from ``inu.1`` / ``build_pointclouds``).
    out_dir:
        Output directory.  Created if it does not exist.  One
        ``{lang}_{term}.npz`` per cell plus ``manifest.csv``.
    max_dim:
        Maximum homology dimension passed to ripser.  ``1`` gives H_0
        (connected components) and H_1 (loops), matching the H_0+H_1
        scope locked at planning time for this study.  Draganov's
        ``2-point_clouds_to_persistence_bars.sh`` runs at ``MAXDIM=2``;
        we drop H_2 because it is not used downstream and significantly
        increases ripser cost.
    overwrite:
        When ``False`` (default), skip cells whose ``.npz`` already
        exists.  Pass ``True`` to force regeneration.

    Returns
    -------
    pd.DataFrame
        Summary with columns ``[lang, term, n_h0, n_h1, file]``, one row
        per cell.  Also written to ``out_dir/manifest.csv``.

    Raises
    ------
    FileNotFoundError
        If ``{pointcloud_dir}/manifest.csv`` does not exist.
    FileNotFoundError
        If a ``.npy`` file listed in the manifest is missing.
    """
    pointcloud_dir = pathlib.Path(pointcloud_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pointcloud_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Pointcloud manifest not found: {manifest_path}"
        )

    pc_manifest = pd.read_csv(manifest_path)
    rows: list[dict[str, Any]] = []

    for _, pc_row in pc_manifest.iterrows():
        lang = str(pc_row["lang"])
        term = str(pc_row["term"])
        n_samples = int(pc_row["n_samples"])

        out_npz = out_dir / f"{lang}_{term}.npz"

        if out_npz.exists() and not overwrite:
            # Read existing .npz for manifest counts
            d = np.load(out_npz)
            n_h0 = len(d["h0"])
            n_h1 = len(d["h1"])
            logger.debug(
                "Skipping existing: %s (n_h0=%d, n_h1=%d)", out_npz, n_h0, n_h1
            )
        else:
            # Locate the .npy point cloud
            npy_path = pointcloud_dir / f"{lang}_{term}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"Point cloud not found for ({lang}, {term!r}): {npy_path}"
                )

            logger.info("Computing diagram: (%s, %r)  n=%d", lang, term, n_samples)

            # Step 1 (← Draganov compute_point-cloud_distance_matrix.py --metric cosine)
            # Load float32 point cloud, compute cosine distance matrix (float64).
            X = np.load(npy_path)  # (N, dim) float32
            D = cosine_distance_matrix(X.astype(np.float64))

            # Step 2 (← Draganov 2-point_clouds_to_persistence_bars.sh / ripser binary)
            # Compute Vietoris-Rips barcode; infinite bars are already stripped.
            bc = rips_barcode(D, max_dim=max_dim)

            # Step 3 (← Draganov ripser_output_to_bars.py)
            # Convert structured arrays to plain (n, 2) float64.
            h0 = _to_plain(bc[0])
            h1 = _to_plain(bc.get(1, np.empty(0, dtype=_BARCODE_DTYPE)))

            np.savez(
                out_npz,
                h0=h0,
                h1=h1,
                lang=lang,
                term=term,
                n_samples=n_samples,
            )

            n_h0 = len(h0)
            n_h1 = len(h1)
            logger.info(
                "  Saved %s: n_h0=%d, n_h1=%d", out_npz.name, n_h0, n_h1
            )

        rows.append(
            {
                "lang": lang,
                "term": term,
                "n_h0": n_h0,
                "n_h1": n_h1,
                "file": str(out_npz),
            }
        )

    result = pd.DataFrame(rows, columns=["lang", "term", "n_h0", "n_h1", "file"])

    manifest_csv = out_dir / "manifest.csv"
    result.to_csv(manifest_csv, index=False)
    logger.info("Manifest written: %s (%d rows)", manifest_csv, len(result))

    return result
