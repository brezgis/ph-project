"""Compute pairwise persistence-diagram distance tensors for the blr eval pipeline.

Computes Wasserstein-2 and bottleneck pairwise distance matrices for every
(layer, head) ∈ 12×12 across all 3-language color samples and caches the
results to disk as ``.npz`` files.  This is the overnight CLI job — do NOT
run inside a notebook.

Usage
-----
    # Default overnight run (n_per_term=20, ~17h at n_jobs=-1):
    python scripts/compute_diagram_distances.py

    # Two-night budget (n_per_term=30, ~38h at n_jobs=-1, N=3000):
    python scripts/compute_diagram_distances.py --n-per-term 30

    # Smoke run — verifies pipeline with tiny settings (seconds):
    python scripts/compute_diagram_distances.py --smoke

    # Re-run even if cached:
    python scripts/compute_diagram_distances.py --force

    # Single metric only:
    python scripts/compute_diagram_distances.py --metrics wasserstein

Memory note
-----------
Each metric tensor at N=600 (n_per_term=20, 30 color terms) is approximately
166 MB in float32 (12 × 12 × 600 × 600 × 2 × 4 bytes).  Both metrics
together ≈ 330 MB total cache.  At N=3000 (n_per_term=30) each metric tensor
is ~4 GB — ensure you have sufficient RAM before launching.

n_jobs=-1 uses all available CPU cores for giotto-tda parallel computation.
On 24 cores this is optimal.  If memory pressure is a concern, reduce n_jobs
(e.g. --n-jobs 8) at the cost of longer wall time.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path when invoked as ``python scripts/...``
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from replication.diagram_distances import (
    load_lang_barcodes,
    subsample_per_term,
    compute_full_distance_tensor,
    cache_path,
    save_distance_tensor,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("compute_diagram_distances")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LANGS = ["en", "ru", "es"]
DOMAIN = "color"
MODEL_TAG = "bert-base-multilingual-cased"
MAX_LEN = 32
N_LAYERS = 12

# Expected sentinel files — one per language (part3 is the last part).
# Script aborts if any are missing (full ripser run not yet complete).
_SENTINEL_GLOB = "{lang}_color_all_heads_12_layers_MAX_LEN_32_bert-base-multilingual-cased_part3of3.json"


def _check_sentinel_files(barcode_dir: pathlib.Path) -> None:
    """Assert that the 3 expected barcode sentinel files exist (one per lang).

    Aborts with a clear error message if the full ripser run is incomplete.
    """
    missing = []
    for lang in LANGS:
        sentinel_name = _SENTINEL_GLOB.format(lang=lang)
        sentinel_path = barcode_dir / sentinel_name
        if not sentinel_path.exists():
            missing.append(str(sentinel_path))

    if missing:
        logger.error(
            "Pre-flight check FAILED — missing barcode sentinel files.\n"
            "The full ripser run (mbert_attention_ripser.ipynb) has not completed.\n"
            "Missing files:\n%s",
            "\n".join(f"  {p}" for p in missing),
        )
        sys.exit(1)

    logger.info("Pre-flight check passed — all 3 sentinel files present.")


def _load_combined_barcodes(
    barcode_dir: pathlib.Path,
    n_per_term: int,
    seed: int,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Load en/ru/es color barcodes, concatenate, and subsample.

    Returns
    -------
    tuple[dict, np.ndarray, pd.DataFrame]
        ``(combined_diagrams, sample_indices, combined_metadata)``
    """
    all_diagrams: dict = {}
    all_metadata_parts: list[pd.DataFrame] = []

    for lang in LANGS:
        logger.info("Loading %s/%s barcodes from %s ...", lang, DOMAIN, barcode_dir)
        diagrams, metadata = load_lang_barcodes(
            barcode_dir=barcode_dir,
            lang=lang,
            domain=DOMAIN,
            model_tag=MODEL_TAG,
            max_len=MAX_LEN,
            n_layers=N_LAYERS,
        )

        n_lang = len(metadata)
        logger.info("  %s/%s: %d samples, %d terms", lang, DOMAIN, n_lang, metadata["term"].nunique())

        # Concatenate per-(layer, head) sample lists in LANGS order. The combined
        # global sample index `i` corresponds to position `i` in every all_diagrams[key]
        # list AND row `i` of combined_metadata after pd.concat below.
        for key, sample_list in diagrams.items():
            if key not in all_diagrams:
                all_diagrams[key] = []
            all_diagrams[key].extend(sample_list)

        all_metadata_parts.append(metadata.reset_index(drop=True))

    combined_metadata = pd.concat(all_metadata_parts, ignore_index=True)
    logger.info("Combined metadata: %d samples across %d languages", len(combined_metadata), len(LANGS))

    # Subsample per-term within each language independently
    # We build a new metadata that includes lang in the groupby key
    combined_metadata["_lang_term"] = combined_metadata["lang"] + "/" + combined_metadata["term"]
    orig_term_col = combined_metadata["term"].copy()
    combined_metadata["term"] = combined_metadata["_lang_term"]

    sample_indices = subsample_per_term(combined_metadata, n_per_term=n_per_term, seed=seed)

    # Restore term column
    combined_metadata["term"] = orig_term_col
    combined_metadata = combined_metadata.drop(columns=["_lang_term"])

    logger.info(
        "Subsampled to %d samples (n_per_term=%d, seed=%d)",
        len(sample_indices), n_per_term, seed,
    )
    return all_diagrams, sample_indices, combined_metadata


def _compute_and_cache(
    all_diagrams: dict,
    sample_indices: np.ndarray,
    combined_metadata: pd.DataFrame,
    metric: str,
    dims: tuple,
    cache_dir: pathlib.Path,
    layers: range,
    heads: range,
    n_jobs: int,
    force: bool,
) -> None:
    """Compute and cache one metric tensor."""
    path = cache_path(cache_dir, metric)

    if path.exists() and not force:
        logger.info("Skipping %s — cache exists at %s (use --force to recompute)", metric, path)
        return

    n = len(sample_indices)
    n_l = len(list(layers))
    n_h = len(list(heads))
    n_d = len(dims)
    size_mb = n_l * n_h * n * n * n_d * 4 / 1e6
    logger.info(
        "Computing %s tensor: shape (%d, %d, %d, %d, %d) ≈ %.0f MB ...",
        metric, n_l, n_h, n, n, n_d, size_mb,
    )

    t0 = time.perf_counter()
    tensor = compute_full_distance_tensor(
        per_layer_head_diagrams=all_diagrams,
        sample_indices=sample_indices,
        metric=metric,
        dims=dims,
        layers=layers,
        heads=heads,
        progress=True,
        n_jobs=n_jobs,
    )
    elapsed = time.perf_counter() - t0

    logger.info(
        "%s done in %.1fs (%.1f min). Saving to %s ...",
        metric, elapsed, elapsed / 60, path,
    )
    save_distance_tensor(tensor, combined_metadata.iloc[sample_indices].reset_index(drop=True), dims, metric, path)
    logger.info("Saved %s tensor to %s", metric, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compute_diagram_distances",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--barcode-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/phase3/barcodes"),
        help="Directory containing barcode JSON files (default: data/phase3/barcodes).",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/phase3/diagram_distances"),
        help="Directory to write cached distance tensors (default: data/phase3/diagram_distances).",
    )
    parser.add_argument(
        "--n-per-term",
        type=int,
        default=20,
        metavar="N",
        help=(
            "Samples per (lang, term) to include. Default 20 (~17h overnight at n_jobs=-1). "
            "For a two-night budget use --n-per-term 30 (~38h, N=3000 total per metric tensor)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling (default: 42).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="wasserstein,bottleneck",
        help=(
            "Comma-separated list of metrics to compute. "
            "Default: 'wasserstein,bottleneck'. "
            "Options: wasserstein, bottleneck."
        ),
    )
    parser.add_argument(
        "--dims",
        type=str,
        default="0,1",
        help="Comma-separated homology dimensions to include (default: '0,1' for H_0 and H_1).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help=(
            "Number of parallel jobs for PairwiseDistance. "
            "Default -1 (use all cores). Reduce if memory pressure is a concern."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached tensors already exist.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke-test mode: n_per_term=5, single (layer=0, head=0), single metric=wasserstein. "
            "Completes in seconds to verify the pipeline works before launching the full overnight run."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply smoke-mode overrides
    if args.smoke:
        logger.info("Smoke mode active — overriding to n_per_term=5, layer=0/head=0, metric=wasserstein.")
        args.n_per_term = 5
        args.metrics = "wasserstein"
        layers_range = range(1)
        heads_range = range(1)
    else:
        layers_range = range(N_LAYERS)
        heads_range = range(N_LAYERS)

    # Parse metrics and dims
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    dims = tuple(int(d.strip()) for d in args.dims.split(",") if d.strip())

    logger.info(
        "Starting compute_diagram_distances: barcode_dir=%s cache_dir=%s "
        "n_per_term=%d seed=%d metrics=%s dims=%s n_jobs=%d",
        args.barcode_dir, args.cache_dir, args.n_per_term,
        args.seed, metrics, dims, args.n_jobs,
    )

    # Pre-flight: check sentinel files (skip in smoke mode for partial data)
    if not args.smoke:
        _check_sentinel_files(args.barcode_dir)
    else:
        logger.info("Smoke mode: skipping sentinel file check (may be partial data).")
        # Verify at least en part1 exists for smoke
        en_part1 = args.barcode_dir / (
            "en_color_all_heads_12_layers_MAX_LEN_32_bert-base-multilingual-cased_part1of3.json"
        )
        if not en_part1.exists():
            logger.error("Smoke mode requires at least en_color_part1of3.json. Not found: %s", en_part1)
            sys.exit(1)

    # Load + subsample
    t_load = time.perf_counter()
    all_diagrams, sample_indices, combined_metadata = _load_combined_barcodes(
        barcode_dir=args.barcode_dir,
        n_per_term=args.n_per_term,
        seed=args.seed,
    )
    logger.info("Load + subsample took %.1fs", time.perf_counter() - t_load)

    # Create cache dir
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute each metric
    t_total = time.perf_counter()
    metric_times: dict[str, float] = {}
    for metric in metrics:
        t_metric = time.perf_counter()
        _compute_and_cache(
            all_diagrams=all_diagrams,
            sample_indices=sample_indices,
            combined_metadata=combined_metadata,
            metric=metric,
            dims=dims,
            cache_dir=args.cache_dir,
            layers=layers_range,
            heads=heads_range,
            n_jobs=args.n_jobs,
            force=args.force,
        )
        metric_times[metric] = time.perf_counter() - t_metric

    total_elapsed = time.perf_counter() - t_total
    logger.info("=" * 60)
    logger.info("DONE. Total wall time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    for metric, t in metric_times.items():
        logger.info("  %s: %.1fs (%.1f min)", metric, t, t / 60)

    if args.smoke:
        # Pair-distance compute scales ~quadratically in N (samples per layer-head).
        # Smoke fixes n_per_term=5; the default overnight run uses n_per_term=20,
        # adding ~(20/5)**2 = 16x on top of the 144 (layer, head) sweep.
        layer_head_factor = 144
        n_scale_for_default = (20 / 5) ** 2
        naive_min = total_elapsed / 60 * layer_head_factor
        adjusted_min = naive_min * n_scale_for_default
        logger.info(
            "Smoke run complete. Naive layer/head extrapolation: ~%.0f min per metric "
            "(layer-head sweep only). Adjusting for default n_per_term=20 vs smoke=5 "
            "(quadratic in N): ~%.0f min per metric, ~%.0f min total for %d metrics.",
            naive_min,
            adjusted_min,
            adjusted_min * len(metrics),
            len(metrics),
        )


if __name__ == "__main__":
    main()
