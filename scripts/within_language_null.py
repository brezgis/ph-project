"""Within-(lang, term) split null for the Kushnareva PD distance pipeline.

For each (lang, term) with enough contexts, randomly partition its context
indices into two halves and compute the mean cross-half PD distance. Repeated
K times this yields a distribution: the noise floor for "how different two
samples of the same term in the same language look."

Compare this to the cross-language same-term distance (e.g. the mean of the
en_red <-> ru_kr <-> es_rojo cross-pairs). If the cross-language signal is
not much larger than the within-language split, the pipeline's apparent
cross-linguistic differences are at the level of pipeline stochasticity.

Uses the existing pre-computed (12, 12, 680, 680, 2) per-context distance
tensor at data/phase3/diagram_distances/wasserstein.npz. Aggregates exactly
the way notebook cell 7 does: mean across all (layer, head) pairs, then
restrict to a single homology dimension (default H_1).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from replication.diagram_distances import load_translation_triples  # noqa: E402

DD_DIR = REPO_ROOT / "data" / "phase3" / "diagram_distances"
OUT_DIR = REPO_ROOT / "results" / "phase3_diagram_distances"
CANON_DIR = REPO_ROOT / "canon-terms"


def split_distance(idx: np.ndarray, mean_dist: np.ndarray, rng: np.random.Generator) -> float:
    """Mean off-diagonal distance between two random equal-size halves of idx."""
    perm = rng.permutation(idx)
    h = len(perm) // 2
    A, B = perm[:h], perm[h : 2 * h]
    return float(mean_dist[np.ix_(A, B)].mean())


def cross_lang_same_term_distance(
    mean_dist: np.ndarray,
    groups: dict[tuple[str, str], np.ndarray],
    triples_df: pd.DataFrame,
    ru_blue_choice: str = "синий",
) -> dict[str, float]:
    """For each translation triple, mean distance over all cross-lang same-triple pairs.

    Mirrors the structure of replication.diagram_distances.per_term_test_statistic
    but returns one number per triple (keyed by english term + russian term).
    """
    blue_rows = triples_df[triples_df["en_term"] == "blue"]
    has_multiple_blues = len(blue_rows) > 1

    out: dict[str, float] = {}
    for _, row in triples_df.iterrows():
        if row["en_term"] == "blue" and has_multiple_blues:
            if row["ru_term"] != ru_blue_choice:
                continue
        en_idx = groups.get(("en", row["en_term"]), np.array([], dtype=int))
        ru_idx = groups.get(("ru", row["ru_term"]), np.array([], dtype=int))
        es_idx = groups.get(("es", row["es_term"]), np.array([], dtype=int))

        pair_means = []
        for a, b in [(en_idx, ru_idx), (en_idx, es_idx), (ru_idx, es_idx)]:
            if len(a) == 0 or len(b) == 0:
                continue
            pair_means.append(mean_dist[np.ix_(a, b)].mean())
        if pair_means:
            key = f"{row['en_term']} ({row['ru_term']}/{row['es_term']})"
            out[key] = float(np.mean(pair_means))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", choices=["wasserstein", "bottleneck"], default="wasserstein")
    ap.add_argument("--dim", type=int, choices=[0, 1], default=1, help="Homology dimension")
    ap.add_argument("--K", type=int, default=200, help="Random splits per (lang, term)")
    ap.add_argument("--min-n", type=int, default=20, help="Min contexts to include a (lang, term)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ru-blue", choices=["синий", "голубой"], default="синий")
    args = ap.parse_args()

    npz_path = DD_DIR / f"{args.metric}.npz"
    if not npz_path.exists():
        sys.exit(f"distance tensor not found: {npz_path}")
    print(f"Loading {npz_path} ...")
    npz = np.load(npz_path, allow_pickle=True)
    tensor = npz["tensor"]  # (L, H, N, N, 2)
    meta = json.loads(npz["metadata_json"].tobytes().decode())
    n_layers, n_heads, n, _, n_dims = tensor.shape
    print(f"  tensor shape={tensor.shape}, contexts={n}, metric={args.metric}, dim=H{args.dim}")

    # Match cell 7 aggregation: mean over (layer, head) -> (N, N)
    mean_dist = tensor[..., args.dim].mean(axis=(0, 1))
    assert mean_dist.shape == (n, n)
    # Symmetrize defensively (the upstream tensor should already be symmetric)
    mean_dist = 0.5 * (mean_dist + mean_dist.T)

    # Group context indices by (lang, term)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        groups[(m["lang"], m["term"])].append(i)
    groups_arr: dict[tuple[str, str], np.ndarray] = {k: np.asarray(v, dtype=int) for k, v in groups.items()}

    # Within-language split null per (lang, term)
    rng = np.random.default_rng(args.seed)
    rows = []
    for (lang, term), idx in sorted(groups_arr.items()):
        if len(idx) < args.min_n:
            rows.append({"lang": lang, "term": term, "n": len(idx), "split_mean": np.nan, "split_std": np.nan, "skipped": True})
            continue
        samples = np.array([split_distance(idx, mean_dist, rng) for _ in range(args.K)])
        rows.append({
            "lang": lang,
            "term": term,
            "n": len(idx),
            "split_mean": float(samples.mean()),
            "split_std": float(samples.std(ddof=1)),
            "split_p05": float(np.percentile(samples, 5)),
            "split_p95": float(np.percentile(samples, 95)),
            "skipped": False,
        })
    null_df = pd.DataFrame(rows)

    # Cross-language same-term distance per triple
    triples_df = load_translation_triples(CANON_DIR, domain="color")
    cross = cross_lang_same_term_distance(mean_dist, groups_arr, triples_df, ru_blue_choice=args.ru_blue)
    cross_df = pd.DataFrame(
        [{"triple": k, "cross_lang_mean": v} for k, v in cross.items()]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    null_path = OUT_DIR / f"within_language_null_{args.metric}_h{args.dim}.csv"
    cross_path = OUT_DIR / f"cross_language_same_term_{args.metric}_h{args.dim}.csv"
    null_df.to_csv(null_path, index=False)
    cross_df.to_csv(cross_path, index=False)
    print(f"Wrote {null_path}")
    print(f"Wrote {cross_path}")

    # ── Print human-readable summary ─────────────────────────────────────
    print()
    print(f"=== Within-(lang, term) split null  [{args.metric}, H{args.dim}, K={args.K}] ===")
    print(f"{'lang':<4} {'term':<14} {'n':>4} {'split_mean':>12} {'split_std':>12} {'p05':>10} {'p95':>10}")
    for _, r in null_df.iterrows():
        if r["skipped"]:
            print(f"{r['lang']:<4} {r['term']:<14} {r['n']:>4}   (skipped, n<{args.min_n})")
        else:
            print(f"{r['lang']:<4} {r['term']:<14} {r['n']:>4} {r['split_mean']:>12.4f} {r['split_std']:>12.4f} {r['split_p05']:>10.4f} {r['split_p95']:>10.4f}")

    used = null_df[~null_df["skipped"]]
    if len(used):
        print()
        print(f"  pooled within-lang split mean : {used['split_mean'].mean():.4f}")
        print(f"  pooled within-lang split std  : {used['split_mean'].std(ddof=1):.4f}")

    print()
    print(f"=== Cross-language same-term distance  [{args.metric}, H{args.dim}] ===")
    print(f"{'triple':<32} {'cross_lang_mean':>16}")
    for _, r in cross_df.iterrows():
        print(f"{r['triple']:<32} {r['cross_lang_mean']:>16.4f}")

    if len(cross_df):
        print()
        print(f"  mean over triples             : {cross_df['cross_lang_mean'].mean():.4f}")

    # ── Bottom-line ratio ────────────────────────────────────────────────
    if len(used) and len(cross_df):
        within = used["split_mean"].mean()
        across = cross_df["cross_lang_mean"].mean()
        print()
        print(f"=== Signal-to-noise summary ===")
        print(f"  within-(lang,term) split distance (noise floor) : {within:.4f}")
        print(f"  cross-language same-term distance (signal)      : {across:.4f}")
        print(f"  ratio (signal / noise)                          : {across / within:.3f}x")
        if across <= within * 1.1:
            print("  -> signal is at the level of pipeline noise; cross-lang differences NOT clearly above the within-lang baseline.")
        else:
            print(f"  -> signal exceeds noise floor by {(across/within - 1)*100:.0f}%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
