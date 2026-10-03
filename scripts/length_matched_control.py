"""Length-matched control for the per-domain (language) permutation test.

mBERT splits Russian into more wordpieces than English or Spanish, so Russian
attention graphs have more nodes and their persistence diagrams more H_0 bars,
whatever the sentence means. A language effect in diagram distances could
partly be a token-count effect. This script re-runs the language test on a
subset of the sampled sentences whose token counts are matched across
languages: for every token count present in all three languages, it keeps the
same number of sentences from each (chosen with a fixed seed).

Uses the cached distance tensor from compute_diagram_distances.py. Token
counts come from the KWIC sentences, tokenized exactly as the attention
notebooks do (special tokens included, truncated at MAX_LEN).

Writes results/phase3_diagram_distances/length_matched_control_<metric>.csv (one row
per (layer, head) plus a head-averaged 'agg' row, for H_0 and H_1) and
length_matched_control_lengths.csv (token-count histogram per language,
before and after matching).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from attention_topology.diagram_distances import (  # noqa: E402
    load_distance_tensor,
    permutation_test_per_domain,
    permutation_test_per_head,
)
from compute_diagram_distances import MAX_LEN, MODEL_TAG  # noqa: E402

DD_DIR = REPO_ROOT / "data" / "phase3" / "diagram_distances"
KWIC_DIR = REPO_ROOT / "data" / "kwic"
OUT_DIR = REPO_ROOT / "results" / "phase3_diagram_distances"


def token_counts(meta: pd.DataFrame, kwic_dir: Path, max_len: int) -> np.ndarray:
    """mBERT token count (with [CLS]/[SEP], capped at max_len) per sampled sentence."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_TAG)
    counts = np.empty(len(meta), dtype=int)
    for lang, rows in meta.groupby("lang"):
        kwic = pd.read_csv(kwic_dir / lang / "color.csv")
        by_term = {term: g["sentence"].tolist() for term, g in kwic.groupby("term", sort=False)}
        sentences = [by_term[r.term][r.sentence_idx_within_term] for r in rows.itertuples()]
        ids = tok(sentences, truncation=True, max_length=max_len)["input_ids"]
        counts[rows.index.to_numpy()] = [len(x) for x in ids]
    return counts


def length_matched_indices(langs: np.ndarray, counts: np.ndarray, seed: int) -> np.ndarray:
    """Indices of a subset with identical token-count histograms in every language."""
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    lang_values = np.unique(langs)
    for n in np.unique(counts):
        pools = [np.flatnonzero((langs == lang) & (counts == n)) for lang in lang_values]
        m = min(len(p) for p in pools)
        for p in pools:
            keep.extend(rng.choice(p, size=m, replace=False))
    return np.sort(np.array(keep, dtype=int))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="wasserstein", choices=["wasserstein", "bottleneck"])
    ap.add_argument("--K", type=int, default=10_000, help="permutations per test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token cap the attention was computed with")
    ap.add_argument("--dd-dir", type=Path, default=DD_DIR)
    ap.add_argument("--kwic-dir", type=Path, default=KWIC_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    tensor, meta, metric, dims = load_distance_tensor(args.dd_dir / f"{args.metric}.npz")
    meta = meta.reset_index(drop=True)
    counts = token_counts(meta, args.kwic_dir, args.max_len)
    langs = meta["lang"].to_numpy()

    idx = length_matched_indices(langs, counts, args.seed)
    sub_meta = meta.iloc[idx].reset_index(drop=True)
    print(f"Length-matched subset: {len(idx)} of {len(meta)} sentences "
          f"({len(idx) // meta['lang'].nunique()} per language)")

    hist = (
        pd.DataFrame({"lang": langs, "n_tokens": counts, "matched": np.isin(np.arange(len(meta)), idx)})
        .groupby(["lang", "n_tokens"])
        .agg(n_all=("matched", "size"), n_matched=("matched", "sum"))
        .reset_index()
    )

    rows = []
    for di, dim in enumerate(dims):
        sub = tensor[:, :, idx][:, :, :, idx, di]
        per_head = permutation_test_per_head(sub, sub_meta, K=args.K, seed=args.seed)
        per_head = per_head[["layer", "head", "observed", "p_value", "effect_size", "passes_bh"]].copy()
        agg = permutation_test_per_domain(sub.mean(axis=(0, 1)), sub_meta, K=args.K, seed=args.seed)
        per_head.loc[len(per_head)] = ["agg", "agg", agg["observed"], agg["p_value"],
                                       agg["effect_size"], agg["p_value"] < 0.05]
        per_head.insert(0, "dim", dim)
        rows.append(per_head)
        n_sig = int(per_head["passes_bh"].iloc[:-1].sum())
        print(f"H{dim}: head-averaged z = {agg['effect_size']:.1f}, p = {agg['p_value']:.4g}; "
              f"{n_sig}/{len(per_head) - 1} heads significant after BH")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(rows, ignore_index=True)
    out.insert(0, "metric", metric)
    out.insert(1, "n_matched", len(idx))
    out.to_csv(args.out_dir / f"length_matched_control_{metric}.csv", index=False)
    hist.to_csv(args.out_dir / "length_matched_control_lengths.csv", index=False)
    print(f"Wrote {args.out_dir / f'length_matched_control_{metric}.csv'}")


if __name__ == "__main__":
    main()
