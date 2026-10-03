"""Language effect in persistence-diagram distances, overall and length-matched.

For one term domain (color, or the adjective control set) and one distance
metric, runs the language permutation test per (layer, head), with BH
correction, and on the head-averaged distance matrix, for H_0 and H_1, on two
subsets of the sampled sentences:

- all: every sampled sentence;
- length_matched: for every mBERT token count present in all three languages,
  the same number of sentences from each (fixed seed). mBERT splits Russian
  into more wordpieces, so Russian attention graphs have more nodes and their
  diagrams more H_0 bars whatever the sentence means; this subset removes that
  difference.

Comparing color with the control set says whether the language effect is
about color at all. Uses the cached distance tensor from
compute_diagram_distances.py; token counts come from the KWIC sentences,
tokenized as the attention notebooks do (special tokens included, truncated
at MAX_LEN).

Writes results/phase3_diagram_distances/language_effect_<domain>_<metric>.csv
and language_effect_<domain>_lengths.csv (token-count histogram per language).
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


def token_counts(meta: pd.DataFrame, kwic_dir: Path, max_len: int, domain: str = "color") -> np.ndarray:
    """mBERT token count (with [CLS]/[SEP], capped at max_len) per sampled sentence."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_TAG)
    counts = np.empty(len(meta), dtype=int)
    for lang, rows in meta.groupby("lang"):
        kwic = pd.read_csv(kwic_dir / lang / f"{domain}.csv")
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


def run_tests(tensor, meta, dims, K: int, seed: int) -> pd.DataFrame:
    """Per-head (BH-corrected) and head-averaged language tests for each dim."""
    rows = []
    for di, dim in enumerate(dims):
        t = tensor[..., di]
        per_head = permutation_test_per_head(t, meta, K=K, seed=seed)
        per_head = per_head[["layer", "head", "observed", "p_value", "effect_size", "passes_bh"]].copy()
        agg = permutation_test_per_domain(t.mean(axis=(0, 1)), meta, K=K, seed=seed)
        per_head.loc[len(per_head)] = ["agg", "agg", agg["observed"], agg["p_value"],
                                       agg["effect_size"], agg["p_value"] < 0.05]
        per_head.insert(0, "dim", dim)
        rows.append(per_head)
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domain", default="color", help="color, or adjective (the control set)")
    ap.add_argument("--metric", default="wasserstein", choices=["wasserstein", "bottleneck"])
    ap.add_argument("--K", type=int, default=10_000, help="permutations per test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=MAX_LEN, help="token cap the attention was computed with")
    ap.add_argument("--dd-dir", type=Path, default=None,
                    help="distance cache dir (default: data/phase3/diagram_distances[/<domain>])")
    ap.add_argument("--kwic-dir", type=Path, default=KWIC_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    dd_dir = args.dd_dir or (DD_DIR if args.domain == "color" else DD_DIR / args.domain)

    tensor, meta, metric, dims = load_distance_tensor(dd_dir / f"{args.metric}.npz")
    meta = meta.reset_index(drop=True)
    counts = token_counts(meta, args.kwic_dir, args.max_len, args.domain)
    langs = meta["lang"].to_numpy()
    idx = length_matched_indices(langs, counts, args.seed)

    results = []
    for subset, sel in (("all", np.arange(len(meta))), ("length_matched", idx)):
        sub_meta = meta.iloc[sel].reset_index(drop=True)
        res = run_tests(tensor[:, :, sel][:, :, :, sel], sub_meta, dims, args.K, args.seed)
        res.insert(0, "subset", subset)
        res.insert(1, "n_sentences", len(sel))
        results.append(res)
        for dim in dims:
            r = res[res["dim"] == dim]
            agg = r[r["layer"] == "agg"].iloc[0]
            n_sig = int(r[r["layer"] != "agg"]["passes_bh"].sum())
            print(f"{args.domain} {subset:14s} (N={len(sel)}) H{dim}: head-averaged z = "
                  f"{agg['effect_size']:.1f}, p = {agg['p_value']:.4g}; {n_sig}/{len(r) - 1} heads significant")

    hist = (
        pd.DataFrame({"lang": langs, "n_tokens": counts, "matched": np.isin(np.arange(len(meta)), idx)})
        .groupby(["lang", "n_tokens"])
        .agg(n_all=("matched", "size"), n_matched=("matched", "sum"))
        .reset_index()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.concat(results, ignore_index=True)
    out.insert(0, "domain", args.domain)
    out.insert(1, "metric", metric)
    out_path = args.out_dir / f"language_effect_{args.domain}_{metric}.csv"
    out.to_csv(out_path, index=False)
    hist.to_csv(args.out_dir / f"language_effect_{args.domain}_lengths.csv", index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
