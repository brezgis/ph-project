"""Tokenization and color-specificity follow-up to the language-effect test.

(A) Interaction: is the language separation larger for color than for the
    non-color adjective control? Normalized separation
        S = (mean between-language distance - mean within-language distance)
            / mean within-language distance
    on the head-averaged distance matrix, length-matched subset; bootstrap over
    sentences (resampled within language) for a CI on S_color - S_adjective.
(B) Tokenization: multiple regression on distance matrices (MRM). For every
    pair of sentences, the head-averaged diagram distance is regressed on
    different-language, same-concept (same term or translation), and pairwise
    differences in wordpiece count, fertility (wordpieces per word), share of
    ## continuation pieces, and target-word wordpiece count. Significance of
    each coefficient by permuting the distance matrix (rows and columns
    together), K permutations.

Run from the ph-project-cleanup repo root with its env:
    uv run python scripts/tokenization_analysis.py [--K 999 --B 1000]
"""
import argparse, re, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path.cwd()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from attention_topology.diagram_distances import load_distance_tensor  # noqa: E402
from language_effect import length_matched_indices  # noqa: E402
from compute_diagram_distances import MAX_LEN, MODEL_TAG  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained(MODEL_TAG)
PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def glosses(domain):
    g = {}
    for lang in ("en", "es", "ru"):
        y = yaml.safe_load(open(REPO / "canon-terms" / lang / f"{domain}.yaml", encoding="utf-8"))
        for t in y["terms"]:
            gl = str(t.get("gloss", t["term"])).lower()
            g[(lang, t["term"])] = re.sub(r"^(dark|light) ", "", gl)
    return g


def features(meta, domain):
    rows = []
    for lang, grp in meta.groupby("lang", sort=False):
        kw = pd.read_csv(REPO / "data" / "kwic" / lang / f"{domain}.csv")
        by_term = {t: g.reset_index(drop=True) for t, g in kw.groupby("term", sort=False)}
        for i, r in grp.iterrows():
            row = by_term[r.term].iloc[r.sentence_idx_within_term]
            ids = TOK(row.sentence, truncation=True, max_length=MAX_LEN)["input_ids"]
            pieces = TOK.convert_ids_to_tokens(ids)[1:-1]
            n_cont = sum(p.startswith("##") for p in pieces)
            n_words = max(1, len(pieces) - n_cont)
            words = row.sentence.split()
            tw = PUNCT.sub("", words[int(row.target_idx)]) if int(row.target_idx) < len(words) else r.term
            tgt = len(TOK.tokenize(tw)) or 1
            rows.append(dict(i=i, n_wp=len(ids), fert=len(pieces) / n_words, cont=n_cont / max(1, len(pieces)), tgt=tgt))
    return pd.DataFrame(rows).set_index("i").sort_index()


def load(domain, metric):
    d = REPO / "data" / "phase3" / "diagram_distances"
    d = d if domain == "color" else d / domain
    t, meta, _, dims = load_distance_tensor(d / f"{metric}.npz")
    meta = meta.reset_index(drop=True)
    return t.mean(axis=(0, 1)), meta, list(dims)   # head-averaged: (N, N, n_dims)


def sep(D, langs, idx=None):
    if idx is not None:
        D = D[np.ix_(idx, idx)]; langs = langs[idx]
        valid = idx[:, None] != idx[None, :]
    else:
        valid = ~np.eye(len(langs), dtype=bool)
    iu = np.triu(valid, 1)
    same = (langs[:, None] == langs[None, :])
    w, b = D[iu & same].mean(), D[iu & ~same].mean()
    return (b - w) / w


def bootstrap_sep(D, langs, B, rng):
    out = np.empty(B)
    li = {l: np.flatnonzero(langs == l) for l in np.unique(langs)}
    for k in range(B):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in li.values()])
        out[k] = sep(D, langs, idx)
    return out


def mrm(D, X, names, K, rng):
    n = D.shape[0]; iu = np.triu_indices(n, 1)
    Xc = np.column_stack([np.ones(len(iu[0]))] + [x[iu] for x in X])
    Xs = Xc.copy(); Xs[:, 1:] = (Xs[:, 1:] - Xs[:, 1:].mean(0)) / Xs[:, 1:].std(0)
    pinv = np.linalg.pinv(Xs)
    def coefs(Dm):
        y = Dm[iu]; return pinv @ ((y - y.mean()) / y.std())
    obs = coefs(D)
    null = np.empty((K, len(obs)))
    for k in range(K):
        p = rng.permutation(n); null[k] = coefs(D[np.ix_(p, p)])
    pv = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (K + 1)
    return pd.DataFrame({"term": ["intercept"] + names, "beta_std": obs, "p_perm": pv})[1:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=999)
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    out = {}
    for metric in ("wasserstein", "bottleneck"):
        S = {}
        for domain in ("color", "adjective"):
            Dall, meta, dims = load(domain, metric)
            langs = meta["lang"].to_numpy()
            f = features(meta, domain)
            idx = length_matched_indices(langs, f["n_wp"].to_numpy(), a.seed)
            gl = glosses(domain)
            concept = np.array([gl.get((r.lang, r.term), r.term) for r in meta.itertuples()])
            if domain == "color" and metric == "wasserstein":
                desc = f.assign(lang=langs).groupby("lang")[["n_wp", "fert", "cont", "tgt"]].mean().round(3)
                descm = f.iloc[idx].assign(lang=langs[idx]).groupby("lang")[["n_wp", "fert", "cont", "tgt"]].mean().round(3)
                print("\nTokenization by language, color, all sentences:\n", desc)
                print("\nTokenization by language, color, length-matched:\n", descm)
            for di, dim in enumerate(dims):
                D = Dall[..., di]
                Dm, lm = D[np.ix_(idx, idx)], langs[idx]
                s_obs = sep(Dm, lm); s_bs = bootstrap_sep(Dm, lm, a.B, rng)
                S[(domain, dim)] = (s_obs, s_bs)
                for subset, sel in (("all", np.arange(len(langs))), ("length_matched", idx)):
                    Ds = D[np.ix_(sel, sel)]; L = langs[sel]; C = concept[sel]; F = f.iloc[sel]
                    diff = lambda v: np.abs(v[:, None] - v[None, :])
                    X = [(L[:, None] != L[None, :]).astype(float), (C[:, None] == C[None, :]).astype(float),
                         diff(F.n_wp.to_numpy(float)), diff(F.fert.to_numpy()), diff(F.cont.to_numpy()), diff(F.tgt.to_numpy(float))]
                    names = ["different_language", "same_concept", "d_wordpieces", "d_fertility", "d_cont_share", "d_target_pieces"]
                    base = mrm(Ds, X[:2], names[:2], a.K, rng)
                    full = mrm(Ds, X, names, a.K, rng)
                    key = f"{metric} {domain} H{dim} {subset}"
                    out[key] = dict(base=base.to_dict("records"), full=full.to_dict("records"))
                    print(f"\n== MRM {key} (N={len(sel)}) ==")
                    print("  language only :", base.round(4).to_string(index=False).replace("\n", "\n                  "))
                    print("  + tokenization:", full.round(4).to_string(index=False).replace("\n", "\n                  "))
        for dim in (0, 1):
            (sc, bc), (sa, ba) = S[("color", dim)], S[("adjective", dim)]
            d = bc - ba
            print(f"\n## Interaction {metric} H{dim}, length-matched: S_color = {sc:.4f} "
                  f"[{np.percentile(bc,2.5):.4f}, {np.percentile(bc,97.5):.4f}], S_adj = {sa:.4f} "
                  f"[{np.percentile(ba,2.5):.4f}, {np.percentile(ba,97.5):.4f}], diff = {sc-sa:+.4f} "
                  f"95% CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}], "
                  f"P(diff<=0) = {(d<=0).mean():.3f}")
            out[f"interaction {metric} H{dim}"] = dict(S_color=sc, S_adj=sa, diff=sc - sa,
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    json.dump(out, open(REPO / "results" / "phase3_diagram_distances" / "tokenization_mrm.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
