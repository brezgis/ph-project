"""Figures for the analysis.

Final paper/web figures (results/figures/final/, PDF + PNG), drawn only from the
committed CSVs in results/phase3_diagram_distances/ (nothing is recomputed):

  fig1_language_effect     language effect (z vs. permutation null) of the
                           head-averaged diagram distance: color terms vs. the
                           control adjectives, all sentences vs. length-matched,
                           H0 and H1, Wasserstein and bottleneck. Also writes the
                           plotted numbers to fig1_language_effect_values.csv.
  fig2_head_effects        per-(layer, head) language effect for color terms on
                           the length-matched subset, H0 and H1; BH-significant
                           heads marked with a dot
  fig3_russian_blue        permutation p-values for the синий/голубой split and
                           the two blue translation triples, 4 metric x dim cells
  fig4_within_language_null  restyled within-language split null forest
                           (same data and logic as scripts/within_language_null_figure.py)

Legacy figure (results/figures/, needs the tokenizer and data/kwic):

  fig_token_lengths        mBERT tokens per KWIC sentence by language, with the
                           old 32-token cap and the current 64-token cap

Inputs: language_effect_<domain>_<metric>.csv (scripts/language_effect.py),
robustness_summary.csv, within_language_null_*.csv, cross_language_same_term_*.csv.

Usage: uv run python scripts/make_figures.py [--only language_effect head_effects
russian_blue within_null token_lengths] [--metric wasserstein|bottleneck]
(--metric only affects the heatmap; fig1 always shows both metrics.)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
RES = REPO_ROOT / "results" / "phase3_diagram_distances"
FIG = REPO_ROOT / "results" / "figures"
FINAL = FIG / "final"

# ---------------------------------------------------------------- style
SURFACE = "#ffffff"
INK = "#1a1a1a"
INK_2 = "#4a4a48"
MUTED = "#8a8984"
GRID = "#ebeae6"
AXIS = "#9c9b96"
# Two-series comparison (validated colorblind-safe on light background)
DOMAIN_COLOR = {"color": "#2F6DB0", "adjective": "#C47A2C"}
DOMAIN_NAME = {"color": "Color terms", "adjective": "Control adjectives"}
LANG_COLOR = {"en": "#2a78d6", "es": "#eb6834", "ru": "#1baf7a"}
LANG_NAME = {"en": "English", "es": "Spanish", "ru": "Russian"}
METRIC_NAME = {"wasserstein": "Wasserstein", "bottleneck": "Bottleneck"}
# Single-hue sequential for one-signed effect sizes (near-white -> blue -> navy)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f5f7fa", "#c9d9ec", "#7ea6d3", "#2F6DB0", "#173b66"])

HALF_W = 3.4  # inches; figures are drawn at the size they are set in the paper

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 7.5,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.5,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})


def save(fig, name: str, out: Path = FINAL) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {(out / name).relative_to(REPO_ROOT)}.pdf/.png")


def _xgrid(ax) -> None:
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def load_effects(metric: str, res: Path = RES) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(res.glob(f"language_effect_*_{metric}.csv"))]
    if not frames:
        raise SystemExit(f"no language_effect_*_{metric}.csv in {res}; run scripts/language_effect.py")
    df = pd.concat(frames, ignore_index=True)
    df["effect_size"] = df["effect_size"].astype(float)
    df["sig"] = df["passes_bh"].astype(str) == "True"
    return df


# ---------------------------------------------------------------- fig 1
ROWS = [(0, "all"), (0, "length_matched"), (1, "all"), (1, "length_matched")]
ROW_Y = [0, 1, 2.4, 3.4]  # small gap between the H0 and H1 groups
SUBSET_NAME = {"all": "all sentences", "length_matched": "length-matched"}


def fig_language_effect(res: Path = RES) -> pd.DataFrame:
    recs = []
    for metric in ("wasserstein", "bottleneck"):
        df = load_effects(metric, res)
        agg = df[df["layer"] == "agg"]
        for _, r in agg.iterrows():
            recs.append({"metric": metric, "domain": r["domain"], "dim": int(r["dim"]), "subset": r["subset"],
                         "n_sentences": int(r["n_sentences"]), "z": float(r["effect_size"]),
                         "p_value": float(r["p_value"]), "passes_bh": bool(r["sig"])})
    vals = pd.DataFrame(recs).sort_values(["metric", "dim", "subset", "domain"], ignore_index=True)

    fig, axes = plt.subplots(2, 1, figsize=(HALF_W, 3.3), sharex=True)
    dodge = {"color": -0.17, "adjective": 0.17}
    xmax = 125
    for ax, metric in zip(axes, ("wasserstein", "bottleneck")):
        _xgrid(ax)
        for y, (dim, sub) in zip(ROW_Y, ROWS):
            for dom in ("color", "adjective"):
                r = vals[(vals.metric == metric) & (vals.dim == dim) & (vals.subset == sub) & (vals.domain == dom)]
                if r.empty:
                    continue
                z = float(r["z"].iloc[0])
                yy = y + dodge[dom]
                ax.plot([0, z], [yy, yy], color=DOMAIN_COLOR[dom], lw=0.8, alpha=0.35,
                        solid_capstyle="butt", zorder=2)
                ax.scatter(z, yy, s=16, color=DOMAIN_COLOR[dom], edgecolor=SURFACE, linewidth=0.6, zorder=3)
                ax.text(z + 2.5, yy, f"{z:.1f}", va="center", ha="left", fontsize=6.3, color=INK_2)
        ax.set_yticks(ROW_Y, [f"$H_{d}$, {SUBSET_NAME[s]}" for d, s in ROWS])
        ax.set_ylim(ROW_Y[-1] + 0.55, -0.55)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{METRIC_NAME[metric]} distance", loc="left", pad=3)
    axes[-1].set_xlabel("Language effect (z), head-averaged distances", loc="right")
    axes[-1].set_xticks([0, 25, 50, 75, 100, 125])
    handles = [Line2D([], [], ls="", marker="o", markersize=4.5, color=DOMAIN_COLOR[d],
                      label=DOMAIN_NAME[d]) for d in ("color", "adjective")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.985), ncol=2,
               handletextpad=0.2, columnspacing=1.5, borderaxespad=0)
    fig.tight_layout(h_pad=0.8)
    save(fig, "fig1_language_effect")
    out = FINAL / "fig1_language_effect_values.csv"
    vals.to_csv(out, index=False, float_format="%.4g")
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return vals


# ---------------------------------------------------------------- fig 2
def _head_grid(df: pd.DataFrame, domain: str, dim: int, subset: str):
    r = df[(df["domain"] == domain) & (df["dim"] == dim) & (df["subset"] == subset) & (df["layer"] != "agg")]
    grid = np.full((12, 12), np.nan)
    sig = np.zeros((12, 12), bool)
    for row in r.itertuples():
        grid[int(row.layer), int(row.head)] = row.effect_size
        sig[int(row.layer), int(row.head)] = row.sig
    return grid, sig


def fig_head_effects(metric: str = "wasserstein", domain: str = "color", subset: str = "length_matched",
                     res: Path = RES) -> None:
    df = load_effects(metric, res)
    grids = {dim: _head_grid(df, domain, dim, subset) for dim in (0, 1)}
    vmax = max(np.nanmax(g) for g, _ in grids.values())
    vmax = float(np.ceil(vmax / 10) * 10)
    # Effect sizes are one-signed apart from a few small negatives (noise around 0),
    # and heavily right-skewed, so: sequential map, square-root scale, negatives clipped to 0.
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax, clip=True)
    fig, axes = plt.subplots(1, 2, figsize=(HALF_W, 2.25))
    for ax, dim in zip(axes, (0, 1)):
        g, s = grids[dim]
        im = ax.pcolormesh(np.arange(13) - 0.5, np.arange(13) - 0.5, np.clip(g, 0, None), cmap=SEQ, norm=norm,
                           edgecolors=SURFACE, linewidth=0.35)
        ii, jj = np.nonzero(s)
        dark = norm(np.clip(g[ii, jj], 0, None)) > 0.55
        ax.scatter(jj, ii, s=2.2, c=np.where(dark, SURFACE, INK), linewidths=0, zorder=3)
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(11.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, 12, 2))
        ax.set_yticks(range(0, 12, 2))
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Head")
        ax.set_title(f"$H_{dim}$", loc="left", pad=11)
        ax.text(0, 1.012, f"{int(s.sum())} of 144 heads significant", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=6.5, color=INK_2)
        print(f"heatmap {domain} {metric} {subset} H{dim}: {int(s.sum())}/144 BH-significant, "
              f"z range {np.nanmin(g):.2f}..{np.nanmax(g):.2f}, median {np.nanmedian(g):.2f}")
    axes[0].set_ylabel("Layer")
    axes[1].tick_params(labelleft=False)
    fig.tight_layout(w_pad=0.6)
    cax = fig.add_axes([0.25, 0.0, 0.55, 0.03])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = [t for t in (0, 2, 5, 10, 20, 40, 60, 80) if t <= vmax]
    cb.set_ticks(ticks, labels=[str(t) for t in ticks])
    cb.ax.tick_params(length=0, pad=1.5, labelsize=6.5)
    cb.outline.set_visible(False)
    cb.set_label("Language effect (z), square-root scale", fontsize=7, labelpad=2)
    suffix = "" if (metric, domain, subset) == ("wasserstein", "color", "length_matched") else \
        f"_{domain}_{metric}_{subset}"
    save(fig, f"fig2_head_effects{suffix}")


# ---------------------------------------------------------------- fig 3
def fig_russian_blue(res: Path = RES) -> pd.DataFrame:
    rob = pd.read_csv(res / "robustness_summary.csv")
    tests = [("ru-blue-zoom", "Russian split: синий vs. голубой"),
             ("per-term-синий", "Triple {blue, azul, синий}"),
             ("per-term-голубой", "Triple {blue, azul, голубой}")]
    combos = [("wasserstein", 0), ("wasserstein", 1), ("bottleneck", 0), ("bottleneck", 1)]
    fig, ax = plt.subplots(figsize=(HALF_W, 2.75))
    _xgrid(ax)
    ylabels, ypos, y, rows = [], [], 0.0, []
    for test, label in tests:
        ax.text(-0.02, y - 1.0, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7, fontweight="bold", color=INK, linespacing=1.1)
        for metric, dim in combos:
            r = rob[(rob.test == test) & (rob.metric == metric) & (rob.dim == dim)]
            if r.empty:
                y += 1
                continue
            p = float(r["p_value"].iloc[0])
            rows.append({"test": test, "metric": metric, "dim": dim, "p_value": p,
                         "effect_size": float(r["effect_size"].iloc[0])})
            sig = p < 0.05
            ax.scatter(p, y, s=15, zorder=3, linewidth=0.9, facecolor=INK if sig else SURFACE, edgecolor=INK)
            ax.text(p * 1.12, y, f"{p:.2f}" if p >= 0.01 else f"{p:.3f}", va="center", ha="left",
                    fontsize=6.3, color=INK_2)
            ylabels.append(f"{METRIC_NAME[metric]} $H_{dim}$")
            ypos.append(y)
            y += 1
        y += 1.6
    ax.axvline(0.05, color=INK_2, lw=0.7, ls=(0, (2.5, 2)), zorder=1)
    ax.text(0.052, -1.55, "p = 0.05", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 1.6)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1], ["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.minorticks_off()
    ax.set_yticks(ypos, ylabels)
    ax.set_ylim(y - 1.6 + 0.2, -1.6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Permutation p-value (log scale)")
    save(fig, "fig3_russian_blue")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- fig 4
TRIPLE_RE = re.compile(r"^(?P<en>[^ ]+) \((?P<ru>[^/]+)/(?P<es>[^)]+)\)$")


def fig_within_language_null(metric: str = "wasserstein", dim: int = 1, res: Path = RES) -> None:
    """Same data and logic as scripts/within_language_null_figure.py, restyled."""
    null_df = pd.read_csv(res / f"within_language_null_{metric}_h{dim}.csv")
    cross_df = pd.read_csv(res / f"cross_language_same_term_{metric}_h{dim}.csv")
    null_df = null_df[~null_df["skipped"].astype(bool)]
    lookup = {(r["lang"], r["term"]): r for _, r in null_df.iterrows()}
    rows = []
    for _, r in cross_df.iterrows():
        m = TRIPLE_RE.match(r["triple"])
        en, ru, es = m["en"], m["ru"], m["es"]
        rows.append({"label": f"{en} / {ru} / {es}", "cross": float(r["cross_lang_mean"]),
                     "en": lookup.get(("en", en)), "ru": lookup.get(("ru", ru)), "es": lookup.get(("es", es))})
    n = len(rows)
    fig, ax = plt.subplots(figsize=(HALF_W, 0.27 * n + 0.75))
    _xgrid(ax)
    off = {"en": -0.22, "es": 0.0, "ru": 0.22}
    for y, row in enumerate(rows):
        for lang in ("en", "es", "ru"):
            st = row[lang]
            if st is None:
                continue
            yy = y + off[lang]
            ax.hlines(yy, st["split_p05"], st["split_p95"], color=LANG_COLOR[lang], lw=1.4, zorder=2)
            ax.scatter(st["split_mean"], yy, s=7, color=LANG_COLOR[lang], linewidths=0, zorder=3)
        ax.scatter(row["cross"], y, s=16, marker="D", color=INK, edgecolor=SURFACE, linewidth=0.5, zorder=4)
    within = np.mean([row[lg]["split_mean"] for row in rows for lg in ("en", "es", "ru") if row[lg] is not None])
    cross = np.mean([row["cross"] for row in rows])
    ax.axvline(within, color=MUTED, lw=0.7, ls=(0, (2.5, 2)), zorder=1)
    ax.axvline(cross, color=INK, lw=0.7, ls=(0, (1, 1.5)), zorder=1)
    print(f"within-language null: pooled within {within:.4f}, pooled cross {cross:.4f} "
          f"({(cross / within - 1) * 100:+.1f}%)")
    ax.set_yticks(range(n), [r["label"] for r in rows])
    ax.set_ylim(n - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Diagram distance ({METRIC_NAME[metric]}, $H_{dim}$)", loc="right")
    handles = [Line2D([], [], color=LANG_COLOR[lg], lw=1.4, marker="o", markersize=2.6, label=LANG_NAME[lg])
               for lg in ("en", "es", "ru")]
    handles.append(Line2D([], [], ls="", marker="D", markersize=3.6, color=INK, label="Cross-language"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.99), ncol=4, handlelength=1.3,
               handletextpad=0.35, columnspacing=1.0, borderaxespad=0)
    fig.tight_layout()
    save(fig, "fig4_within_language_null")


# ---------------------------------------------------------------- legacy
def fig_token_lengths(max_len_old: int = 32, max_len_new: int = 64) -> None:
    from transformers import AutoTokenizer

    from compute_diagram_distances import MODEL_TAG

    tok = AutoTokenizer.from_pretrained(MODEL_TAG)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    bins = np.arange(8, 81)
    peak = 0
    for lang in ("en", "es", "ru"):
        sents = pd.read_csv(REPO_ROOT / "data" / "kwic" / lang / "color.csv")["sentence"].tolist()
        n = np.array([len(x) for x in tok(sents)["input_ids"]])
        h, _ = np.histogram(n, bins=bins)
        share = h / h.sum() * 100
        peak = max(peak, share.max())
        ax.plot(bins[:-1], share, color=LANG_COLOR[lang], lw=2, solid_capstyle="round",
                solid_joinstyle="round", label=f"{LANG_NAME[lang]} (median {int(np.median(n))})")
        cut = (n > max_len_old).mean() * 100
        print(f"{lang}: median {np.median(n):.0f} tokens, {cut:.0f}% over {max_len_old}")
    for x, txt in ((max_len_old, f"old cap ({max_len_old})"), (max_len_new, f"current cap ({max_len_new})")):
        ax.axvline(x, color=INK_2, lw=1, ls=(0, (2, 2)))
        ax.text(x + 0.6, peak * 1.02, txt, color=INK_2, fontsize=8.5, va="bottom")
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, peak * 1.15)
    ax.set_xlabel("mBERT tokens per KWIC sentence (incl. [CLS], [SEP])")
    ax.set_ylabel("Share of sentences (%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=9, handlelength=1.6,
              borderaxespad=0.3)
    ax.set_title("The same 21-word window is longer in Russian once mBERT tokenizes it",
                 loc="left", pad=26)
    save(fig, "fig_token_lengths", out=FIG)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="wasserstein", choices=["wasserstein", "bottleneck"],
                    help="metric for the per-head heatmap (fig1 always shows both)")
    ap.add_argument("--results-dir", type=Path, default=RES)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: language_effect head_effects russian_blue within_null token_lengths")
    args = ap.parse_args(argv)
    want = set(args.only or ["language_effect", "head_effects", "russian_blue", "within_null"])
    if "language_effect" in want:
        vals = fig_language_effect(args.results_dir)
        print(vals.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if "head_effects" in want:
        fig_head_effects(args.metric, res=args.results_dir)
    if "russian_blue" in want:
        print(fig_russian_blue(args.results_dir).to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if "within_null" in want:
        fig_within_language_null(res=args.results_dir)
    if "token_lengths" in want:
        fig_token_lengths()


if __name__ == "__main__":
    main()
