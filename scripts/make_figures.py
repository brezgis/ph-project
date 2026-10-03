"""Figures for the re-run analysis.

  fig_language_effect      head-averaged language effect (z) for color terms vs
                           the control adjectives, all sentences vs length-matched,
                           H0 and H1
  fig_head_effects         per-(layer, head) language effect, color vs control,
                           length-matched H1, significant heads outlined
  fig_head_effects_color   the color panel alone (used on the website)
  fig_token_lengths        mBERT tokens per KWIC sentence by language, with the
                           old 32-token cap and the current 64-token cap

Inputs: results/phase3_diagram_distances/language_effect_<domain>_<metric>.csv
(scripts/language_effect.py) and data/kwic/<lang>/color.csv. Writes PNG + PDF to
results/figures/.

Colors follow one fixed scheme: languages en/es/ru are always blue/orange/aqua;
the two term sets are filled vs hollow ink, so the encodings never collide.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
RES = REPO_ROOT / "results" / "phase3_diagram_distances"
FIG = REPO_ROOT / "results" / "figures"

# Palette (validated reference instance; light chart surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LANG_COLOR = {"en": "#2a78d6", "es": "#eb6834", "ru": "#1baf7a"}
LANG_NAME = {"en": "English", "es": "Spanish", "ru": "Russian"}
DOMAIN_NAME = {"color": "Color terms", "adjective": "Control adjectives"}
# Diverging effect scale: blue = farther apart across languages, red = closer, gray = 0
DIVERGING = LinearSegmentedColormap.from_list(
    "effect", ["#9e2a29", "#e34948", "#f2b8b5", "#f0efec", "#9ec5f4", "#3987e5", "#104281"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote results/figures/{name}.png/.pdf")


def load_effects(metric: str, res: Path = RES) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(res.glob(f"language_effect_*_{metric}.csv"))]
    if not frames:
        raise SystemExit(f"no language_effect_*_{metric}.csv in {RES}; run scripts/language_effect.py")
    return pd.concat(frames, ignore_index=True)


def fig_language_effect(df: pd.DataFrame, metric: str) -> None:
    agg = df[df["layer"] == "agg"].copy()
    agg["effect_size"] = agg["effect_size"].astype(float)
    subsets = [("all", "All sentences"), ("length_matched", "Length-matched")]
    domains = [d for d in ("color", "adjective") if d in set(agg["domain"])]
    xmax = agg["effect_size"].max() * 1.12
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.6), sharey=True)
    for ax, dim in zip(axes, (0, 1)):
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for yi, (sub, _) in enumerate(subsets):
            vals = {}
            for d in domains:
                r = agg[(agg["dim"] == dim) & (agg["subset"] == sub) & (agg["domain"] == d)]
                if len(r):
                    vals[d] = float(r["effect_size"].iloc[0])
            if len(vals) == 2:  # connector shows the gap between term sets
                ax.plot([vals["color"], vals["adjective"]], [yi, yi], color=AXIS, lw=2, zorder=1,
                        solid_capstyle="round")
            for d, v in vals.items():
                filled = d == "color"
                ax.scatter(v, yi, s=70, zorder=3, linewidths=1.6,
                           facecolor=INK if filled else SURFACE, edgecolor=INK)
                ax.annotate(f"{v:.0f}", (v, yi), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK_2)
        ax.set_title(f"H{dim}  ({'connected components' if dim == 0 else 'loops'})", loc="left")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(subsets) - 0.4)
        ax.invert_yaxis()
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Language effect (z vs. permutation null)")
    axes[0].set_yticks(range(len(subsets)), [s for _, s in subsets])
    handles = [
        plt.Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=INK, markeredgecolor=INK,
                   label=DOMAIN_NAME["color"]),
        plt.Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=INK,
                   markeredgewidth=1.6, label=DOMAIN_NAME["adjective"]),
    ]
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.suptitle(f"How strongly persistence diagrams differ by language ({metric.capitalize()} distance)",
                 x=0.01, ha="left", y=0.99, fontsize=12, fontweight="semibold", color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.0, 0.92), fontsize=9,
               handletextpad=0.3, columnspacing=1.4)
    save(fig, "fig_language_effect")


def _head_grid(df: pd.DataFrame, domain: str, dim: int, subset: str):
    r = df[(df["domain"] == domain) & (df["dim"] == dim) & (df["subset"] == subset) & (df["layer"] != "agg")]
    grid = np.full((12, 12), np.nan)
    sig = np.zeros((12, 12), bool)
    for row in r.itertuples():
        grid[int(row.layer), int(row.head)] = float(row.effect_size)
        sig[int(row.layer), int(row.head)] = str(row.passes_bh) == "True"
    return grid, sig


def _draw_heads(ax, grid, sig, norm):
    # Cells separated by surface-colored gaps; heads that are NOT significant
    # get a small muted dot (most heads are significant, so mark the exception).
    im = ax.pcolormesh(np.arange(13) - 0.5, np.arange(13) - 0.5, grid, cmap=DIVERGING, norm=norm,
                       edgecolors=SURFACE, linewidth=1.5)
    ii, jj = np.nonzero(~sig & ~np.isnan(grid))
    ax.scatter(jj, ii, s=6, color=INK_2, linewidths=0, zorder=3)
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(11.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 12, 2))
    ax.set_yticks(range(0, 12, 2))
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_head_effects(df: pd.DataFrame, dim: int = 1, subset: str = "length_matched") -> None:
    domains = [d for d in ("color", "adjective") if d in set(df["domain"])]
    grids = {d: _head_grid(df, d, dim, subset) for d in domains}
    lim = max(np.nanmax(np.abs(g)) for g, _ in grids.values())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(1, len(domains), figsize=(4.0 * len(domains) + 0.8, 4.2))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, domains):
        g, s = grids[d]
        im = _draw_heads(ax, g, s, norm)
        ax.set_title(f"{DOMAIN_NAME[d]}: {int(s.sum())}/144 heads significant", loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("Language effect (z)", color=INK_2)
    cb.outline.set_visible(False)
    label = "length-matched" if subset == "length_matched" else "all sentences"
    fig.suptitle(f"Language effect per attention head (H{dim}, {label}; dot = not significant, BH q = 0.05)",
                 x=0.02, ha="left", fontsize=12, fontweight="semibold", color=INK)
    save(fig, "fig_head_effects")

    # Website thumbnail: color panel only, no titles
    g, s = grids["color"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = _draw_heads(ax, g, s, norm)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label("Language effect (z)", color=INK_2)
    cb.outline.set_visible(False)
    save(fig, "fig_head_effects_color")


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
    save(fig, "fig_token_lengths")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="wasserstein", choices=["wasserstein", "bottleneck"])
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: language_effect head_effects token_lengths")
    args = ap.parse_args(argv)
    want = set(args.only or ["language_effect", "head_effects", "token_lengths"])
    if want & {"language_effect", "head_effects"}:
        df = load_effects(args.metric, args.results_dir or RES)
        if "language_effect" in want:
            fig_language_effect(df, args.metric)
        if "head_effects" in want:
            fig_head_effects(df)
    if "token_lengths" in want:
        fig_token_lengths()


if __name__ == "__main__":
    main()
