"""Generate a 34-panel persistence-diagram scatter grid for the Draganov pipeline.

One panel per (lang, term) cell. Each panel shows H_0 (components) and H_1 (loops)
birth/death pairs as a scatter plot, with the diagonal y=x for reference.
The Russian-blues centerpiece cells (en/blue, es/azul, ru/синий, ru/голубой)
get a highlighted border.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CACHE = Path("data/phase3/draganov_diagrams")
OUT = Path("results/figures")

CANON = {
    "en": ["black", "white", "red", "green", "yellow", "blue",
           "brown", "pink", "gray", "orange", "purple"],
    "ru": ["чёрный", "белый", "красный", "зелёный", "жёлтый", "синий",
           "голубой", "коричневый", "розовый", "серый", "оранжевый", "фиолетовый"],
    "es": ["negro", "blanco", "rojo", "verde", "amarillo", "azul",
           "marrón", "rosa", "gris", "naranja", "morado"],
}
CENTERPIECE = {("en", "blue"), ("es", "azul"),
               ("ru", "синий"), ("ru", "голубой")}

H0_COLOR = "#1f77b4"
H1_COLOR = "#d62728"
HILITE = "#ff7f0e"

LANG_LABEL = {"en": "English", "ru": "Russian", "es": "Spanish"}


def load_all():
    pds = {}
    for npz in CACHE.glob("*.npz"):
        d = np.load(npz, allow_pickle=True)
        pds[(str(d["lang"]), str(d["term"]))] = (
            d["h0"], d["h1"], int(d["n_samples"])
        )
    return pds


def main():
    pds = load_all()

    all_deaths = np.concatenate([
        np.concatenate([pds[k][0][:, 1], pds[k][1][:, 1]])
        for k in pds if pds[k][0].size or pds[k][1].size
    ])
    xmax = float(all_deaths.max()) * 1.05

    n_cols = 12
    fig, axes = plt.subplots(
        3, n_cols, figsize=(n_cols * 1.55, 3 * 1.85),
        sharex=True, sharey=True
    )

    for row, lang in enumerate(("en", "ru", "es")):
        terms = CANON[lang]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(terms):
                ax.axis("off")
                continue
            term = terms[col]
            key = (lang, term)
            if key not in pds:
                ax.axis("off")
                continue

            h0, h1, n = pds[key]
            ax.plot([0, xmax], [0, xmax], "k-", lw=0.4, alpha=0.4, zorder=1)
            if h0.size:
                ax.scatter(h0[:, 0], h0[:, 1], s=2.5, c=H0_COLOR,
                           alpha=0.55, zorder=2, label="H₀")
            if h1.size:
                ax.scatter(h1[:, 0], h1[:, 1], s=4.0, c=H1_COLOR,
                           alpha=0.75, zorder=3, label="H₁")
            ax.set_xlim(-0.02, xmax)
            ax.set_ylim(-0.02, xmax)
            ax.set_aspect("equal")
            ax.tick_params(axis="both", labelsize=6)

            title = f"{term}\nn={n}, |H₁|={len(h1)}"
            ax.set_title(title, fontsize=7, pad=2)

            if key in CENTERPIECE:
                for spine in ax.spines.values():
                    spine.set_edgecolor(HILITE)
                    spine.set_linewidth(2.0)

        axes[row, 0].set_ylabel(
            f"{LANG_LABEL[lang]}\nDeath",
            fontsize=9, labelpad=8
        )

    for col in range(n_cols):
        axes[2, col].set_xlabel("Birth", fontsize=7)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=H0_COLOR, alpha=0.55, label="H₀ (components)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=H1_COLOR, alpha=0.75, label="H₁ (loops)"),
        plt.Line2D([0], [0], marker="s", linestyle="",
                   markerfacecolor="none", markeredgecolor=HILITE,
                   markeredgewidth=2, label="Russian-blues centerpiece"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Per-(lang, term) persistence diagrams — Draganov contextual mBERT pipeline\n"
        "Cosine-distance Vietoris–Rips on mean-pooled embedding clouds  "
        "(34 cells: 11 EN + 12 RU + 11 ES)",
        fontsize=11, y=1.005
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    for ext in ("pdf", "png"):
        path = OUT / f"draganov_replication_pd_scatter_grid.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
