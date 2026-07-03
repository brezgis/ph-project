"""4-panel persistence-diagram zoom on the Russian-blues centerpiece cells.

en/blue, es/azul, ru/синий, ru/голубой — birth/death pairs side-by-side
so the topological structure of each cloud is directly comparable.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CACHE = Path("data/phase3/draganov_diagrams")
OUT = Path("results/figures")

CENTERPIECE = [
    ("en", "blue", "English  blue"),
    ("es", "azul", "Spanish  azul"),
    ("ru", "синий", "Russian  синий  (dark blue)"),
    ("ru", "голубой", "Russian  голубой  (light blue)"),
]

H0_COLOR = "#1f77b4"
H1_COLOR = "#d62728"


def load(lang, term):
    p = CACHE / f"{lang}_{term}.npz"
    d = np.load(p, allow_pickle=True)
    return d["h0"], d["h1"], int(d["n_samples"])


def main():
    cells = [(l, t, label, *load(l, t)) for (l, t, label) in CENTERPIECE]

    all_deaths = np.concatenate([
        np.concatenate([c[3][:, 1], c[4][:, 1]])
        for c in cells
    ])
    xmax = float(all_deaths.max()) * 1.05

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)

    for ax, (lang, term, label, h0, h1, n) in zip(axes, cells):
        ax.plot([0, xmax], [0, xmax], "k-", lw=0.5, alpha=0.4, zorder=1)
        ax.scatter(h0[:, 0], h0[:, 1], s=10, c=H0_COLOR,
                   alpha=0.55, zorder=2, label=f"H₀ ({len(h0)})")
        ax.scatter(h1[:, 0], h1[:, 1], s=14, c=H1_COLOR,
                   alpha=0.75, zorder=3, label=f"H₁ ({len(h1)})")

        ax.set_xlim(-0.02, xmax)
        ax.set_ylim(-0.02, xmax)
        ax.set_aspect("equal")
        ax.set_xlabel("Birth", fontsize=10)
        ax.set_title(f"{label}\nn={n}", fontsize=11)
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        ax.grid(alpha=0.2, linewidth=0.4)

    axes[0].set_ylabel("Death", fontsize=10)

    fig.suptitle(
        "Russian-blues centerpiece: per-cell persistence diagrams\n"
        "Cosine-distance Vietoris–Rips on mean-pooled mBERT embedding clouds",
        fontsize=12, y=1.02
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT / f"draganov_replication_pd_scatter_centerpiece.{ext}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
