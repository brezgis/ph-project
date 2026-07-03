"""Forest plot of the within-(lang, term) split null vs cross-language signal.

For each color triple, three horizontal whiskers show the within-language
split-distance distribution (mean and 5–95% range from K=200 random halves
of contexts of the same term in the same language). A diamond marks the
cross-language same-term observed distance.

The visual story: the cross-lang signal sits just outside (or inside) the
within-lang noise band — the gap is a few percent.

Outputs:
  results/figures/within_language_null_forest.pdf
  results/figures/within_language_null_forest.png
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "results" / "phase3_diagram_distances"
OUT_DIR = REPO_ROOT / "results" / "figures"

METRIC = "wasserstein"
DIM = 1

LANG_COLORS = {"en": "#1f77b4", "ru": "#d62728", "es": "#2ca02c"}
LANG_LABELS = {"en": "English", "ru": "Russian", "es": "Spanish"}
SIGNAL_COLOR = "#000000"

TRIPLE_RE = re.compile(r"^(?P<en>[^ ]+) \((?P<ru>[^/]+)/(?P<es>[^)]+)\)$")


def parse_triple(key: str) -> tuple[str, str, str]:
    m = TRIPLE_RE.match(key)
    if not m:
        raise ValueError(f"unparseable triple key: {key!r}")
    return m["en"], m["ru"], m["es"]


def main() -> None:
    null_path = DATA_DIR / f"within_language_null_{METRIC}_h{DIM}.csv"
    cross_path = DATA_DIR / f"cross_language_same_term_{METRIC}_h{DIM}.csv"
    null_df = pd.read_csv(null_path)
    cross_df = pd.read_csv(cross_path)

    null_df = null_df[~null_df["skipped"].astype(bool)].copy()
    null_lookup = {(r["lang"], r["term"]): r for _, r in null_df.iterrows()}

    rows = []
    for _, r in cross_df.iterrows():
        en, ru, es = parse_triple(r["triple"])
        rows.append({
            "label": f"{en}  /  {ru}  /  {es}",
            "en": null_lookup.get(("en", en)),
            "ru": null_lookup.get(("ru", ru)),
            "es": null_lookup.get(("es", es)),
            "cross": float(r["cross_lang_mean"]),
        })

    n = len(rows)
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * n + 1.6))

    y_positions = np.arange(n)[::-1]
    lang_offset = {"en": +0.22, "ru": 0.0, "es": -0.22}

    for y, row in zip(y_positions, rows):
        for lang in ("en", "ru", "es"):
            stats = row[lang]
            if stats is None:
                continue
            yy = y + lang_offset[lang]
            ax.hlines(
                yy, stats["split_p05"], stats["split_p95"],
                color=LANG_COLORS[lang], lw=2.2, alpha=0.85, zorder=2,
            )
            ax.plot(
                stats["split_mean"], yy,
                marker="o", ms=5.0,
                mfc=LANG_COLORS[lang], mec="white", mew=0.7,
                zorder=3,
            )
        ax.plot(
            row["cross"], y,
            marker="D", ms=8.5,
            mfc=SIGNAL_COLOR, mec="white", mew=1.0,
            zorder=4,
        )

    pooled_within = np.mean([
        row[lang]["split_mean"]
        for row in rows for lang in ("en", "ru", "es")
        if row[lang] is not None
    ])
    pooled_cross = np.mean([row["cross"] for row in rows])
    ratio_pct = (pooled_cross / pooled_within - 1) * 100

    ax.axvline(pooled_within, color="grey", ls="--", lw=0.9, alpha=0.7, zorder=1)
    ax.axvline(pooled_cross, color=SIGNAL_COLOR, ls=":", lw=0.9, alpha=0.7, zorder=1)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=9)
    ax.set_xlabel(
        f"Mean cross-half persistence-diagram distance  "
        f"({METRIC}, $H_{DIM}$, layer/head averaged)"
    )
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=LANG_COLORS[l], lw=2.2,
                   marker="o", ms=5, mec="white",
                   label=f"{LANG_LABELS[l]}  within-(lang, term) split  (5–95%)")
        for l in ("en", "ru", "es")
    ] + [
        plt.Line2D([0], [0], color=SIGNAL_COLOR, lw=0,
                   marker="D", ms=8, mec="white", mew=1.0,
                   label="Cross-language same-term observed"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.10),
        ncol=2, frameon=False, fontsize=9,
    )

    ax.set_title(
        f"Within-language split null vs cross-language signal  "
        f"(pooled signal {ratio_pct:+.1f}% above noise floor)",
        fontsize=11, pad=10,
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / "within_language_null_forest.pdf"
    png = OUT_DIR / "within_language_null_forest.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
