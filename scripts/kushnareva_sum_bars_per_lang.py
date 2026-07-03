"""Cross-linguistic analog of Kushnareva 2021 Figure 4 — one chart per (layer, head).

For every (layer, head) in mBERT, render a standalone figure overlaying per-language
histograms of "sum of bars in H_k" pooled across all color-term samples within
each language. 144 charts × 2 homology dimensions = 288 PNG/PDF files.

Inputs:  results/phase3_ripser/{lang}_color_..._ripser.npy
         shape (12 layers, 12 heads, n_samples, 14 ripser features).
         Feature axis follows notebooks/mbert_attention_ripser.ipynb cell 31.
Outputs: results/figures/kushnareva_sum_bars_per_lang/h{0,1}/L{ll}_H{hh}.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RIPSER_DIR = Path("results/phase3_ripser")
OUT_BASE = Path("results/figures/kushnareva_sum_bars_per_lang")

RIPSER_FILE = ("{lang}_color_all_heads_12_layers_MAX_LEN_32_"
               "bert-base-multilingual-cased_ripser.npy")

FEATURE_NAMES = [
    "h0_s", "h0_e", "h0_t_d",
    "h0_n_d_m_t0.75", "h0_n_d_m_t0.5", "h0_n_d_l_t0.25",
    "h1_t_b", "h1_n_b_m_t0.25", "h1_n_b_l_t0.95", "h1_n_b_l_t0.70",
    "h1_s", "h1_e", "h1_v", "h1_nb",
]

LANGS = ("en", "ru", "es")
LANG_LABEL = {"en": "English", "ru": "Russian", "es": "Spanish"}
LANG_COLOR = {"en": "#1f77b4", "ru": "#d62728", "es": "#2ca02c"}
LANG_N = {"en": 2200, "ru": 2267, "es": 2161}


def load_sums(feature: str) -> dict[str, np.ndarray]:
    idx = FEATURE_NAMES.index(feature)
    sums = {}
    for lang in LANGS:
        arr = np.load(RIPSER_DIR / RIPSER_FILE.format(lang=lang))
        sums[lang] = arr[..., idx]  # (12, 12, n_samples)
    return sums


def render_one(layer: int, head: int, feature: str,
               sums: dict[str, np.ndarray], lo: float, hi: float,
               out_dir: Path) -> None:
    bins = np.linspace(lo, hi, 50)
    homology = "H_0" if feature == "h0_s" else "H_1"

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for lang in LANGS:
        vals = sums[lang][layer, head]
        ax.hist(
            vals, bins=bins, density=True,
            color=LANG_COLOR[lang], alpha=0.5,
            histtype="stepfilled", linewidth=0,
            label=f"{LANG_LABEL[lang]} (n={LANG_N[lang]})",
        )
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"$\\sum$ bars in ${homology}$  (per-sentence ripser feature {feature})",
                  fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="best")
    ax.set_title(
        f"$\\sum$ bars in ${homology}$ — Layer {layer}, Head {head}\n"
        "Cross-linguistic comparison on color-term KWIC contexts",
        fontsize=11,
    )
    fig.tight_layout()

    stem = f"L{layer:02d}_H{head:02d}"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for feature, subdir in [("h0_s", "h0"), ("h1_s", "h1")]:
        out_dir = OUT_BASE / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        sums = load_sums(feature)
        all_vals = np.concatenate([s.ravel() for s in sums.values()])
        lo = float(np.percentile(all_vals, 0.5))
        hi = float(np.percentile(all_vals, 99.5))

        for layer in range(12):
            for head in range(12):
                render_one(layer, head, feature, sums, lo, hi, out_dir)
        print(f"  saved {12 * 12} charts × 2 ext to {out_dir}/")


if __name__ == "__main__":
    main()
