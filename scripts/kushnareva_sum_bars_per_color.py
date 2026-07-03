"""Per-color cross-linguistic distributions of "sum of bars in H_k" — one chart per color.

For each Berlin & Kay basic color anchor, render a standalone figure overlaying
English / Russian / Spanish distributions of the per-sentence sum-of-bars ripser
feature, pooled across all 144 (layer, head) attention heads. Russian голубой
gets its own panel as a separate Russian-only second-blue chart.

Inputs:  results/phase3_ripser/{lang}_color_..._ripser.npy
         data/kwic/{lang}/color.csv  (column "term", aligned 1:1 to ripser axis)
Outputs: results/figures/kushnareva_sum_bars_per_color/h{0,1}/{slug}.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RIPSER_DIR = Path("results/phase3_ripser")
KWIC_DIR = Path("data/kwic")
OUT_BASE = Path("results/figures/kushnareva_sum_bars_per_color")

RIPSER_FILE = ("{lang}_color_all_heads_12_layers_MAX_LEN_32_"
               "bert-base-multilingual-cased_ripser.npy")

FEATURE_NAMES = [
    "h0_s", "h0_e", "h0_t_d",
    "h0_n_d_m_t0.75", "h0_n_d_m_t0.5", "h0_n_d_l_t0.25",
    "h1_t_b", "h1_n_b_m_t0.25", "h1_n_b_l_t0.95", "h1_n_b_l_t0.70",
    "h1_s", "h1_e", "h1_v", "h1_nb",
]

LANG_COLOR = {"en": "#1f77b4", "ru": "#d62728", "es": "#2ca02c"}
LANG_LABEL = {"en": "English", "ru": "Russian", "es": "Spanish"}

# Berlin & Kay anchored cross-language alignment (Stage VII basics).
# Russian "blue" splits — синий sits on the canonical blue panel; голубой
# gets its own Russian-only second-blue panel.
BK_PANELS = [
    ("01_black",         "black",
     {"en": ["black"],  "ru": ["чёрный"],     "es": ["negro"]}),
    ("02_white",         "white",
     {"en": ["white"],  "ru": ["белый"],      "es": ["blanco"]}),
    ("03_red",           "red",
     {"en": ["red"],    "ru": ["красный"],    "es": ["rojo"]}),
    ("04_green",         "green",
     {"en": ["green"],  "ru": ["зелёный"],    "es": ["verde"]}),
    ("05_yellow",        "yellow",
     {"en": ["yellow"], "ru": ["жёлтый"],     "es": ["amarillo"]}),
    ("06_blue_синий",    "blue (синий)",
     {"en": ["blue"],   "ru": ["синий"],      "es": ["azul"]}),
    ("07_blue_голубой",  "blue (голубой) — Russian only",
     {"en": [],         "ru": ["голубой"],    "es": []}),
    ("08_brown",         "brown",
     {"en": ["brown"],  "ru": ["коричневый"], "es": ["marrón"]}),
    ("09_purple",        "purple",
     {"en": ["purple"], "ru": ["фиолетовый"], "es": ["morado"]}),
    ("10_pink",          "pink",
     {"en": ["pink"],   "ru": ["розовый"],    "es": ["rosa"]}),
    ("11_orange",        "orange",
     {"en": ["orange"], "ru": ["оранжевый"],  "es": ["naranja"]}),
    ("12_gray",          "gray",
     {"en": ["gray"],   "ru": ["серый"],      "es": ["gris"]}),
]


def load_per_color(feature: str) -> dict:
    idx = FEATURE_NAMES.index(feature)
    pooled = {}
    sentence_counts = {}
    for lang in ("en", "ru", "es"):
        arr = np.load(RIPSER_DIR / RIPSER_FILE.format(lang=lang))
        feat = arr[..., idx].transpose(2, 0, 1)        # (n_samples, 12, 12)
        feat = feat.reshape(feat.shape[0], -1)         # (n_samples, 144)
        df = pd.read_csv(KWIC_DIR / lang / "color.csv")
        assert len(df) == feat.shape[0], \
            f"{lang}: csv {len(df)} != ripser {feat.shape[0]}"
        terms = df["term"].values
        for slug, _label, mapping in BK_PANELS:
            wanted = mapping[lang]
            if not wanted:
                pooled[(slug, lang)] = np.empty(0, dtype=np.float64)
                sentence_counts[(slug, lang)] = 0
                continue
            mask = np.isin(terms, wanted)
            pooled[(slug, lang)] = feat[mask].ravel()
            sentence_counts[(slug, lang)] = int(mask.sum())
    return pooled, sentence_counts


def render_one(slug: str, label: str, feature: str,
               pooled: dict, counts: dict, lo: float, hi: float,
               out_dir: Path) -> None:
    bins = np.linspace(lo, hi, 60)
    homology = "H_0" if feature == "h0_s" else "H_1"

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for lang in ("en", "ru", "es"):
        vals = pooled[(slug, lang)]
        if vals.size == 0:
            continue
        ax.hist(
            vals, bins=bins, density=True,
            color=LANG_COLOR[lang], alpha=0.5,
            histtype="stepfilled", linewidth=0,
            label=f"{LANG_LABEL[lang]} (n={counts[(slug, lang)]} sentences)",
        )
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"$\\sum$ bars in ${homology}$  "
                  f"(per-sentence × per-head ripser feature {feature})",
                  fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=10, frameon=False, loc="best")
    ax.set_title(
        f"$\\sum$ bars in ${homology}$ — {label}\n"
        "Pooled across all 144 (layer × head) attention heads",
        fontsize=11,
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{slug}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for feature, subdir in [("h0_s", "h0"), ("h1_s", "h1")]:
        out_dir = OUT_BASE / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        pooled, counts = load_per_color(feature)
        all_vals = np.concatenate([v for v in pooled.values() if v.size])
        lo = float(np.percentile(all_vals, 0.5))
        hi = float(np.percentile(all_vals, 99.5))

        for slug, label, _mapping in BK_PANELS:
            render_one(slug, label, feature, pooled, counts, lo, hi, out_dir)
        print(f"  saved {len(BK_PANELS)} charts × 2 ext to {out_dir}/")


if __name__ == "__main__":
    main()
