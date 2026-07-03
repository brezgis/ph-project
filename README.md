# Topology of Color

Cross-linguistic category structure in mBERT attention, measured with persistent homology.

Course project for COSI 115a (NLP Fundamentals), Brandeis University, Spring 2026.
Anna Brežģis — [brezgis.com](https://brezgis.com)

## The question

Languages carve the continuous color spectrum into discrete, unevenly drawn categories.
Russian famously draws one boundary that English and Spanish don't: dark blue (*синий*) and
light blue (*голубой*) are obligatory, distinct basic color terms, and Winawer et al. (2007)
showed the split has measurable perceptual consequences. If a category boundary is salient
enough in a language, does it leave a trace in the *shape* of a multilingual transformer's
attention?

Concretely, three nested questions:

1. Does distributional evidence reveal systematic, distinct category boundaries cross-linguistically?
2. Is this structure present in transformer attention?
3. Can we detect it with topological data analysis?

## Method

The pipeline adapts **Kushnareva et al. (2021)**, *Artificial Text Detection via Examining the
Topology of Attention Maps* (EMNLP). Each mBERT attention head over a sentence is treated as a
weighted graph; persistent homology summarizes the graph's multi-scale structure as a
persistence diagram; Wasserstein and bottleneck distances make diagrams comparable; permutation
tests (K = 10,000) ask whether language labels carry topological signal.

- **Terms:** the Berlin & Kay (1969) basic color terms — 11 for English, 11 for Castilian
  Spanish (Lillo et al. 2018), 12 for Russian (Paramei 2005; Winawer et al. 2007).
  Term lists are literature-grounded and documented in `canon-terms/`.
- **Data:** ~200 KWIC sentences per term per language from the Leipzig Corpora Collection
  1M news slices (2019, 2020, 2023) — 6,628 attention signatures across 34 (language, term)
  cells.
- **Model:** `bert-base-multilingual-cased`, so all three languages share one representational
  space.
- **Second track:** a Draganov & Skiena (2024)-style pipeline over point clouds of
  contextualized mBERT embeddings, one persistence diagram per (language, term) cell
  (`draganov_replication/`).
- **Baselines:** hierarchical clustering, static-embedding persistent homology, and
  MUSE-aligned cross-lingual distances on fastText vectors (`baselines/`).

## Findings, briefly

- **The language signal is statistically unambiguous but small.** The per-color permutation
  test gives z = 56.8, p < 10⁻⁴, and 100 of 144 attention heads are individually significant
  at BH q = 0.05 — yet within- vs cross-language mean diagram distances differ by only
  1–9% depending on metric and homology dimension. Two diagrams of the same color in the same
  language are nearly as far apart as two diagrams from different languages.
- **The predicted синий/голубой separation does not exceed noise** (permutation p ≈ 0.27
  within-Russian). In the embedding-cloud track the two Russian blues are in fact the *most
  similar* pair among {en/blue, es/azul, ru/синий, ru/голубой}.
- **Translation triples cohere asymmetrically:** {blue, azul, синий} is topologically closer
  than chance (p = 0.008), while {blue, azul, голубой} is not (p = 0.68) — consistent with
  evidence that English *blue* is roughly coextensive with синий.
- Aggregate feature distances recover the expected language ordering:
  en–es < ru–es < en–ru.

Summary tables and figures live in `results/`; the full analysis is in the accompanying paper
(*Topology of Color: Cross-Linguistic Category Structure in mBERT Attention*).

## Repository layout

```
ph-project/
├── reference/             Kushnareva et al. (2021) originals — frozen, never edited
├── replication/           notebook-faithful replication of their pipeline + adapted helpers
├── phase1_kwic/           KWIC extraction package (Leipzig news → per-term sentence sets)
├── canon-terms/           literature-grounded color/emotion/kinship term lists (en/ru/es)
├── baselines/             fastText baselines: clustering, static PH, MUSE-aligned distances
├── notebooks/             main analysis notebooks (phases 1–3)
├── draganov_replication/  contextual-embedding point-cloud topology track
├── scripts/               CLI runners and figure scripts
├── results/               summary CSVs and final figures (intermediates gitignored)
├── tests/                 pytest suite for all pipeline stages
└── docs/                  long-form LaTeX overview of the project (build with make pdf)
```

Raw corpora, attention tensors, barcodes, and distance matrices (~45 GB) are gitignored;
everything under `data/` is regenerable from the scripts and notebooks below.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
pytest tests/
```

Pipeline order:

1. `scripts/download_leipzig.sh`, `scripts/download_fasttext.sh` — corpora and vectors
2. `scripts/extract_kwic.py` — KWIC extraction (validate with
   `notebooks/phase1_kwic_validation.ipynb`)
3. `notebooks/baseline_*.ipynb` — fastText baselines
4. `notebooks/mbert_attention_thresholds.ipynb`, `notebooks/mbert_attention_ripser.ipynb` —
   attention extraction and topological features (GPU; several hours)
5. `scripts/compute_diagram_distances.py` — pairwise persistence-diagram distances
   (~17 h overnight on 24 cores), then `notebooks/phase3_diagram_distances.ipynb` and
   `notebooks/phase3_comparison.ipynb` for the statistics
6. `draganov_replication/notebooks/draganov_color_per_term.ipynb` — the embedding-cloud track

Hardware used: one RTX 5070 Ti (16 GB), 64 GB RAM.

## Scope note

The project was designed for three semantic domains (color; emotion via Wierzbicka's NSM;
kinship via Murdock 1949) and descoped to color in May 2026: news-genre corpora attest color
vocabulary well but leave 50–79% of emotion and kinship terms under target, especially in
Russian. The emotion and kinship canon lists remain in `canon-terms/` for follow-up work.
One documented casualty: ru *фиолетовый* reached only n = 104 sentences and is kept with a
caveat (see `canon-terms/ru/color.yaml`).

## References

- Kushnareva et al. (2021). *Artificial Text Detection via Examining the Topology of Attention
  Maps.* EMNLP. — original code preserved under `reference/`, see
  `reference/KUSHNAREVA_README.md` for license and provenance
- Berlin & Kay (1969). *Basic Color Terms: Their Universality and Evolution.*
- Winawer et al. (2007). *Russian blues reveal effects of language on color discrimination.* PNAS.
- Paramei (2005). *Singing the Russian blues.* Cross-Cultural Research.
- Lillo et al. (2018). *Basic color terms in three dialects of the Spanish language.* Frontiers in Psychology.
- Draganov & Skiena (2024). *The Shape of Word Embeddings.* Findings of EMNLP.
- Carlsson (2009). *Topology and data.* Bulletin of the AMS.

## AI use

Code was developed with Claude Code (Claude Opus 4.7) inside a task-decomposed,
human-in-the-loop workflow; all generated code was reviewed through GitHub pull requests
before merging. See the paper's AI Use Statement for details.
