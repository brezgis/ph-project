# draganov_replication

This directory contains the Draganov-style contextual mBERT topology pipeline for
the cross-linguistic color study.  It is methodologically parallel to `replication/`
(Kushnareva-adapted attention-graph PH) but operates on a different substrate: instead
of per-sentence attention graphs, it builds per-(language, term) clouds of
contextualized mBERT embeddings and applies persistent homology to those clouds.

## What this is

For each of the 34 (language, color-term) cells — 11 English, 12 Russian, 11 Spanish
Berlin & Kay basic color terms — we extract one contextual embedding per corpus
sentence, mean-pool the target color-term WordPiece span to a 768-d vector, and
collect all such vectors into a point cloud.  We then compute Vietoris–Rips
persistence diagrams (H_0 + H_1) for each cloud using the cosine distance metric,
and compute a 34×34 grid of pairwise persistence-diagram distances (bottleneck,
sliced Wasserstein, persistence image, and bars statistics).  The result is a second,
independent line of evidence for cross-linguistic topology differences that
complements the Kushnareva attention-graph permutation tests.

## Relation to `replication/` and `draganov/`

`replication/` is our notebook-faithful adaptation of Kushnareva et al. (2021).  It
asks: what is the topology of the attention graph *inside* each sentence that contains
a color term?  `draganov_replication/` asks a complementary question: what is the
topology of the *cloud of contextualized representations* of each color term across all
sentences?  The two pipelines use different substrates (attention weights vs. final-layer
embeddings), different granularity (per-sentence diagrams vs. one diagram per term), and
different distance frameworks (permutation test on diagram distributions vs. pairwise PD
distances).

`draganov/` is a frozen upstream clone of Draganov & Skiena (2024)'s public code for
their Findings of EMNLP 2024 paper on non-isometry of static word embeddings.  We do
not import from it; we reimplement the relevant helpers here to avoid taking a
dependency on code that targets a different task (static fastText embeddings over 81
Indo-European languages).  Read `draganov/` for reference; never edit it.

## Module layout

- `pointclouds.py` — `build_pointclouds()`: reads the phase3 mBERT embedding parts
  and parquet manifests, pools the target WP span per sentence, and saves one `.npy`
  per (lang, term) cell to `data/phase3/draganov_pointclouds/`.
- `diagrams.py` — `compute_diagrams()`: loads each point cloud, computes the cosine
  distance matrix, runs Vietoris–Rips PH, and saves one `.npz` per cell to
  `data/phase3/draganov_diagrams/`.
- `pd_distances.py` — `compute_pd_distance_grid()`: computes the 34×34 pairwise PD
  distance matrices (4 distance notions × 2 homology dimensions = 8 matrices) and
  saves them to `data/phase3/draganov_pd_distances/`.
- `notebooks/draganov_color_per_term.ipynb` — end-to-end runner and analysis notebook.

## Related epics

- `ph-project-inu` — this epic (Draganov-contextual-mBERT).
- `ph-project-blr` — Kushnareva attention-graph PD pipeline (the primary analysis).
- `ph-project-1iq` — Draganov-style static-fastText sub-baseline.
