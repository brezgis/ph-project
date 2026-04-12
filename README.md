# ph-project

Cross-linguistic semantic comparison via persistent homology on mBERT attention graphs.

**Course:** COSI 115a NLP Fundamentals, Brandeis University
**Researcher:** Anna Brezgis
**Due:** May 6, 2026

## Hypothesis

Distributional patterns in language encode culturally-specific attentional structures. These should produce measurably different *topology* in attention graphs across languages — not just different distances, but different shapes.

**Key test.** Russian obligatorily distinguishes синий (dark blue) from голубой (light blue) as separate basic color terms; English and Spanish don't. We predict an extra connected component (H₀ feature) in the Russian color attention graph that isn't present in English or Spanish.

## Method

Notebook-faithful adaptation of [Kushnareva et al. (2021)](https://aclanthology.org/2021.emnlp-main.50/) "Artificial Text Detection via Examining the Topology of Attention Maps." Their original task: binary text classification. Our adapted task: cross-linguistic semantic comparison across three domains (color, emotion, kinship) and three languages (English, Russian, Spanish).

Pipeline:

1. Extract ~200 KWIC sentences per canonical term per language from Leipzig Corpora
2. Baseline: pre-aligned fastText vectors, cosine similarity, clustering
3. Main: mBERT attention matrices → distance matrices → persistent homology → topological features
4. Compare cross-linguistically; test pre-registered predictions

Canon term lists are derived from published literature (Berlin & Kay for color, Wierzbicka NSM for emotion, Murdock for kinship), not assembled informally. See `canon-terms/`.

## Pre-registered predictions

1. **Russian blue split.** Russian color domain shows additional H₀ structure (extra connected component) for blue terms compared to English/Spanish, reflecting the obligatory голубой/синий distinction.
2. **Emotion divergence > color divergence.** Colors have more universal structure (Berlin & Kay hierarchy); emotions are more culturally variable. Expect larger cross-linguistic distances in the emotion domain.
3. **Russian kinship complexity.** Russian kinship shows more topological complexity than English (more categorical distinctions), comparable to or less than Spanish.
4. **Head specialization.** Certain attention heads consistently carry domain-specific topological signatures across languages (building on Clark et al. 2019).

## Setup

```bash
cd ~/ph-project
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Project structure

```
ph-project/
├── reference/          Kushnareva originals (frozen, never edited)
├── notebooks/          adapted notebooks (notebook-faithful rewrites)
├── canon-terms/        literature-grounded term lists per language per domain
├── data/               corpora, KWIC, attention matrices (gitignored)
├── results/            experiment outputs
├── tests/              tests for adapted code
├── .beads/             issue tracking (jdelfino agent-workflow)
└── AGENTS.md           beads workflow documentation
```

## Workflow

This project uses the [jdelfino/agent-workflow](https://github.com/jdelfino/agent-workflow) setup with **bd (beads)** for issue tracking. Slash commands: `/plan`, `/work`, `/bug`, `/fire`. See `AGENTS.md` for the beads workflow and `CLAUDE.md` for project-specific conventions.

## License

This work is for academic coursework. The Kushnareva code in `reference/` is preserved under its original license — see `reference/KUSHNAREVA_README.md`.
