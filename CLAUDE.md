# ph-project

## What this is

A cross-linguistic semantic comparison study using persistent homology on mBERT attention graphs. **Hypothesis:** distributional patterns in language encode culturally-specific attentional structures that show up as measurably different *topology* in attention graphs — not just different distances, but different shapes.

COSI 115a NLP Fundamentals final project. Anna Brezgis, Brandeis. Due May 6, 2026. Compute: north (RTX 5070 Ti, 16GB VRAM, 64GB RAM).

## Methodology

Notebook-faithful adaptation of **Kushnareva et al. (2021) — Artificial Text Detection via Examining the Topology of Attention Maps** (EMNLP). Their original code lives in `reference/`, frozen, never edited. Our adapted notebooks live in `notebooks/`.

Their task: binary classification of human vs. machine text.
Our task: cross-linguistic comparison of semantic domains (color, emotion, kinship) across English, Russian, and Spanish.

Canon term lists are derived from published linguistics/anthropology literature, not assembled informally — see `canon-terms/`.

## Hard rules

- **Never edit `reference/` — no exceptions for AI agents.** The tree is kept for line-by-line comparison and reproducibility verification against the Kushnareva original. Anna (the author) may add comments or docstrings for her own comprehension; agents must treat the entire tree as frozen (no edits of any kind, including comments, formatting, or whitespace). When reviewing diffs in this tree, agents should assume any change was intentional and by Anna, and should not suggest reverting it.
- **Notebook-faithful.** When adapting Kushnareva's notebooks, preserve their structure and function names. Change only the inputs (data, terms, languages) and the final analysis step (cross-linguistic comparison instead of binary classification).
- **No reaching into `~/clawd/projects/tda-project/`.** That's the prior attempt at this project, kept as historical reference only. New work happens here. Do not import code from there.
- **Term lists must cite specific publications.** Each canon term in `canon-terms/` should be traceable to a source. Document deviations from the source explicitly.
- **Use bd (beads) for all task tracking.** No markdown TODO files, no scattered notes. See `AGENTS.md`.

## Project structure

```
ph-project/
├── reference/          Kushnareva originals (frozen)
├── notebooks/          adapted notebooks (notebook-faithful rewrites)
├── canon-terms/        literature-grounded term lists per language per domain
├── data/               corpora, KWIC, attention matrices (gitignored)
├── results/            experiment outputs
├── tests/              tests for adapted code
├── .beads/             beads issue tracking
├── .venv/              Python environment
└── AGENTS.md           beads workflow documentation
```

## Setup

```bash
cd ~/ph-project
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Quality gates

Before committing code changes:
- `pytest tests/` if tests exist for the area you touched
- `python -m py_compile <file>` for syntax check on edited Python files
- For dependency changes: update `requirements.txt` and verify a fresh install works

## Workflow

This project uses the [jdelfino/agent-workflow](https://github.com/jdelfino/agent-workflow) setup with **bd (beads)** for issue tracking. Slash commands:
- `/plan <description>` — create epic with sub-issues
- `/work <id>` — implement an issue
- `/bug <description>` — investigate a bug
- `/fire` — emergency context dump
- `/merge` — process PR queue

Useful bd commands:
- `bd ready` — show unblocked tasks
- `bd show <id>` — show task details
- `bd list --status open` — all open tasks

See `AGENTS.md` for the full beads workflow.

## Key references

- **Kushnareva et al. (2021)** — Artificial Text Detection via Examining the Topology of Attention Maps. EMNLP. Core methodology.
- **Berlin & Kay (1969)** — Basic Color Terms. Anchor for color domain canon list.
- **Wierzbicka NSM** — Natural Semantic Metalanguage primes. Anchor for emotion domain canon list.
- **Murdock (1949)** — Social Structure. Anchor for kinship domain canon list.
- **Draganov & Skiena (2024)** — The Shape of Word Embeddings: TDA on 81 Indo-European languages. arxiv:2404.00500
- **Clark et al. (2019)** — What Does BERT Look At? Attention head specialization.

## About Anna

First-year MS student in Computational Linguistics at Brandeis. Russian/Spanish multilingual, deeper in linguistics than engineering. She chose to rewrite this project from scratch (rather than reuse the prior tda-project attempt at `~/clawd/projects/tda-project/`) specifically so she'd know it intimately. Prefers warmth over efficiency, prose over bullet points, and being walked through the *why* of decisions. She's collaborative, not delegating — guide her through thinking rather than just handing her answers.
