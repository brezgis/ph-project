# Topology of Color

Does a language's color vocabulary leave a trace in the *shape* of a multilingual
transformer's attention?

Russian has two basic terms for blue, *синий* (dark blue) and *голубой* (light blue), where
English and Spanish have one, and Russian speakers are measurably faster at telling the two
shades apart (Winawer et al. 2007). In this project I treat each mBERT attention head over a
sentence as a weighted graph, summarize its shape with persistent homology, and ask whether
those shapes differ by language, and whether the Russian blue split shows up in them.

Final project for COSI 115b (Fundamentals of NLP II) at Brandeis, Spring 2026. The paper is
*Topology of Color: Cross-Linguistic Category Structure in mBERT Attention*
([PDF](https://brezgis.com/assets/papers/brezgis-topology-of-color.pdf)).

## What I found

- **There is a language signal, but it's small.** A permutation test over all attention
  heads is overwhelmingly significant (z = 56.8, p < 10⁻⁴), and 100 of 144 heads are
  significant on their own after Benjamini–Hochberg correction. But within- and
  cross-language distances between persistence diagrams differ by only 1–9%: two diagrams
  of the same color in the same language are nearly as far apart as two from different
  languages.
- **синий and голубой don't separate beyond noise** (within-Russian permutation p ≈ 0.27).
- **The translation triples are asymmetric.** {blue, azul, синий} is topologically closer
  than chance (p = 0.008); {blue, azul, голубой} is not (p = 0.68). That fits the evidence
  that English *blue* lines up with синий.
- Aggregate feature distances order the languages en–es < ru–es < en–ru.

Summary tables and figures are in `results/`.

## Replicating Kushnareva et al. (2021)

The pipeline adapts Kushnareva et al. (2021), *Artificial Text Detection via Examining the
Topology of Attention Maps*. Before adapting it, I reproduced their original experiment
end to end: telling human-written text from GPT-2 output using topological features of
BERT attention. Their code targets 2021-era libraries, so getting it to run took a
compatibility shim for current `transformers`, a patched `ripser++` build for CUDA 12, and
a memory fix for the barcode step.

On 500 human and 500 GPT-2 texts per split, with all three of their feature families
(threshold, barcode and template features), logistic regression reaches **80.2% test
accuracy** (MCC 0.60), with hyperparameters chosen on the validation split. The paper reports
roughly 85–95% with 20,000 texts per class. Everything is in [`replication/`](replication/).

## Layout

```
attention_topology/   attention extraction, persistence-diagram distances, permutation tests
phase1_kwic/          KWIC extraction: Leipzig news corpora → sentences per color term
canon-terms/          basic color term lists for en/es/ru, with sources
baselines/            fastText baselines: clustering, static-embedding PH, aligned distances
notebooks/            the analysis, in pipeline order (see below)
scripts/              downloads, extraction, the long distance computation, figures
replication/          the Kushnareva et al. (2021) reproduction
vendor/tda4atd/       their original code (git submodule, unmodified)
results/              summary CSVs and figures
tests/                pytest suite
```

## Running it

You need [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU for the attention steps.

```bash
git clone --recurse-submodules https://github.com/brezgis/ph-project.git
cd ph-project
uv sync              # add --extra gpu for ripser++ (needs nvcc and a CUDA 12 toolkit)
uv run pytest
```

Then, in order:

1. `bash scripts/download_leipzig.sh && bash scripts/download_fasttext.sh`
2. `uv run python scripts/extract_kwic.py --lang en --domain color` (and `--lang ru`, `--lang es`),
   checked with `notebooks/phase1_kwic_validation.ipynb`
3. `notebooks/baseline_*.ipynb`: the fastText baselines
4. `notebooks/mbert_attention_thresholds.ipynb`, `notebooks/mbert_attention_ripser.ipynb`:
   attention extraction and topological features (GPU, several hours)
5. `uv run python scripts/compute_diagram_distances.py` (about 17 hours on 24 cores; `--smoke`
   runs a quick check first), then `notebooks/phase3_diagram_distances.ipynb` and
   `notebooks/phase3_comparison.ipynb` for the statistics

Raw corpora, attention tensors and barcodes (about 45 GB) aren't committed; everything
under `data/` can be regenerated with the steps above. I ran it on one RTX 5070 Ti (16 GB)
with 64 GB of RAM.

## Notes

- The project started with three domains: color, emotion and kinship. I narrowed it to color
  because news corpora attest too few emotion and kinship terms, especially in Russian. The
  fastText baselines ran before that cut, so they still include all three, and the emotion and
  kinship term lists stay in `canon-terms/` because those notebooks read them.
- ru *фиолетовый* reached only 104 sentences (the target was 200) and is kept with a caveat;
  see `canon-terms/ru/color.yaml`.

KNOWN_ISSUES_PLACEHOLDER

## References

- Kushnareva et al. (2021). Artificial Text Detection via Examining the Topology of Attention
  Maps. *EMNLP*.
- Berlin & Kay (1969). *Basic Color Terms: Their Universality and Evolution.*
- Winawer et al. (2007). Russian blues reveal effects of language on color discrimination.
  *PNAS*.
- Paramei (2005). Singing the Russian blues. *Cross-Cultural Research*.
- Lillo et al. (2018). Basic color terms in three dialects of the Spanish language.
  *Frontiers in Psychology*.
- Carlsson (2009). Topology and data. *Bulletin of the AMS*.

## AI use

I wrote this code with Claude Code, reviewing every change through pull requests. The
paper's AI use statement has the details.

## License

MIT; see [LICENSE](LICENSE). The Kushnareva et al. code in `vendor/tda4atd` is theirs and
isn't covered by it.
