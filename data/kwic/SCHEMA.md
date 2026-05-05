# KWIC output schema

This document specifies the on-disk format for Phase 1 (`ph-project-5f9`)
KWIC extraction outputs. Phase 3 (`ph-project-mwk`) consumes these files.
Implementations of `phase1_kwic` MUST conform to this schema; the
threshold notebook in Phase 3 (`notebooks/mbert_attention_thresholds.ipynb`)
expects exactly these columns and paths.

## Layout

```
data/
└── kwic/
    ├── en/
    │   ├── color.csv          # one row per (term, KWIC sentence)
    │   ├── color.report.json  # per-term hit counts + provenance
    │   ├── emotion.csv
    │   ├── emotion.report.json
    │   ├── kinship.csv
    │   └── kinship.report.json
    ├── ru/
    │   └── ... (same)
    └── es/
        └── ... (same)
```

**Granularity rule:** one CSV per `(language, domain)` pair. Nine CSVs
total. Per-term files would multiply by ~20 with no analytical benefit;
a single master file would force Phase 3 to re-filter on every cell.

The CSVs and JSON sidecars are gitignored (see `.gitignore`); only
this `SCHEMA.md` is tracked. The data files regenerate from the Leipzig
corpora and the canon-term YAMLs and are not part of source.

## CSV schema

Each `<lang>/<domain>.csv` is UTF-8, comma-separated, with a header row.
Columns are fixed:

| column          | type   | description                                                                  |
|-----------------|--------|------------------------------------------------------------------------------|
| `term`          | string | Canon term (surface form from `canon-terms/<lang>/<domain>.yaml`).           |
| `labels`        | string | Substrate-compat alias of `term` (identical content; see below).             |
| `sentence`      | string | The KWIC string, ±10 whitespace tokens around the target. See "Window".      |
| `target_idx`    | int    | Zero-based index of the target token within `sentence` (post-tokenization). |
| `corpus_source` | string | Originating Leipzig corpus ID for this row's sentence; multi-corpus extractions emit multiple distinct values across rows. e.g., `eng_news_2020_1M`. |

### Why these columns and only these

* `term` is the per-row human-readable anchor and the canonical name
  for the canon-term surface form.
* `labels` is **identical in content** to `term` and exists purely
  for substrate compatibility. Kushnareva's threshold notebook reads
  `data['labels']` (it was a binary class label in the original task);
  Phase 3 (`ph-project-mwk.2`) is notebook-faithful per CLAUDE.md and
  preserves that reference. Emitting `labels` here means `mwk.2`'s
  edit list (model swap, max-tokens cap, do_lower_case=False) does
  NOT need to grow a column-rename step. The disk cost is one
  redundant string column per row, which is negligible.
* `sentence` is the input fed to mBERT. It MUST be non-empty and MUST
  NOT roundtrip to NaN through pandas — `sentence` will be passed
  through `re.sub` in the substrate notebook (see
  `replication/scripts/prepare_csv.py:36-47` for the same constraint).
* `target_idx` lets downstream analyses point attention queries at the
  canon token specifically (e.g., "what does mBERT attend to when it
  reads `красный`?"). Optional for cross-linguistic comparison but
  cheap to record and load-bearing for any per-target analysis we add
  later.
* `corpus_source` lets us trace any sentence back to its corpus of
  origin. Each row records the originating Leipzig ID for its sentence;
  multi-corpus extractions produce rows with multiple distinct values.

No `domain` or `language` column: those are encoded in the file path,
and adding them per-row is redundant when the entire file shares them.

## KWIC window

The KWIC unit is a span of **±10 whitespace-tokenized words** around
the target token, taken from the source Leipzig sentence:

* If the target appears at index `i` in the source sentence's
  whitespace-tokenized form, the KWIC string is the join of tokens
  `[max(0, i-10) : i+11]`.
* This gives a window of up to 21 source tokens, which fits inside
  `max_tokens_amount = 32` after mBERT subword expansion plus the two
  special tokens (`[CLS]`, `[SEP]`). The 32-token bound is set by
  `ph-project-mwk.2` (the threshold notebook's tokenizer config) and
  is what we are extracting to fit.
* For multi-word canon terms (e.g., `двоюродный брат`), `i` points to
  the **head word** — Russian/English right-headed, Spanish left-headed
  per `baselines.distances.HEAD_POSITION`. The window is taken around
  the head, and the full multi-word surface form lives inside the
  window (canon multi-word terms are short enough that this holds).
* Sentences shorter than 5 source tokens after the target are dropped
  (filters fragments). The window is right-truncated naturally if the
  source sentence ends before `i+11`.

`target_idx` is the index of the target's head-word position in the
**emitted KWIC string**, not in the source sentence. After window
extraction the index is recomputed. This matters for sentences where
the target is near the start of the source: in those cases
`target_idx == i` (no left padding was trimmed).

## Tokenization for window selection

Window selection uses **whitespace tokenization** of the Leipzig
sentence, **not** mBERT subword tokenization. Reasons:

* Leipzig sentences are clean text already; whitespace tokenization is
  reproducible without a tokenizer dependency.
* Subword counts vary per-language and would make "±10" mean different
  span sizes in en vs. ru. Whitespace tokens are linguistically more
  comparable.
* The mBERT tokenizer is applied later (in the Phase 3 notebook) and
  is responsible for the final 32-token cap.

Matching itself is **lemma-based**, not whitespace-based — see
`phase1_kwic.matchers` (`ph-project-5f9.X` for the matcher subtask). The
flow is:

1. Whitespace-tokenize the Leipzig sentence.
2. Lemmatize each token (pymorphy3 for ru; spaCy `*_core_news_md` for
   en/es).
3. Search for canon-term lemma matches.
4. On match at source-token index `i`, slice the whitespace tokens at
   `[i-10 : i+11]` and join with single spaces.

The emitted `sentence` is therefore a possibly-respaced version of the
original — single spaces only. Punctuation that was attached to a
token in the source stays attached.

## Sample size and under-target handling

* **Target:** 200 KWIC sentences per `(lang, domain, term)`.
* **Floor:** none. Terms with fewer than 200 corpus hits keep all
  hits and are flagged in the sidecar `.report.json`.
* **Sampling:** when corpus hits exceed 200, downsample with a
  per-term subseed derived deterministically from the user-provided
  seed and the term surface form. Implementations MUST NOT use
  Python's builtin `hash()` (it is salted per-process unless
  `PYTHONHASHSEED` is set). Use a stable hash, e.g.::

      digest = hashlib.sha256(f"{seed}|{term}".encode("utf-8")).digest()
      subseed = int.from_bytes(digest[:8], "big")
      rng = random.Random(subseed)
      sample = rng.sample(candidates, 200)

  This guarantees the same term yields the same sample across reruns
  and across machines.
* **Dedup:** exact-string deduplication on the `sentence` field within
  each CSV, applied **before** sampling. Prevents Leipzig duplicates
  from inflating apparent hit counts.

## Sidecar report file

Each CSV is paired with a `<domain>.report.json` capturing extraction
provenance and per-term diagnostics. Schema:

```json
{
  "language": "ru",
  "domain": "kinship",
  "corpus_source": ["rus_news_2019_1M", "rus_news_2020_1M", "rus_news_2023_1M"],
  "corpus_total_sentences": 1000000,
  "extracted_at": "2026-05-03T18:00:00Z",
  "seed": 0,
  "n_samples_target": 200,
  "window": {"left": 10, "right": 10, "unit": "whitespace_tokens"},
  "min_post_target_tokens": 5,
  "matchers": {"ru": "pymorphy3", "en": "spacy:en_core_web_md", "es": "spacy:es_core_news_md"},
  "terms": [
    {
      "term": "мать",
      "n_corpus_hits": 4521,
      "n_kept_after_dedup": 3987,
      "n_emitted": 200,
      "under_target": false
    },
    {
      "term": "свёкор",
      "n_corpus_hits": 14,
      "n_kept_after_dedup": 14,
      "n_emitted": 14,
      "under_target": true
    }
  ]
}
```

The report is **the** place to look for "did we hit the 200-per-term
target?" — the validation notebook in Phase 1 reads it directly.

**Multi-corpus provenance note:** the top-level `corpus_source` field in the
sidecar is a list of the input corpus IDs in the order they were ingested
(e.g., `["rus_news_2019_1M", "rus_news_2020_1M", "rus_news_2023_1M"]`).
The per-row `corpus_source` column in the CSV records the originating corpus
for each individual sentence — so a multi-year extraction will have rows
with different `corpus_source` values, one per year. The two uses are
complementary: the sidecar list documents the full input set; the CSV column
enables per-sentence provenance tracing.

## Reproducibility

A KWIC CSV is a deterministic function of:

1. The pinned Leipzig corpus tarball (sha256 recorded in the download
   script — see `ph-project-5f9.2`).
2. The canon-term YAML (versioned in git under `canon-terms/`).
3. The matcher version (pymorphy3 + spaCy model versions, captured in
   the `matchers` field of the sidecar).
4. The seed (default 0, recorded in the sidecar).

Re-running `scripts/extract_kwic.py --lang <l> --domain <d>` with the
same inputs MUST produce a byte-identical CSV.

## Validation rules (asserted in tests)

The schema-tests subtask (`ph-project-5f9.X` — to be filed) enforces:

* All nine CSVs exist after a full extraction run.
* Each CSV has exactly the five columns named above, in that order:
  `term`, `labels`, `sentence`, `target_idx`, `corpus_source`.
* `labels` equals `term` row-for-row (the substrate-compat alias).
* No row has a NaN or empty-string `sentence`.
* Each `term` value appears in the corresponding canon-term YAML.
* `target_idx` is a non-negative integer less than the number of
  whitespace tokens in `sentence`.
* The `corpus_source` value is one of the pinned IDs for the language
  (`CORPUS_SOURCE_IDS[lang]`).
* The Russian CSVs contain Cyrillic characters; the Spanish CSVs
  contain at least one accented character; English CSVs are predominantly
  ASCII (a soft check — proper nouns may include diacritics).
* Per-term row counts in the CSV match `n_emitted` in the sidecar.
