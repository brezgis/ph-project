"""
phase1_kwic/extract.py
----------------------
KWIC extraction pipeline for ph-project Phase 1.

Implements the corpus iteration → match → dedup → sample → emit pipeline
that produces the per-(lang, domain) CSVs described in data/kwic/SCHEMA.md.

Public API
----------
extract_kwic(lang, domain, corpus_paths, corpus_source_ids, ...)
    Stream one or more Leipzig sentences files, match canon terms, dedup,
    sample, and return (df, report) conforming to SCHEMA.md.
    Each row's corpus_source records the originating corpus ID.

Tokenization strategy
---------------------
KWIC windows are computed over **whitespace tokens** (per SCHEMA.md).
Lemma matching is also done per-whitespace-token: for each whitespace
token in a sentence, we call matcher.lemmatize(ws_token) and extract
the first alphabetic lemma — exactly as load_canon does for Term.lemmas.

This keeps the index space consistent: whitespace token index i in the
source sentence corresponds directly to lemma index i in our lemma list,
regardless of whether the matcher is PymorphyMatcher (whitespace-based)
or SpacyMatcher (spaCy-tokenizer-based, which splits punctuation).

Multi-word term matching
------------------------
For a multi-word term like "dark red" (lemmas = ("dark", "red")):
  - HEAD_POSITION["en"] = "right" → head lemma is "red" (last token)
  - On a hit for "red" at whitespace index i, verify "dark" appears at
    index i-1 (the modifier, which is 1 before the head for right-headed)
  - More generally for an n-word term, head is at position (n-1) from
    term start for right-headed, or position 0 for left-headed.

The head index within the term lemma list:
  - right-headed: head_offset = len(lemmas) - 1  (last token)
  - left-headed:  head_offset = 0                  (first token)

When the head lemma hits at source index i:
  - term_start_in_source = i - head_offset  (for right-headed)
  - term_start_in_source = i               (for left-headed)

Verification: source_lemmas[term_start_in_source : term_start_in_source + n]
must equal term.lemmas exactly.

The emitted target_idx is the head's position in the *windowed* KWIC string.
"""
from __future__ import annotations

import hashlib
import pathlib
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from phase1_kwic.canon import Term, load_canon
from phase1_kwic.matchers import HEAD_POSITION, _SPACY_MODEL, get_matcher

# Number of sentences to accumulate before calling lemmatize_many.
# Large enough to amortize nlp.pipe() setup cost; small enough to keep
# memory bounded on 1M-sentence corpora (~400 MB peak at 512 sentences
# × ~800 chars each).
_LEMMATIZE_CHUNK_SIZE = 512

if TYPE_CHECKING:
    from phase1_kwic.matchers import Matcher


# ---------------------------------------------------------------------------
# Pinned corpus IDs — multi-year list per language
# ---------------------------------------------------------------------------
# Each language uses 3 years: 2019, 2020, 2023.  Year coverage is symmetric
# across en/ru/es (the only set of years with full overlap on Leipzig), which
# keeps cross-linguistic comparisons methodology-clean.

CORPUS_SOURCE_IDS: dict[str, list[str]] = {
    "en": ["eng_news_2019_1M", "eng_news_2020_1M", "eng_news_2023_1M"],
    "ru": ["rus_news_2019_1M", "rus_news_2020_1M", "rus_news_2023_1M"],
    "es": ["spa_news_2019_1M", "spa_news_2020_1M", "spa_news_2023_1M"],
}

# Default data root for CLI use
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "leipzig"


def default_corpus_paths(lang: str) -> list[pathlib.Path]:
    """Return the default corpus paths for *lang* per SCHEMA.md layout.

    Returns one path per pinned corpus year, in the same order as
    ``CORPUS_SOURCE_IDS[lang]``.
    """
    return [
        _DEFAULT_CORPUS_DIR / lang / f"{corpus_id}-sentences.txt"
        for corpus_id in CORPUS_SOURCE_IDS[lang]
    ]


# ---------------------------------------------------------------------------
# Matcher-version strings for the sidecar report
# ---------------------------------------------------------------------------

def _matcher_versions() -> dict[str, str]:
    """Return the {lang: version_string} map for the sidecar 'matchers' field."""
    versions: dict[str, str] = {}
    try:
        import pymorphy3
        versions["ru"] = f"pymorphy3=={pymorphy3.__version__}"
    except (AttributeError, ImportError):
        versions["ru"] = "pymorphy3"
    try:
        import spacy
        for lang, model in _SPACY_MODEL.items():
            versions[lang] = f"spacy:{model}"
    except (AttributeError, ImportError):
        for lang, model in _SPACY_MODEL.items():
            versions[lang] = f"spacy:{model}"
    return versions


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_kwic(
    lang: str,
    domain: str,
    corpus_paths: "Sequence[pathlib.Path]",
    corpus_source_ids: "Sequence[str]",
    n_samples: int = 200,
    seed: int = 0,
    window: int = 10,
    min_post_target_tokens: int = 5,
    *,
    _matcher_override: "Matcher | None" = None,
    _terms_override: "list[Term] | None" = None,
) -> tuple[pd.DataFrame, dict]:
    """Stream one or more Leipzig sentences files, match canon terms, dedup, sample, emit.

    Parameters
    ----------
    lang : str
        BCP-47-style language code (``"en"``, ``"ru"``, or ``"es"``).
    domain : str
        Semantic domain (``"color"``, ``"emotion"``, or ``"kinship"``).
    corpus_paths : Sequence[pathlib.Path]
        Paths to Leipzig sentences TSV files (``idx<TAB>sentence``, UTF-8).
        Must be non-empty and the same length as ``corpus_source_ids``.
    corpus_source_ids : Sequence[str]
        Pinned corpus ID strings, one per path, e.g. ``["eng_news_2020_1M"]``.
        Each emitted row records the originating corpus ID in
        ``corpus_source``.
    n_samples : int, default 200
        Maximum KWIC hits to emit per canon term.
    seed : int, default 0
        Master seed for deterministic per-term subseeds.
    window : int, default 10
        Left and right window size in whitespace tokens.
    min_post_target_tokens : int, default 5
        Drop sentences with fewer than this many whitespace tokens after the
        target (filters sentence fragments).
    _matcher_override : Matcher or None
        Inject a custom matcher (used in tests to avoid loading real models).
    _terms_override : list[Term] or None
        Inject a custom term list (used in tests to skip YAML loading).

    Returns
    -------
    df : pd.DataFrame
        KWIC hits with columns exactly as in SCHEMA.md:
        ``[term, labels, sentence, target_idx, corpus_source]``.
    report : dict
        Sidecar report conforming to the SCHEMA.md sidecar schema.
    """
    # ------------------------------------------------------------------
    # 0. Validate corpus_paths / corpus_source_ids
    # ------------------------------------------------------------------
    corpus_paths = list(corpus_paths)
    corpus_source_ids = list(corpus_source_ids)
    if len(corpus_paths) == 0:
        raise ValueError("corpus_paths must be non-empty")
    if len(corpus_paths) != len(corpus_source_ids):
        raise ValueError(
            f"corpus_paths and corpus_source_ids must have the same length; "
            f"got {len(corpus_paths)} paths and {len(corpus_source_ids)} IDs"
        )
    # ------------------------------------------------------------------
    # 1. Load canon terms and matcher
    # ------------------------------------------------------------------
    if _matcher_override is not None:
        matcher: "Matcher" = _matcher_override
    else:
        matcher = get_matcher(lang)

    if _terms_override is not None:
        terms: list[Term] = _terms_override
    else:
        terms = load_canon(lang, domain, matcher=matcher)

    # ------------------------------------------------------------------
    # 2. Build head-lemma → [term_index] lookup dict.
    #    Multi-word terms are keyed by their HEAD lemma only; single-word
    #    terms are keyed by their only lemma.
    # ------------------------------------------------------------------
    head_pos = HEAD_POSITION.get(lang, "right")  # default right-headed

    # head_lemma → list of (term_index, head_offset) pairs in `terms`
    head_lemma_to_term_idxs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for term_idx, term in enumerate(terms):
        if not term.lemmas:
            continue
        n = len(term.lemmas)
        head_offset = n - 1 if head_pos == "right" else 0
        head_lemma = term.lemmas[head_offset]
        head_lemma_to_term_idxs[head_lemma].append((term_idx, head_offset))

    # ------------------------------------------------------------------
    # 3. Per-term hit accumulator: term_idx → list of
    #    (kwic_str, target_idx_in_kwic, corpus_source_id)
    # ------------------------------------------------------------------
    hits: dict[int, list[tuple[str, int, str]]] = {i: [] for i in range(len(terms))}
    # n_corpus_hits[term_idx] — count BEFORE dedup
    n_corpus_hits: dict[int, int] = {i: 0 for i in range(len(terms))}

    # ------------------------------------------------------------------
    # 4. Stream corpus files using lemmatize_many for batched processing.
    #
    #    Strategy: iterate over each (corpus_path, corpus_source_id) pair.
    #    For each file, read in chunks of _LEMMATIZE_CHUNK_SIZE valid
    #    sentences, call matcher.lemmatize_many on each chunk, then
    #    process the (sentence, ws_lemmas) pairs.
    #
    #    Per-row provenance: each row records the corpus_source_id of the
    #    file it came from.  The active ID is tracked via a nonlocal binding
    #    (_active_corpus_source_id) updated before each file is processed.
    #
    #    corpus_total_sentences sums across all files automatically.
    # ------------------------------------------------------------------
    corpus_total_sentences = 0
    _active_corpus_source_id: str = corpus_source_ids[0]

    def _iter_valid_sentences(fh):
        """Yield sentence_str for valid non-empty corpus lines."""
        for raw_line in fh:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue
            parts = raw_line.split("\t", 1)
            if len(parts) != 2:
                continue
            sentence = parts[1]
            if not sentence.strip():
                continue
            yield sentence

    def _process_sentence(sentence: str, ws_lemmas: list[str]) -> None:
        """Apply hit-detection logic for one sentence with pre-computed lemmas."""
        nonlocal corpus_total_sentences
        corpus_total_sentences += 1

        ws_tokens = sentence.split()

        # Scan for head lemma hits
        for ws_idx, lemma in enumerate(ws_lemmas):
            if lemma not in head_lemma_to_term_idxs:
                continue

            # One or more terms have this head lemma
            for term_idx, head_offset in head_lemma_to_term_idxs[lemma]:
                term = terms[term_idx]
                n_tok = len(term.lemmas)

                # Compute start position of the full multi-word term
                term_start = ws_idx - head_offset

                # Bounds check
                if term_start < 0 or term_start + n_tok > len(ws_lemmas):
                    continue

                # Verify the full lemma sequence matches
                if ws_lemmas[term_start: term_start + n_tok] != list(term.lemmas):
                    continue

                # Head is at ws_idx — apply min_post_target_tokens filter
                # (tokens after head = total_tokens - ws_idx - 1)
                post_target = len(ws_tokens) - ws_idx - 1
                if post_target < min_post_target_tokens:
                    continue

                # Compute KWIC window (around the HEAD position)
                left = max(0, ws_idx - window)
                right = min(len(ws_tokens), ws_idx + window + 1)
                kwic_tokens = ws_tokens[left:right]
                kwic_str = " ".join(kwic_tokens)

                # target_idx in the windowed KWIC string
                target_idx_in_kwic = ws_idx - left

                n_corpus_hits[term_idx] += 1
                # Record the active corpus source ID for per-row provenance
                hits[term_idx].append((kwic_str, target_idx_in_kwic, _active_corpus_source_id))

    for corpus_path, corpus_source_id in zip(corpus_paths, corpus_source_ids):
        _active_corpus_source_id = corpus_source_id
        with corpus_path.open("r", encoding="utf-8") as fh:
            chunk: list[str] = []

            for sentence in _iter_valid_sentences(fh):
                chunk.append(sentence)
                if len(chunk) >= _LEMMATIZE_CHUNK_SIZE:
                    # Batch-lemmatize the chunk.  lemmatize_many yields one list per
                    # sentence, where each list entry is (surface, lemma) for one
                    # whitespace token.  We extract just the lemma strings.
                    for sentence_i, pairs in zip(chunk, matcher.lemmatize_many(chunk)):
                        ws_lemmas_i: list[str] = [lemma for _, lemma in pairs]
                        _process_sentence(sentence_i, ws_lemmas_i)
                    chunk = []

            # Process the final partial chunk (if any)
            if chunk:
                for sentence_i, pairs in zip(chunk, matcher.lemmatize_many(chunk)):
                    ws_lemmas_i = [lemma for _, lemma in pairs]
                    _process_sentence(sentence_i, ws_lemmas_i)

    # ------------------------------------------------------------------
    # 5. Per-term: dedup → sample → collect
    # ------------------------------------------------------------------
    rows: list[tuple[str, str, str, int, str]] = []  # (term, labels, sentence, target_idx, corpus_source)

    term_report_entries: list[dict] = []

    for term_idx, term in enumerate(terms):
        term_hits = hits[term_idx]
        n_raw = n_corpus_hits[term_idx]

        # Dedup on kwic_str (preserve first occurrence to keep stable order).
        # Each hit is (kwic_str, target_idx, corpus_source_id); dedup on
        # kwic_str alone — cross-year duplicate sentences for the same term
        # are collapsed here automatically.
        seen_kwic: set[str] = set()
        deduped: list[tuple[str, int, str]] = []
        for kwic_str, tidx, hit_corpus_id in term_hits:
            if kwic_str not in seen_kwic:
                seen_kwic.add(kwic_str)
                deduped.append((kwic_str, tidx, hit_corpus_id))

        n_after_dedup = len(deduped)

        # Sample if over target
        if len(deduped) > n_samples:
            digest = hashlib.sha256(
                f"{seed}|{term.surface}".encode("utf-8")
            ).digest()
            subseed = int.from_bytes(digest[:8], "big")
            rng = random.Random(subseed)
            sampled = rng.sample(deduped, n_samples)
        else:
            sampled = deduped

        n_emitted = len(sampled)
        under_target = n_emitted < n_samples

        # Collect rows — per-row corpus_source reflects originating corpus
        for kwic_str, tidx, hit_corpus_id in sampled:
            rows.append((term.surface, term.surface, kwic_str, tidx, hit_corpus_id))

        term_report_entries.append({
            "term": term.surface,
            "n_corpus_hits": n_raw,
            "n_kept_after_dedup": n_after_dedup,
            "n_emitted": n_emitted,
            "under_target": under_target,
        })

    # ------------------------------------------------------------------
    # 6. Build DataFrame with exact column order from SCHEMA.md
    # ------------------------------------------------------------------
    df = pd.DataFrame(
        rows,
        columns=["term", "labels", "sentence", "target_idx", "corpus_source"],
    )
    # Ensure target_idx is integer dtype even for empty DataFrame
    df["target_idx"] = df["target_idx"].astype(int)

    # ------------------------------------------------------------------
    # 7. Build sidecar report
    # ------------------------------------------------------------------
    report: dict = {
        "language": lang,
        "domain": domain,
        "corpus_source": list(corpus_source_ids),  # input list, in order
        "corpus_total_sentences": corpus_total_sentences,
        "extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "n_samples_target": n_samples,
        "window": {"left": window, "right": window, "unit": "whitespace_tokens"},
        "min_post_target_tokens": min_post_target_tokens,
        "matchers": _matcher_versions(),
        "terms": term_report_entries,
    }

    return df, report
