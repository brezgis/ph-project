"""
phase1_kwic/extract.py
----------------------
KWIC extraction pipeline for ph-project Phase 1.

Implements the corpus iteration → match → dedup → sample → emit pipeline
that produces the per-(lang, domain) CSVs described in data/kwic/SCHEMA.md.

Public API
----------
extract_kwic(lang, domain, corpus_path, corpus_source_id, ...)
    Stream the Leipzig sentences file, match canon terms, dedup, sample,
    and return (df, report) conforming to SCHEMA.md.

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
from typing import TYPE_CHECKING

import pandas as pd

from phase1_kwic.canon import Term, load_canon, _lemmatize_ws_token
from phase1_kwic.matchers import HEAD_POSITION, _SPACY_MODEL, get_matcher

if TYPE_CHECKING:
    from phase1_kwic.matchers import Matcher


# ---------------------------------------------------------------------------
# Pinned corpus IDs — one per language
# ---------------------------------------------------------------------------

CORPUS_SOURCE_ID: dict[str, str] = {
    "en": "eng_news_2020_1M",
    "ru": "rus_news_2020_1M",
    "es": "spa_news_2020_1M",
}

# Default data root for CLI use
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "leipzig"


def default_corpus_path(lang: str) -> pathlib.Path:
    """Return the default corpus path for *lang* per SCHEMA.md layout."""
    corpus_id = CORPUS_SOURCE_ID[lang]
    return _DEFAULT_CORPUS_DIR / lang / f"{corpus_id}-sentences.txt"


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
    corpus_path: pathlib.Path,
    corpus_source_id: str,
    n_samples: int = 200,
    seed: int = 0,
    window: int = 10,
    min_post_target_tokens: int = 5,
    *,
    _matcher_override: "Matcher | None" = None,
    _terms_override: "list[Term] | None" = None,
) -> tuple[pd.DataFrame, dict]:
    """Stream the Leipzig sentences file, match canon terms, dedup, sample, emit.

    Parameters
    ----------
    lang : str
        BCP-47-style language code (``"en"``, ``"ru"``, or ``"es"``).
    domain : str
        Semantic domain (``"color"``, ``"emotion"``, or ``"kinship"``).
    corpus_path : pathlib.Path
        Path to the Leipzig sentences TSV file (``idx<TAB>sentence``, UTF-8).
    corpus_source_id : str
        Pinned corpus ID string, e.g. ``"eng_news_2020_1M"``.
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
    # 3. Per-term hit accumulator: term_idx → list of (kwic_str, head_ws_idx)
    # ------------------------------------------------------------------
    # hits[term_idx] = list of (kwic_str, target_idx_in_kwic)
    hits: dict[int, list[tuple[str, int]]] = {i: [] for i in range(len(terms))}
    # n_corpus_hits[term_idx] — count BEFORE dedup
    n_corpus_hits: dict[int, int] = {i: 0 for i in range(len(terms))}

    # ------------------------------------------------------------------
    # 4. Stream the corpus file
    # ------------------------------------------------------------------
    corpus_total_sentences = 0

    with corpus_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            raw_line = raw_line.rstrip("\n")
            if not raw_line.strip():
                continue  # skip empty lines (mirrors prepare_csv.py:36-47)

            # Leipzig TSV: idx<TAB>sentence
            parts = raw_line.split("\t", 1)
            if len(parts) != 2:
                continue
            sentence = parts[1]

            # Apply the same empty/whitespace-only filter from prepare_csv.py:36-47
            if not sentence.strip():
                continue

            corpus_total_sentences += 1

            ws_tokens = sentence.split()

            # Lemmatize each whitespace token
            ws_lemmas: list[str] = [
                _lemmatize_ws_token(tok, matcher) for tok in ws_tokens
            ]

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
                    hits[term_idx].append((kwic_str, target_idx_in_kwic))

    # ------------------------------------------------------------------
    # 5. Per-term: dedup → sample → collect
    # ------------------------------------------------------------------
    rows: list[tuple[str, str, str, int, str]] = []  # (term, labels, sentence, target_idx, corpus_source)

    term_report_entries: list[dict] = []

    for term_idx, term in enumerate(terms):
        term_hits = hits[term_idx]
        n_raw = n_corpus_hits[term_idx]

        # Dedup on kwic_str (preserve first occurrence to keep stable order)
        seen_kwic: set[str] = set()
        deduped: list[tuple[str, int]] = []
        for kwic_str, tidx in term_hits:
            if kwic_str not in seen_kwic:
                seen_kwic.add(kwic_str)
                deduped.append((kwic_str, tidx))

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

        # Collect rows
        for kwic_str, tidx in sampled:
            rows.append((term.surface, term.surface, kwic_str, tidx, corpus_source_id))

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
        "corpus_source": corpus_source_id,
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
