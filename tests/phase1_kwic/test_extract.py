"""
Tests for phase1_kwic.extract — KWIC extraction pipeline.

Runs against a synthetic 50-sentence corpus written to a tmp file.
Uses a StubMatcher to avoid loading spaCy/pymorphy3 for every test.
All tests verify the exact behavior specified in data/kwic/SCHEMA.md.

Design choices
--------------
- StubMatcher lemmatizes each whitespace token as token.lower() — mimics
  PymorphyMatcher behavior (whitespace-aligned, no punctuation split).
- Corpus sentences are constructed to give:
    "red"    — 20 occurrences (over-target for n_samples_test=10)
    "blue"   — 5 occurrences (under-target)
    "purple" — 1 occurrence
  Plus 24 filler sentences with none of those terms.
- Multi-word terms use the same stub via whitespace alignment, making it
  easy to plant known positions.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random

import pandas as pd
import pytest

from phase1_kwic.canon import Term


# ---------------------------------------------------------------------------
# Stub Matcher — whitespace-aligned lemmatizer, no spaCy/pymorphy3 load
# ---------------------------------------------------------------------------

class StubMatcher:
    """Fake Matcher that lowercases each whitespace token.

    Behaves like PymorphyMatcher in token-space alignment:
    one (surface, lemma) pair per whitespace token.

    This keeps tests fast and free of model dependencies.
    """

    def lemmatize(self, sentence: str) -> list[tuple[str, str]]:
        return [(tok, tok.lower()) for tok in sentence.split()]


# ---------------------------------------------------------------------------
# Helpers — canon term constructors and corpus builders
# ---------------------------------------------------------------------------

def _make_term(surface: str, lemmas: tuple[str, ...]) -> Term:
    """Build a Term with minimal fields for test purposes."""
    return Term(
        surface=surface,
        gloss=None,
        source="test",
        notes=None,
        lemmas=lemmas,
    )


def _build_corpus_lines(
    n_red: int = 20,
    n_blue: int = 5,
    n_purple: int = 1,
    total: int = 50,
    extra_tokens: int = 8,  # tokens after the target term in each sentence
) -> list[str]:
    """Build Leipzig-format TSV lines with planted target terms.

    Leipzig format: ``<idx>TAB<sentence>``

    Each sentence containing a target word looks like:
        "word0 word1 ... <TARGET> after0 after1 ... afterN"
    The target is at whitespace position 3 (0-indexed), surrounded by
    enough tokens on each side to pass the min_post_target_tokens filter.
    """
    lines: list[str] = []
    idx = 1

    def _sentence_with_target(target: str, i: int) -> str:
        # 3 prefix tokens + target + extra_tokens suffix tokens
        prefix = f"tok_a_{i} tok_b_{i} tok_c_{i}"
        suffix = " ".join(f"suf_{j}_{i}" for j in range(extra_tokens))
        return f"{prefix} {target} {suffix}"

    for i in range(n_red):
        lines.append(f"{idx}\t{_sentence_with_target('red', i)}")
        idx += 1
    for i in range(n_blue):
        lines.append(f"{idx}\t{_sentence_with_target('blue', i)}")
        idx += 1
    for i in range(n_purple):
        lines.append(f"{idx}\t{_sentence_with_target('purple', i)}")
        idx += 1

    # Filler sentences (no target terms)
    used = n_red + n_blue + n_purple
    for i in range(total - used):
        lines.append(f"{idx}\t filler sentence number {i} without any target word")
        idx += 1

    return lines


@pytest.fixture()
def corpus_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a synthetic 50-sentence Leipzig corpus to a temp file."""
    path = tmp_path / "test-sentences.txt"
    lines = _build_corpus_lines()
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture()
def color_terms() -> list[Term]:
    """A minimal 3-term color canon matching the synthetic corpus."""
    return [
        _make_term("red",    ("red",)),
        _make_term("blue",   ("blue",)),
        _make_term("purple", ("purple",)),
    ]


# ---------------------------------------------------------------------------
# Import target — done here so a missing module fails tests immediately
# ---------------------------------------------------------------------------

from phase1_kwic.extract import extract_kwic  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: run extract_kwic with the stub matcher and fixed terms
# ---------------------------------------------------------------------------

def _run(
    corpus_file: pathlib.Path,
    color_terms: list[Term],
    *,
    n_samples: int = 10,
    seed: int = 0,
    window: int = 10,
    min_post_target_tokens: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Run extract_kwic with a StubMatcher, bypassing real lang/domain loading."""
    return extract_kwic(
        lang="en",
        domain="color",
        corpus_path=corpus_file,
        corpus_source_id="test_corpus_1M",
        n_samples=n_samples,
        seed=seed,
        window=window,
        min_post_target_tokens=min_post_target_tokens,
        _matcher_override=StubMatcher(),
        _terms_override=color_terms,
    )


# ---------------------------------------------------------------------------
# Phase 1 tests (these must FAIL before implementation)
# ---------------------------------------------------------------------------


class TestColumnSchema:
    """CSV columns must be exactly [term, labels, sentence, target_idx, corpus_source]."""

    def test_column_names_and_order(self, corpus_file, color_terms):
        df, _ = _run(corpus_file, color_terms)
        expected = ["term", "labels", "sentence", "target_idx", "corpus_source"]
        assert list(df.columns) == expected, (
            f"Expected columns {expected}, got {list(df.columns)}"
        )

    def test_labels_equals_term_row_for_row(self, corpus_file, color_terms):
        df, _ = _run(corpus_file, color_terms)
        assert (df["labels"] == df["term"]).all(), (
            "labels column must be an identical copy of term column"
        )

    def test_no_nan_or_empty_sentence(self, corpus_file, color_terms):
        df, _ = _run(corpus_file, color_terms)
        assert df["sentence"].notna().all(), "sentence must not contain NaN"
        assert (df["sentence"].str.strip() != "").all(), (
            "sentence must not be empty or whitespace-only"
        )

    def test_corpus_source_matches_arg(self, corpus_file, color_terms):
        df, _ = _run(corpus_file, color_terms)
        assert (df["corpus_source"] == "test_corpus_1M").all()

    def test_target_idx_dtype_int(self, corpus_file, color_terms):
        df, _ = _run(corpus_file, color_terms)
        assert pd.api.types.is_integer_dtype(df["target_idx"]), (
            f"target_idx must be integer dtype, got {df['target_idx'].dtype}"
        )


class TestTargetIdxBounds:
    """target_idx must be a valid index into the whitespace tokens of sentence."""

    def test_target_idx_in_bounds(self, corpus_file, color_terms):
        df, _ = _run(corpus_file, color_terms)
        for _, row in df.iterrows():
            n_tokens = len(row["sentence"].split())
            assert 0 <= row["target_idx"] < n_tokens, (
                f"target_idx={row['target_idx']} out of bounds "
                f"for sentence with {n_tokens} tokens: {row['sentence']!r}"
            )

    def test_sentence_at_target_idx_matches_term(self, corpus_file, color_terms):
        """The whitespace token at target_idx should be (approximately) the term
        surface form — at minimum, the lowercased token should match the term."""
        df, _ = _run(corpus_file, color_terms)
        for _, row in df.iterrows():
            tokens = row["sentence"].split()
            target_token = tokens[row["target_idx"]].lower()
            assert target_token == row["term"].lower(), (
                f"Token at target_idx {row['target_idx']} is {target_token!r}, "
                f"expected {row['term'].lower()!r}. Sentence: {row['sentence']!r}"
            )


class TestWindowSize:
    """KWIC window must not exceed 2*window+1 tokens."""

    def test_window_max_tokens(self, corpus_file, color_terms):
        window = 10
        df, _ = _run(corpus_file, color_terms, window=window)
        max_tokens = 2 * window + 1
        for _, row in df.iterrows():
            n_tokens = len(row["sentence"].split())
            assert n_tokens <= max_tokens, (
                f"Sentence has {n_tokens} tokens, exceeds max {max_tokens}: "
                f"{row['sentence']!r}"
            )

    def test_window_is_single_spaced(self, corpus_file, color_terms):
        """Emitted KWIC string must be a single-space join (no double spaces)."""
        df, _ = _run(corpus_file, color_terms)
        for _, row in df.iterrows():
            assert "  " not in row["sentence"], (
                f"Double space in sentence: {row['sentence']!r}"
            )


class TestDeduplication:
    """Hits must be deduplicated on kwic_str before sampling."""

    def test_no_duplicate_sentences_per_term(self, tmp_path, color_terms):
        """If the corpus has duplicate sentences for a term, only one is emitted."""
        # Build corpus with 10 identical sentences for "red"
        dup_sentence = "tok_a tok_b tok_c red suf0 suf1 suf2 suf3 suf4 suf5"
        lines = [f"{i+1}\t{dup_sentence}" for i in range(10)]
        # Add 5 unique sentences for blue
        for i in range(5):
            lines.append(
                f"{11+i}\ttok_a_{i} tok_b_{i} tok_c_{i} blue suf0 suf1 suf2 suf3 suf4 suf5"
            )
        corpus = tmp_path / "dup-corpus.txt"
        corpus.write_text("\n".join(lines), encoding="utf-8")

        terms = [
            _make_term("red",  ("red",)),
            _make_term("blue", ("blue",)),
        ]
        df, report = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )

        red_rows = df[df["term"] == "red"]
        # 10 identical raw sentences → deduplicated to 1
        assert len(red_rows) == 1, (
            f"Expected 1 unique red sentence after dedup, got {len(red_rows)}"
        )

        # Report reflects the dedup
        red_info = next(t for t in report["terms"] if t["term"] == "red")
        assert red_info["n_corpus_hits"] == 10
        assert red_info["n_kept_after_dedup"] == 1
        assert red_info["n_emitted"] == 1
        assert red_info["under_target"] is True


class TestSampling:
    """Sampling must respect n_samples cap, be deterministic, seed-sensitive."""

    def test_over_target_capped(self, corpus_file, color_terms):
        """Term with >n_samples hits must be capped at n_samples."""
        df, _ = _run(corpus_file, color_terms, n_samples=10)
        red_rows = df[df["term"] == "red"]
        assert len(red_rows) == 10, (
            f"Expected 10 red rows (capped), got {len(red_rows)}"
        )

    def test_under_target_kept_all(self, corpus_file, color_terms):
        """Term with <n_samples hits keeps all and is flagged under_target."""
        df, report = _run(corpus_file, color_terms, n_samples=10)
        blue_rows = df[df["term"] == "blue"]
        assert len(blue_rows) == 5, (
            f"Expected 5 blue rows (all kept), got {len(blue_rows)}"
        )
        blue_info = next(t for t in report["terms"] if t["term"] == "blue")
        assert blue_info["under_target"] is True

    def test_determinism_same_seed(self, corpus_file, color_terms):
        """Same seed → byte-identical row order."""
        df1, _ = _run(corpus_file, color_terms, seed=42, n_samples=10)
        df2, _ = _run(corpus_file, color_terms, seed=42, n_samples=10)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))

    def test_determinism_different_seed_differs(self, corpus_file, color_terms):
        """Different seeds → different samples (probabilistically)."""
        df1, _ = _run(corpus_file, color_terms, seed=0, n_samples=10)
        df2, _ = _run(corpus_file, color_terms, seed=999, n_samples=10)
        red1 = df1[df1["term"] == "red"]["sentence"].tolist()
        red2 = df2[df2["term"] == "red"]["sentence"].tolist()
        # Both have 10 samples from 20 unique sentences — different seeds should differ
        assert red1 != red2, (
            "Expected different samples for different seeds, but they are the same."
        )

    def test_sha256_subseed_used(self, corpus_file, color_terms):
        """Sampling is based on SHA-256 subseed, not plain random.Random(seed).

        Replicates the full SHA-256 → subseed → random.Random → sample chain
        externally against the known synthetic corpus and asserts that the
        emitted KWIC sentences match that external replication exactly.

        Mutation test: if the implementation is changed to use
        ``random.Random(seed)`` instead of the SHA-256-derived subseed, this
        test FAILS because the two RNGs diverge immediately.
        """
        seed = 7
        n_samples = 10
        term_surface = "red"

        df, _ = _run(corpus_file, color_terms, seed=seed, n_samples=n_samples)
        red_rows = df[df["term"] == "red"]
        assert len(red_rows) == n_samples

        # Reconstruct the exact deduped candidate list the pipeline sees.
        # The corpus has 20 unique "red" sentences; each KWIC string is:
        #   "tok_a_i tok_b_i tok_c_i red suf_0_i ... suf_7_i"
        # with window=10 (default in _run), so the window covers the full
        # sentence (max 12 tokens < 21 tokens window).
        # Build them in corpus order (same as pipeline).
        extra_tokens = 8  # matches _build_corpus_lines default
        window = 10       # matches _run default
        deduped_candidates: list[tuple[str, int]] = []
        for i in range(20):  # 20 red sentences in corpus order
            prefix = f"tok_a_{i} tok_b_{i} tok_c_{i}"
            suffix = " ".join(f"suf_{j}_{i}" for j in range(extra_tokens))
            sentence = f"{prefix} red {suffix}"
            ws_tokens = sentence.split()
            ws_idx = 3  # "red" is at position 3
            left = max(0, ws_idx - window)
            right = min(len(ws_tokens), ws_idx + window + 1)
            kwic_str = " ".join(ws_tokens[left:right])
            target_idx = ws_idx - left
            deduped_candidates.append((kwic_str, target_idx))

        # Replicate the SHA-256 subseed derivation
        digest = hashlib.sha256(f"{seed}|{term_surface}".encode("utf-8")).digest()
        subseed = int.from_bytes(digest[:8], "big")
        rng = random.Random(subseed)
        expected_sample = rng.sample(deduped_candidates, n_samples)
        expected_sentences = [kwic for kwic, _ in expected_sample]

        actual_sentences = list(red_rows["sentence"])
        assert actual_sentences == expected_sentences, (
            "Emitted KWIC sentences do not match the external SHA-256 replication. "
            "If the impl uses random.Random(seed) instead of the SHA-256-derived "
            "subseed, the samples will diverge."
        )


class TestMinPostTargetFilter:
    """Sentences with fewer than min_post_target_tokens after target are dropped."""

    def test_short_sentences_dropped(self, tmp_path, color_terms):
        """Sentences with only 1 token after the target are filtered out."""
        # Short sentence: target at index 3, only 1 token after (< 5)
        short = "tok_a tok_b tok_c red only_one_suffix"
        # Long sentence: target at index 3, 8 tokens after (>= 5)
        long_s = "tok_a tok_b tok_c red s0 s1 s2 s3 s4 s5 s6 s7"
        lines = [
            f"1\t{short}",
            f"2\t{long_s}",
        ]
        corpus = tmp_path / "short-corpus.txt"
        corpus.write_text("\n".join(lines), encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, report = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )

        # Only the long sentence should survive
        assert len(df) == 1, f"Expected 1 row (long sentence only), got {len(df)}"
        assert "s0 s1 s2" in df.iloc[0]["sentence"]

    def test_short_filter_default_threshold(self, tmp_path, color_terms):
        """Default min_post_target_tokens=5 — exactly 4 tokens after drops it."""
        # 4 tokens after target — below threshold
        just_under = "tok_a tok_b tok_c red a b c d"
        # 5 tokens after target — meets threshold
        just_over = "tok_a tok_b tok_c red a b c d e"
        lines = [
            f"1\t{just_under}",
            f"2\t{just_over}",
        ]
        corpus = tmp_path / "threshold-corpus.txt"
        corpus.write_text("\n".join(lines), encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, _ = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )
        assert len(df) == 1
        tokens = df.iloc[0]["sentence"].split()
        target_pos = df.iloc[0]["target_idx"]
        post = len(tokens) - target_pos - 1
        assert post >= 5, f"Post-target tokens = {post}, expected >= 5"


class TestSidecarReport:
    """Sidecar report must contain all fields specified in SCHEMA.md."""

    def test_report_top_level_fields(self, corpus_file, color_terms):
        _, report = _run(corpus_file, color_terms)
        required = {
            "language", "domain", "corpus_source", "corpus_total_sentences",
            "extracted_at", "seed", "n_samples_target",
            "window", "min_post_target_tokens", "matchers", "terms",
        }
        missing = required - set(report.keys())
        assert not missing, f"Report missing fields: {missing}"

    def test_report_window_subfields(self, corpus_file, color_terms):
        _, report = _run(corpus_file, color_terms)
        w = report["window"]
        assert w["left"] == 10
        assert w["right"] == 10
        assert w["unit"] == "whitespace_tokens"

    def test_report_term_subfields(self, corpus_file, color_terms):
        _, report = _run(corpus_file, color_terms)
        for term_info in report["terms"]:
            for field in ("term", "n_corpus_hits", "n_kept_after_dedup",
                          "n_emitted", "under_target"):
                assert field in term_info, (
                    f"Term report missing field {field!r}: {term_info}"
                )

    def test_report_n_emitted_matches_dataframe(self, corpus_file, color_terms):
        """n_emitted in sidecar must match actual row counts in the DataFrame."""
        df, report = _run(corpus_file, color_terms, n_samples=10)
        for term_info in report["terms"]:
            t = term_info["term"]
            actual = len(df[df["term"] == t])
            assert term_info["n_emitted"] == actual, (
                f"Term {t!r}: report says n_emitted={term_info['n_emitted']}, "
                f"but DataFrame has {actual} rows"
            )

    def test_report_under_target_flag(self, corpus_file, color_terms):
        """under_target is True for blue (5 hits < 10 target), False for red."""
        _, report = _run(corpus_file, color_terms, n_samples=10)
        term_map = {t["term"]: t for t in report["terms"]}
        assert term_map["red"]["under_target"] is False
        assert term_map["blue"]["under_target"] is True
        assert term_map["purple"]["under_target"] is True

    def test_report_corpus_total_sentences(self, corpus_file, color_terms):
        """corpus_total_sentences must count all lines in the corpus file."""
        _, report = _run(corpus_file, color_terms)
        assert report["corpus_total_sentences"] == 50

    def test_report_extracted_at_is_iso8601(self, corpus_file, color_terms):
        import re
        _, report = _run(corpus_file, color_terms)
        ts = report["extracted_at"]
        # Accepts both 'Z' and '+00:00' suffix
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$"
        assert re.match(pattern, ts), (
            f"extracted_at={ts!r} does not look like ISO 8601 UTC"
        )

    def test_report_seed_recorded(self, corpus_file, color_terms):
        _, report = _run(corpus_file, color_terms, seed=42)
        assert report["seed"] == 42


class TestMultiwordTerms:
    """Multi-word terms key off the HEAD lemma (right-headed for en/ru, left for es)."""

    def test_multiword_en_right_headed(self, tmp_path):
        """'dark red' is right-headed → head is 'red'; match on 'red' token."""
        # Sentence: "tok0 tok1 tok2 dark red suf0 suf1 suf2 suf3 suf4 suf5"
        # 'dark' at ws-index 3, 'red' (head) at ws-index 4
        sentences = [
            "1\ttok0 tok1 tok2 dark red suf0 suf1 suf2 suf3 suf4 suf5",
        ]
        corpus = tmp_path / "mw-corpus.txt"
        corpus.write_text("\n".join(sentences), encoding="utf-8")

        # multi-word term: lemmas = ("dark", "red"), head = "red" (right)
        terms = [_make_term("dark red", ("dark", "red"))]
        df, report = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )

        assert len(df) == 1, f"Expected 1 row for 'dark red', got {len(df)}"
        # target_idx should point to 'red' (or within the window — recomputed)
        row = df.iloc[0]
        target_token = row["sentence"].split()[row["target_idx"]].lower()
        assert target_token == "red", (
            f"target_idx {row['target_idx']} points to {target_token!r}, expected 'red'"
        )
        # term should be the surface form
        assert row["term"] == "dark red"

    def test_multiword_es_left_headed(self, tmp_path):
        """Spanish multi-word term 'rojo oscuro' is left-headed → head is 'rojo'."""
        # 'rojo' at ws-index 3, 'oscuro' at ws-index 4
        sentences = [
            "1\ttok0 tok1 tok2 rojo oscuro suf0 suf1 suf2 suf3 suf4 suf5",
        ]
        corpus = tmp_path / "es-corpus.txt"
        corpus.write_text("\n".join(sentences), encoding="utf-8")

        terms = [_make_term("rojo oscuro", ("rojo", "oscuro"))]
        df, report = extract_kwic(
            lang="es",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )

        assert len(df) == 1, f"Expected 1 row for 'rojo oscuro', got {len(df)}"
        row = df.iloc[0]
        target_token = row["sentence"].split()[row["target_idx"]].lower()
        assert target_token == "rojo", (
            f"target_idx {row['target_idx']} points to {target_token!r}, expected 'rojo'"
        )

    def test_multiword_partial_match_rejected(self, tmp_path):
        """Only the head lemma appears but not the full sequence — must not match."""
        # "tok0 tok1 tok2 red suf0 suf1 suf2 suf3 suf4 suf5"
        # 'red' at index 3 but 'dark' is not before it
        sentences = [
            "1\ttok0 tok1 tok2 red suf0 suf1 suf2 suf3 suf4 suf5",
        ]
        corpus = tmp_path / "partial-corpus.txt"
        corpus.write_text("\n".join(sentences), encoding="utf-8")

        # Multi-word: "dark red" — head 'red' found but 'dark' is not at index 2
        terms = [_make_term("dark red", ("dark", "red"))]
        df, _ = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )
        assert len(df) == 0, (
            f"Expected 0 rows (partial multi-word match), got {len(df)}"
        )


class TestStableRowOrder:
    """Rows are emitted in deterministic order: canon term order, then per-term sample order."""

    def test_term_order_follows_canon_order(self, corpus_file, color_terms):
        """Terms appear in the DataFrame in canon term list order."""
        df, _ = _run(corpus_file, color_terms, n_samples=5)
        seen_terms: list[str] = []
        for t in df["term"]:
            if not seen_terms or t != seen_terms[-1]:
                seen_terms.append(t)
        # Extract the unique ordered terms
        unique_ordered = list(dict.fromkeys(seen_terms))
        expected_order = [t.surface for t in color_terms if t.surface in unique_ordered]
        assert unique_ordered == expected_order, (
            f"Terms not in canon order. Got: {unique_ordered}, expected: {expected_order}"
        )


class TestEmptyCorpus:
    """Edge case: corpus with no matching sentences returns empty DataFrame."""

    def test_empty_result_for_no_matches(self, tmp_path):
        lines = [
            "1\tthis sentence has nothing relevant",
            "2\tanother boring sentence",
        ]
        corpus = tmp_path / "no-match.txt"
        corpus.write_text("\n".join(lines), encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, report = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )
        assert len(df) == 0
        assert list(df.columns) == ["term", "labels", "sentence", "target_idx", "corpus_source"]
        red_info = next(t for t in report["terms"] if t["term"] == "red")
        assert red_info["n_corpus_hits"] == 0
        assert red_info["n_emitted"] == 0
        assert red_info["under_target"] is True

    def test_empty_corpus_file(self, tmp_path):
        corpus = tmp_path / "empty.txt"
        corpus.write_text("", encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, report = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )
        assert len(df) == 0
        assert report["corpus_total_sentences"] == 0


class TestPunctuationHandling:
    """Verify that tokens with attached punctuation are still matched correctly.

    This exercises the _lemmatize_ws_token path where a whitespace token like
    'red,' must still resolve to the lemma 'red' and trigger a match.
    The StubMatcher lowercases the whole token ('red,' → 'red,') which would
    miss — so we use a PunctuationStripMatcher that strips trailing punctuation
    to simulate real matcher behavior.
    """

    def test_token_with_trailing_comma_matched(self, tmp_path):
        """A matcher that strips trailing punctuation lets 'red,' match 'red'."""

        class PunctuationStripMatcher:
            """Strips trailing non-alpha chars before lowercasing, like real matchers."""
            def lemmatize(self, sentence: str) -> list[tuple[str, str]]:
                result = []
                for tok in sentence.split():
                    stripped = tok.rstrip(".,!?;:'\"")
                    lemma = stripped.lower() if stripped else tok.lower()
                    result.append((tok, lemma))
                return result

        # 'red,' with the trailing comma — matcher strips comma → lemma 'red'
        sentences = [
            "1\ttok0 tok1 tok2 red, suf0 suf1 suf2 suf3 suf4 suf5",
        ]
        corpus = tmp_path / "punct-corpus.txt"
        corpus.write_text("\n".join(sentences), encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, _ = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=PunctuationStripMatcher(),
            _terms_override=terms,
        )
        assert len(df) == 1, (
            f"Expected 1 row for 'red,' with punctuation-stripping matcher, got {len(df)}"
        )
        # target should point at 'red,' token in the KWIC string
        row = df.iloc[0]
        tokens = row["sentence"].split()
        assert tokens[row["target_idx"]].lower().rstrip(".,") == "red"


class TestCorpusSourceConstants:
    """Sanity-check the CORPUS_SOURCE_ID dict and default_corpus_path helper."""

    def test_corpus_source_id_has_all_langs(self):
        from phase1_kwic.extract import CORPUS_SOURCE_ID
        from phase1_kwic import SUPPORTED_LANGUAGES
        assert set(CORPUS_SOURCE_ID.keys()) == set(SUPPORTED_LANGUAGES)

    def test_corpus_source_id_values(self):
        from phase1_kwic.extract import CORPUS_SOURCE_ID
        assert CORPUS_SOURCE_ID["en"] == "eng_news_2020_1M"
        assert CORPUS_SOURCE_ID["ru"] == "rus_news_2020_1M"
        assert CORPUS_SOURCE_ID["es"] == "spa_news_2020_1M"

    def test_default_corpus_path_format(self):
        from phase1_kwic.extract import default_corpus_path
        p = default_corpus_path("en")
        assert p.name == "eng_news_2020_1M-sentences.txt"
        assert p.parent.name == "en"


# ---------------------------------------------------------------------------
# New tests: items 2, 3, 5, 6, 7 from review findings
# ---------------------------------------------------------------------------


class TestPerTermSubseedIndependence:
    """Each term gets its own subseed derived from its surface form.

    Two different terms with the same global seed must produce different
    samples, proving the term surface is mixed into the subseed rather than
    sharing global RNG state.
    """

    def test_different_terms_get_different_samples(self, tmp_path):
        """Two over-target terms with the same seed produce different sample sets."""
        # Build a corpus where both "red" and "blue" appear 20 times each.
        lines: list[str] = []
        for i in range(20):
            lines.append(
                f"{i+1}\ttok_a_{i} tok_b_{i} tok_c_{i} red suf_0_{i} suf_1_{i} suf_2_{i} suf_3_{i} suf_4_{i} suf_5_{i}"
            )
        for i in range(20):
            lines.append(
                f"{21+i}\ttok_a_{i} tok_b_{i} tok_c_{i} blue suf_0_{i} suf_1_{i} suf_2_{i} suf_3_{i} suf_4_{i} suf_5_{i}"
            )
        corpus = tmp_path / "two-term-corpus.txt"
        corpus.write_text("\n".join(lines), encoding="utf-8")

        terms = [
            _make_term("red",  ("red",)),
            _make_term("blue", ("blue",)),
        ]
        n_samples = 10
        df, _ = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=n_samples,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )

        red_rows = df[df["term"] == "red"]
        blue_rows = df[df["term"] == "blue"]
        assert len(red_rows) == n_samples
        assert len(blue_rows) == n_samples

        # The kwic_str for "red" and "blue" sentences have the same structure
        # (same prefix/suffix tokens), differing only in the target word.
        # If both terms used the same global RNG, sampling index order would
        # be identical, and stripping the target word would yield the same
        # context prefixes/suffixes. With independent subseeds they must differ.
        red_indices = [
            int(s.split()[0].split("_")[-1])  # extract i from "tok_a_i"
            for s in red_rows["sentence"]
        ]
        blue_indices = [
            int(s.split()[0].split("_")[-1])
            for s in blue_rows["sentence"]
        ]
        assert red_indices != blue_indices, (
            "Red and blue samples have the same index order, suggesting they "
            "share a global RNG rather than per-term subseeds."
        )


class TestLeftWindowClip:
    """target_idx == 0 is valid when target is at the start of a sentence.

    The synthetic corpus always plants the target at ws-index 3, so the
    left-clip path (max(0, ws_idx - window)) is never exercised non-trivially
    in other tests.  This test constructs a sentence where the target IS the
    first token (ws_idx = 0) and asserts target_idx == 0 in the emitted row.
    """

    def test_target_at_index_zero(self, tmp_path):
        """A sentence starting with the target word emits target_idx == 0."""
        # "red" at ws-index 0; 8 tokens follow (satisfies min_post_target=5)
        sentences = [
            "1\tred suf0 suf1 suf2 suf3 suf4 suf5 suf6 suf7",
        ]
        corpus = tmp_path / "leftclip-corpus.txt"
        corpus.write_text("\n".join(sentences), encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, _ = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )

        assert len(df) == 1, f"Expected 1 row, got {len(df)}"
        row = df.iloc[0]
        assert row["target_idx"] == 0, (
            f"Expected target_idx=0 for sentence-initial target, got {row['target_idx']}"
        )
        # And the token at index 0 must be the target
        assert row["sentence"].split()[0].lower() == "red"


class TestEmptyCorpusColumnSchema:
    """Empty corpus file must return a DataFrame with the correct column schema."""

    def test_empty_corpus_file_has_correct_columns(self, tmp_path):
        """Empty file returns empty DataFrame with full column schema."""
        corpus = tmp_path / "empty.txt"
        corpus.write_text("", encoding="utf-8")

        terms = [_make_term("red", ("red",))]
        df, _ = extract_kwic(
            lang="en",
            domain="color",
            corpus_path=corpus,
            corpus_source_id="test_corpus",
            n_samples=200,
            seed=0,
            window=10,
            min_post_target_tokens=5,
            _matcher_override=StubMatcher(),
            _terms_override=terms,
        )
        assert len(df) == 0
        expected_cols = ["term", "labels", "sentence", "target_idx", "corpus_source"]
        assert list(df.columns) == expected_cols, (
            f"Empty DataFrame has wrong columns: {list(df.columns)}"
        )
        assert pd.api.types.is_integer_dtype(df["target_idx"]), (
            f"target_idx must be integer dtype on empty DataFrame, got {df['target_idx'].dtype}"
        )


class TestReportWindowNonDefault:
    """Report window.left/right reflect the actual window argument, not just the default."""

    def test_report_window_reflects_non_default_value(self, corpus_file, color_terms):
        """Calling with window=5 must emit window.left=5 and window.right=5."""
        _, report = _run(corpus_file, color_terms, window=5)
        w = report["window"]
        assert w["left"] == 5, f"Expected window.left=5, got {w['left']}"
        assert w["right"] == 5, f"Expected window.right=5, got {w['right']}"
        assert w["unit"] == "whitespace_tokens"


class TestReportMatchersField:
    """report['matchers'] must be a non-empty dict."""

    def test_matchers_is_dict_with_entries(self, corpus_file, color_terms):
        """report['matchers'] is a dict and has at least one non-empty entry."""
        _, report = _run(corpus_file, color_terms)
        matchers = report["matchers"]
        assert isinstance(matchers, dict), (
            f"report['matchers'] must be a dict, got {type(matchers)}"
        )
        assert len(matchers) >= 1, "report['matchers'] must have at least one entry"
        # All values must be non-empty strings
        for lang, ver in matchers.items():
            assert isinstance(ver, str) and ver, (
                f"matchers[{lang!r}] is empty or not a string: {ver!r}"
            )


# ---------------------------------------------------------------------------
# Item 4: CLI happy-path test for scripts/extract_kwic.main()
# ---------------------------------------------------------------------------


class TestExtractKwicCLI:
    """Happy-path test for scripts/extract_kwic.main().

    Uses _matcher_override and _terms_override via a thin wrapper so the
    test does not load spaCy or pymorphy3.  The CLI is invoked via the
    main() function directly (not subprocess), using _matcher_override and
    _terms_override injected via monkeypatch on extract_kwic.
    """

    def test_main_writes_csv_and_report(self, tmp_path, monkeypatch):
        """main() writes <lang>/<domain>.csv and <lang>/<domain>.report.json."""
        import sys

        # Build a tiny synthetic corpus in tmp_path
        lines = [
            f"{i+1}\ttok_a tok_b tok_c red suf_0 suf_1 suf_2 suf_3 suf_4 suf_5"
            for i in range(5)
        ]
        corpus_path = tmp_path / "test-sentences.txt"
        corpus_path.write_text("\n".join(lines), encoding="utf-8")

        output_dir = tmp_path / "kwic_out"

        # Monkeypatch extract_kwic inside scripts.extract_kwic so that the
        # CLI's call goes through our stub.  We wrap the real function to
        # inject _matcher_override and _terms_override without changing the
        # CLI code.
        import scripts.extract_kwic as cli_module
        import phase1_kwic.extract as ext_module

        original_extract = ext_module.extract_kwic
        terms_stub = [_make_term("red", ("red",))]
        matcher_stub = StubMatcher()

        def patched_extract_kwic(*args, **kwargs):
            kwargs["_matcher_override"] = matcher_stub
            kwargs["_terms_override"] = terms_stub
            return original_extract(*args, **kwargs)

        monkeypatch.setattr(cli_module, "extract_kwic", patched_extract_kwic)

        cli_module.main([
            "--lang", "en",
            "--domain", "color",
            "--corpus-path", str(corpus_path),
            "--corpus-source-id", "test_corpus_1M",
            "--n-samples", "10",
            "--seed", "0",
            "--output-dir", str(output_dir),
        ])

        # Assert output files exist at schema-mandated paths
        csv_path = output_dir / "en" / "color.csv"
        report_path = output_dir / "en" / "color.report.json"
        assert csv_path.exists(), f"CSV not written: {csv_path}"
        assert report_path.exists(), f"Report not written: {report_path}"

        # Assert CSV has the correct column order
        import csv
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        expected_cols = ["term", "labels", "sentence", "target_idx", "corpus_source"]
        assert header == expected_cols, f"CSV header mismatch: {header}"

        # Assert report is valid JSON with required top-level fields
        import json as _json
        with report_path.open(encoding="utf-8") as f:
            report = _json.load(f)
        for field in ("language", "domain", "corpus_source", "terms", "matchers"):
            assert field in report, f"Report missing field: {field!r}"
