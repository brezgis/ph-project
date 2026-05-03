"""
phase1_kwic/matchers.py
-----------------------
Per-language lemma matchers for ph-project Phase 1 KWIC extraction.

Each matcher turns a running text into a list of (surface_token, lemma) pairs.
Matching logic — finding which canon terms hit in a sentence — lives in the
extraction pipeline (5f9.4), NOT here.  The matcher's only job is lemmatization.

Public API
----------
Matcher
    Protocol class defining the one-method interface.

PymorphyMatcher
    Russian lemmatizer via pymorphy3.  Uses whitespace tokenization.

SpacyMatcher
    English / Spanish lemmatizer via spaCy.  Uses spaCy's tokenizer
    (which splits punctuation from words — see tokenization note below).

get_matcher(lang)
    Factory function returning the appropriate Matcher for *lang*.

HEAD_POSITION
    Imported from baselines.distances (NOT redefined here).  Exposed here
    so downstream modules can do ``from phase1_kwic.matchers import HEAD_POSITION``
    without reaching into baselines directly.

Tokenization difference: PymorphyMatcher vs. SpacyMatcher
----------------------------------------------------------
The two matchers use different tokenization strategies, by design:

* **PymorphyMatcher** (Russian) — whitespace split.  The Leipzig Russian
  corpora use Cyrillic script with punctuation largely separated by spaces,
  so whitespace split is appropriate.  This means ``"мать,"`` would be
  tokenized as one token ``"мать,"`` and pymorphy3 would still lemmatize it
  (it strips trailing punctuation internally), but for the KWIC window the
  whitespace token count is what matters.

* **SpacyMatcher** (English, Spanish) — spaCy's built-in tokenizer, which
  splits punctuation away from words.  So ``"Mother,"`` becomes two tokens:
  ``"Mother"`` (index 0) and ``","`` (index 1).  This means the ``lemmas``
  list returned by SpacyMatcher is indexed by *spaCy token* position, not
  by whitespace position.

  Consequence for load_canon: when computing Term.lemmas for multi-word
  terms like ``"maternal uncle"`` or ``"двоюродный брат"``, only non-space
  tokens are returned, so the lemma count equals the whitespace-split count.
  Single-word terms: same count (1).  Punctuation-only tokens are excluded.

  Consequence for sentence scanning in 5f9.4: the extraction pipeline uses
  whitespace-tokenized positions for KWIC window arithmetic (per SCHEMA.md).
  The matcher is called for lemma lookup; index reconciliation between spaCy
  token space and whitespace token space is the extraction pipeline's job.
"""
from __future__ import annotations

from typing import Protocol

from baselines.distances import HEAD_POSITION  # imported, NOT redefined

# spaCy model names per language
_SPACY_MODEL: dict[str, str] = {
    "en": "en_core_web_md",
    "es": "es_core_news_md",
}


# ---------------------------------------------------------------------------
# Matcher Protocol
# ---------------------------------------------------------------------------

class Matcher(Protocol):
    """Protocol for per-language lemma matchers.

    A Matcher converts a sentence string into a list of (surface_token,
    lemma) pairs — one pair per token emitted by the matcher's tokenizer.

    The returned list is indexed in the matcher's own token space:
    - PymorphyMatcher: whitespace tokens
    - SpacyMatcher: spaCy tokens (punctuation separated)

    This is the ONLY method the Protocol requires.  Matching (finding canon
    terms in the list) happens in 5f9.4's extraction loop, not here.
    """

    def lemmatize(self, sentence: str) -> list[tuple[str, str]]:
        """Return [(surface_token, lemma), ...] for *sentence*.

        Parameters
        ----------
        sentence : str
            A running-text sentence (or a canon-term surface form when
            computing Term.lemmas in load_canon).

        Returns
        -------
        list[tuple[str, str]]
            Each element is (surface_token, lemma).  Tokens and lemmas are
            lowercased where the language's orthography permits; Russian
            Cyrillic is passed through as-is.
        """
        ...


# ---------------------------------------------------------------------------
# PymorphyMatcher — Russian (whitespace tokenization)
# ---------------------------------------------------------------------------

class PymorphyMatcher:
    """Russian lemmatizer using pymorphy3.

    Tokenization: whitespace split (``sentence.split()``).

    The MorphAnalyzer is created once and cached as an instance attribute.
    pymorphy3 recommends reusing a single MorphAnalyzer object rather than
    constructing one per call (it loads dictionaries on first construction).

    Lemma is taken as ``parse(token)[0].normal_form`` — the highest-scoring
    parse's normal form, which is the canonical nominative singular for nouns
    and the infinitive for verbs.
    """

    def __init__(self) -> None:
        import pymorphy3
        self._analyzer = pymorphy3.MorphAnalyzer()

    def lemmatize(self, sentence: str) -> list[tuple[str, str]]:
        """Whitespace-tokenize *sentence* and return (token, lemma) pairs.

        Parameters
        ----------
        sentence : str
            Input sentence or surface form.

        Returns
        -------
        list[tuple[str, str]]
            One entry per whitespace token.  The lemma is
            ``MorphAnalyzer.parse(token)[0].normal_form``.
        """
        result: list[tuple[str, str]] = []
        for token in sentence.split():
            parses = self._analyzer.parse(token)
            lemma = parses[0].normal_form if parses else token.lower()
            result.append((token, lemma))
        return result


# ---------------------------------------------------------------------------
# SpacyMatcher — English / Spanish (spaCy tokenization)
# ---------------------------------------------------------------------------

class SpacyMatcher:
    """English or Spanish lemmatizer using spaCy.

    Tokenization: spaCy's built-in tokenizer, which splits punctuation from
    words.  E.g. ``"Mother,"`` → [("Mother", "mother"), (",", ",")].

    The model is lazy-loaded on first call to ``lemmatize``.  Parser and NER
    are disabled for speed; the tagger is kept because it is required for
    accurate English/Spanish lemmatization.

    Supported languages and their spaCy model names:
        "en" → "en_core_web_md"
        "es" → "es_core_news_md"
    """

    def __init__(self, lang: str) -> None:
        if lang not in _SPACY_MODEL:
            raise ValueError(
                f"SpacyMatcher: lang must be one of {sorted(_SPACY_MODEL)!r}, "
                f"got {lang!r}."
            )
        self._lang = lang
        self._model_name = _SPACY_MODEL[lang]
        self._nlp = None  # lazy-loaded on first lemmatize() call

    def _load(self) -> None:
        """Load the spaCy model if not already loaded."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(
                self._model_name,
                disable=["parser", "ner"],
            )

    def lemmatize(self, sentence: str) -> list[tuple[str, str]]:
        """Tokenize *sentence* with spaCy and return (token.text, token.lemma_) pairs.

        Whitespace-only tokens (``token.is_space``) are excluded so that the
        returned list length equals the number of real tokens in the sentence.
        This ensures that for a multi-word term like ``"maternal uncle"``
        (2 whitespace tokens), the returned list has exactly 2 entries, which
        matches the whitespace-split count used by load_canon for Term.lemmas.

        Parameters
        ----------
        sentence : str
            Input sentence or surface form.

        Returns
        -------
        list[tuple[str, str]]
            One entry per non-whitespace spaCy token: ``(token.text, token.lemma_)``.
            Lemmas are returned as-is from spaCy (typically lowercased for
            content words in en/es).
        """
        self._load()
        doc = self._nlp(sentence)
        return [
            (token.text, token.lemma_)
            for token in doc
            if not token.is_space
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_matcher(lang: str) -> Matcher:
    """Return the appropriate Matcher instance for *lang*.

    Parameters
    ----------
    lang : str
        Language code.  Must be one of ``{"en", "ru", "es"}``.

    Returns
    -------
    Matcher
        ``PymorphyMatcher()`` for Russian; ``SpacyMatcher(lang)`` for
        English and Spanish.

    Raises
    ------
    ValueError
        If *lang* is not in the supported set.
    """
    if lang == "ru":
        return PymorphyMatcher()
    if lang in {"en", "es"}:
        return SpacyMatcher(lang)
    raise ValueError(
        f"get_matcher: lang must be one of {{'en', 'ru', 'es'}}, got {lang!r}."
    )
