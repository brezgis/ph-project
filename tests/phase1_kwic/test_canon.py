"""
Tests for phase1_kwic.canon — Term dataclass and load_canon function.

Row-count expectations:
  en color=11, en emotion=18, en kinship=27
  es color=11, es emotion=22, es kinship=32
  ru color=12, ru emotion=19, ru kinship=34
"""
import dataclasses

import pytest

from phase1_kwic.canon import Term, load_canon
from phase1_kwic.matchers import get_matcher
from phase1_kwic import SUPPORTED_LANGUAGES, DOMAINS
import baselines


# ---------------------------------------------------------------------------
# Module-scoped matcher fixtures — load each backend once per session.
# Without these, the parametrized row-count test reloads the spaCy English
# and Spanish models 3× each (once per domain), inflating runtime.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def matchers():
    return {lang: get_matcher(lang) for lang in ("en", "es", "ru")}


# ---------------------------------------------------------------------------
# Row-count parametrize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,domain,expected_count", [
    ("en", "color",   11),
    ("en", "emotion", 18),
    ("en", "kinship", 27),
    ("es", "color",   11),
    ("es", "emotion", 22),
    ("es", "kinship", 32),
    ("ru", "color",   12),
    ("ru", "emotion", 19),
    ("ru", "kinship", 34),
])
def test_load_canon_row_count(lang, domain, expected_count, matchers):
    """Each (lang, domain) YAML loads exactly the expected number of terms."""
    terms = load_canon(lang, domain, matcher=matchers[lang])
    assert len(terms) == expected_count, (
        f"load_canon({lang!r}, {domain!r}) returned {len(terms)} terms, "
        f"expected {expected_count}"
    )


# ---------------------------------------------------------------------------
# Term dataclass fields
# ---------------------------------------------------------------------------

def test_term_has_required_fields(matchers):
    """Term must have surface, gloss, source, notes, lemmas attributes."""
    terms = load_canon("en", "color", matcher=matchers["en"])
    t = terms[0]
    assert hasattr(t, "surface")
    assert hasattr(t, "gloss")
    assert hasattr(t, "source")
    assert hasattr(t, "notes")
    assert hasattr(t, "lemmas")


def test_term_surface_is_string(matchers):
    """Term.surface is a non-empty string."""
    terms = load_canon("en", "color", matcher=matchers["en"])
    for t in terms:
        assert isinstance(t.surface, str) and t.surface, (
            f"Expected non-empty string surface, got {t.surface!r}"
        )


def test_term_lemmas_is_tuple(matchers):
    """Term.lemmas is a tuple (frozen)."""
    terms = load_canon("en", "color", matcher=matchers["en"])
    for t in terms:
        assert isinstance(t.lemmas, tuple), (
            f"Expected tuple for lemmas, got {type(t.lemmas)}"
        )


def test_term_lemmas_length_matches_whitespace_tokens(matchers):
    """Term.lemmas must have the same length as whitespace-split surface tokens.

    This covers single-word and multi-word terms. The matcher lemmatizes each
    whitespace token separately, so the count must match.
    """
    # Check all 9 files to catch multi-word terms too (e.g., ru kinship has
    # "двоюродный брат" which is 2 whitespace tokens → lemmas must be length 2).
    for lang in ("en", "es", "ru"):
        for domain in ("color", "emotion", "kinship"):
            terms = load_canon(lang, domain, matcher=matchers[lang])
            for t in terms:
                ws_count = len(t.surface.split())
                assert len(t.lemmas) == ws_count, (
                    f"[{lang}/{domain}] Term {t.surface!r}: "
                    f"whitespace tokens={ws_count}, lemmas={len(t.lemmas)}"
                )


# ---------------------------------------------------------------------------
# Term is frozen (immutable)
# ---------------------------------------------------------------------------

def test_term_is_frozen(matchers):
    """Term dataclass must be frozen — setting attributes raises FrozenInstanceError."""
    terms = load_canon("en", "color", matcher=matchers["en"])
    t = terms[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.surface = "hacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation: invalid lang / domain raises ValueError
# ---------------------------------------------------------------------------

def test_invalid_lang_raises():
    """load_canon with an unsupported language raises ValueError."""
    with pytest.raises(ValueError, match="lang"):
        load_canon("fr", "color")


def test_invalid_domain_raises():
    """load_canon with an unsupported domain raises ValueError."""
    with pytest.raises(ValueError, match="domain"):
        load_canon("en", "food")


def test_invalid_both_raises():
    """load_canon with both invalid lang and domain raises ValueError."""
    with pytest.raises(ValueError):
        load_canon("de", "weather")


# ---------------------------------------------------------------------------
# Content spot-checks
# ---------------------------------------------------------------------------

def test_en_color_terms_include_red(matchers):
    """English color canon must contain 'red' as a surface form."""
    terms = load_canon("en", "color", matcher=matchers["en"])
    surfaces = [t.surface for t in terms]
    assert "red" in surfaces


def test_ru_kinship_has_multiword_term(matchers):
    """Russian kinship must contain the multi-word term 'двоюродный брат'."""
    terms = load_canon("ru", "kinship", matcher=matchers["ru"])
    surfaces = [t.surface for t in terms]
    assert "двоюродный брат" in surfaces


def test_source_field_non_empty(matchers):
    """Every term must have a non-empty source field."""
    for lang in ("en", "es", "ru"):
        for domain in ("color", "emotion", "kinship"):
            terms = load_canon(lang, domain, matcher=matchers[lang])
            for t in terms:
                assert t.source, (
                    f"[{lang}/{domain}] Term {t.surface!r} has empty source"
                )


# ---------------------------------------------------------------------------
# phase1_kwic package exports
# ---------------------------------------------------------------------------

def test_supported_languages_is_same_object_as_baselines():
    """phase1_kwic.SUPPORTED_LANGUAGES must be the same object as baselines.SUPPORTED_LANGUAGES.

    This verifies we re-export rather than duplicate.
    """
    assert SUPPORTED_LANGUAGES is baselines.SUPPORTED_LANGUAGES


def test_domains_frozenset():
    """DOMAINS must be a frozenset of exactly the three expected domain names."""
    assert isinstance(DOMAINS, frozenset)
    assert DOMAINS == frozenset({"color", "emotion", "kinship"})


# ---------------------------------------------------------------------------
# Hyphenated compound term handling (en/kinship)
# ---------------------------------------------------------------------------

def test_hyphenated_term_lemmas_one_per_whitespace_token(matchers):
    """en/kinship hyphenated terms like 'father-in-law' have exactly 1 lemma.

    spaCy's tokenizer splits 'father-in-law' into 5 sub-tokens.
    canon.py must lemmatize per whitespace token so lemmas length == 1.
    """
    terms = load_canon("en", "kinship", matcher=matchers["en"])
    hyphenated = [t for t in terms if "-" in t.surface and " " not in t.surface]
    assert hyphenated, "Expected at least one hyphenated en/kinship term"
    for t in hyphenated:
        assert len(t.lemmas) == 1, (
            f"Term {t.surface!r}: expected 1 lemma (1 whitespace token), "
            f"got {len(t.lemmas)}: {t.lemmas}"
        )


# ---------------------------------------------------------------------------
# Lemma value spot checks (catches identity-mapping or swapped-field bugs)
# ---------------------------------------------------------------------------

def test_en_color_red_lemma_value(matchers):
    """The 'red' canon term must lemmatize to ('red',), not the surface itself
    routed unchanged through a no-op matcher."""
    terms = load_canon("en", "color", matcher=matchers["en"])
    red = next(t for t in terms if t.surface == "red")
    assert red.lemmas == ("red",), red.lemmas


def test_ru_kinship_multiword_lemma_value(matchers):
    """'двоюродный брат' has two whitespace tokens; pymorphy3 lemmatizes
    each independently → ('двоюродный', 'брат')."""
    terms = load_canon("ru", "kinship", matcher=matchers["ru"])
    cousin = next(t for t in terms if t.surface == "двоюродный брат")
    assert cousin.lemmas == ("двоюродный", "брат"), cousin.lemmas


def test_es_emotion_terms_have_glosses(matchers):
    """Every es/emotion term must have a non-empty gloss (per SCHEMA convention:
    glosses required for ru/es). Guards against gloss/notes field swap.
    """
    terms = load_canon("es", "emotion", matcher=matchers["es"])
    for t in terms:
        assert t.gloss, f"es/emotion term {t.surface!r} missing gloss"


# ---------------------------------------------------------------------------
# Matcher injection — load_canon(matcher=...) bypasses default factory
# ---------------------------------------------------------------------------

def test_load_canon_uses_injected_matcher():
    """When a matcher is passed in, load_canon must use it (not call the
    default get_matcher). Verified by injecting a stub that records every
    sentence it sees and returns sentinel lemmas.
    """
    seen: list[str] = []

    class StubMatcher:
        def lemmatize(self, sentence):
            seen.append(sentence)
            return [(sentence, "STUB")]

    terms = load_canon("en", "color", matcher=StubMatcher())
    assert len(terms) == 11
    # Every term in en/color is single-word, so each surface produced one
    # whitespace token, which became one stubbed lemmatize call.
    assert len(seen) == 11
    for t in terms:
        assert t.lemmas == ("STUB",), (
            f"Stub matcher should produce ('STUB',), got {t.lemmas}"
        )
