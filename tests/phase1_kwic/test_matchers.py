"""
Tests for phase1_kwic.matchers — PymorphyMatcher, SpacyMatcher, get_matcher.

SpaCy models load once per session via module-scoped fixtures to keep the
total test runtime under 30 seconds.

Tokenization notes (documented here so test expectations are transparent):
  - SpacyMatcher uses spaCy's tokenizer (which splits punctuation from words),
    so "Mother," becomes two tokens: "Mother" and ",". The lemma "mother" is
    at index 0 in spaCy token space.
  - PymorphyMatcher uses whitespace split, so "мать" at index 2 is the
    whitespace token at position 2 in "Я любил мать всю жизнь".
"""
import pytest

from phase1_kwic.matchers import (
    PymorphyMatcher,
    SpacyMatcher,
    get_matcher,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — load each spaCy model once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def en_matcher():
    """SpacyMatcher for English, loaded once."""
    return SpacyMatcher("en")


@pytest.fixture(scope="module")
def es_matcher():
    """SpacyMatcher for Spanish, loaded once."""
    return SpacyMatcher("es")


@pytest.fixture(scope="module")
def ru_matcher():
    """PymorphyMatcher for Russian, loaded once."""
    return PymorphyMatcher()


# ---------------------------------------------------------------------------
# Matcher Protocol — interface contract
# ---------------------------------------------------------------------------

def test_matcher_has_lemmatize(en_matcher):
    """SpacyMatcher must expose a `lemmatize` method."""
    assert hasattr(en_matcher, "lemmatize") and callable(en_matcher.lemmatize)


def test_ru_matcher_has_lemmatize(ru_matcher):
    """PymorphyMatcher must expose a `lemmatize` method."""
    assert hasattr(ru_matcher, "lemmatize") and callable(ru_matcher.lemmatize)


# ---------------------------------------------------------------------------
# lemmatize return type
# ---------------------------------------------------------------------------

def test_en_lemmatize_returns_list_of_tuples(en_matcher):
    """lemmatize must return list[tuple[str, str]]."""
    result = en_matcher.lemmatize("I love red apples")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Expected (surface, lemma) tuple, got {item!r}"
        )
        surface, lemma = item
        assert isinstance(surface, str)
        assert isinstance(lemma, str)


def test_ru_lemmatize_returns_list_of_tuples(ru_matcher):
    """PymorphyMatcher.lemmatize must return list[tuple[str, str]]."""
    result = ru_matcher.lemmatize("Я любил мать всю жизнь")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, tuple) and len(item) == 2


# ---------------------------------------------------------------------------
# English sentence tests (SpacyMatcher)
# ---------------------------------------------------------------------------

def test_en_red_apples(en_matcher):
    """'I love red apples' — 'red' lemma at spaCy token index 2."""
    pairs = en_matcher.lemmatize("I love red apples")
    lemmas = [lemma.lower() for _, lemma in pairs]
    assert len(lemmas) > 2, f"Expected at least 3 tokens, got {lemmas}"
    assert lemmas[2] == "red", (
        f"Expected 'red' at index 2, got {lemmas[2]!r}. Full lemmas: {lemmas}"
    )


def test_en_mother_case_insensitive(en_matcher):
    """'Mother, please come here' — 'mother' lemma at spaCy token index 0.

    spaCy's tokenizer splits 'Mother,' into 'Mother' (index 0) and ','
    (index 1), so the 'mother' lemma lands at index 0.
    """
    pairs = en_matcher.lemmatize("Mother, please come here")
    lemmas = [lemma.lower() for _, lemma in pairs]
    assert len(lemmas) > 0, "Expected at least 1 token"
    assert lemmas[0] == "mother", (
        f"Expected 'mother' at index 0, got {lemmas[0]!r}. Full lemmas: {lemmas}"
    )


# ---------------------------------------------------------------------------
# Russian sentence tests (PymorphyMatcher)
# ---------------------------------------------------------------------------

def test_ru_mat_at_index_2(ru_matcher):
    """'Я любил мать всю жизнь' — 'мать' lemma at whitespace token index 2.

    PymorphyMatcher uses whitespace tokenization: tokens are
    ['Я', 'любил', 'мать', 'всю', 'жизнь'].
    """
    pairs = ru_matcher.lemmatize("Я любил мать всю жизнь")
    lemmas = [lemma.lower() for _, lemma in pairs]
    assert len(lemmas) > 2
    assert lemmas[2] == "мать", (
        f"Expected 'мать' at index 2, got {lemmas[2]!r}. Full lemmas: {lemmas}"
    )


def test_ru_krasnaya_inflection(ru_matcher):
    """'красная машина проехала' — 'красный' lemma at whitespace token index 0.

    pymorphy3 should normalize the feminine nominative 'красная' to the
    canonical masculine nominative 'красный'.
    """
    pairs = ru_matcher.lemmatize("красная машина проехала")
    lemmas = [lemma.lower() for _, lemma in pairs]
    assert len(lemmas) > 0
    assert lemmas[0] == "красный", (
        f"Expected 'красный' at index 0, got {lemmas[0]!r}. Full lemmas: {lemmas}"
    )


# ---------------------------------------------------------------------------
# Spanish sentence tests (SpacyMatcher)
# ---------------------------------------------------------------------------

def test_es_miedo_at_index_2(es_matcher):
    """'Tengo mucho miedo' — 'miedo' lemma at spaCy token index 2."""
    pairs = es_matcher.lemmatize("Tengo mucho miedo")
    lemmas = [lemma.lower() for _, lemma in pairs]
    assert len(lemmas) > 2
    assert lemmas[2] == "miedo", (
        f"Expected 'miedo' at index 2, got {lemmas[2]!r}. Full lemmas: {lemmas}"
    )


# ---------------------------------------------------------------------------
# get_matcher factory
# ---------------------------------------------------------------------------

def test_get_matcher_ru_returns_pymorphy():
    """get_matcher('ru') returns a PymorphyMatcher."""
    m = get_matcher("ru")
    assert isinstance(m, PymorphyMatcher)


def test_get_matcher_en_returns_spacy():
    """get_matcher('en') returns a SpacyMatcher."""
    m = get_matcher("en")
    assert isinstance(m, SpacyMatcher)


def test_get_matcher_es_returns_spacy():
    """get_matcher('es') returns a SpacyMatcher."""
    m = get_matcher("es")
    assert isinstance(m, SpacyMatcher)


def test_get_matcher_unknown_lang_raises():
    """get_matcher with an unsupported language raises ValueError."""
    with pytest.raises(ValueError, match="lang"):
        get_matcher("de")


# ---------------------------------------------------------------------------
# HEAD_POSITION import check
# ---------------------------------------------------------------------------

def test_head_position_imported_not_redefined():
    """HEAD_POSITION must be imported from baselines.distances, not redefined."""
    import phase1_kwic.matchers as mod
    import baselines.distances as bd_mod
    # They must be the same object (not a copy)
    assert mod.HEAD_POSITION is bd_mod.HEAD_POSITION, (
        "HEAD_POSITION in matchers must be the same object as baselines.distances.HEAD_POSITION"
    )
