"""
phase1_kwic/canon.py
--------------------
Canon-term YAML loader for ph-project Phase 1 KWIC extraction.

Each canon-term file lives at::

    canon-terms/<lang>/<domain>.yaml

and conforms to the schema in canon-terms/SCHEMA.md.

Public API
----------
Term
    Frozen dataclass representing one canon term.  The `.lemmas` field
    contains one lemma per whitespace token of `.surface`, computed by
    the per-language matcher at load time.

load_canon(lang, domain)
    Load all terms from the appropriate YAML and return a list of Term
    objects with `.lemmas` already populated.

Design note — how lemmas are filled
------------------------------------
`load_canon` accepts an optional *matcher* argument.  When a matcher is
provided, it calls ``matcher.lemmatize(term.surface)`` to compute the
lemmas tuple.  When no matcher is provided, `load_canon` uses the default
matcher for the language (via ``get_matcher(lang)`` from
``phase1_kwic.matchers``), so callers get lemmas without any boilerplate.

This "eager, default matcher" design was chosen over lazy on-access
computation because:

1. It keeps Term fully immutable (frozen dataclass with no computed state).
2. It surfaces matcher errors at load time, not at match time.
3. The matcher is already loaded once for the extraction loop anyway;
   passing it in avoids loading it twice.

Passing an explicit matcher is useful for tests (inject a stub) and for the
extraction pipeline (reuse the already-loaded matcher instance).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from phase1_kwic import SUPPORTED_LANGUAGES, DOMAINS

if TYPE_CHECKING:
    # Avoid a circular import at runtime: matchers.py imports canon.py for
    # the Term type, but canon.py only needs Matcher for the type annotation.
    from phase1_kwic.matchers import Matcher


def _pick_lemma(ws_token: str, pairs: list[tuple[str, str]]) -> str:
    """Pick the canonical lemma from a list of (surface, lemma) pairs.

    This is the single source of truth for the lemma-selection rule:

    1. If *pairs* is empty, return ``ws_token.lower()``.
    2. Otherwise prefer the first pair whose lemma contains at least one
       alphabetic character (skips punctuation-only sub-tokens like ``","``).
    3. Fall back to ``pairs[0][1]`` if no alphabetic lemma is found.

    Both :func:`_lemmatize_ws_token` (canon-term loading) and
    :meth:`SpacyMatcher.lemmatize_many <phase1_kwic.matchers.SpacyMatcher>`
    (batched corpus lemmatization) delegate to this helper so that the
    lemma-selection rule is defined in exactly one place.

    Parameters
    ----------
    ws_token : str
        The original whitespace-split token.  Used only as a fallback when
        *pairs* is empty.
    pairs : list[tuple[str, str]]
        ``(surface_token, lemma)`` pairs produced by the matcher for
        *ws_token* (or for the spaCy sub-tokens that overlap it).

    Returns
    -------
    str
        The selected canonical lemma.
    """
    if not pairs:
        return ws_token.lower()
    return next(
        (lem for _, lem in pairs if any(ch.isalpha() for ch in lem)),
        pairs[0][1],
    )


def _lemmatize_ws_token(ws_token: str, matcher: "Matcher") -> str:
    """Return the canonical lemma for a single whitespace token.

    Calls ``matcher.lemmatize(ws_token)`` and delegates pair selection to
    :func:`_pick_lemma`.

    This is the call-site for canon-term loading (``load_canon``) and for
    any code that needs to lemmatize one token at a time.  For batched corpus
    processing use ``matcher.lemmatize_many``, which also uses
    :func:`_pick_lemma` internally.

    Parameters
    ----------
    ws_token : str
        A single whitespace-split token (a surface form or a corpus word).
    matcher : Matcher
        The per-language matcher whose ``lemmatize`` method is used.

    Returns
    -------
    str
        The canonical lemma for *ws_token*.
    """
    pairs = matcher.lemmatize(ws_token)
    return _pick_lemma(ws_token, pairs)

# Root of the repo — two levels up from this file (phase1_kwic/canon.py)
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_CANON_DIR = _REPO_ROOT / "canon-terms"


@dataclass(frozen=True)
class Term:
    """One canon term loaded from a ``canon-terms/<lang>/<domain>.yaml`` file.

    Attributes
    ----------
    surface : str
        The raw term string as it appears in the YAML (lowercased by
        convention; may be multi-word, e.g. ``"двоюродный брат"``).
    gloss : str or None
        English gloss for non-English terms.  Optional for English entries.
    source : str
        Short-form citation (e.g. ``"Berlin & Kay 1969"``) identifying which
        source licenses this term.  Must match one entry in the file's
        ``sources:`` list.
    notes : str or None
        Optional free-text annotation — morphological notes, register
        caveats, etc.
    lemmas : tuple[str, ...]
        One lemma per whitespace token of ``surface``.  Length always equals
        ``len(surface.split())``.  Populated at load time by the
        per-language matcher.
    """

    surface: str
    gloss: str | None
    source: str
    notes: str | None
    lemmas: tuple[str, ...]


def load_canon(
    lang: str,
    domain: str,
    matcher: "Matcher | None" = None,
) -> list[Term]:
    """Load canon terms for *lang* and *domain* from the YAML file.

    Parameters
    ----------
    lang : str
        BCP-47-style language code.  Must be in ``SUPPORTED_LANGUAGES``
        (``{"en", "ru", "es"}``).
    domain : str
        Semantic domain name.  Must be in ``DOMAINS``
        (``{"color", "emotion", "kinship"}``).
    matcher : Matcher or None, optional
        A ``phase1_kwic.matchers.Matcher``-compatible object whose
        ``lemmatize(sentence)`` method is used to compute ``.lemmas`` for
        each term's surface form.  When *None*, the default matcher for
        *lang* is constructed via ``get_matcher(lang)``.

    Returns
    -------
    list[Term]
        All terms in the YAML, in file order, each with ``.lemmas``
        populated.

    Raises
    ------
    ValueError
        If *lang* is not in ``SUPPORTED_LANGUAGES`` or *domain* is not in
        ``DOMAINS``.
    FileNotFoundError
        If the expected YAML file does not exist at
        ``canon-terms/<lang>/<domain>.yaml``.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"lang must be one of {sorted(SUPPORTED_LANGUAGES)!r}, got {lang!r}."
        )
    if domain not in DOMAINS:
        raise ValueError(
            f"domain must be one of {sorted(DOMAINS)!r}, got {domain!r}."
        )

    yaml_path = _CANON_DIR / lang / f"{domain}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Canon-term YAML not found: {yaml_path}"
        )

    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Resolve matcher lazily here to avoid a circular import at module load.
    # The import is deferred to inside the function body on purpose.
    if matcher is None:
        from phase1_kwic.matchers import get_matcher
        matcher = get_matcher(lang)

    terms: list[Term] = []
    for entry in data.get("terms", []):
        surface: str = entry["term"]
        gloss: str | None = entry.get("gloss")
        source: str = entry["source"]
        notes: str | None = entry.get("notes")

        # Lemmatize each whitespace token of the surface form individually.
        #
        # We call matcher.lemmatize() once per whitespace token rather than
        # once for the whole surface string.  This guarantees that
        # len(lemmas) == len(surface.split()) for every term, including
        # hyphenated compound terms like "father-in-law":
        #
        #   surface.split() == ["father-in-law"]         (1 whitespace token)
        #   SpacyMatcher.lemmatize("father-in-law") →    5 spaCy tokens
        #
        # By lemmatizing each whitespace token independently and taking only
        # the first non-punctuation lemma from each, we produce exactly one
        # lemma per whitespace token regardless of the matcher's internal
        # tokenizer granularity.
        #
        # For PymorphyMatcher (whitespace-based), this is a no-op: splitting
        # on whitespace first and calling lemmatize on each piece produces the
        # same result as calling lemmatize on the whole surface.
        #
        # For SpacyMatcher (spaCy-based), sub-lemmas from a hyphenated chunk
        # are collapsed by taking the first lemma from the result list for
        # that chunk.
        ws_tokens = surface.split()
        lemmas: tuple[str, ...] = tuple(
            _lemmatize_ws_token(ws_token, matcher) for ws_token in ws_tokens
        )

        terms.append(Term(
            surface=surface,
            gloss=gloss,
            source=source,
            notes=notes,
            lemmas=lemmas,
        ))

    return terms
