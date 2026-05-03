"""
distances — pairwise distance matrices and term vector extraction.

Used by all three sub-baselines (A, B, C) as the shared input layer.
"""

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform

from baselines import SUPPORTED_LANGUAGES


logger = logging.getLogger(__name__)


def _warn_oov(term: str, lang: str, domain_ctx: str, detail: str = "") -> None:
    """Emit an OOV warning via the module logger."""
    extra = f" {detail}" if detail else ""
    logger.warning(
        "OOV: term=%r, lang=%r%s%s — excluded", term, lang, domain_ctx, extra
    )

# Per-language NP head position:
#   EN: right-headed  ("maternal uncle"   → "uncle")
#   RU: right-headed  ("двоюродный брат"  → "брат")
#   ES: left-headed   ("tío materno"      → "tío")
HEAD_POSITION: dict[str, str] = {
    "en": "right",
    "ru": "right",
    "es": "left",
}

assert HEAD_POSITION.keys() == set(SUPPORTED_LANGUAGES), (
    f"HEAD_POSITION keys {sorted(HEAD_POSITION)!r} must equal "
    f"SUPPORTED_LANGUAGES {sorted(SUPPORTED_LANGUAGES)!r}"
)


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix for a set of vectors.

    Parameters
    ----------
    X : np.ndarray, shape (n_terms, dim)
        Row matrix of term vectors (one row per term).  Must be 2-D.

    Returns
    -------
    D : np.ndarray, shape (n_terms, n_terms)
        Symmetric pairwise cosine distance matrix.  D[i, j] = 1 - cos(X[i], X[j]).
        Diagonal is zero.  Values are in [0, 2] in principle but in [0, 1] for
        non-negative embedding spaces.

        For n=0 returns shape (0, 0); for n=1 returns shape (1, 1) of zeros.

    Raises
    ------
    ValueError
        If *X* is not 2-dimensional.
    """
    if X.ndim != 2:
        raise ValueError(
            f"cosine_distance_matrix requires a 2-D array; got ndim={X.ndim}"
        )
    n = X.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=float)
    condensed = pdist(X, metric="cosine")
    return squareform(condensed)


def _lookup_vector(word: str, kv):
    """Return the vector for *word* from *kv*, or None on genuine OOV.

    For FastText .bin models, ``kv[word]`` always succeeds via subword
    composition and never raises KeyError.  For MUSE .vec KeyedVectors,
    missing words raise KeyError.

    Returns
    -------
    np.ndarray or None
    """
    try:
        return kv[word]
    except KeyError:
        return None


def extract_term_vectors(
    terms: list[str],
    vectors,
    strategy: str = "head",
    lang: str = "en",
    head_position: str | None = None,
    domain: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a (term × dim) matrix from a loaded word-vector model.

    Handles single-word and multi-word terms.  Multi-word handling is
    controlled by *strategy* and, for the "head" strategy, by *lang*.

    Parameters
    ----------
    terms : list[str]
        Canon term strings, e.g. ``["red", "maternal uncle", "joy"]``.
    vectors : gensim KeyedVectors or FastText model
        A loaded word-vector model.  FastText .bin models support subword
        composition and return a finite vector for any string; MUSE-aligned
        .vec models do not and will produce OOV misses.
    strategy : str, default "head"
        How to handle multi-word terms:

        ``"head"``
            Take the head word according to per-language NP typology.
            Head position is looked up from::

                HEAD_POSITION = {"en": "right", "ru": "right", "es": "left"}

            EN: right-headed — ``"maternal uncle"`` → ``"uncle"``
            RU: right-headed — ``"двоюродный брат"`` → ``"брат"``
            ES: left-headed  — ``"tío materno"``    → ``"tío"``

            Single-word terms are returned as-is regardless of language.

        ``"mean"``
            Average vectors of all component words.  Component words that
            are OOV in a no-subword model are skipped; if *all* components
            are OOV the term is dropped (mask=False).

        ``"skip"``
            Drop multi-word terms entirely (mask=False for those entries).

    lang : str, default "en"
        BCP-47-style language code used by the ``"head"`` strategy.
        Must be one of ``"en"``, ``"ru"``, ``"es"`` (expandable).
    head_position : str or None, default None
        Optional per-call override for head position.  When provided,
        overrides the ``HEAD_POSITION[lang]`` lookup.  Must be ``"left"``
        or ``"right"`` if given.
    domain : str or None, default None
        Optional domain label included in OOV log messages for context.
        Has no effect on computation.

    Returns
    -------
    matrix : np.ndarray, shape (n_found, dim)
        Row matrix of extracted vectors; one row per *found* term.
        OOV/dropped terms are excluded — the matrix never contains zero
        padding rows.
    found_mask : np.ndarray[bool], shape (n_terms,)
        Boolean mask aligned to *terms*.  ``found_mask[i]`` is ``True`` if
        ``terms[i]`` contributed a row to *matrix*, ``False`` if it was
        dropped (OOV or excluded by strategy).  ``matrix.shape[0] ==
        found_mask.sum()``.

    Notes
    -----
    OOV behaviour differs by model type:

    * FastText .bin (subword) — subword composition always yields a finite
      vector; no misses expected.  ``found_mask`` will be all-True.
    * MUSE-aligned .vec (no subword) — missing tokens are emitted via the
      ``baselines.distances`` logger at WARNING (including lang, and the
      optional *domain* context) and flagged False in ``found_mask``.  They
      are excluded from *matrix*; no zero-vector rows are inserted.
    """
    dim = vectors.vector_size
    n = len(terms)
    found_mask = np.zeros(n, dtype=bool)
    collected: list[np.ndarray] = []

    # Resolve head position for this call
    if head_position is not None:
        _head_pos = head_position
    elif lang in HEAD_POSITION:
        _head_pos = HEAD_POSITION[lang]
    else:
        # Unknown language: fall back to right-headed and warn
        _head_pos = "right"
        logger.warning(
            "unknown lang=%r; defaulting to head_position='right'", lang
        )

    _domain_ctx = f", domain={domain!r}" if domain else ""

    for i, term in enumerate(terms):
        words = term.split()

        if len(words) == 1:
            # Single-word term: attempt direct lookup
            vec = _lookup_vector(term, vectors)
            if vec is None:
                _warn_oov(term, lang, _domain_ctx)
            else:
                found_mask[i] = True
                collected.append(vec)
            continue

        # Multi-word term: dispatch on strategy
        if strategy == "skip":
            # Drop entirely
            continue

        elif strategy == "head":
            if _head_pos == "right":
                head_word = words[-1]
            else:  # "left"
                head_word = words[0]

            vec = _lookup_vector(head_word, vectors)
            if vec is None:
                _warn_oov(term, lang, _domain_ctx, f"(head word={head_word!r})")
            else:
                found_mask[i] = True
                collected.append(vec)

        elif strategy == "mean":
            component_vecs: list[np.ndarray] = []
            for word in words:
                wvec = _lookup_vector(word, vectors)
                if wvec is None:
                    logger.warning(
                        "OOV component: word=%r in term=%r, lang=%r%s — skipped",
                        word, term, lang, _domain_ctx,
                    )
                else:
                    component_vecs.append(wvec)

            if not component_vecs:
                _warn_oov(term, lang, _domain_ctx, "(all components OOV)")
            else:
                avg = np.mean(np.stack(component_vecs, axis=0), axis=0)
                found_mask[i] = True
                collected.append(avg)

        else:
            raise ValueError(
                f"Unknown strategy={strategy!r}. "
                f"Expected one of 'head', 'mean', 'skip'."
            )

    if collected:
        matrix = np.stack(collected, axis=0)
    else:
        matrix = np.empty((0, dim), dtype=np.float32)

    return matrix, found_mask
