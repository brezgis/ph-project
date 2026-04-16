"""
distances — pairwise distance matrices and term vector extraction.

Used by all three sub-baselines (A, B, C) as the shared input layer.
"""

import numpy as np


def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix for a set of vectors.

    Parameters
    ----------
    X : np.ndarray, shape (n_terms, dim)
        Row matrix of term vectors (one row per term).

    Returns
    -------
    D : np.ndarray, shape (n_terms, n_terms)
        Symmetric pairwise cosine distance matrix.  D[i, j] = 1 - cos(X[i], X[j]).
        Diagonal is zero.  Values are in [0, 2] in principle but in [0, 1] for
        non-negative embedding spaces.
    """
    raise NotImplementedError


def extract_term_vectors(
    terms: list[str],
    vectors,
    strategy: str = "head",
    lang: str = "en",
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
    * MUSE-aligned .vec (no subword) — missing tokens are logged to stderr
      (including lang, domain context in caller) and flagged False in
      ``found_mask``.  They are excluded from *matrix*; no zero-vector rows
      are inserted.
    """
    raise NotImplementedError
