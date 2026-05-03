"""
topology — persistent homology pipeline for the static-embedding baselines.

Wraps Ripser (via ripser or ripserplusplus) to compute Vietoris–Rips barcodes
from distance matrices, then extracts the Kushnareva-style feature vector.

Feature-name convention (mirrors reference/ripser_count.py)
------------------------------------------------------------
Format: ``h{dim}_{type}_{args}``

Dimension prefix
~~~~~~~~~~~~~~~~
``h0`` — H0 (connected components), ``h1`` — H1 (loops / 1-cycles).

Type codes
~~~~~~~~~~
``s``
    Sum of bar lengths: ``h0_s``, ``h1_s``.
``m``
    Mean of bar lengths: ``h0_m``, ``h1_m``.
``v``
    Variance (std in the reference) of bar lengths: ``h0_v``, ``h1_v``.
``e``
    Persistence entropy: ``h0_e``, ``h1_e``.
``n``
    Count of bars whose birth or death time is more/less than a threshold.
    Args: ``b``/``d`` (birth or death), ``m``/``l`` (more or less), ``t<val>``.
    Examples: ``h0_n_d_m_t0.5``, ``h1_n_b_l_t0.75``.
``t``
    Birth or death time of the *longest* finite bar.
    Args: ``b``/``d``.
    Examples: ``h0_t_b``, ``h0_t_d``, ``h1_t_b``, ``h1_t_d``.

Default feature set produced by barcode_features()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The exact feature list is fixed by the ssa.5 implementation so that
baseline B is directly comparable to the main mBERT result.  The
template is::

    h{d}_s, h{d}_m, h{d}_v, h{d}_e,
    h{d}_n_d_m_t<t>, h{d}_n_b_l_t<t>,
    h{d}_t_d, h{d}_t_b

for ``d ∈ {0, 1}`` and thresholds ``t`` chosen in ssa.5 (the reference
notebook uses several, e.g. ``0.25``, ``0.5``, ``0.75``).
"""

import numpy as np
import ripser


# Structured dtype used for all barcode arrays (mirrors reference/ripser_count.py).
_BARCODE_DTYPE = np.dtype([("birth", "f8"), ("death", "f8")])

# Thresholds for the n-type features (matches reference/ripser_count.py usage).
_THRESHOLDS = (0.25, 0.5, 0.75)


def rips_barcode(D: np.ndarray, max_dim: int = 1) -> dict:
    """Compute the Vietoris–Rips persistence barcode for a distance matrix.

    Infinite bars (the single H0 bar that never dies) are stripped before
    returning, mirroring the ``barcode_pop_inf`` step in
    ``reference/ripser_count.py``.

    Parameters
    ----------
    D : np.ndarray, shape (n_terms, n_terms)
        Symmetric pairwise distance matrix with zero diagonal.  Values
        should be in [0, 1] (cosine distances); the function does not
        rescale.
    max_dim : int, default 1
        Maximum homology dimension to compute.  ``max_dim=1`` gives H0 and
        H1 (components and loops).  Higher dimensions are possible but
        expensive.

    Returns
    -------
    barcode : dict
        ``{dim: np.ndarray}`` for dim in ``range(max_dim + 1)``.
        Per ssa.5, each value is a NumPy **structured array** with
        ``'birth'`` and ``'death'`` float fields so that downstream code
        can index by name (``barcode[dim]['death']``) exactly like
        ``reference/ripser_count.py``.  Infinite bars are already removed.
        Implementations that wrap the ``ripser`` package (which returns
        plain ``(n, 2)`` arrays via ``result['dgms']``) must convert to
        the structured-array layout before returning.

    Raises
    ------
    ValueError
        If *D* fails any of the following pre-ripser checks:
        - Not a 2-D ndarray
        - Not square
        - Not symmetric (within a small tolerance)
        - Contains negative values
        - Contains non-finite values (NaN or inf)
    """
    # ------------------------------------------------------------------
    # Input validation — firewall against the silent-zero-output bug
    # (ph-project-mwk.3).  All checks happen BEFORE calling ripser.
    # ------------------------------------------------------------------
    if D.ndim != 2:
        raise ValueError(
            f"D must be a 2-D ndarray; got ndim={D.ndim}"
        )
    n_rows, n_cols = D.shape
    if n_rows != n_cols:
        raise ValueError(
            f"D must be square; got shape ({n_rows}, {n_cols})"
        )
    if not np.isfinite(D).all():
        raise ValueError(
            "D must be finite (no NaN or inf); "
            f"found {np.isnan(D).sum()} NaN(s) and {np.isinf(D).sum()} inf(s)"
        )
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError(
            "D must be symmetric; max asymmetry = "
            f"{np.max(np.abs(D - D.T)):.3e}"
        )
    if (D < 0.0).any():
        raise ValueError(
            "D must be non-negative; found values as small as "
            f"{D.min():.6f}"
        )

    # ------------------------------------------------------------------
    # Run Ripser (CPU ripser package, not ripserplusplus — per ssa.1 pin)
    # ------------------------------------------------------------------
    result = ripser.ripser(D, distance_matrix=True, maxdim=max_dim)
    dgms = result["dgms"]  # list of (n_i, 2) plain float arrays

    # ------------------------------------------------------------------
    # Convert to structured arrays and strip infinite bars (barcode_pop_inf)
    # ------------------------------------------------------------------
    barcode: dict[int, np.ndarray] = {}
    for dim in range(max_dim + 1):
        plain = dgms[dim]  # shape (n_bars, 2), may be empty
        # Strip bars whose death is +inf
        finite_mask = np.isfinite(plain[:, 1])
        finite = plain[finite_mask]
        # Build structured array
        sa = np.empty(len(finite), dtype=_BARCODE_DTYPE)
        if len(finite) > 0:
            sa["birth"] = finite[:, 0]
            sa["death"] = finite[:, 1]
        barcode[dim] = sa

    return barcode


def barcode_features(barcode: dict) -> dict:
    """Extract the Kushnareva-style scalar feature vector from a barcode.

    Feature-name convention mirrors ``reference/ripser_count.py``:
    ``h{dim}_{type}_{args}`` — see module docstring for the full spec.

    h{d}_v computes std (not variance), matching Kushnareva naming.

    Parameters
    ----------
    barcode : dict
        Output of ``rips_barcode``: ``{dim: structured np.ndarray}`` with
        ``'birth'`` and ``'death'`` float fields per bar.  Infinite bars
        must already be stripped (``rips_barcode`` does this).

    Returns
    -------
    features : dict
        ``{feature_name: float}`` for the Kushnareva feature set chosen
        in ssa.5 (see module docstring for the template and ssa.5 for the
        exact thresholds).  Missing dimensions (e.g. H1 when no loops
        exist) contribute ``0.0`` for every feature in that dimension.

    Notes
    -----
    ``h{d}_v`` computes std (not variance), matching Kushnareva naming.
    The letter ``v`` is preserved for parity with
    ``reference/ripser_count.py::barcode_std``; do not silently rename.

    Feature semantics (see reference/ripser_count.py for the authoritative
    implementation):

    * ``*_s``  — sum of (death − birth) across all finite bars.
    * ``*_m``  — mean of (death − birth).
    * ``*_v``  — std of (death − birth) (reference uses std, not variance).
    * ``*_e``  — persistence entropy: −Σ (l_i/L) log(l_i/L) where l_i are
                 bar lengths and L = Σ l_i.  The reference does not guard
                 against an empty barcode; the ssa.5 implementation must
                 return ``0.0`` in that case.
    * ``*_n_{b|d}_{m|l}_t<t>`` — count of bars whose birth/death is
                 more/less than threshold ``t``.
    * ``*_t_b`` — birth time of the longest bar.
    * ``*_t_d`` — death time of the longest bar.

    Exact feature list (deterministic order, for downstream comparison):

    For each d in {0, 1}:
        h{d}_s
        h{d}_m
        h{d}_v          (std, NOT variance — Kushnareva naming convention)
        h{d}_e
        h{d}_n_d_m_t0.25
        h{d}_n_d_m_t0.5
        h{d}_n_d_m_t0.75
        h{d}_n_b_l_t0.25
        h{d}_n_b_l_t0.5
        h{d}_n_b_l_t0.75
        h{d}_t_b
        h{d}_t_d
    """
    features: dict[str, float] = {}

    for d in (0, 1):
        prefix = f"h{d}"
        bc = barcode.get(d, np.empty(0, dtype=_BARCODE_DTYPE))

        if len(bc) == 0:
            # Empty dimension: all features are 0.0
            # Two separate loops to match the grouped insertion order of the
            # non-empty branch (all n_d_m thresholds, then all n_b_l thresholds).
            features[f"{prefix}_s"] = 0.0
            features[f"{prefix}_m"] = 0.0
            features[f"{prefix}_v"] = 0.0
            features[f"{prefix}_e"] = 0.0
            for t in _THRESHOLDS:
                features[f"{prefix}_n_d_m_t{t}"] = 0.0
            for t in _THRESHOLDS:
                features[f"{prefix}_n_b_l_t{t}"] = 0.0
            features[f"{prefix}_t_b"] = 0.0
            features[f"{prefix}_t_d"] = 0.0
            continue

        lengths = bc["death"] - bc["birth"]

        # s: sum of bar lengths
        features[f"{prefix}_s"] = float(np.sum(lengths))

        # m: mean of bar lengths
        features[f"{prefix}_m"] = float(np.mean(lengths))

        # v: STANDARD DEVIATION of bar lengths (named 'v' per Kushnareva; NOT variance)
        features[f"{prefix}_v"] = float(np.std(lengths))

        # e: persistence entropy -Σ (l_i/L) log(l_i/L); 0.0 when empty or L==0
        L = np.sum(lengths)
        if L <= 0.0:
            features[f"{prefix}_e"] = 0.0
        else:
            p = lengths / L
            # Guard against log(0) — should not occur if L>0 and lengths>=0,
            # but be safe.
            p = p[p > 0]
            features[f"{prefix}_e"] = max(0.0, float(-np.sum(p * np.log(p))))

        # n_d_m_t<T>: count of bars with death >= T
        for t in _THRESHOLDS:
            features[f"{prefix}_n_d_m_t{t}"] = float(np.sum(bc["death"] >= t))

        # n_b_l_t<T>: count of bars with birth <= T
        for t in _THRESHOLDS:
            features[f"{prefix}_n_b_l_t{t}"] = float(np.sum(bc["birth"] <= t))

        # t_b, t_d: birth/death of the longest finite bar (argmax of length)
        max_idx = int(np.argmax(lengths))
        features[f"{prefix}_t_b"] = float(bc["birth"][max_idx])
        features[f"{prefix}_t_d"] = float(bc["death"][max_idx])

    return features
