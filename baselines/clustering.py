"""
clustering — agglomerative linkage and k-means silhouette sweep.

Used by Baseline A (language-specific descriptive) and Baseline C
(cross-lingual aligned) to characterise semantic cluster structure.
"""

import numpy as np


def agglomerative_linkage(D: np.ndarray, method: str = "average") -> np.ndarray:
    """Compute a hierarchical clustering linkage matrix from a distance matrix.

    Parameters
    ----------
    D : np.ndarray, shape (n_terms, n_terms)
        Symmetric pairwise distance matrix (e.g. cosine distances from
        ``cosine_distance_matrix``).  Diagonal should be zero.
    method : str, default "average"
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
        Common values: ``"average"`` (UPGMA), ``"ward"``, ``"complete"``,
        ``"single"``.  Note: ``"ward"`` requires Euclidean distances.

    Returns
    -------
    Z : np.ndarray, shape (n_terms - 1, 4)
        Scipy-style linkage matrix.  Each row is
        ``[cluster_i, cluster_j, distance, n_observations]``.
        Pass directly to ``scipy.cluster.hierarchy.dendrogram``.
    """
    raise NotImplementedError


def kmeans_silhouette_sweep(X: np.ndarray, k_range: range) -> dict:
    """Run k-means for each k in k_range and return silhouette scores.

    Parameters
    ----------
    X : np.ndarray, shape (n_terms, dim)
        Row matrix of term vectors.  For Baseline A/C these are the raw
        fastText / MUSE vectors; do *not* pass a distance matrix here.
    k_range : range
        Range of cluster counts to evaluate, e.g. ``range(2, 8)``.
        k=1 is excluded because silhouette score is undefined for a single
        cluster.

    Returns
    -------
    scores : dict
        Mapping ``{k: silhouette_score}`` for each k in *k_range*.
        Silhouette scores are in [-1, 1]; higher is better-separated.
        Uses ``sklearn.metrics.silhouette_score`` with ``metric="cosine"``.
    """
    raise NotImplementedError
