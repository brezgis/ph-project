"""Tests for baselines/clustering.py — agglomerative_linkage + kmeans_silhouette_sweep.

Covers:
  * agglomerative_linkage returns ndarray of shape (n-1, 4)
  * kmeans_silhouette_sweep returns dict with correct keys
  * well-separated 2-cluster data yields silhouette > 0.5 at k=2
  * k_range with a single value works
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_cluster_data(n_per_cluster: int = 20, dim: int = 2, seed: int = 0):
    """Two well-separated clusters for cosine distance.

    Each cluster's vectors point in a different orthogonal direction from the
    origin, which is exactly what makes them well-separated in cosine space.
    Blobs are 10 units apart along orthogonal axes so k=2 cosine silhouette
    should be >> 0.5.
    """
    rng = np.random.default_rng(seed)
    # Cluster A: vectors pointing along +x axis (high dim-0, small others)
    cluster_a = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
    cluster_a[:, 0] += 10.0
    # Cluster B: vectors pointing along +y axis (high dim-1, small others)
    cluster_b = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
    cluster_b[:, 1] += 10.0
    X = np.vstack([cluster_a, cluster_b])
    return X


def _distance_matrix_from_X(X: np.ndarray) -> np.ndarray:
    """Build a cosine distance matrix from X using the module under test."""
    from baselines.distances import cosine_distance_matrix
    return cosine_distance_matrix(X)


# ---------------------------------------------------------------------------
# Import smoke
# ---------------------------------------------------------------------------

def test_clustering_module_importable():
    """Smoke: module tree is valid and the public API is present."""
    from baselines.clustering import agglomerative_linkage, kmeans_silhouette_sweep
    assert callable(agglomerative_linkage)
    assert callable(kmeans_silhouette_sweep)


# ---------------------------------------------------------------------------
# agglomerative_linkage
# ---------------------------------------------------------------------------

class TestAgglomerativeLinkage:
    """Unit tests for agglomerative_linkage."""

    def test_returns_ndarray(self):
        """Return value is a numpy ndarray."""
        from baselines.clustering import agglomerative_linkage

        X = _two_cluster_data()
        D = _distance_matrix_from_X(X)
        Z = agglomerative_linkage(D)
        assert isinstance(Z, np.ndarray), f"Expected ndarray, got {type(Z)}"

    def test_shape_n_minus_1_by_4(self):
        """Linkage matrix shape is (n-1, 4) for n observations."""
        from baselines.clustering import agglomerative_linkage

        X = _two_cluster_data(n_per_cluster=15)
        n = X.shape[0]
        D = _distance_matrix_from_X(X)
        Z = agglomerative_linkage(D)
        assert Z.shape == (n - 1, 4), (
            f"Expected shape ({n-1}, 4), got {Z.shape}"
        )

    def test_shape_small_n(self):
        """Shape invariant holds for small n=5."""
        from baselines.clustering import agglomerative_linkage

        rng = np.random.default_rng(1)
        X = rng.standard_normal((5, 4)).astype(np.float32)
        D = _distance_matrix_from_X(X)
        Z = agglomerative_linkage(D)
        assert Z.shape == (4, 4)

    def test_default_method_is_average(self):
        """Default method='average' produces a valid linkage."""
        from baselines.clustering import agglomerative_linkage
        import inspect
        sig = inspect.signature(agglomerative_linkage)
        assert sig.parameters["method"].default == "average"

    def test_custom_method_complete(self):
        """method='complete' is accepted and returns correct shape."""
        from baselines.clustering import agglomerative_linkage

        X = _two_cluster_data(n_per_cluster=10)
        D = _distance_matrix_from_X(X)
        Z = agglomerative_linkage(D, method="complete")
        assert Z.shape == (X.shape[0] - 1, 4)

    def test_last_merge_unites_all_observations(self):
        """Last row of linkage matrix should have n_observations == n."""
        from baselines.clustering import agglomerative_linkage

        X = _two_cluster_data(n_per_cluster=8)
        n = X.shape[0]
        D = _distance_matrix_from_X(X)
        Z = agglomerative_linkage(D)
        assert int(Z[-1, 3]) == n, (
            f"Last merge should combine all {n} observations, got {Z[-1, 3]}"
        )

    def test_first_merge_distance_is_within_cluster(self):
        """First merge is within a tight pair; final merge crosses well-separated clusters.

        Specifically pins the squareform-condense step: skipping it and passing the
        full square matrix to scipy linkage would inflate within-cluster distances and
        fail the Z[0, 2] < 0.01 assertion.
        """
        from baselines.clustering import agglomerative_linkage

        X = np.array([
            [1.0, 0.0, 0.0], [1.0, 0.01, 0.0],   # tight pair A
            [0.0, 1.0, 0.0], [0.0, 1.0, 0.01],    # tight pair B
        ], dtype=np.float64)
        D = _distance_matrix_from_X(X)
        Z = agglomerative_linkage(D, method="average")
        assert Z[0, 2] < 0.01, (
            f"First merge should be within a tight pair (< 0.01), got {Z[0, 2]}"
        )
        assert Z[-1, 2] > 0.5, (
            f"Final merge should cross the two clusters (> 0.5), got {Z[-1, 2]}"
        )

    def test_raises_on_fewer_than_two_observations(self):
        """agglomerative_linkage raises ValueError for n < 2 distance matrices."""
        from baselines.clustering import agglomerative_linkage

        # n=0
        D_empty = np.zeros((0, 0), dtype=float)
        with pytest.raises(ValueError):
            agglomerative_linkage(D_empty)

        # n=1
        D_single = np.zeros((1, 1), dtype=float)
        with pytest.raises(ValueError):
            agglomerative_linkage(D_single)


# ---------------------------------------------------------------------------
# kmeans_silhouette_sweep
# ---------------------------------------------------------------------------

class TestKmeansSilhouetteSweep:
    """Unit tests for kmeans_silhouette_sweep."""

    def test_returns_dict(self):
        """Return value is a dict."""
        from baselines.clustering import kmeans_silhouette_sweep

        X = _two_cluster_data()
        scores = kmeans_silhouette_sweep(X, k_range=range(2, 4))
        assert isinstance(scores, dict)

    def test_keys_match_k_range(self):
        """Dict keys are exactly the values in k_range."""
        from baselines.clustering import kmeans_silhouette_sweep

        X = _two_cluster_data()
        k_range = range(2, 6)
        scores = kmeans_silhouette_sweep(X, k_range=k_range)
        assert set(scores.keys()) == set(k_range), (
            f"Expected keys {set(k_range)}, got {set(scores.keys())}"
        )

    def test_scores_in_valid_range(self):
        """Silhouette scores are in [-1, 1]."""
        from baselines.clustering import kmeans_silhouette_sweep

        X = _two_cluster_data()
        scores = kmeans_silhouette_sweep(X, k_range=range(2, 5))
        for k, s in scores.items():
            assert -1.0 <= s <= 1.0, f"Score out of [-1,1] at k={k}: {s}"

    def test_well_separated_clusters_silhouette_above_threshold(self):
        """Well-separated 2-cluster data should yield silhouette > 0.5 at k=2."""
        from baselines.clustering import kmeans_silhouette_sweep

        X = _two_cluster_data(n_per_cluster=30, seed=0)
        scores = kmeans_silhouette_sweep(X, k_range=range(2, 5))
        assert scores[2] > 0.5, (
            f"Expected silhouette > 0.5 for well-separated k=2 data, got {scores[2]:.4f}"
        )

    def test_single_k_in_range(self):
        """k_range with a single value works without error."""
        from baselines.clustering import kmeans_silhouette_sweep

        X = _two_cluster_data()
        scores = kmeans_silhouette_sweep(X, k_range=range(2, 3))
        assert set(scores.keys()) == {2}

    def test_reproducible_with_fixed_random_state(self):
        """kmeans_silhouette_sweep calls KMeans with random_state=0.

        Uses unittest.mock.patch to verify the contract directly, independent
        of whether the fixture is numerically ambiguous.
        """
        from unittest.mock import patch, MagicMock
        from baselines.clustering import kmeans_silhouette_sweep

        X = _two_cluster_data()

        captured_kwargs = []

        def _capturing_init(self_obj, **kwargs):
            captured_kwargs.append(kwargs)
            # Delegate to real KMeans so the rest of the call chain works.
            from sklearn.cluster import KMeans as _RealKMeans
            real_instance = _RealKMeans.__new__(_RealKMeans)
            _RealKMeans.__init__(real_instance, **kwargs)
            self_obj.__dict__.update(real_instance.__dict__)
            self_obj._real = real_instance

        with patch("baselines.clustering.KMeans") as MockKMeans:
            # Make the mock behave like a real KMeans by forwarding calls.
            from sklearn.cluster import KMeans as _RealKMeans

            def _side_effect(**kwargs):
                return _RealKMeans(**kwargs)

            MockKMeans.side_effect = _side_effect
            kmeans_silhouette_sweep(X, k_range=range(2, 4))

        assert len(MockKMeans.call_args_list) == 2, (
            "Expected one KMeans call per k value"
        )
        for call in MockKMeans.call_args_list:
            _, kw = call
            assert kw.get("random_state") == 0, (
                f"KMeans must be called with random_state=0, got {kw!r}"
            )
