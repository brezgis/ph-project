"""Placeholder tests for baselines/clustering.py.

Real tests arrive in ssa.4.
"""
import pytest


def test_clustering_module_importable():
    """Smoke: module tree is valid and the public API is present."""
    from baselines.clustering import agglomerative_linkage, kmeans_silhouette_sweep
    assert callable(agglomerative_linkage)
    assert callable(kmeans_silhouette_sweep)


@pytest.mark.skip(reason="awaits ssa.4 implementation")
def test_agglomerative_linkage_placeholder():
    """Placeholder — full unit tests land with ssa.4."""
    pass
