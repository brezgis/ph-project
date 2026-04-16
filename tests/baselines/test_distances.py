"""Placeholder tests for baselines/distances.py.

Real tests arrive in ssa.3 (extract_term_vectors) and ssa.4
(cosine_distance_matrix).
"""
import pytest


def test_distances_module_importable():
    """Smoke: module tree is valid and the public API is present."""
    from baselines.distances import cosine_distance_matrix, extract_term_vectors
    assert callable(cosine_distance_matrix)
    assert callable(extract_term_vectors)


@pytest.mark.skip(reason="awaits ssa.3 implementation")
def test_extract_term_vectors_placeholder():
    """Placeholder — full unit tests land with ssa.3."""
    pass


@pytest.mark.skip(reason="awaits ssa.4 implementation")
def test_cosine_distance_matrix_placeholder():
    """Placeholder — full unit tests land with ssa.4."""
    pass
