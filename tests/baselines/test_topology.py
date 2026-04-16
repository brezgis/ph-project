"""Placeholder tests for baselines/topology.py.

Real tests arrive in ssa.5.
"""
import pytest


def test_topology_module_importable():
    """Smoke: module tree is valid and the public API is present."""
    from baselines.topology import rips_barcode, barcode_features
    assert callable(rips_barcode)
    assert callable(barcode_features)


@pytest.mark.skip(reason="awaits ssa.5 implementation")
def test_barcode_features_placeholder():
    """Placeholder — full unit tests land with ssa.5."""
    pass
