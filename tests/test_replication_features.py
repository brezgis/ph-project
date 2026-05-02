"""Tests for replication threshold-notebook feature outputs.

These tests verify that all three splits (train, valid, test) have
consistent feature tensor shapes after running
features_calculation_by_thresholds.ipynb on each subset.

Expected shape: (12, 12, 6, N_samples, 6)
  - 12 layers x 12 heads x 6 features x N_samples x 6 thresholds
  - N_samples = 1000 for all splits (500 human + 500 machine)

By default, per-subset checks skip when the .npy files are absent so
local development stays green before the notebook has been run for that
split. Set PH_REQUIRE_FEATURES=1 to flip absent files into hard failures
— useful for CI or post-run verification where the files MUST exist.
"""
import os
import numpy as np
import pytest

REQUIRE_FEATURES = os.environ.get("PH_REQUIRE_FEATURES") == "1"


def _missing(subset: str, path):
    """Either skip or fail when a feature file is absent, per env var."""
    msg = f"Features file for '{subset}' not yet produced."
    if REQUIRE_FEATURES:
        pytest.fail(msg + " (PH_REQUIRE_FEATURES=1)")
    pytest.skip(msg)

FEATURES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "replication",
    "outputs",
    "features",
)

EXPECTED_SHAPE_PREFIX = (12, 12, 6)  # layers, heads, features
EXPECTED_N_SAMPLES = 1000
EXPECTED_SHAPE_SUFFIX = (6,)  # thresholds
EXPECTED_SHAPE = EXPECTED_SHAPE_PREFIX + (EXPECTED_N_SAMPLES,) + EXPECTED_SHAPE_SUFFIX

SUBSETS = ["train", "valid", "test"]


def _find_features_file(subset: str) -> str | None:
    """Return path to the features .npy for `subset`, or None if absent."""
    if not os.path.isdir(FEATURES_DIR):
        return None
    for fname in os.listdir(FEATURES_DIR):
        if fname.startswith(subset + "_all_heads") and fname.endswith(".npy"):
            return os.path.join(FEATURES_DIR, fname)
    return None


@pytest.mark.parametrize("subset", SUBSETS)
def test_features_shape(subset):
    """Feature tensor must have shape (12, 12, 6, 1000, 6) — layers x heads x features x samples x thresholds.

    Skips when the file does not yet exist; the threshold notebook needs
    to have been run on this subset first.
    """
    path = _find_features_file(subset)
    if path is None:
        _missing(subset, path)
    arr = np.load(path, allow_pickle=True)
    assert arr.shape == EXPECTED_SHAPE, (
        f"subset='{subset}': expected shape {EXPECTED_SHAPE}, got {arr.shape}"
    )


@pytest.mark.parametrize("subset", SUBSETS)
def test_features_no_inf(subset):
    """Feature tensor must not contain +/-inf values."""
    path = _find_features_file(subset)
    if path is None:
        _missing(subset, path)
    arr = np.load(path, allow_pickle=True).astype(float)
    n_inf = np.sum(np.isinf(arr))
    assert n_inf == 0, f"subset='{subset}': found {n_inf} +/-inf values in feature tensor."


@pytest.mark.parametrize("subset", SUBSETS)
def test_features_not_all_zero(subset):
    """No head/layer slice should be entirely zero (would indicate a ripser bug)."""
    path = _find_features_file(subset)
    if path is None:
        _missing(subset, path)
    arr = np.load(path, allow_pickle=True)
    # Shape: (12, 12, 6, 1000, 6) — check that no (layer, head) slice is all-zero
    for layer in range(arr.shape[0]):
        for head in range(arr.shape[1]):
            slice_ = arr[layer, head]
            assert np.any(slice_ != 0), (
                f"subset='{subset}', layer={layer}, head={head}: "
                f"feature slice is entirely zero."
            )
