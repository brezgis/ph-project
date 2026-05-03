"""Tests for replication.grab_weights_compat.grab_attention_weights.

These tests load bert-base-multilingual-cased once (via a session-scoped
fixture) and verify that grab_attention_weights returns an array with the
correct shape, dtype, value bounds, and a stable snapshot value at a fixed
coordinate.

Because loading mBERT weighs ~700 MB and requires a GPU (or CPU with patience),
these tests are **skipped by default**. Set the env var::

    PH_RUN_GRAB_WEIGHTS=1 pytest tests/test_grab_weights_compat.py

to opt in. This mirrors the PH_REQUIRE_FEATURES pattern used in
tests/test_replication_features.py.

If the model is not cached under ~/.cache/huggingface and
PH_RUN_GRAB_WEIGHTS is not set, the tests skip with a clear message.
"""
import os

import numpy as np
import pytest

RUN_GRAB_WEIGHTS = os.environ.get("PH_RUN_GRAB_WEIGHTS") == "1"

# ---------------------------------------------------------------------------
# Skip gate — mirror of PH_REQUIRE_FEATURES pattern in test_replication_features.py
# ---------------------------------------------------------------------------

def _check_mbert_cached() -> bool:
    """Return True if mBERT appears to be cached locally under ~/.cache/huggingface."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if not os.path.isdir(hf_cache):
        return False
    for entry in os.listdir(hf_cache):
        if "bert-base-multilingual-cased" in entry:
            return True
    return False


_SKIP_REASON = (
    "Set PH_RUN_GRAB_WEIGHTS=1 to run grab_weights tests "
    "(loads/downloads ~700 MB mBERT model)."
)

# If the caller opted in explicitly we always run. If not, we skip unless the
# model is already cached (fast local development path).
_should_skip = not RUN_GRAB_WEIGHTS and not _check_mbert_cached()

pytestmark = pytest.mark.skipif(_should_skip, reason=_SKIP_REASON)


# ---------------------------------------------------------------------------
# Session-scoped fixture — load mBERT once for the entire test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mbert():
    """Load bert-base-multilingual-cased once; yield (model, tokenizer, device)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        do_lower_case=False,
    )
    model = AutoModel.from_pretrained(
        "bert-base-multilingual-cased",
        output_attentions=True,
    )
    model.eval()
    torch.manual_seed(42)
    model.to(device)

    yield model, tokenizer, device


# ---------------------------------------------------------------------------
# Fixed test sentences (English / Russian / Spanish)
# ---------------------------------------------------------------------------

SENTENCES = [
    "The color red is warm.",          # English
    "Красный цвет тёплый.",            # Russian
    "El color rojo es cálido.",        # Spanish
]

MAX_LEN = 16

# Expected mBERT dimensions
N_LAYERS = 12
N_HEADS = 12
BATCH = len(SENTENCES)  # 3


# ---------------------------------------------------------------------------
# Helper that calls the shim under test
# ---------------------------------------------------------------------------

def _get_attention(mbert):
    """Call grab_attention_weights and return the result array."""
    from replication.grab_weights_compat import grab_attention_weights

    model, tokenizer, device = mbert
    return grab_attention_weights(model, tokenizer, SENTENCES, MAX_LEN, device=device)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_output_shape(mbert):
    """Output array must have shape (n_layers, batch, n_heads, MAX_LEN, MAX_LEN).

    For mBERT: (12, 3, 12, 16, 16).
    """
    arr = _get_attention(mbert)
    expected = (N_LAYERS, BATCH, N_HEADS, MAX_LEN, MAX_LEN)
    assert arr.shape == expected, (
        f"Expected shape {expected}, got {arr.shape}."
    )


def test_output_dtype(mbert):
    """Output dtype must be float16."""
    arr = _get_attention(mbert)
    assert arr.dtype == np.float16, (
        f"Expected dtype float16, got {arr.dtype}."
    )


def test_values_bounded_zero_one(mbert):
    """All values must lie in [0, 1] — attention weights are softmax outputs."""
    arr = _get_attention(mbert).astype(np.float32)
    assert arr.min() >= 0.0, f"Found negative attention weight: min={arr.min()}"
    assert arr.max() <= 1.0, f"Found attention weight > 1: max={arr.max()}"


def test_rows_sum_to_one(mbert):
    """Each attention row (over key positions) must sum to ~1.0.

    Uses atol=1e-2 because float16 introduces rounding at each of the 16
    positions — worst-case accumulated error is O(MAX_LEN * eps_fp16) ≈ 8e-3.
    """
    arr = _get_attention(mbert).astype(np.float32)
    # arr shape: (n_layers, batch, n_heads, MAX_LEN, MAX_LEN)
    # sum over last axis (keys) — each query row should sum to 1
    row_sums = arr.sum(axis=-1)  # (12, 3, 12, 16)
    assert np.allclose(row_sums, 1.0, atol=1e-2), (
        f"Attention rows do not sum to 1.0 within atol=1e-2. "
        f"Max deviation: {np.abs(row_sums - 1.0).max():.4f}"
    )


def test_snapshot_element(mbert):
    """Snapshot: arr[0, 0, 0, 0, 0] must match a recorded baseline value.

    Coordinate: layer=0, sample=0, head=0, query_row=0, key_col=0.
    This is the [CLS] → [CLS] attention weight in the first head of layer 0
    for the English sentence.

    The snapshot value was recorded by running this test with
    PH_RUN_GRAB_WEIGHTS=1 on north (RTX 5070 Ti, transformers 5.5.3,
    torch 2.11, mBERT bert-base-multilingual-cased).

    float16 + GPU non-determinism can shift values by ~1e-3; we use
    np.isclose with atol=1e-3 and torch.manual_seed(42) + model.eval()
    to keep the value stable across runs on the same hardware.
    """
    SNAPSHOT_VALUE = 0.09893798828125  # float16 recorded value — see module docstring

    arr = _get_attention(mbert)
    actual = float(arr[0, 0, 0, 0, 0])
    assert np.isclose(actual, SNAPSHOT_VALUE, atol=1e-3), (
        f"Snapshot mismatch at arr[0,0,0,0,0]: "
        f"expected {SNAPSHOT_VALUE}, got {actual}."
    )
