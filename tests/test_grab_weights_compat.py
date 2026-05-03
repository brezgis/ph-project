"""Tests for replication.grab_weights_compat.grab_attention_weights.

These tests load bert-base-multilingual-cased once (via a session-scoped
fixture) and verify that grab_attention_weights returns an array with the
correct shape, dtype, value bounds, and stable snapshot values at fixed
coordinates.

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
    # model.eval() makes BERT inference fully deterministic for fixed inputs and
    # weights (dropout is disabled). torch.manual_seed(42) does nothing for
    # determinism here, but is kept as cheap insurance against future
    # stochastic-path additions (e.g. if a stochastic layer is ever added).
    torch.manual_seed(42)
    model.to(device)

    yield model, tokenizer, device


@pytest.fixture(scope="session")
def attn(mbert):
    """Call grab_attention_weights exactly once for the entire test session.

    Returns the (12, 3, 12, 16, 16) float16 array produced from SENTENCES with
    MAX_LEN=16. All tests that need the attention array should take this fixture
    instead of calling _get_attention(mbert) individually — that avoids five
    separate forward passes through mBERT.
    """
    return _get_attention(mbert)


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

def test_output_shape(attn):
    """Output array must have shape (n_layers, batch, n_heads, MAX_LEN, MAX_LEN).

    For mBERT: (12, 3, 12, 16, 16).
    """
    expected = (N_LAYERS, BATCH, N_HEADS, MAX_LEN, MAX_LEN)
    assert attn.shape == expected, (
        f"Expected shape {expected}, got {attn.shape}."
    )


def test_output_dtype(attn):
    """Output dtype must be float16."""
    assert attn.dtype == np.float16, (
        f"Expected dtype float16, got {attn.dtype}."
    )


def test_values_bounded_zero_one(attn):
    """All values must lie in [0, 1] — attention weights are softmax outputs."""
    arr = attn.astype(np.float32)
    assert arr.min() >= 0.0, f"Found negative attention weight: min={arr.min()}"
    assert arr.max() <= 1.0, f"Found attention weight > 1: max={arr.max()}"


def test_rows_sum_to_one(attn):
    """Each attention row (over key positions) must sum to ~1.0.

    Uses atol=1e-2 because float16 introduces rounding at each of the 16
    positions — worst-case accumulated error is O(MAX_LEN * eps_fp16) ≈ 8e-3.
    """
    arr = attn.astype(np.float32)
    # arr shape: (n_layers, batch, n_heads, MAX_LEN, MAX_LEN)
    # sum over last axis (keys) — each query row should sum to 1
    row_sums = arr.sum(axis=-1)  # (12, 3, 12, 16)
    assert np.allclose(row_sums, 1.0, atol=1e-2), (
        f"Attention rows do not sum to 1.0 within atol=1e-2. "
        f"Max deviation: {np.abs(row_sums - 1.0).max():.4f}"
    )


def test_snapshot_element(attn):
    """Snapshot: a scalar at a body coordinate catches value-level regressions.

    Coordinate: layer=6, sample=0, head=6, query_row=2, key_col=3.
    This reaches into the middle of the tensor (not a corner) so it would
    catch axis-ordering bugs — e.g. a batch/head transposition would produce
    a different value here than at arr[0,0,0,0,0].

    Baseline recorded with PH_RUN_GRAB_WEIGHTS=1 on north
    (RTX 5070 Ti, transformers 5.5.3, torch 2.11,
    mBERT bert-base-multilingual-cased).

    float16 + GPU non-determinism can shift values by ~1e-3; model.eval()
    makes inference deterministic for fixed inputs and weights.
    """
    SNAPSHOT_SCALAR = 0.0401611328125  # arr[6, 0, 6, 2, 3] — float16 recorded value

    actual = float(attn[6, 0, 6, 2, 3])
    assert np.isclose(actual, SNAPSHOT_SCALAR, atol=1e-3), (
        f"Snapshot mismatch at arr[6,0,6,2,3]: "
        f"expected {SNAPSHOT_SCALAR}, got {actual}."
    )


def test_snapshot_first_row(attn):
    """Snapshot: the full first attention-row vector catches axis-transposition bugs.

    A whole-row comparison (16 values) at arr[layer=0, sample=0, head=0, query_row=0, :]
    provides much stronger axis-ordering coverage than a single scalar.  A
    batch/head/query/key transposition would change multiple positions
    simultaneously, making it easy to spot.

    Baseline recorded with PH_RUN_GRAB_WEIGHTS=1 on north
    (RTX 5070 Ti, transformers 5.5.3, torch 2.11,
    mBERT bert-base-multilingual-cased).
    """
    # arr[0, 0, 0, 0, :] — layer=0, sample=0, head=0, query_row=0, all key cols
    # Trailing zeros are padding positions (MAX_LEN=16; sentence has 8 tokens).
    SNAPSHOT_ROW = np.array(
        [
            0.09893798828125,        # key_col=0  [CLS]→[CLS]
            0.006988525390625,       # key_col=1
            0.0011882781982421875,   # key_col=2
            0.0010843276977539062,   # key_col=3
            0.005039215087890625,    # key_col=4
            0.00316619873046875,     # key_col=5
            0.0311126708984375,      # key_col=6
            0.8525390625,            # key_col=7  (last real token, [SEP])
            0.0,                     # key_col=8  (padding)
            0.0,                     # key_col=9
            0.0,                     # key_col=10
            0.0,                     # key_col=11
            0.0,                     # key_col=12
            0.0,                     # key_col=13
            0.0,                     # key_col=14
            0.0,                     # key_col=15
        ],
        dtype=np.float32,
    )

    actual_row = attn[0, 0, 0, 0, :].astype(np.float32)
    assert np.allclose(actual_row, SNAPSHOT_ROW, atol=1e-3), (
        f"Snapshot mismatch for arr[0,0,0,0,:].\n"
        f"  Expected: {SNAPSHOT_ROW.tolist()}\n"
        f"  Got:      {actual_row.tolist()}"
    )
