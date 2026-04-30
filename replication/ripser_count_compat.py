"""CPU shim for ripserplusplus — mirrors reference/ripser_count.py's public API.

ripserplusplus (GPU-accelerated Vietoris-Rips) does not build on CUDA 12.9 /
Blackwell (sm_120): the thrust::sort symbols were removed from thrust's
device-execution-policy namespace in recent CUDA releases and the upstream
package has not been updated.  This shim replaces the single call to
rpp.run(...) with scikit-tda's CPU ``ripser.ripser``, which ships as a
pre-built wheel and is already pinned in requirements.txt.

API contract
------------
Every public symbol that reference/ripser_count.py exports is re-exported
here with an identical signature and return-value format:

  barcode_pop_inf(barcode)
  barcode_number(barcode, dim, bd, ml, t)
  barcode_time(barcode, dim, bd)
  barcode_number_of_barcodes(barcode, dim)
  barcode_entropy(barcode, dim)
  barcode_sum(barcode, dim)
  barcode_mean(barcode, dim)
  barcode_std(barcode, dim)
  count_ripser_features(barcodes, feature_list)
  matrix_to_ripser(matrix, ntokens, lower_bound)
  run_ripser_on_matrix(matrix, dim)
  get_barcodes(matricies, ntokens_array, dim, lower_bound, layer_head)
  calculate_features_r(adj_matricies, dim, lower_bound, ripser_features,
                        ntokens_array, logfile)

Barcode format
--------------
reference/ripser_count.py works with *structured arrays* whose dtype has
named fields ``birth`` and ``death`` (float32).  ripser.ripser() returns
plain float64 arrays of shape (n, 2) with columns [birth, death].  This
shim converts the ripser output into the structured-array format so all
downstream helpers (barcode_sum, barcode_mean, …) work unchanged.

Performance note
----------------
CPU Vietoris-Rips is considerably slower than the GPU original — expect
roughly 10–50× slower on large matrices.  For the replication's smoke-test
subset (500+500 texts, max 128 tokens) this is acceptable.  The cross-
linguistic mBERT pipeline (ph-project-mwk) may revisit this choice.

Usage in the barcode notebook
-------------------------------
Replace the import::

    import ripser_count

with::

    import sys, os
    sys.path.insert(0, os.path.abspath("../.."))   # replication/ root
    import replication.ripser_count_compat as ripser_count

or simply place this file on ``sys.path`` so::

    import ripser_count_compat as ripser_count

resolves before ``reference/ripser_count`` does.
"""

import sys
import os

import numpy as np
from ripser import ripser as _ripser_cpu
from tqdm import tqdm

# utils.cutoff_matrix lives in reference/ — import from there.
_ref_dir = os.path.join(os.path.dirname(__file__), "..", "reference")
if _ref_dir not in sys.path:
    sys.path.insert(0, _ref_dir)
from utils import cutoff_matrix  # noqa: E402 (import after path manipulation)

# ---------------------------------------------------------------------------
# Barcode dtype — mirrors what ripserplusplus returns.
# ---------------------------------------------------------------------------

_BARCODE_DTYPE = np.dtype([("birth", "<f4"), ("death", "<f4")])


def _dgm_to_structured(dgm: np.ndarray) -> np.ndarray:
    """Convert a plain (N, 2) float array from ripser into a structured array.

    ripser.ripser returns float64 arrays with columns [birth, death].
    reference/ripser_count.py expects structured arrays with fields 'birth'
    and 'death' (float32).  This function bridges the two.
    """
    if dgm.shape[0] == 0:
        return np.array([], dtype=_BARCODE_DTYPE)
    out = np.empty(dgm.shape[0], dtype=_BARCODE_DTYPE)
    out["birth"] = dgm[:, 0].astype(np.float32)
    out["death"] = dgm[:, 1].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Barcode feature helpers — identical logic to reference/ripser_count.py
# ---------------------------------------------------------------------------

def barcode_pop_inf(barcode):
    """Delete all infinite barcodes."""
    for dim in barcode:
        if len(barcode[dim]):
            barcode[dim] = barcode[dim][barcode[dim]["death"] != np.inf]
    return barcode


def barcode_number(barcode, dim=0, bd="death", ml="m", t=0.5):
    """Number of barcodes in h{dim} with time of birth/death more/less than threshold."""
    if len(barcode[dim]):
        if ml == "m":
            return np.sum(barcode[dim][bd] >= t)
        elif ml == "l":
            return np.sum(barcode[dim][bd] <= t)
        else:
            raise Exception("Wrong more/less type in barcode_number calculation")
    else:
        return 0.0


def barcode_time(barcode, dim=0, bd="birth"):
    """Time of birth/death in h{dim} of the longest barcode."""
    if len(barcode[dim]):
        max_len_idx = np.argmax(barcode[dim]["death"] - barcode[dim]["birth"])
        return barcode[dim][bd][max_len_idx]
    else:
        return 0.0


def barcode_number_of_barcodes(barcode, dim=0):
    return len(barcode[dim])


def barcode_entropy(barcode, dim=0):
    # Note: uses non-in-place division (lengths = lengths / total) instead of
    # reference's in-place form.  Behaviorally equivalent, since lengths is
    # already a fresh subtraction result.
    lengths = barcode[dim]["death"] - barcode[dim]["birth"]
    lengths = lengths / np.sum(lengths)
    return -np.sum(lengths * np.log(lengths))


def barcode_sum(barcode, dim=0):
    """Sum of lengths of barcodes in h{dim}."""
    if len(barcode[dim]):
        return np.sum(barcode[dim]["death"] - barcode[dim]["birth"])
    else:
        return 0.0


def barcode_mean(barcode, dim=0):
    """Mean of lengths of barcodes in h{dim}."""
    if len(barcode[dim]):
        return np.mean(barcode[dim]["death"] - barcode[dim]["birth"])
    else:
        return 0.0


def barcode_std(barcode, dim=0):
    """Std of lengths of barcodes in h{dim}."""
    if len(barcode[dim]):
        return np.std(barcode[dim]["death"] - barcode[dim]["birth"])
    else:
        return 0.0


def count_ripser_features(barcodes, feature_list=["h0_m"]):
    """Calculate all provided ripser features.

    Identical logic to reference/ripser_count.py — operates on structured
    barcode arrays, so it works after converting ripser output via
    run_ripser_on_matrix.

    Note: raises ValueError on unknown feature type, vs reference's accidental
    NameError fall-through.
    """
    barcodes = [barcode_pop_inf(barcode) for barcode in barcodes]
    features = []
    for feature in feature_list:
        parts = feature.split("_")
        dim, ftype, fargs = int(parts[0][1:]), parts[1], parts[2:]
        if ftype == "s":
            feat = [barcode_sum(b, dim) for b in barcodes]
        elif ftype == "m":
            feat = [barcode_mean(b, dim) for b in barcodes]
        elif ftype == "v":
            feat = [barcode_std(b, dim) for b in barcodes]
        elif ftype == "n":
            bd, ml, t = fargs[0], fargs[1], float(fargs[2][1:])
            if bd == "b":
                bd = "birth"
            elif bd == "d":
                bd = "death"
            else:
                raise ValueError(f"Unknown bd character: {bd!r}")
            feat = [barcode_number(b, dim, bd, ml, t) for b in barcodes]
        elif ftype == "t":
            if fargs[0] == "b":
                bd = "birth"
            elif fargs[0] == "d":
                bd = "death"
            else:
                raise ValueError(f"Unknown bd character: {fargs[0]!r}")
            feat = [barcode_time(b, dim, bd) for b in barcodes]
        elif ftype == "nb":
            feat = [barcode_number_of_barcodes(b, dim) for b in barcodes]
        elif ftype == "e":
            feat = [barcode_entropy(b, dim) for b in barcodes]
        else:
            raise ValueError(f"Unknown ripser feature type: {ftype!r}")
        features.append(feat)
    return np.swapaxes(np.array(features), 0, 1)  # samples × n_features


# ---------------------------------------------------------------------------
# Matrix pre-processing — same as reference/ripser_count.py
# ---------------------------------------------------------------------------

def matrix_to_ripser(matrix, ntokens, lower_bound=0.0):
    """Convert an attention matrix to a symmetric distance matrix for ripser.

    Mirrors reference/ripser_count.py::matrix_to_ripser exactly, except we
    use float64 throughout (ripser.ripser expects float64) and avoid
    np.int (deprecated alias removed in NumPy 1.24).

    Note: uses float64 throughout to avoid the deprecated np.int alias used by
    the reference.
    """
    matrix = cutoff_matrix(matrix, ntokens)
    matrix = (matrix > lower_bound).astype(np.float64) * matrix.astype(np.float64)
    matrix = 1.0 - matrix
    matrix -= np.diag(np.diag(matrix))  # zero on diagonal
    matrix = np.minimum(matrix.T, matrix)  # symmetrize
    return matrix


# ---------------------------------------------------------------------------
# ripser call — the only place that differs from reference/ripser_count.py
# ---------------------------------------------------------------------------

def run_ripser_on_matrix(matrix, dim):
    """Run persistent homology on a distance matrix.

    Returns a dict keyed by homology dimension (int) whose values are
    structured numpy arrays with 'birth' and 'death' fields (float32) —
    the same format ripserplusplus returns.

    Uses scikit-tda's CPU ``ripser.ripser`` instead of rpp.run().
    """
    result = _ripser_cpu(matrix, maxdim=dim, distance_matrix=True)
    dgms = result["dgms"]
    barcode = {}
    for d, dgm in enumerate(dgms):
        barcode[d] = _dgm_to_structured(dgm)
    return barcode


# ---------------------------------------------------------------------------
# get_barcodes / calculate_features_r — same signatures as reference
# ---------------------------------------------------------------------------

def get_barcodes(matricies, ntokens_array=[], dim=1, lower_bound=0.0, layer_head=(0, 0)):
    """Get barcodes from a batch of attention matrices."""
    barcodes = []
    for i, matrix in enumerate(matricies):
        matrix = matrix_to_ripser(matrix, ntokens_array[i], lower_bound)
        barcode = run_ripser_on_matrix(matrix, dim)
        barcodes.append(barcode)
    return barcodes


def calculate_features_r(adj_matricies, dim, lower_bound, ripser_features,
                          ntokens_array, logfile="log.txt"):
    """Calculate ripser barcode features for adj_matricies.

    Returns array of shape (n_layers, n_heads, n_samples, n_features) —
    identical shape to reference/ripser_count.py::calculate_features_r.
    """
    features = []
    for layer in tqdm(range(adj_matricies.shape[1])):
        features.append([])
        for head in range(adj_matricies.shape[2]):
            matricies = adj_matricies[:, layer, head, :, :]
            barcodes = get_barcodes(matricies, ntokens_array, dim, lower_bound,
                                    (layer, head))
            lh_features = count_ripser_features(barcodes, ripser_features)
            features[-1].append(lh_features)
    return np.asarray(features)  # layer × head × samples × n_features
