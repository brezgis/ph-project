"""Apply replication-specific edits to Kushnareva notebooks.

Supports two notebooks:
- features_calculation_by_thresholds.ipynb
- features_prediction.ipynb

Edits for both notebooks:
1. Prepend a setup cell that adds reference/ and replication/ to sys.path
   and monkey-patches networkx.from_numpy_matrix for networkx 3.x compat.

Edits for features_calculation_by_thresholds.ipynb only:
2. Swap `from grab_weights import ...` -> `from grab_weights_compat import ...`.
3. Repoint input_dir/output_dir at replication/data/processed and replication/outputs.
4. Change subset from "test_5k" to "test".
5. Patch deprecated tokenizer API (batch_encode_plus / pad_to_max_length / ndarray).
6. Patch hardcoded cuda:1 -> cuda:0 (single-GPU machine).

Edits for features_prediction.ipynb only:
2. Change train_subset from "test_5k" to "train".
3. Change test_subset from "test_5k" to "test".
4. Change input_dir from "./small_gpt_web/" to "../data/processed/".
5. Repoint feature .npy file paths from input_dir + "features/" to ../outputs/features/.

Idempotent: re-running on an already-patched notebook leaves it unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default notebooks — kept for backward compatibility when no --notebook arg given.
THRESHOLD_NOTEBOOK = REPO_ROOT / "notebooks" / "features_calculation_by_thresholds.ipynb"
PREDICTION_NOTEBOOK = REPO_ROOT / "notebooks" / "features_prediction.ipynb"

# ---------------------------------------------------------------------------
# Shared setup cell (both notebooks)
# ---------------------------------------------------------------------------

SETUP_CELL_MARKER = "# replication setup — added by patch_notebook.py"
SETUP_CELL_SOURCE = f"""{SETUP_CELL_MARKER}
import os, sys
_HERE = os.path.abspath(os.getcwd())
for p in (os.path.abspath(os.path.join(_HERE, "..", "..", "reference")),
          os.path.abspath(os.path.join(_HERE, ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)

# networkx 3.x removed from_numpy_matrix; reference/stats_count.py still calls it.
# Behavior of from_numpy_array on a 2D ndarray matches the old from_numpy_matrix.
import networkx as _nx
if not hasattr(_nx, "from_numpy_matrix"):
    _nx.from_numpy_matrix = _nx.from_numpy_array
"""

# ---------------------------------------------------------------------------
# Threshold notebook patches
# ---------------------------------------------------------------------------

OLD_IMPORT = "from grab_weights import grab_attention_weights, text_preprocessing"
NEW_IMPORT = "from grab_weights_compat import grab_attention_weights, text_preprocessing"

OLD_INPUT = 'input_dir = "small_gpt_web/"'
NEW_INPUT = 'input_dir = "../data/processed/"'
OLD_OUTPUT = 'output_dir = "small_gpt_web/"'
NEW_OUTPUT = 'output_dir = "../outputs/"'

OLD_SUBSET = 'subset = "test_5k"'
NEW_SUBSET = 'subset = "test"'

# Modern transformers replaced batch_encode_plus(...pad_to_max_length=True)
# with __call__(...padding="max_length"). The notebook has its own helper
# (get_token_length) that uses the deprecated API directly; patch it in place.
OLD_BATCH_ENCODE = "tokenizer.batch_encode_plus("
NEW_BATCH_ENCODE = "tokenizer("
OLD_PAD_KW = "pad_to_max_length=True"
NEW_PAD_KW = 'padding="max_length"'
# The new tokenizer __call__ rejects numpy ndarrays — needs list[str]. The
# notebook passes data['sentence'].values (ndarray) into get_token_length.
OLD_TOK_BATCH = "tokenizer(batch_texts,"
NEW_TOK_BATCH = "tokenizer(list(batch_texts),"

# The original notebook hardcodes cuda:1 (paper authors had multiple GPUs).
# This machine has one GPU; cuda:0 is the only valid ordinal.
OLD_DEVICE = "device='cuda:1'"
NEW_DEVICE = "device='cuda:0'"

THRESHOLD_PATCHES = [
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_INPUT, NEW_INPUT),
    (OLD_OUTPUT, NEW_OUTPUT),
    (OLD_SUBSET, NEW_SUBSET),
    (OLD_BATCH_ENCODE, NEW_BATCH_ENCODE),
    (OLD_PAD_KW, NEW_PAD_KW),
    (OLD_TOK_BATCH, NEW_TOK_BATCH),
    (OLD_DEVICE, NEW_DEVICE),
]

# ---------------------------------------------------------------------------
# Prediction notebook patches
# ---------------------------------------------------------------------------

# train_subset and test_subset are on different variables than the threshold
# notebook's single "subset". Match exactly to avoid touching other occurrences.
OLD_TRAIN_SUBSET = 'train_subset = "test_5k"'
NEW_TRAIN_SUBSET = 'train_subset = "train"'

OLD_TEST_SUBSET = 'test_subset  = "test_5k"'
NEW_TEST_SUBSET = 'test_subset  = "test"'

OLD_PRED_INPUT = 'input_dir = "./small_gpt_web/"'
NEW_PRED_INPUT = 'input_dir = "../data/processed/"'

# The threshold notebook writes features to ../outputs/features/<subset>_*.npy.
# The prediction notebook builds paths as input_dir + "features/" + subset + suffix,
# which after patching input_dir would point to ../data/processed/features/ — wrong.
# Replace the base expression so all six file vars use ../outputs/features/ directly.
OLD_FEATURE_BASE = 'input_dir + "features/" + '
NEW_FEATURE_BASE = '"../outputs/features/" + '

PREDICTION_PATCHES = [
    (OLD_TRAIN_SUBSET, NEW_TRAIN_SUBSET),
    (OLD_TEST_SUBSET, NEW_TEST_SUBSET),
    (OLD_PRED_INPUT, NEW_PRED_INPUT),
    (OLD_FEATURE_BASE, NEW_FEATURE_BASE),
]


def patch(nb_path: Path) -> bool:
    """Apply all applicable patches to the notebook at *nb_path*.

    Returns True if the notebook was modified, False if it was already patched
    (idempotent).
    """
    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    # Setup cell — shared by both notebooks.
    if not (nb.cells and SETUP_CELL_MARKER in nb.cells[0].source):
        nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL_SOURCE))
        changed = True

    # Determine which patch set to apply based on notebook name.
    # Exactly two notebooks are supported; any other name is an error.
    name = nb_path.name
    _SUPPORTED = {
        "features_calculation_by_thresholds.ipynb": THRESHOLD_PATCHES,
        "features_prediction.ipynb": PREDICTION_PATCHES,
    }
    if name not in _SUPPORTED:
        raise ValueError(
            f"No patch set defined for notebook {name!r}; "
            f"supported: {sorted(_SUPPORTED)}"
        )
    patch_pairs = _SUPPORTED[name]

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        before = cell.source
        after = before
        for old, new in patch_pairs:
            if old in after:
                after = after.replace(old, new)
        if after != before:
            cell.source = after
            changed = True

    if changed:
        nbformat.write(nb, nb_path)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch a Kushnareva replication notebook for this repo's layout."
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=THRESHOLD_NOTEBOOK,
        help=(
            "Path to the notebook to patch. "
            "Defaults to features_calculation_by_thresholds.ipynb. "
            "Pass features_prediction.ipynb to patch the prediction notebook."
        ),
    )
    args = parser.parse_args()

    nb_path: Path = args.notebook
    if not nb_path.exists():
        sys.exit(f"notebook not found: {nb_path}")
    try:
        changed = patch(nb_path)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"{nb_path.name}: {'patched' if changed else 'already patched (no-op)'}")


if __name__ == "__main__":
    main()
