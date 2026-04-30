"""Apply replication-specific edits to a copy of the threshold notebook.

Three edits, all minimal:
1. Prepend a setup cell that adds reference/ and replication/ to sys.path.
2. Swap `from grab_weights import ...` -> `from grab_weights_compat import ...`.
3. Repoint input_dir/output_dir at replication/data/processed and replication/outputs.

Idempotent: re-running on an already-patched notebook leaves it unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "features_calculation_by_thresholds.ipynb"

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


def patch(nb_path: Path) -> bool:
    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    if not (nb.cells and SETUP_CELL_MARKER in nb.cells[0].source):
        nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL_SOURCE))
        changed = True

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        before = cell.source
        after = before
        for old, new in [
            (OLD_IMPORT, NEW_IMPORT),
            (OLD_INPUT, NEW_INPUT),
            (OLD_OUTPUT, NEW_OUTPUT),
            (OLD_SUBSET, NEW_SUBSET),
            (OLD_BATCH_ENCODE, NEW_BATCH_ENCODE),
            (OLD_PAD_KW, NEW_PAD_KW),
            (OLD_TOK_BATCH, NEW_TOK_BATCH),
            (OLD_DEVICE, NEW_DEVICE),
        ]:
            if old in after:
                after = after.replace(old, new)
        if after != before:
            cell.source = after
            changed = True

    if changed:
        nbformat.write(nb, nb_path)
    return changed


def main() -> None:
    if not NOTEBOOK.exists():
        sys.exit(f"notebook not found: {NOTEBOOK}")
    changed = patch(NOTEBOOK)
    print(f"{NOTEBOOK.name}: {'patched' if changed else 'already patched (no-op)'}")


if __name__ == "__main__":
    main()
