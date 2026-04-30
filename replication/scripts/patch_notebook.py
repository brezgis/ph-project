"""Apply replication-specific edits to Kushnareva notebooks.

Supports three notebooks:
- features_calculation_by_thresholds.ipynb
- features_calculation_ripser_and_templates.ipynb
- features_prediction.ipynb
- features_calculation_ripser_and_templates.ipynb

Edits for all notebooks:
1. Prepend a setup cell that adds reference/ and replication/ to sys.path
   and monkey-patches networkx.from_numpy_matrix for networkx 3.x compat.

Edits for features_calculation_by_thresholds.ipynb only:
2. Swap `from grab_weights import ...` -> `from grab_weights_compat import ...`.
3. Repoint input_dir/output_dir at replication/data/processed and replication/outputs.
4. Change subset from "test_5k" to "test".
5. Patch deprecated tokenizer API (batch_encode_plus / pad_to_max_length / ndarray).
6. Patch hardcoded cuda:1 -> cuda:0 (single-GPU machine).
7. Insert a disk-budget warning markdown cell right after the setup cell.
8. Append a cleanup cell that deletes per-subset attention .npy files once
   features are saved (gated on KEEP_ATTENTION flag).

Edits for features_calculation_ripser_and_templates.ipynb only:
2-5. Same import/path/subset/tokenizer fixes as the threshold notebook.
6. Repoint cell 41's hardcoded template-phase paths to derive from `subset`.
7. Repoint hardcoded "small_gpt_web/features/" template save paths.
8. Same Pool worker-count + model cleanup pattern.

Edits for features_prediction.ipynb only:
2. Change train_subset from "test_5k" to "train".
3. Change test_subset from "test_5k" to "test".
4. Change input_dir from "./small_gpt_web/" to "../data/processed/".
5. Repoint feature .npy file paths from input_dir + "features/" to ../outputs/features/.

Edits for features_calculation_ripser_and_templates.ipynb only:
2. Refactor get_only_barcodes to accept (filename, indices, ...) and load
   via mmap_mode='r', so the parent never materialises the full tensor.
3. Replace the barcode loop (cell 29 in 1-indexed reference) with an
   indices-passing variant: no parent np.load, no split_matricies_and_lengths.
   Bumps number_of_splits from 2 to 20 for production use.

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
RIPSER_NOTEBOOK = REPO_ROOT / "notebooks" / "features_calculation_ripser_and_templates.ipynb"
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

# numpy 1.20 deprecated np.int as an alias for builtin int; 1.24 removed it.
# reference/ripser_count.py still calls .astype(np.int). Restore the alias.
import numpy as _np
if not hasattr(_np, "int"):
    _np.int = int
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

# multiprocessing.Pool fork()s the parent kernel — with the BERT model still
# resident in RAM and system swap commonly tight on this host, 20 workers
# OOM-kills the kernel. Free the model (not needed for topology features:
# those read attention tensors back from disk) and use a more conservative
# worker count. 8 still gives plenty of parallelism on a 24-core box.
# Two variants — the threshold notebook has no trailing space after 20,
# the ripser notebook does. Both map to the same replacement.
OLD_POOL_CELL = "num_of_workers = 20\npool = Pool(num_of_workers)"
OLD_POOL_CELL_TRAILING_SPACE = "num_of_workers = 20 \npool = Pool(num_of_workers)"
NEW_POOL_CELL = (
    "import gc\n"
    "try:\n"
    "    del model  # not needed for topology features (read from disk)\n"
    "except NameError:\n"
    "    pass\n"
    "try:\n"
    "    import torch\n"
    "    torch.cuda.empty_cache()\n"
    "except Exception:\n"
    "    pass\n"
    "gc.collect()\n"
    "\n"
    "num_of_workers = 8\n"
    "pool = Pool(num_of_workers)"
)

THRESHOLD_PATCHES = [
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_INPUT, NEW_INPUT),
    (OLD_OUTPUT, NEW_OUTPUT),
    (OLD_SUBSET, NEW_SUBSET),
    (OLD_BATCH_ENCODE, NEW_BATCH_ENCODE),
    (OLD_PAD_KW, NEW_PAD_KW),
    (OLD_TOK_BATCH, NEW_TOK_BATCH),
    (OLD_DEVICE, NEW_DEVICE),
    (OLD_POOL_CELL, NEW_POOL_CELL),
]

# ---------------------------------------------------------------------------
# Ripser/templates notebook patches
# ---------------------------------------------------------------------------

# Cell 41 of the ripser notebook hardcodes a second set of paths for the
# template-feature phase, baking the subset name into the attention filename.
# Rewrite to derive everything from `subset` (defined in cell 8) so re-runs
# only need that one variable changed.
OLD_RIPSER_TEMPLATE_CONFIG = (
    "attention_dir = 'small_gpt_web/attentions/'\n"
    "attention_name = 'test_5k_all_heads_12_layers_MAX_LEN_128_bert-base-uncased'\n"
    "\n"
    "texts_name = 'small_gpt_web/test_5k.csv'\n"
    "\n"
    "MAX_LEN = 128"
)
NEW_RIPSER_TEMPLATE_CONFIG = (
    "attention_dir = '../outputs/attentions/'\n"
    "attention_name = subset + '_all_heads_12_layers_MAX_LEN_128_bert-base-uncased'\n"
    "\n"
    "texts_name = '../data/processed/' + subset + '.csv'\n"
    "\n"
    "MAX_LEN = 128"
)

# Two cells construct the template .npy save path with a hardcoded prefix.
OLD_TEMPLATE_SAVE = '"small_gpt_web/features/" + '
NEW_TEMPLATE_SAVE = '"../outputs/features/" + '

RIPSER_PATCHES = [
    (OLD_IMPORT, NEW_IMPORT),
    (OLD_INPUT, NEW_INPUT),
    (OLD_OUTPUT, NEW_OUTPUT),
    (OLD_SUBSET, NEW_SUBSET),
    (OLD_BATCH_ENCODE, NEW_BATCH_ENCODE),
    (OLD_PAD_KW, NEW_PAD_KW),
    (OLD_TOK_BATCH, NEW_TOK_BATCH),
    (OLD_POOL_CELL_TRAILING_SPACE, NEW_POOL_CELL),
    (OLD_RIPSER_TEMPLATE_CONFIG, NEW_RIPSER_TEMPLATE_CONFIG),
    (OLD_TEMPLATE_SAVE, NEW_TEMPLATE_SAVE),
]

# ---------------------------------------------------------------------------
# Disk-budget warning (threshold notebook only)
# ---------------------------------------------------------------------------

# Each subset persists ~4.7GB of float16 attention tensors before features
# are extracted. f93 filled the host disk to 100% running this notebook on
# train+valid splits; surface the constraint at the top so a future reader
# (or rerun) is not surprised. The cleanup cell at the bottom reclaims
# space once features are saved.
DISK_BUDGET_MD_MARKER = "<!-- replication disk-budget — added by patch_notebook.py -->"
DISK_BUDGET_MD_SOURCE = f"""{DISK_BUDGET_MD_MARKER}
## Disk budget

This notebook persists attention tensors to `../outputs/attentions/` before
extracting topology features. Each subset writes ~4.7GB
(samples × 12 layers × 12 heads × 128² × float16). Running all three splits
(train, valid, test) without cleanup needs ~14GB free.

The final cell deletes the per-subset attention files after features are
saved. Set `KEEP_ATTENTION = True` in that cell to retain them — useful
when chaining into `features_calculation_ripser_and_templates.ipynb` on
the same subset to avoid recomputing attention.
"""

# ---------------------------------------------------------------------------
# Cleanup cell (threshold notebook only)
# ---------------------------------------------------------------------------

# Appended at the very end so it runs only if the notebook executed
# successfully through to feature save. Uses adj_filenames (built earlier in
# the feature-extraction phase) so the deletion list is exactly the files
# this run produced — no globbing, no risk of touching another subset's
# attention files.
CLEANUP_CELL_MARKER = "# replication cleanup — added by patch_notebook.py"
CLEANUP_CELL_SOURCE = f"""{CLEANUP_CELL_MARKER}
# Delete per-subset attention .npy files now that features are saved.
# Set KEEP_ATTENTION = True to retain them (e.g. to skip recompute when
# running features_calculation_ripser_and_templates.ipynb on the same
# subset). See the disk-budget cell at the top.
KEEP_ATTENTION = False

import os as _os
if KEEP_ATTENTION:
    print(f"Cleanup: KEEP_ATTENTION=True, retaining {{len(adj_filenames)}} attention file(s) for subset={{subset!r}}.")
else:
    _removed = 0
    for _f in adj_filenames:
        if _os.path.exists(_f):
            _os.remove(_f)
            _removed += 1
    print(f"Cleanup: removed {{_removed}} attention file(s) for subset={{subset!r}}.")
"""

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

# ---------------------------------------------------------------------------
# Ripser notebook patches (features_calculation_ripser_and_templates.ipynb)
# ---------------------------------------------------------------------------

# Cell 27 patch — replace only the get_only_barcodes function definition.
# The other helpers in the same cell (format_barcodes, save_barcodes,
# unite_barcodes, split_matricies_and_lengths) are left untouched.
#
# OLD and NEW strings sourced verbatim from the live notebook via:
#   import nbformat
#   nb = nbformat.read("replication/notebooks/features_calculation_ripser_and_templates.ipynb", as_version=4)
#   print(repr(nb.cells[27].source))
#
OLD_GET_ONLY_BARCODES = (
    'def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):\n'
    '    """Get barcodes from adj matricies for each layer, head"""\n'
    '    barcodes = {}\n'
    '    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])\n'
    '    for (layer, head) in itertools.product(layers, heads):\n'
    '        matricies = adj_matricies[:, layer, head, :, :]\n'
    '        barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))\n'
    '    return barcodes'
)

NEW_GET_ONLY_BARCODES = (
    'def get_only_barcodes(filename, indices, ntokens_array, dim, lower_bound):\n'
    '    """Get barcodes from a slice of an attention .npy. Loads via\n'
    '    mmap so the child process never materializes more than its\n'
    '    slice; parent never materializes the array body at all."""\n'
    '    adj_matricies = np.load(filename, mmap_mode=\'r\')[indices]\n'
    '    barcodes = {}\n'
    '    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])\n'
    '    for (layer, head) in itertools.product(layers, heads):\n'
    '        matricies = adj_matricies[:, layer, head, :, :]\n'
    '        barcodes[(layer, head)] = ripser_count.get_barcodes(\n'
    '            matricies, ntokens_array, dim, lower_bound, (layer, head))\n'
    '    return barcodes'
)

# Cell 28 patch — replace the entire barcode loop cell.
# The OLD string is the entire cell 28 source (0-indexed), sourced verbatim
# from the live notebook. The NEW string removes the parent-side np.load
# and split_matricies_and_lengths call, passes (filename, indices) instead,
# and bumps number_of_splits to 20 for production use.
OLD_BARCODE_LOOP = (
    "queue = Queue()\n"
    "number_of_splits = 2\n"
    "for i, filename in enumerate(tqdm(adj_filenames, desc='Calculating barcodes')):\n"
    "    barcodes = defaultdict(list)\n"
    "    adj_matricies = np.load(filename, allow_pickle=True) # samples X \n"
    '    print(f"Matricies loaded from: {filename}")\n'
    "    ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]\n"
    "    splitted = split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits)\n"
    "    for matricies, ntokens in tqdm(splitted, leave=False):\n"
    "        p = Process(\n"
    "            target=subprocess_wrap,\n"
    "            args=(\n"
    "                queue,\n"
    "                get_only_barcodes,\n"
    "                (matricies, ntokens, dim, lower_bound)\n"
    "            )\n"
    "        )\n"
    "        p.start()\n"
    "        barcodes_part = queue.get() # block until putted and get barcodes from the queue\n"
    '#         print("Features got.")\n'
    "        p.join() # release resources\n"
    '#         print("The process is joined.")\n'
    "        p.close() # releasing resources of ripser\n"
    '#         print("The proccess is closed.")\n'
    "        \n"
    "        barcodes = unite_barcodes(barcodes, barcodes_part)\n"
    "    part = filename.split('_')[-1].split('.')[0]\n"
    "    save_barcodes(barcodes, barcodes_file + '_' + part + '.json')"
)

NEW_BARCODE_LOOP = (
    "queue = Queue()\n"
    "number_of_splits = 20\n"
    "for i, filename in enumerate(tqdm(adj_filenames, desc='Calculating barcodes')):\n"
    "    barcodes = defaultdict(list)\n"
    '    print(f"Processing: {filename}")\n'
    "    ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]\n"
    "    splitted_ids = np.array_split(np.arange(len(ntokens)), number_of_splits)\n"
    "    for indices in tqdm(splitted_ids, leave=False):\n"
    "        p = Process(\n"
    "            target=subprocess_wrap,\n"
    "            args=(\n"
    "                queue,\n"
    "                get_only_barcodes,\n"
    "                (filename, indices, ntokens[indices], dim, lower_bound)\n"
    "            )\n"
    "        )\n"
    "        p.start()\n"
    "        barcodes_part = queue.get()\n"
    "        p.join()\n"
    "        p.close()\n"
    "        barcodes = unite_barcodes(barcodes, barcodes_part)\n"
    "    part = filename.split('_')[-1].split('.')[0]\n"
    "    save_barcodes(barcodes, barcodes_file + '_' + part + '.json')"
)

RIPSER_PATCHES = [
    (OLD_GET_ONLY_BARCODES, NEW_GET_ONLY_BARCODES),
    (OLD_BARCODE_LOOP, NEW_BARCODE_LOOP),
]


def _has_cell_with_marker(nb: nbformat.NotebookNode, marker: str) -> bool:
    """True if any cell in *nb* contains *marker* in its source."""
    return any(marker in cell.source for cell in nb.cells)


def patch(nb_path: Path) -> bool:
    """Apply all applicable patches to the notebook at *nb_path*.

    Returns True if the notebook was modified, False if it was already patched
    (idempotent).
    """
    nb = nbformat.read(nb_path, as_version=4)
    changed = False

    # Setup cell — shared by all notebooks. Insert if missing, refresh if outdated.
    if nb.cells and SETUP_CELL_MARKER in nb.cells[0].source:
        if nb.cells[0].source.rstrip() != SETUP_CELL_SOURCE.rstrip():
            nb.cells[0].source = SETUP_CELL_SOURCE
            changed = True
    else:
        nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL_SOURCE))
        changed = True

    # Determine which patch set to apply based on notebook name.
    # Three notebooks are supported; any other name is an error.
    name = nb_path.name
    _SUPPORTED = {
        "features_calculation_by_thresholds.ipynb": THRESHOLD_PATCHES,
        "features_calculation_ripser_and_templates.ipynb": RIPSER_PATCHES,
        "features_prediction.ipynb": PREDICTION_PATCHES,
        "features_calculation_ripser_and_templates.ipynb": RIPSER_PATCHES,
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

    # Threshold-notebook-only: disk-budget markdown + cleanup cell.
    if name == "features_calculation_by_thresholds.ipynb":
        if not _has_cell_with_marker(nb, DISK_BUDGET_MD_MARKER):
            # Insert right after the setup cell (which is at index 0).
            nb.cells.insert(1, nbformat.v4.new_markdown_cell(DISK_BUDGET_MD_SOURCE))
            changed = True
        if not _has_cell_with_marker(nb, CLEANUP_CELL_MARKER):
            nb.cells.append(nbformat.v4.new_code_cell(CLEANUP_CELL_SOURCE))
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
            "Pass features_calculation_ripser_and_templates.ipynb or "
            "features_prediction.ipynb to patch those instead."
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
