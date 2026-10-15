# Replicating Kushnareva et al. (2021)

Before adapting their method to color terms, I reproduced the original experiment from
Kushnareva et al. (2021), *Artificial Text Detection via Examining the Topology of
Attention Maps*: telling human-written text from GPT-2 output using topological features
of BERT attention graphs.

## Result

Data: WebText (human) vs. GPT-2 Small output from OpenAI's
[gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset), 500 texts per class
in each of train, validation and test, truncated to 128 BERT tokens. Features: all three of
the paper's families (threshold features, barcode features, and distance-to-pattern
template features) from `bert-base-uncased`, fed to logistic regression.

| | accuracy | MCC |
|---|---|---|
| hyperparameters chosen on validation, scored on test | **80.2%** | 0.60 |
| best setting on test (optimistic) | 81.9% | 0.64 |

The paper reports roughly 85–95% with 20,000 texts per class; with 1,000 per class this
is in line. The scores are in `notebooks/features_prediction.ipynb`. (Its notebook
template picks the best of a hyperparameter grid on the test set, so the first row is the
fair number.)

## Getting their code to run

Their code (in `../vendor/tda4atd`, unmodified) targets Python 3.8 and 2021 libraries.
What it took:

- **`transformers`:** `batch_encode_plus(..., pad_to_max_length=True)` no longer exists;
  `../attention_topology/grab_weights_compat.py` swaps in the current tokenizer call.
- **`ripser++`:** the PyPI release doesn't compile against CUDA 12; the project pins the
  upstream commit that fixes it (the `gpu` extra in `pyproject.toml`).
- **`networkx` and NumPy:** `from_numpy_matrix` and `np.int` are gone; the notebooks
  restore both aliases in their first cell.
- **Memory:** the barcode loop loaded every attention matrix in the parent process before
  forking workers and ran out of memory; it now passes file names and indices and each
  worker memory-maps its slice (`tests/test_ripser_mmap.py` guards this).

`scripts/patch_notebook.py` applies these edits to fresh copies of their notebooks, so the
differences from the originals are explicit.

## Running it

From the repository root, after `uv sync --extra gpu`:

```bash
cd replication
bash scripts/download_webtext.sh                    # ~600 MB into data/raw/
uv run python scripts/prepare_csv.py --max-per-class 500
```

Then run, in order: `notebooks/features_calculation_by_thresholds.ipynb`,
`notebooks/features_calculation_ripser_and_templates.ipynb` (the slow one), and
`notebooks/features_prediction.ipynb`. `scripts/launch_jupyter.sh` starts Jupyter with a
memory cap (48 GB by default, via `systemd-run`; set `JUPYTER_MEM_MAX` to change it), so a
runaway cell kills the kernel instead of freezing the machine.
`scripts/install_kernel_spec.py` installs the same capped kernel for VS Code.

## Layout

```
scripts/
  download_webtext.sh        fetches the WebText / GPT-2 JSONL files
  prepare_csv.py             builds labeled train/valid/test CSVs
  patch_notebook.py          applies the compatibility edits to their notebooks
  predict_threshold_only.py  quick classifier on threshold features only
  launch_jupyter.sh          Jupyter with a memory cap
  launch_kernel.sh, install_kernel_spec.py   the capped kernel for VS Code
notebooks/                   their notebooks, patched; the executed copies hold our results
data/, outputs/              gitignored: raw data, CSVs, attention tensors, features
```
