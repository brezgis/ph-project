# Kushnareva 2021 replication

This directory replicates the original Kushnareva et al. (2021) pipeline as it
was intended: **binary classification of human vs. machine-generated text**
using topological features of BERT attention graphs. Tracked under bd issue
[ph-project-4zo](../.beads/issues.jsonl).

## Why this exists

The main project (`notebooks/`) adapts Kushnareva to a cross-linguistic
comparison task. Before doing that adaptation, it's useful to run the original
pipeline end-to-end on its native task. Three reasons:

1. **Stack sanity check** — verify CUDA, ripser, and the mBERT load path all
   work in this environment before introducing our own changes.
2. **Feature intuition** — see the actual shape and scale of the topology
   feature tensors instead of inferring them from the paper.
3. **Reference numbers** — establish a baseline accuracy on this hardware
   that any future port of `grab_weights.py` (issue
   [ph-project-mwk.1](../.beads/issues.jsonl)) can be diffed against.

This is **not** a project deliverable. It exists to build understanding.

## Relationship to `reference/`

`reference/` holds the original Kushnareva code and is **frozen** (see project
root CLAUDE.md). Notebooks here in `replication/notebooks/` are editable
copies — only data paths and a few small things change so the pipeline can
actually run.

## Dataset

**WebText (human) vs. GPT-2 Small 117M (machine)**, from OpenAI's public
[gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset)
release. Per the paper (Table 1), each text is truncated to 128
BertTokenizer tokens. Their split is 20K/2.5K/2.5K per class for
train/valid/test.

We start much smaller than that — see "Run order" below.

## Run order

```bash
# 0. Activate the project venv (from project root)
cd /home/anna/ph-project
source .venv/bin/activate

# 1. Download the raw JSONL files (~600MB total to replication/data/raw/)
cd replication
bash scripts/download_webtext.sh

# 2. Build (text, label) CSVs the notebooks expect
#    --max-per-class subsamples each class for a quick first pass.
python scripts/prepare_csv.py --max-per-class 500

# 3. Edit the notebook copies (see "Notebook edits needed" below), then run
#    in order:
#      notebooks/features_calculation_by_thresholds.ipynb        (cheap)
#      notebooks/features_calculation_ripser_and_templates.ipynb (slow)
#      notebooks/features_prediction.ipynb                       (classifier)
```

## Running notebooks

Use `scripts/launch_jupyter.sh` instead of invoking `jupyter lab` directly.
The wrapper places the kernel in a systemd user scope with a memory cap, so a
runaway cell (such as the ripser barcode loop) kills only the kernel — it does
not freeze the host.

```bash
cd /home/anna/ph-project/replication
bash scripts/launch_jupyter.sh            # defaults: 48 GB RAM cap, 4 GB swap cap
```

Tune the limits with env vars if needed:

```bash
JUPYTER_MEM_MAX=32G JUPYTER_SWAP_MAX=2G bash scripts/launch_jupyter.sh
```

`JUPYTER_MEM_MAX` sets `MemoryMax` (default `48G`); `JUPYTER_SWAP_MAX` sets
`MemorySwapMax` (default `4G`). Both are passed to `systemd-run --user --scope`.

**Requirement:** the host must run cgroup v2 (`stat -fc %T /sys/fs/cgroup`
must return `cgroup2fs`). The script will exit immediately with a clear error
on a cgroup v1 host. North satisfies this requirement.

### Capped kernel for VS Code / Jupyter

Install once:

```bash
python replication/scripts/install_kernel_spec.py
```

Then in VS Code, click the kernel picker (top right of any notebook) and
choose **Python 3 (ph-project, capped)**. The same option appears in
JupyterLab's kernel dropdown.

Each kernel start spawns inside a `systemd-run --user --scope` cgroup with
`MemoryMax=48G` and `MemorySwapMax=4G`. Override via environment variables:

```bash
PH_KERNEL_MEM_MAX=32G PH_KERNEL_SWAP_MAX=2G
```

Set these in your shell profile or VS Code's terminal environment before
starting the kernel. Requires cgroup v2 (the launcher refuses to run on
cgroup v1).

## Notebook edits needed

The notebooks in `replication/notebooks/` are unmodified copies of `reference/`.
For them to run against the data this directory produces, three things need to
change in each:

1. **Path to the reference modules.** Add a setup cell at the top:
   ```python
   import sys, os
   sys.path.insert(0, os.path.abspath("../../reference"))
   sys.path.insert(0, os.path.abspath(".."))
   ```
   This makes `stats_count` and `ripser_count` importable from the frozen
   `reference/` tree, and `grab_weights_compat` importable from `replication/`.

   Also replace the line:
   ```python
   from grab_weights import grab_attention_weights, text_preprocessing
   ```
   with:
   ```python
   from grab_weights_compat import grab_attention_weights, text_preprocessing
   ```
   `grab_weights_compat` is a tiny shim that uses the modern transformers
   tokenizer API (`__call__` + `padding="max_length"`) instead of the
   removed `batch_encode_plus(..., pad_to_max_length=True)`. See
   `grab_weights_compat.py` for the four-line difference. `reference/`
   stays untouched.

2. **Input/output directories.** In each notebook's config cell, change:
   ```python
   input_dir  = "small_gpt_web/"   # original
   output_dir = "small_gpt_web/"   # original
   ```
   to:
   ```python
   input_dir  = "../data/processed/"
   output_dir = "../outputs/"
   ```

3. **Output subdirectories.** The notebooks write into `output_dir + "attentions/"`
   and `output_dir + "features/"`. Make sure those exist before running:
   ```bash
   mkdir -p outputs/attentions outputs/features
   ```

Column names (`sentence`, `labels`) already match what `prepare_csv.py`
produces — no notebook changes needed for data loading.

## Layout

```
replication/
├── README.md            this file
├── .gitignore           ignores data/ and outputs/
├── scripts/
│   ├── download_webtext.sh       pulls JSONL files from OpenAI's public mirror
│   ├── prepare_csv.py            combines human + machine JSONL into labeled CSVs
│   ├── patch_notebook.py         applies replication-specific edits to the reference notebooks
│   ├── predict_threshold_only.py threshold-only classifier path (working end-to-end pipeline)
│   ├── launch_jupyter.sh         systemd-run wrapper that caps kernel memory (see "Running notebooks")
│   ├── launch_kernel.sh          systemd-run wrapper for individual kernel launches (VS Code / Jupyter)
│   └── install_kernel_spec.py    installs the capped kernel spec to ~/.local/share/jupyter/kernels/
├── notebooks/           editable copies of the four reference/ notebooks
├── data/                gitignored — raw JSONL and processed CSVs land here
└── outputs/             gitignored — feature .npy files and classifier outputs
```
