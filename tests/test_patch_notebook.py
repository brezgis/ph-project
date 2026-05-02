"""Tests for replication/scripts/patch_notebook.py.

Covers:
- patch() applied to features_calculation_by_thresholds.ipynb (already patched in 4zo)
- patch() applied to features_prediction.ipynb (new in 3nh)
- Idempotency: re-running patch() leaves notebooks unchanged (bool + byte equality)
- CLI: patch_notebook.py accepts a --notebook argument and patches content correctly
- Dispatcher: unknown notebook names raise ValueError
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "replication" / "scripts" / "patch_notebook.py"
THRESHOLD_NB = REPO_ROOT / "replication" / "notebooks" / "features_calculation_by_thresholds.ipynb"
PREDICTION_NB = REPO_ROOT / "replication" / "notebooks" / "features_prediction.ipynb"
RIPSER_NB = REPO_ROOT / "replication" / "notebooks" / "features_calculation_ripser_and_templates.ipynb"
# reference/ notebooks are the frozen originals — always unpatched.
REF_PREDICTION_NB = REPO_ROOT / "reference" / "features_prediction.ipynb"
REF_THRESHOLD_NB = REPO_ROOT / "reference" / "features_calculation_by_thresholds.ipynb"
REF_RIPSER_NB = REPO_ROOT / "reference" / "features_calculation_ripser_and_templates.ipynb"


def _load_nb(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


# ---------------------------------------------------------------------------
# Module-level helper: load patch_notebook module from script path
# ---------------------------------------------------------------------------

def _load_patch_module():
    """Import patch_notebook.py as a module. Returns the loaded module."""
    spec = importlib.util.spec_from_file_location("patch_notebook", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_code_source(nb: nbformat.NotebookNode) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _find_source_containing(nb: nbformat.NotebookNode, text: str) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code" and text in c.source]


# ---------------------------------------------------------------------------
# features_prediction.ipynb patches
# ---------------------------------------------------------------------------

class TestPredictionNotebookPatched:
    """Verify the prediction notebook receives all expected patches."""

    @pytest.fixture(autouse=True)
    def patched_nb(self, tmp_path):
        """Copy the frozen reference prediction notebook to tmp, apply patch, load result.

        Using REF_PREDICTION_NB (reference/) guarantees the source is always
        unpatched, making the fixture independent of the state of the working
        replication/ notebook.
        """
        src = REF_PREDICTION_NB
        dst = tmp_path / "features_prediction.ipynb"
        shutil.copy(src, dst)
        mod = _load_patch_module()
        mod.patch(dst)
        self.nb = _load_nb(dst)
        self.mod = mod

    def test_setup_cell_prepended(self):
        """First cell must be the replication setup cell."""
        assert self.mod.SETUP_CELL_MARKER in self.nb.cells[0].source

    def test_train_subset_is_train(self):
        """train_subset must be 'train', not 'test_5k'."""
        sources = _find_source_containing(self.nb, "train_subset")
        assert sources, "No cell with train_subset"
        assert 'train_subset = "train"' in sources[0]

    def test_test_subset_is_test(self):
        """test_subset must be 'test', not 'test_5k'."""
        sources = _find_source_containing(self.nb, "test_subset")
        assert sources, "No cell with test_subset"
        assert 'test_subset  = "test"' in sources[0]

    def test_input_dir_updated(self):
        """input_dir must point to ../data/processed/."""
        sources = _find_source_containing(self.nb, "input_dir")
        assert sources, "No cell with input_dir"
        assert '../data/processed/' in sources[0]
        assert 'small_gpt_web' not in sources[0]

    def test_feature_files_point_to_outputs(self):
        """Feature .npy paths must use ../outputs/features/, not input_dir + features/."""
        sources = _find_source_containing(self.nb, "old_f_train_file")
        assert sources, "No cell with old_f_train_file"
        cell_src = sources[0]
        assert '"../outputs/features/"' in cell_src or "'../outputs/features/'" in cell_src, (
            "Feature file paths should reference ../outputs/features/ directly"
        )

    def test_no_small_gpt_web_in_feature_paths(self):
        """small_gpt_web must not appear anywhere in patched notebook."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert "small_gpt_web" not in all_sources

    def test_idempotent(self, tmp_path):
        """Applying patch twice produces identical notebook bytes.

        Uses the frozen reference/ original (always unpatched) as the source
        so changed1 is reliably True regardless of the state of the working
        replication/ notebook.
        """
        dst = tmp_path / "features_prediction.ipynb"
        shutil.copy(REF_PREDICTION_NB, dst)
        mod = _load_patch_module()
        changed1 = mod.patch(dst)
        bytes_after_first = dst.read_bytes()
        changed2 = mod.patch(dst)
        bytes_after_second = dst.read_bytes()
        assert changed1 is True
        assert changed2 is False, "Second patch run should be a no-op"
        assert bytes_after_first == bytes_after_second, (
            "Byte content must be identical after first and second patch runs"
        )


# ---------------------------------------------------------------------------
# features_calculation_by_thresholds.ipynb patches
# ---------------------------------------------------------------------------

class TestThresholdNotebookPatched:
    """Verify the threshold notebook receives all expected patches."""

    @pytest.fixture(autouse=True)
    def patched_nb(self, tmp_path):
        """Copy the frozen reference threshold notebook to tmp, apply patch, load result."""
        src = REF_THRESHOLD_NB
        dst = tmp_path / "features_calculation_by_thresholds.ipynb"
        shutil.copy(src, dst)
        mod = _load_patch_module()
        mod.patch(dst)
        self.nb = _load_nb(dst)
        self.mod = mod

    def test_setup_cell_prepended(self):
        """First cell must be the replication setup cell."""
        assert self.mod.SETUP_CELL_MARKER in self.nb.cells[0].source

    def test_import_swap_old_gone(self):
        """Old grab_weights import must not appear in patched notebook."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert "from grab_weights import grab_attention_weights" not in all_sources

    def test_import_swap_new_present(self):
        """New grab_weights_compat import must appear in patched notebook."""
        sources = _find_source_containing(self.nb, "grab_weights_compat")
        assert sources, "No cell with grab_weights_compat import"
        assert "from grab_weights_compat import grab_attention_weights" in sources[0]

    def test_input_dir_old_gone(self):
        """Old small_gpt_web input_dir must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert 'input_dir = "small_gpt_web/"' not in all_sources

    def test_input_dir_new_present(self):
        """New input_dir must point to ../data/processed/."""
        sources = _find_source_containing(self.nb, 'input_dir = "../data/processed/"')
        assert sources, 'No cell with input_dir = "../data/processed/"'

    def test_output_dir_old_gone(self):
        """Old small_gpt_web output_dir must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert 'output_dir = "small_gpt_web/"' not in all_sources

    def test_output_dir_new_present(self):
        """New output_dir must point to ../outputs/."""
        sources = _find_source_containing(self.nb, 'output_dir = "../outputs/"')
        assert sources, 'No cell with output_dir = "../outputs/"'

    def test_subset_old_gone(self):
        """Old subset test_5k assignment must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert 'subset = "test_5k"' not in all_sources

    def test_subset_new_present(self):
        """New subset must be 'test'."""
        sources = _find_source_containing(self.nb, 'subset = "test"')
        assert sources, 'No cell with subset = "test"'

    def test_batch_encode_plus_old_gone(self):
        """Deprecated tokenizer.batch_encode_plus must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert "tokenizer.batch_encode_plus(" not in all_sources

    def test_batch_encode_plus_new_present(self):
        """Modern tokenizer( call must appear."""
        sources = _find_source_containing(self.nb, "tokenizer(")
        assert sources, "No cell with tokenizer( call"

    def test_pad_to_max_length_old_gone(self):
        """Deprecated pad_to_max_length=True must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert "pad_to_max_length=True" not in all_sources

    def test_pad_to_max_length_new_present(self):
        """Modern padding='max_length' must appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert 'padding="max_length"' in all_sources

    def test_list_batch_texts_old_gone(self):
        """Raw tokenizer(batch_texts, must not appear (ndarray not accepted)."""
        all_sources = " ".join(_get_code_source(self.nb))
        # Old form passes ndarray directly
        assert "tokenizer(batch_texts," not in all_sources

    def test_list_batch_texts_new_present(self):
        """list(batch_texts) wrapping must appear."""
        sources = _find_source_containing(self.nb, "tokenizer(list(batch_texts),")
        assert sources, "No cell with tokenizer(list(batch_texts),"

    def test_cuda1_old_gone(self):
        """Hardcoded cuda:1 must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert "device='cuda:1'" not in all_sources

    def test_cuda0_new_present(self):
        """cuda:0 must appear."""
        sources = _find_source_containing(self.nb, "device='cuda:0'")
        assert sources, "No cell with device='cuda:0'"

    def test_disk_budget_markdown_inserted_after_setup(self):
        """A disk-budget markdown cell must sit immediately after the setup cell."""
        assert self.nb.cells[1].cell_type == "markdown", (
            f"Cell 1 should be markdown, got {self.nb.cells[1].cell_type}"
        )
        assert self.mod.DISK_BUDGET_MD_MARKER in self.nb.cells[1].source

    def test_disk_budget_markdown_mentions_size_and_cleanup(self):
        """The disk-budget cell must communicate the per-subset size and the
        existence of a cleanup mechanism so a future reader is not surprised."""
        import re
        md = self.nb.cells[1].source
        assert re.search(r"\d+(\.\d+)?\s?GB", md), (
            "Disk-budget cell should call out the per-subset size in GB"
        )
        assert "KEEP_ATTENTION" in md, "Disk-budget cell should reference the override flag"

    def test_cleanup_cell_appended_at_end(self):
        """A cleanup code cell must be the last cell in the notebook."""
        last = self.nb.cells[-1]
        assert last.cell_type == "code", f"Last cell should be code, got {last.cell_type}"
        assert self.mod.CLEANUP_CELL_MARKER in last.source

    def test_cleanup_cell_uses_keep_attention_flag(self):
        """Cleanup must be gated on KEEP_ATTENTION so a user can opt out."""
        src = self.nb.cells[-1].source
        assert "KEEP_ATTENTION = False" in src, "Default must be cleanup-enabled"
        assert "if KEEP_ATTENTION" in src, "Flag must control the deletion path"

    def test_cleanup_cell_iterates_adj_filenames(self):
        """Cleanup must delete files from adj_filenames (the per-subset list
        built during feature extraction) — not glob a directory, which could
        touch other subsets' files."""
        src = self.nb.cells[-1].source
        assert "adj_filenames" in src
        assert "_os.remove(_f)" in src

    def test_idempotent(self, tmp_path):
        """Applying patch twice produces identical notebook bytes."""
        dst = tmp_path / "features_calculation_by_thresholds.ipynb"
        shutil.copy(REF_THRESHOLD_NB, dst)
        mod = _load_patch_module()
        changed1 = mod.patch(dst)
        bytes_after_first = dst.read_bytes()
        changed2 = mod.patch(dst)
        bytes_after_second = dst.read_bytes()
        assert changed1 is True
        assert changed2 is False, "Second patch run should be a no-op"
        assert bytes_after_first == bytes_after_second, (
            "Byte content must be identical after first and second patch runs"
        )

    def test_disk_budget_and_cleanup_not_added_to_prediction(self, tmp_path):
        """Disk-budget markdown and cleanup cell are threshold-notebook-only;
        the prediction notebook must not receive them."""
        dst = tmp_path / "features_prediction.ipynb"
        shutil.copy(REF_PREDICTION_NB, dst)
        mod = _load_patch_module()
        mod.patch(dst)
        nb = _load_nb(dst)
        all_sources = " ".join(c.source for c in nb.cells)
        assert mod.DISK_BUDGET_MD_MARKER not in all_sources
        assert mod.CLEANUP_CELL_MARKER not in all_sources


# ---------------------------------------------------------------------------
# Cleanup cell behavior — exec the cell body, verify KEEP_ATTENTION semantics
# ---------------------------------------------------------------------------

class TestCleanupCellBehavior:
    """Execute the cleanup cell body against fake adj_filenames to verify it
    actually deletes files when KEEP_ATTENTION=False, retains them when True,
    and tolerates entries that no longer exist on disk."""

    def _patched_cleanup_source(self, tmp_path):
        """Return the cleanup cell's source string from a freshly-patched threshold notebook."""
        dst = tmp_path / "features_calculation_by_thresholds.ipynb"
        shutil.copy(REF_THRESHOLD_NB, dst)
        mod = _load_patch_module()
        mod.patch(dst)
        nb = _load_nb(dst)
        # Cleanup cell is the last code cell and contains the marker.
        for cell in reversed(nb.cells):
            if cell.cell_type == "code" and mod.CLEANUP_CELL_MARKER in cell.source:
                return cell.source
        raise AssertionError("Cleanup cell not found in patched notebook")

    def _make_npys(self, tmp_path, names):
        paths = []
        for n in names:
            p = tmp_path / n
            p.write_bytes(b"\x00" * 16)  # tiny valid file
            paths.append(str(p))
        return paths

    def test_keep_false_deletes_all_files(self, tmp_path):
        src = self._patched_cleanup_source(tmp_path)
        files = self._make_npys(tmp_path, ["a.npy", "b.npy", "c.npy"])
        ns = {"adj_filenames": list(files), "subset": "train"}
        exec(src, ns)
        for f in files:
            assert not os.path.exists(f), f"{f} should have been deleted"

    def test_keep_true_retains_all_files(self, tmp_path):
        src = self._patched_cleanup_source(tmp_path)
        files = self._make_npys(tmp_path, ["a.npy", "b.npy"])
        # Override KEEP_ATTENTION after the cell sets it. Easiest: replace
        # the literal in the source so the cell starts with KEEP_ATTENTION=True.
        src_keep = src.replace("KEEP_ATTENTION = False", "KEEP_ATTENTION = True")
        assert "KEEP_ATTENTION = True" in src_keep, "Sanity: KEEP_ATTENTION literal must be in cell"
        ns = {"adj_filenames": list(files), "subset": "train"}
        exec(src_keep, ns)
        for f in files:
            assert os.path.exists(f), f"{f} should have been retained when KEEP_ATTENTION=True"

    def test_missing_file_does_not_raise(self, tmp_path):
        """A stale entry in adj_filenames whose .npy was already removed must
        not crash the cleanup loop."""
        src = self._patched_cleanup_source(tmp_path)
        real = self._make_npys(tmp_path, ["real.npy"])
        ghost = str(tmp_path / "ghost.npy")  # never created
        ns = {"adj_filenames": real + [ghost], "subset": "test"}
        exec(src, ns)  # must not raise
        assert not os.path.exists(real[0])


# ---------------------------------------------------------------------------
# features_calculation_ripser_and_templates.ipynb patches
# ---------------------------------------------------------------------------

class TestRipserNotebookPatched:
    """Verify the ripser notebook receives the mmap OOM-fix patches."""

    @pytest.fixture(autouse=True)
    def patched_nb(self, tmp_path):
        """Copy the frozen reference ripser notebook to tmp, apply patch, load result."""
        src = REF_RIPSER_NB
        dst = tmp_path / "features_calculation_ripser_and_templates.ipynb"
        shutil.copy(src, dst)
        mod = _load_patch_module()
        mod.patch(dst)
        self.nb = _load_nb(dst)
        self.mod = mod

    def test_setup_cell_prepended(self):
        """First cell must be the replication setup cell."""
        assert self.mod.SETUP_CELL_MARKER in self.nb.cells[0].source

    def test_get_only_barcodes_new_signature(self):
        """get_only_barcodes must accept (filename, indices, ...) not (adj_matricies, ...)."""
        sources = _find_source_containing(self.nb, "def get_only_barcodes(")
        assert sources, "No cell defining get_only_barcodes"
        src = sources[0]
        assert "def get_only_barcodes(filename, indices, ntokens_array, dim, lower_bound)" in src, (
            "get_only_barcodes must have (filename, indices, ntokens_array, dim, lower_bound) signature"
        )

    def test_get_only_barcodes_uses_mmap_load(self):
        """get_only_barcodes body must use np.load(filename, mmap_mode='r')[indices]."""
        sources = _find_source_containing(self.nb, "def get_only_barcodes(")
        assert sources, "No cell defining get_only_barcodes"
        src = sources[0]
        assert "np.load(filename, mmap_mode='r')[indices]" in src, (
            "get_only_barcodes must load via mmap_mode='r' to avoid materialising full array"
        )

    def test_get_only_barcodes_old_signature_gone(self):
        """Old get_only_barcodes(adj_matricies, ...) signature must not appear."""
        all_sources = " ".join(_get_code_source(self.nb))
        assert "def get_only_barcodes(adj_matricies," not in all_sources, (
            "Old get_only_barcodes signature with adj_matricies must be removed"
        )

    def test_barcode_loop_no_parent_np_load(self):
        """Barcode loop cell must not call np.load in the parent (pre-load removed).

        The barcode loop cell is identified by `queue = Queue()` which only
        appears in that cell, not in the helper-functions cell above it.
        """
        sources = _find_source_containing(self.nb, "queue = Queue()")
        assert sources, "No cell with queue = Queue() (barcode loop)"
        src = sources[0]
        assert "np.load(filename, allow_pickle=True)" not in src, (
            "Parent-side np.load(filename, allow_pickle=True) must be removed from barcode loop"
        )

    def test_barcode_loop_no_split_matricies_and_lengths(self):
        """Barcode loop must not call split_matricies_and_lengths (fancy-index copies gone).

        Uses queue = Queue() as the unique marker for the loop cell; the
        helper-functions cell above it also contains 'number_of_splits'.
        """
        sources = _find_source_containing(self.nb, "queue = Queue()")
        assert sources, "No cell with queue = Queue() (barcode loop)"
        src = sources[0]
        assert "split_matricies_and_lengths(" not in src, (
            "split_matricies_and_lengths builds fancy-index copies; must not be called from loop"
        )

    def test_barcode_loop_passes_filename_and_indices(self):
        """Barcode loop must pass (filename, indices, ...) to each Process."""
        sources = _find_source_containing(self.nb, "queue = Queue()")
        assert sources, "No cell with queue = Queue() (barcode loop)"
        src = sources[0]
        assert "filename, indices," in src, (
            "Barcode loop must pass filename and indices to the worker Process"
        )

    def test_barcode_loop_passes_ntokens_sliced_by_indices(self):
        """Barcode loop must pass ntokens[indices], not the full ntokens array.

        The whole point of the refactor is that the child receives only the
        slice it needs.  Passing ntokens (unsliced) would silently regress the
        semantic intent even though filename and indices are present.
        """
        sources = _find_source_containing(self.nb, "queue = Queue()")
        assert sources, "No cell with queue = Queue() (barcode loop)"
        src = sources[0]
        assert "ntokens[indices]" in src, (
            "Barcode loop must pass ntokens[indices] (sliced), not the full ntokens array"
        )

    def test_number_of_splits_is_20(self):
        """number_of_splits must be 20 (bumped for production from the reference value of 2)."""
        sources = _find_source_containing(self.nb, "queue = Queue()")
        assert sources, "No cell with queue = Queue() (barcode loop)"
        src = sources[0]
        assert "number_of_splits = 20" in src, (
            "number_of_splits must be 20"
        )

    def test_idempotent(self, tmp_path):
        """Applying patch twice produces identical notebook bytes."""
        dst = tmp_path / "features_calculation_ripser_and_templates.ipynb"
        shutil.copy(REF_RIPSER_NB, dst)
        mod = _load_patch_module()
        changed1 = mod.patch(dst)
        bytes_after_first = dst.read_bytes()
        changed2 = mod.patch(dst)
        bytes_after_second = dst.read_bytes()
        assert changed1 is True
        assert changed2 is False, "Second patch run should be a no-op"
        assert bytes_after_first == bytes_after_second, (
            "Byte content must be identical after first and second patch runs"
        )

    def test_disk_budget_and_cleanup_not_added_to_ripser(self, tmp_path):
        """Disk-budget markdown and cleanup cell are threshold-notebook-only;
        the ripser notebook must not receive them."""
        all_sources = " ".join(c.source for c in self.nb.cells)
        assert self.mod.DISK_BUDGET_MD_MARKER not in all_sources
        assert self.mod.CLEANUP_CELL_MARKER not in all_sources


# ---------------------------------------------------------------------------
# Dispatcher: unknown notebook name must raise ValueError
# ---------------------------------------------------------------------------

class TestDispatcherRejectsUnknownNotebook:
    """patch() must fail loudly for unsupported notebook names."""

    def test_unknown_notebook_raises_value_error(self, tmp_path):
        """patch() on a truly unknown notebook basename must raise ValueError."""
        # Use REF_THRESHOLD_NB content but save under a name not in _SUPPORTED
        src = REF_THRESHOLD_NB
        dst = tmp_path / "totally_unknown_notebook.ipynb"
        shutil.copy(src, dst)
        mod = _load_patch_module()
        with pytest.raises(ValueError, match="No patch set defined for notebook"):
            mod.patch(dst)

    def test_unknown_notebook_cli_exits_nonzero(self, tmp_path):
        """CLI on a truly unknown notebook basename must exit non-zero."""
        src = REF_THRESHOLD_NB
        dst = tmp_path / "totally_unknown_notebook.ipynb"
        shutil.copy(src, dst)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--notebook", str(dst)],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_ripser_notebook_is_now_supported(self, tmp_path):
        """features_calculation_ripser_and_templates.ipynb must now be in _SUPPORTED."""
        dst = tmp_path / "features_calculation_ripser_and_templates.ipynb"
        shutil.copy(REF_RIPSER_NB, dst)
        mod = _load_patch_module()
        # Must NOT raise ValueError — ripser notebook is now supported
        changed = mod.patch(dst)
        assert isinstance(changed, bool)


# ---------------------------------------------------------------------------
# CLI: --notebook argument
# ---------------------------------------------------------------------------

class TestCLI:
    """patch_notebook.py should accept a --notebook CLI argument."""

    def test_cli_accepts_notebook_arg_threshold(self, tmp_path):
        """Running with --notebook pointing to threshold nb should patch content."""
        dst = tmp_path / "features_calculation_by_thresholds.ipynb"
        shutil.copy(REF_THRESHOLD_NB, dst)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--notebook", str(dst)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        # Verify post-patch content is present
        nb = _load_nb(dst)
        all_sources = " ".join(_get_code_source(nb))
        assert (
            "from grab_weights_compat import grab_attention_weights" in all_sources
            or 'input_dir = "../data/processed/"' in all_sources
        ), "Expected at least one post-patch string in patched threshold notebook"

    def test_cli_accepts_notebook_arg_prediction(self, tmp_path):
        """Running with --notebook pointing to prediction nb should succeed."""
        dst = tmp_path / "features_prediction.ipynb"
        shutil.copy(REF_PREDICTION_NB, dst)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--notebook", str(dst)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_missing_notebook_exits_nonzero(self, tmp_path):
        """Running with --notebook pointing to nonexistent file exits nonzero."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--notebook", str(tmp_path / "nonexistent.ipynb")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
