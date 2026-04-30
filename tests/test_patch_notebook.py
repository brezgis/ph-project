"""Tests for replication/scripts/patch_notebook.py.

Covers:
- patch() applied to features_calculation_by_thresholds.ipynb (already patched in 4zo)
- patch() applied to features_prediction.ipynb (new in 3nh)
- Idempotency: re-running patch() leaves notebooks unchanged
- CLI: patch_notebook.py accepts a --notebook argument
"""

from __future__ import annotations

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
# reference/ notebooks are the frozen originals — always unpatched.
REF_PREDICTION_NB = REPO_ROOT / "reference" / "features_prediction.ipynb"
REF_THRESHOLD_NB = REPO_ROOT / "reference" / "features_calculation_by_thresholds.ipynb"


def _load_nb(path: Path) -> nbformat.NotebookNode:
    return nbformat.read(path, as_version=4)


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
        # Import patch() directly from the script
        import importlib.util
        spec = importlib.util.spec_from_file_location("patch_notebook", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
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
        """Applying patch twice produces identical notebook.

        Uses the frozen reference/ original (always unpatched) as the source
        so changed1 is reliably True regardless of the state of the working
        replication/ notebook.
        """
        dst = tmp_path / "features_prediction_idem.ipynb"
        shutil.copy(REF_PREDICTION_NB, dst)
        import importlib.util
        spec = importlib.util.spec_from_file_location("patch_notebook", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        changed1 = mod.patch(dst)
        changed2 = mod.patch(dst)
        assert changed1 is True
        assert changed2 is False, "Second patch run should be a no-op"


# ---------------------------------------------------------------------------
# CLI: --notebook argument
# ---------------------------------------------------------------------------

class TestCLI:
    """patch_notebook.py should accept a --notebook CLI argument."""

    def test_cli_accepts_notebook_arg_threshold(self, tmp_path):
        """Running with --notebook pointing to threshold nb should succeed."""
        dst = tmp_path / "threshold.ipynb"
        shutil.copy(REF_THRESHOLD_NB, dst)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--notebook", str(dst)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_accepts_notebook_arg_prediction(self, tmp_path):
        """Running with --notebook pointing to prediction nb should succeed."""
        dst = tmp_path / "prediction.ipynb"
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
