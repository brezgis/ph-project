"""Tests for replication/scripts/install_kernel_spec.py.

Tests cover:
- kernel.json is valid JSON
- argv has shape [<launcher>, "-f", "{connection_file}"]
- display_name matches "Python 3 (ph-project, capped)"
- language == "python"
- argv[0] points at an existing executable file
- second install is idempotent (file content identical)
- install fails fast if launcher does not exist
- install fails fast if launcher exists but is not executable
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Add the scripts dir to sys.path so we can import install_kernel_spec.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "replication" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

LAUNCHER = SCRIPTS_DIR / "launch_kernel.sh"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kernel_json_is_valid(tmp_path):
    """install() must produce a valid JSON file."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    kernel_json = target_dir / "kernel.json"
    assert kernel_json.exists(), "kernel.json was not created"
    data = json.loads(kernel_json.read_text())
    assert isinstance(data, dict), "kernel.json root must be an object"


def test_argv_shape(tmp_path):
    """argv must be [<launcher>, "-f", "{connection_file}"]."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    data = json.loads((target_dir / "kernel.json").read_text())
    argv = data["argv"]
    assert len(argv) == 3, f"Expected 3 elements in argv, got {len(argv)}"
    assert argv[1] == "-f"
    assert argv[2] == "{connection_file}"


def test_argv_points_at_existing_executable(tmp_path):
    """argv[0] must point at an existing executable file."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    data = json.loads((target_dir / "kernel.json").read_text())
    launcher_path = Path(data["argv"][0])
    assert launcher_path.exists(), f"Launcher not found: {launcher_path}"
    assert os.access(launcher_path, os.X_OK), f"Launcher not executable: {launcher_path}"


def test_display_name(tmp_path):
    """display_name must be 'Python 3 (ph-project, capped)'."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    data = json.loads((target_dir / "kernel.json").read_text())
    assert data["display_name"] == "Python 3 (ph-project, capped)"


def test_language(tmp_path):
    """language must be 'python'."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    data = json.loads((target_dir / "kernel.json").read_text())
    assert data["language"] == "python"


def test_idempotent_install(tmp_path):
    """Running install() twice must produce identical kernel.json content."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    first = (target_dir / "kernel.json").read_text()
    install_kernel_spec.install(target_dir=target_dir, launcher=LAUNCHER)
    second = (target_dir / "kernel.json").read_text()
    assert first == second, "Second install produced different content"


def test_fails_if_launcher_missing(tmp_path):
    """install() must exit if the launcher path does not exist."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    fake_launcher = tmp_path / "nonexistent" / "launch_kernel.sh"
    with pytest.raises(SystemExit):
        install_kernel_spec.install(target_dir=target_dir, launcher=fake_launcher)


def test_fails_if_launcher_not_executable(tmp_path):
    """install() must exit if the launcher exists but is not executable."""
    import install_kernel_spec

    target_dir = tmp_path / "kernels" / "ph-project-capped"
    fake_launcher = tmp_path / "launch_kernel.sh"
    fake_launcher.write_text("#!/usr/bin/env bash\necho hello\n")
    fake_launcher.chmod(0o644)  # readable but NOT executable
    with pytest.raises(SystemExit):
        install_kernel_spec.install(target_dir=target_dir, launcher=fake_launcher)
