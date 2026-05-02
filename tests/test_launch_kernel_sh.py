"""Tests for replication/scripts/launch_kernel.sh.

Tests cover:
- Script exists and is executable
- Bash syntax is valid (bash -n)
- cgroup v2 guard: script exits 1 with a clear error message on a v1-
  looking system (simulated by overriding stat output via a shim)
- cgroup v2 guard: script does NOT bail on a cgroup2fs host (i.e. this
  machine, north)
- Default and custom MemoryMax / MemorySwapMax via PH_KERNEL_MEM_MAX /
  PH_KERNEL_SWAP_MAX env vars
- "$@" forwarding to ipykernel_launcher
- venv python is correctly resolved (present -> proceeds; missing -> exit 1)

Note: the PATH-prepend shim trick only intercepts commands looked up via
PATH (stat, systemd-run). It will NOT intercept the launcher's $VENV_PYTHON
because that is an ABSOLUTE path computed at runtime. To test venv resolution
and the "venv python missing" failure case, we copy launch_kernel.sh into a
tmp directory tree that has (or lacks) a fake .venv/bin/python alongside it,
then run the copy. systemd-run is still PATH-shimmed so no real scope is
created.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "replication"
    / "scripts"
    / "launch_kernel.sh"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shim_dir(
    *,
    stat_shim: str | None = None,
    echo_systemd_run: bool = True,
) -> Path:
    """Create a temp dir with optional stat and systemd-run shims."""
    shim_dir = Path(tempfile.mkdtemp(prefix="shim_"))
    shim_dir.chmod(0o755)

    if echo_systemd_run:
        sdr = shim_dir / "systemd-run"
        sdr.write_text("#!/usr/bin/env bash\necho systemd-run \"$@\"\n")
        sdr.chmod(0o755)

    if stat_shim is not None:
        stat_bin = shim_dir / "stat"
        stat_bin.write_text(
            textwrap.dedent(
                f"""\
                #!/usr/bin/env bash
                # Shim: always return the configured fstype string.
                echo "{stat_shim}"
                """
            )
        )
        stat_bin.chmod(0o755)

    return shim_dir


def _run_script(
    args: list[str] | None = None,
    env_override: dict[str, str] | None = None,
    stat_shim: str | None = None,
    echo_only: bool = True,
) -> subprocess.CompletedProcess:
    """Run launch_kernel.sh in a controlled way.

    If *echo_only* is True (default), systemd-run is shimmed so the script
    prints its arguments instead of actually launching a kernel.

    If *stat_shim* is a non-empty string it is prepended to PATH as a tiny
    ``stat`` replacement to simulate a non-cgroup2fs system.

    NOTE: this helper runs the REAL script at SCRIPT path, which resolves
    $VENV_PYTHON relative to the script's own location. The real worktree
    has .venv/bin/python (symlinked), so venv resolution succeeds. For tests
    that need to control venv presence/absence, use _run_script_copy instead.
    """
    env = dict(os.environ)
    if env_override:
        env.update(env_override)

    shim_dir: Path | None = None
    if echo_only or stat_shim is not None:
        shim_dir = _make_shim_dir(
            stat_shim=stat_shim,
            echo_systemd_run=echo_only,
        )
        env["PATH"] = str(shim_dir) + ":" + env.get("PATH", "")

    result = subprocess.run(
        ["bash", str(SCRIPT)] + (args or []),
        capture_output=True,
        text=True,
        env=env,
    )

    if shim_dir is not None:
        shutil.rmtree(shim_dir, ignore_errors=True)

    return result


def _run_script_copy(
    tmp_path: Path,
    *,
    venv_present: bool,
    args: list[str] | None = None,
    env_override: dict[str, str] | None = None,
    stat_shim: str | None = None,
) -> subprocess.CompletedProcess:
    """Copy launch_kernel.sh into a tmp tree and run the copy.

    This is needed because $VENV_PYTHON is resolved as an absolute path
    relative to the script's own location, so PATH shims can't intercept it.
    We build a fake project tree under tmp_path:

        tmp_path/
        ├── replication/scripts/launch_kernel.sh   (copy)
        └── .venv/bin/python                       (fake, if venv_present)
    """
    scripts_dir = tmp_path / "replication" / "scripts"
    scripts_dir.mkdir(parents=True)
    copy = scripts_dir / "launch_kernel.sh"
    shutil.copy2(SCRIPT, copy)
    copy.chmod(0o755)

    if venv_present:
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        fake_python = venv_bin / "python"
        fake_python.write_text("#!/usr/bin/env bash\necho fake-python \"$@\"\n")
        fake_python.chmod(0o755)

    env = dict(os.environ)
    if env_override:
        env.update(env_override)

    shim_dir = _make_shim_dir(
        stat_shim=stat_shim,
        echo_systemd_run=True,
    )
    env["PATH"] = str(shim_dir) + ":" + env.get("PATH", "")

    result = subprocess.run(
        ["bash", str(copy)] + (args or []),
        capture_output=True,
        text=True,
        env=env,
    )

    shutil.rmtree(shim_dir, ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_script_exists():
    assert SCRIPT.exists(), f"Script not found: {SCRIPT}"


def test_script_is_executable():
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, f"Script not executable by owner: {SCRIPT}"


def test_bash_syntax():
    """bash -n must exit 0 -- pure syntax check, no execution."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


def test_cgroup_guard_fails_on_v1_system():
    """Script must exit 1 with a clear error when fstype is not cgroup2fs."""
    result = _run_script(stat_shim="tmpfs")
    assert result.returncode == 1, (
        f"Expected exit 1 on non-cgroup2fs, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "cgroup2fs" in result.stderr, (
        f"Error message should mention cgroup2fs.\nstderr: {result.stderr}"
    )
    assert "ERROR" in result.stderr, (
        f"Error message should contain ERROR.\nstderr: {result.stderr}"
    )


def test_cgroup_guard_passes_on_v2_system():
    """Script must NOT exit 1 on this host (north, cgroup2fs)."""
    result = _run_script()
    assert result.returncode == 0, (
        f"Script bailed on cgroup2fs host (should not).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_default_mem_max_passed_to_systemd_run():
    """Without PH_KERNEL_MEM_MAX set, script must pass MemoryMax=48G."""
    env = dict(os.environ)
    env.pop("PH_KERNEL_MEM_MAX", None)
    result = _run_script(env_override=env)
    assert "MemoryMax=48G" in result.stdout, (
        f"Default MemoryMax not found.\nstdout: {result.stdout}"
    )


def test_default_swap_max_passed_to_systemd_run():
    """Without PH_KERNEL_SWAP_MAX set, script must pass MemorySwapMax=4G."""
    env = dict(os.environ)
    env.pop("PH_KERNEL_SWAP_MAX", None)
    result = _run_script(env_override=env)
    assert "MemorySwapMax=4G" in result.stdout, (
        f"Default MemorySwapMax not found.\nstdout: {result.stdout}"
    )


def test_custom_mem_max_env_var():
    """PH_KERNEL_MEM_MAX=32G must override the default."""
    result = _run_script(env_override={"PH_KERNEL_MEM_MAX": "32G"})
    assert result.returncode == 0
    assert "MemoryMax=32G" in result.stdout, (
        f"Custom MemoryMax not found.\nstdout: {result.stdout}"
    )


def test_custom_swap_max_env_var():
    """PH_KERNEL_SWAP_MAX=8G must override the default."""
    result = _run_script(env_override={"PH_KERNEL_SWAP_MAX": "8G"})
    assert result.returncode == 0
    assert "MemorySwapMax=8G" in result.stdout, (
        f"Custom MemorySwapMax not found.\nstdout: {result.stdout}"
    )


def test_extra_args_forwarded_to_ipykernel():
    """Arguments after the script name must be forwarded after ipykernel_launcher."""
    result = _run_script(args=["-f", "/tmp/conn.json"])
    assert result.returncode == 0
    assert "-f" in result.stdout, (
        f"Extra args not forwarded.\nstdout: {result.stdout}"
    )
    assert "/tmp/conn.json" in result.stdout, (
        f"Extra args not forwarded.\nstdout: {result.stdout}"
    )
    assert "ipykernel_launcher" in result.stdout, (
        f"ipykernel_launcher not in command.\nstdout: {result.stdout}"
    )


def test_venv_python_missing(tmp_path):
    """Script must exit 1 with clear error when .venv/bin/python is absent."""
    result = _run_script_copy(tmp_path, venv_present=False)
    assert result.returncode == 1, (
        f"Expected exit 1 when venv python missing, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "venv" in result.stderr.lower() or "python" in result.stderr.lower(), (
        f"Error message should mention venv or python.\nstderr: {result.stderr}"
    )


def test_venv_python_resolved_correctly(tmp_path):
    """Script must find and exec through the venv python when it exists."""
    result = _run_script_copy(tmp_path, venv_present=True)
    assert result.returncode == 0, (
        f"Script failed with venv python present.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # The systemd-run shim echoes; the venv python path should be in the
    # output because it's passed as an argument to systemd-run.
    venv_python_path = str(tmp_path / ".venv" / "bin" / "python")
    assert venv_python_path in result.stdout, (
        f"Expected venv python path in systemd-run args.\nstdout: {result.stdout}"
    )
