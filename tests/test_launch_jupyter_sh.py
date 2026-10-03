"""Tests for replication/scripts/launch_jupyter.sh.

Tests cover:
- Script is executable
- Bash syntax is valid (bash -n)
- cgroup v2 guard: script exits 1 with a clear error message on a v1-
  looking system (simulated by overriding stat output via a shim)
- cgroup v2 guard: script does NOT bail on a cgroup2fs host (i.e. the
  development machine)
- JUPYTER_MEM_MAX / JUPYTER_SWAP_MAX env vars are threaded through to
  the systemd-run invocation

Note: live "launch Jupyter and OOM a cell" and "systemctl shows
MemoryMax=48G" tests are manual — flagged in the PR for the user to run.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path
import textwrap

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "replication" / "scripts" / "launch_jupyter.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_script(
    args: list[str] | None = None,
    env_override: dict[str, str] | None = None,
    stat_shim: str | None = None,
    echo_only: bool = True,
) -> subprocess.CompletedProcess:
    """Run launch_jupyter.sh in a controlled way.

    If *echo_only* is True (the default), ``systemd-run`` is shimmed so
    the script prints its arguments instead of actually launching Jupyter.
    This makes the test hermetic.

    If *stat_shim* is a non-empty string it is prepended to PATH as a
    tiny ``stat`` replacement so we can simulate a non-cgroup2fs system
    without touching the real /sys/fs/cgroup.
    """
    env = dict(os.environ)
    if env_override:
        env.update(env_override)

    shim_dir: Path | None = None

    if echo_only or stat_shim is not None:
        import tempfile

        shim_dir = Path(tempfile.mkdtemp(prefix="shim_"))
        shim_dir.chmod(0o755)

        if echo_only:
            # Shim systemd-run: echo its args rather than running them.
            sdr = shim_dir / "systemd-run"
            sdr.write_text("#!/usr/bin/env bash\necho systemd-run \"$@\"\n")
            sdr.chmod(0o755)
            # Also shim jupyter so the script can reach exec without error.
            jup = shim_dir / "jupyter"
            jup.write_text("#!/usr/bin/env bash\necho jupyter \"$@\"\n")
            jup.chmod(0o755)

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

        env["PATH"] = str(shim_dir) + ":" + env.get("PATH", "")

    result = subprocess.run(
        ["bash", str(SCRIPT)] + (args or []),
        capture_output=True,
        text=True,
        env=env,
    )

    # Clean up shim dir.
    if shim_dir is not None:
        import shutil

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
    """bash -n must exit 0 — pure syntax check, no execution."""
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"bash -n failed:\n{result.stderr}"
    )


def test_cgroup_guard_fails_on_v1_system():
    """Script must exit 1 with a clear error when fstype is not cgroup2fs."""
    result = _run_script(stat_shim="tmpfs")  # not cgroup2fs
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
    """Script must NOT exit 1 on a cgroup2fs host."""
    result = _run_script()  # uses real /sys/fs/cgroup stat
    assert result.returncode == 0, (
        f"Script bailed on cgroup2fs host (should not).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_default_mem_max_passed_to_systemd_run():
    """Without JUPYTER_MEM_MAX set, script must pass MemoryMax=48G."""
    env = dict(os.environ)
    env.pop("JUPYTER_MEM_MAX", None)
    result = _run_script(env_override=env)
    assert "MemoryMax=48G" in result.stdout, (
        f"Default MemoryMax not found.\nstdout: {result.stdout}"
    )


def test_default_swap_max_passed_to_systemd_run():
    """Without JUPYTER_SWAP_MAX set, script must pass MemorySwapMax=4G."""
    env = dict(os.environ)
    env.pop("JUPYTER_SWAP_MAX", None)
    result = _run_script(env_override=env)
    assert "MemorySwapMax=4G" in result.stdout, (
        f"Default MemorySwapMax not found.\nstdout: {result.stdout}"
    )


def test_custom_mem_max_env_var():
    """JUPYTER_MEM_MAX=32G must override the default."""
    result = _run_script(env_override={"JUPYTER_MEM_MAX": "32G"})
    assert result.returncode == 0
    assert "MemoryMax=32G" in result.stdout, (
        f"Custom MemoryMax not found.\nstdout: {result.stdout}"
    )


def test_custom_swap_max_env_var():
    """JUPYTER_SWAP_MAX=8G must override the default."""
    result = _run_script(env_override={"JUPYTER_SWAP_MAX": "8G"})
    assert result.returncode == 0
    assert "MemorySwapMax=8G" in result.stdout, (
        f"Custom MemorySwapMax not found.\nstdout: {result.stdout}"
    )


def test_extra_args_forwarded_to_jupyter():
    """Arguments after the script name must be forwarded to jupyter lab."""
    result = _run_script(args=["--port", "8889"])
    assert result.returncode == 0
    # The shim echoes: systemd-run <flags> jupyter lab --port 8889
    assert "--port" in result.stdout, (
        f"Extra args not forwarded.\nstdout: {result.stdout}"
    )
    assert "8889" in result.stdout, (
        f"Extra args not forwarded.\nstdout: {result.stdout}"
    )
