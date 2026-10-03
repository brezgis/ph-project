"""Install ph-project-capped Jupyter kernel spec to user dir.

Writes ~/.local/share/jupyter/kernels/ph-project-capped/kernel.json
with argv pointing at launch_kernel.sh (absolute path, resolved at
install time) and display_name="Python 3 (ph-project, capped)".

Idempotent: re-running overwrites with the same content.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "replication" / "scripts" / "launch_kernel.sh"
KERNEL_DIR = (
    Path.home() / ".local" / "share" / "jupyter" / "kernels" / "ph-project-capped"
)


def install(
    target_dir: Path = KERNEL_DIR, launcher: Path = LAUNCHER
) -> Path:
    """Write kernel.json into *target_dir*, pointing at *launcher*.

    Returns the path to the written kernel.json file.
    Exits with a non-zero status if the launcher is missing or not executable.
    """
    if not launcher.exists():
        sys.exit(f"Launcher not found: {launcher}")
    if not os.access(launcher, os.X_OK):
        sys.exit(f"Launcher not executable: {launcher}")

    target_dir.mkdir(parents=True, exist_ok=True)

    spec = {
        "argv": [str(launcher), "-f", "{connection_file}"],
        "display_name": "Python 3 (ph-project, capped)",
        "language": "python",
    }

    target = target_dir / "kernel.json"
    target.write_text(json.dumps(spec, indent=2) + "\n")
    return target


if __name__ == "__main__":
    path = install()
    print(f"Installed kernel spec at {path}")
