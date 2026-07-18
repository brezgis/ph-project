#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# cgroup v2 guard — same rationale as launch_jupyter.sh.
# systemd-run -p MemorySwapMax requires cgroup v2. Bail early on
# systems where /sys/fs/cgroup is not cgroup2fs (would silently
# miss the swap cap, leaving the host exposed).
if [ "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)" != "cgroup2fs" ]; then
  echo "ERROR: /sys/fs/cgroup is not cgroup2fs; this kernel launcher requires cgroup v2." >&2
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "ERROR: venv python not found at $VENV_PYTHON" >&2
  exit 1
fi

# Cap kernel memory so a runaway notebook kills the kernel cleanly
# without freezing the host. 48 GB leaves headroom for the desktop session
# on a 64 GB box; tune via PH_KERNEL_MEM_MAX / PH_KERNEL_SWAP_MAX.
MEM_MAX="${PH_KERNEL_MEM_MAX:-48G}"
SWAP_MAX="${PH_KERNEL_SWAP_MAX:-4G}"

exec systemd-run --user --scope --quiet \
  -p "MemoryMax=${MEM_MAX}" \
  -p "MemorySwapMax=${SWAP_MAX}" \
  "$VENV_PYTHON" -m ipykernel_launcher "$@"
