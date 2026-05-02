#!/usr/bin/env bash
set -euo pipefail

# systemd-run -p MemorySwapMax requires cgroup v2. Bail early on
# systems where /sys/fs/cgroup is not cgroup2fs (would silently
# miss the swap cap, leaving north exposed).
if [ "$(stat -fc %T /sys/fs/cgroup 2>/dev/null)" != "cgroup2fs" ]; then
  echo "ERROR: /sys/fs/cgroup is not cgroup2fs; this wrapper requires cgroup v2." >&2
  echo "       Either switch to cgroup v2 or invoke jupyter directly without the cap." >&2
  exit 1
fi

# Cap kernel memory so a runaway notebook (e.g. the ripser barcode
# loop) kills the kernel cleanly without freezing north. 48 GB
# leaves headroom for the desktop session on a 64 GB box; tune via
# JUPYTER_MEM_MAX env.
MEM_MAX="${JUPYTER_MEM_MAX:-48G}"
SWAP_MAX="${JUPYTER_SWAP_MAX:-4G}"

exec systemd-run --user --scope --quiet \
  -p "MemoryMax=${MEM_MAX}" \
  -p "MemorySwapMax=${SWAP_MAX}" \
  jupyter lab "$@"
