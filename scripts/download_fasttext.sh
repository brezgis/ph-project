#!/usr/bin/env bash
# scripts/download_fasttext.sh
# -----------------------------------------------------------------------
# Reproducible download of fastText vectors used in ph-project Phase 2.
#
# Files downloaded
# ----------------
# CC-300 binary (.bin) — language-specific, supports subword OOV:
#   data/fasttext/cc/cc.en.300.bin   (~4.2 GB)
#   data/fasttext/cc/cc.ru.300.bin   (~4.2 GB)
#   data/fasttext/cc/cc.es.300.bin   (~4.2 GB)
#   Source: https://fasttext.cc/docs/en/crawl-vectors.html
#
# MUSE supervised-aligned (.vec) — cross-lingual, no subword fallback:
#   data/fasttext/aligned/wiki.multi.en.vec   (~650 MB)
#   data/fasttext/aligned/wiki.multi.ru.vec   (~650 MB)
#   data/fasttext/aligned/wiki.multi.es.vec   (~650 MB)
#   Source: https://github.com/facebookresearch/MUSE#download
#
# SHA-256 verification
# --------------------
# Expected SHA-256 values are stored in data/fasttext/README.md.
# If a value is still a TODO placeholder this script computes and PRINTS
# the actual hash of the downloaded file rather than silently accepting it.
# Populate the README.md values after the first successful download.
#
# Usage
# -----
#   cd ~/ph-project
#   bash scripts/download_fasttext.sh
#
# Environment variables
# ---------------------
#   FASTTEXT_DATA_DIR  — override the data root (default: data/fasttext
#                        relative to this script's parent directory)
#   SKIP_SHA           — set to "1" to skip sha256 verification entirely
#
# -----------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${FASTTEXT_DATA_DIR:-$REPO_ROOT/data/fasttext}"
SKIP_SHA="${SKIP_SHA:-0}"

CC_DIR="$DATA_DIR/cc"
ALIGNED_DIR="$DATA_DIR/aligned"

mkdir -p "$CC_DIR" "$ALIGNED_DIR"

# -----------------------------------------------------------------------
# SHA-256 expected values.
# Set to "TODO" until confirmed.  Script will compute and print if TODO.
# Populate these after the first run.
# -----------------------------------------------------------------------
declare -A EXPECTED_SHA256=(
    ["cc.en.300.bin"]="14c7167b130056944cbdc37b7451f055867fe9a4e3fed3bbc1ecc0e74f6763ca"
    ["cc.ru.300.bin"]="208df9419e13196de5b63008880999ebcf8383d083762c9bb0c210f84f280279"
    ["cc.es.300.bin"]="b8c800affac505d60c8a929cb90ff2ef616b4e8b5224d8f3a0a5e911a8a6546e"
    ["wiki.multi.en.vec"]="b9558d40469e9ed6cb3963cc85b28a2e3841811a7d6d3b9ce3c54bf7784caacd"
    ["wiki.multi.ru.vec"]="0cd989c36691df2d5ea64e0a59b9629df79e2faa48269ff41c2a2210bc6626eb"
    ["wiki.multi.es.vec"]="96f1c2dd9651b653ebcd1fd720a20bb4195f9c457d9df4757768fc03b758a326"
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

verify_or_print() {
    local file="$1"
    local expected="$2"
    local basename
    basename="$(basename "$file")"

    if [[ "$SKIP_SHA" == "1" ]]; then
        echo "  [sha256 skip] SKIP_SHA=1 — not verifying $basename"
        return 0
    fi

    local actual
    actual="$(sha256_of "$file")"

    if [[ "$expected" == "TODO" ]]; then
        echo ""
        echo "  *** SHA-256 PLACEHOLDER for $basename ***"
        echo "  Actual sha256: $actual"
        echo "  Add this line to data/fasttext/README.md:"
        echo "  | $basename | $actual |"
        echo ""
    else
        if [[ "$actual" == "$expected" ]]; then
            echo "  [sha256 ok] $basename"
        else
            echo "ERROR: SHA-256 mismatch for $basename" >&2
            echo "  expected: $expected" >&2
            echo "  actual:   $actual" >&2
            exit 1
        fi
    fi
}

download_file() {
    # download_file URL DEST SHA_KEY
    # Used for files served without compression (MUSE .vec).
    # Downloads URL to DEST.tmp, moves to DEST on success (atomic), then
    # verifies sha256. Idempotent: no-ops if DEST already exists.
    local url="$1"
    local dest="$2"
    local sha_key="$3"

    # Idempotency: check for the FINAL destination file (the .bin or .vec).
    if [[ -f "$dest" ]]; then
        echo "  Already present: $dest"
        verify_or_print "$dest" "${EXPECTED_SHA256[$sha_key]}"
        return 0
    fi

    # Determine the actual download path (may differ from dest for .gz).
    # The caller passes dest as the .bin path; we download the .gz separately.
    # For MUSE .vec files, dest IS the downloaded file (no compression).
    # We always download to a .tmp to avoid leaving partial files.
    local download_dest="$dest.tmp"

    # Clean up any leftover .tmp on error or function return.
    trap 'rm -f "$download_dest"' ERR RETURN

    echo "  Downloading: $url"
    echo "  -> $dest"
    if command -v wget &>/dev/null; then
        wget --progress=bar:force -O "$download_dest" "$url"
    else
        curl -L --progress-bar -o "$download_dest" "$url"
    fi

    mv "$download_dest" "$dest"
    # Trap no longer needs to clean up after successful move.
    trap - ERR RETURN

    verify_or_print "$dest" "${EXPECTED_SHA256[$sha_key]}"
}

# -----------------------------------------------------------------------
# download_cc LANG
#
# Downloads cc.LANG.300.bin.gz (if the final .bin is absent), decompresses
# to cc.LANG.300.bin, verifies sha256 of the .bin, then removes the .gz.
# -----------------------------------------------------------------------

download_cc() {
    local lang="$1"
    local gz_url="$CC_BASE_URL/cc.$lang.300.bin.gz"
    local gz_dest="$CC_DIR/cc.$lang.300.bin.gz"
    local bin_dest="$CC_DIR/cc.$lang.300.bin"
    local sha_key="cc.$lang.300.bin"

    # Idempotency: if final .bin already exists, just verify and skip.
    if [[ -f "$bin_dest" ]]; then
        echo "  Already present: $bin_dest"
        verify_or_print "$bin_dest" "${EXPECTED_SHA256[$sha_key]}"
        return 0
    fi

    # Download the .gz to a .tmp staging file.
    local gz_tmp="$gz_dest.tmp"
    trap 'rm -f "$gz_tmp"' ERR RETURN

    echo "  Downloading: $gz_url"
    echo "  -> $bin_dest (via $gz_dest)"
    if command -v wget &>/dev/null; then
        wget --progress=bar:force -O "$gz_tmp" "$gz_url"
    else
        curl -L --progress-bar -o "$gz_tmp" "$gz_url"
    fi

    # Decompress to .bin (gunzip -c streams to stdout; write atomically via .tmp).
    local bin_tmp="$bin_dest.tmp"
    trap 'rm -f "$gz_tmp" "$bin_tmp"' ERR RETURN
    echo "  Decompressing to $bin_dest ..."
    gunzip -c "$gz_tmp" > "$bin_tmp"
    mv "$bin_tmp" "$bin_dest"
    rm -f "$gz_tmp"
    trap - ERR RETURN

    verify_or_print "$bin_dest" "${EXPECTED_SHA256[$sha_key]}"
}

# -----------------------------------------------------------------------
# CC-300 (.bin) — fasttext.cc
# -----------------------------------------------------------------------

echo ""
echo "=== CC-300 vectors (fasttext.cc) ==="

CC_BASE_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"

download_cc en
download_cc ru
download_cc es

# -----------------------------------------------------------------------
# MUSE aligned (.vec) — github.com/facebookresearch/MUSE
# -----------------------------------------------------------------------

echo ""
echo "=== MUSE supervised-aligned vectors ==="

MUSE_BASE_URL="https://dl.fbaipublicfiles.com/arrival/vectors"

download_file \
    "$MUSE_BASE_URL/wiki.multi.en.vec" \
    "$ALIGNED_DIR/wiki.multi.en.vec" \
    "wiki.multi.en.vec"

download_file \
    "$MUSE_BASE_URL/wiki.multi.ru.vec" \
    "$ALIGNED_DIR/wiki.multi.ru.vec" \
    "wiki.multi.ru.vec"

download_file \
    "$MUSE_BASE_URL/wiki.multi.es.vec" \
    "$ALIGNED_DIR/wiki.multi.es.vec" \
    "wiki.multi.es.vec"

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

echo ""
echo "=== Download complete ==="
echo ""
echo "Files in $CC_DIR:"
ls -lh "$CC_DIR" 2>/dev/null || echo "  (empty)"

echo ""
echo "Files in $ALIGNED_DIR:"
ls -lh "$ALIGNED_DIR" 2>/dev/null || echo "  (empty)"

echo ""
echo "If any SHA-256 values above say 'PLACEHOLDER', copy them into"
echo "data/fasttext/README.md and commit the update."
