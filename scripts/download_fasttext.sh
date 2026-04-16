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
    ["cc.en.300.bin"]="TODO"
    ["cc.ru.300.bin"]="TODO"
    ["cc.es.300.bin"]="TODO"
    ["wiki.multi.en.vec"]="TODO"
    ["wiki.multi.ru.vec"]="TODO"
    ["wiki.multi.es.vec"]="TODO"
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
            echo "  actual:   $actual"  >&2
            exit 1
        fi
    fi
}

download_file() {
    local url="$1"
    local dest="$2"
    local sha_key="$3"

    local basename
    basename="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        echo "  Already present: $dest"
        verify_or_print "$dest" "${EXPECTED_SHA256[$sha_key]}"
        return 0
    fi

    echo "  Downloading: $url"
    echo "  -> $dest"
    # Use wget with progress bar; fall back to curl if wget is unavailable.
    if command -v wget &>/dev/null; then
        wget --progress=bar:force -O "$dest" "$url"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi

    verify_or_print "$dest" "${EXPECTED_SHA256[$sha_key]}"
}

# -----------------------------------------------------------------------
# CC-300 (.bin) — fasttext.cc
# -----------------------------------------------------------------------

echo ""
echo "=== CC-300 vectors (fasttext.cc) ==="

CC_BASE_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl"

download_file \
    "$CC_BASE_URL/cc.en.300.bin.gz" \
    "$CC_DIR/cc.en.300.bin.gz" \
    "cc.en.300.bin"

download_file \
    "$CC_BASE_URL/cc.ru.300.bin.gz" \
    "$CC_DIR/cc.ru.300.bin.gz" \
    "cc.ru.300.bin"

download_file \
    "$CC_BASE_URL/cc.es.300.bin.gz" \
    "$CC_DIR/cc.es.300.bin.gz" \
    "cc.es.300.bin"

# Decompress if .gz exists but .bin does not
for lang in en ru es; do
    gz="$CC_DIR/cc.$lang.300.bin.gz"
    bin="$CC_DIR/cc.$lang.300.bin"
    if [[ -f "$gz" && ! -f "$bin" ]]; then
        echo "  Decompressing $gz ..."
        gunzip -k "$gz"
    fi
done

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
