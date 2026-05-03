#!/usr/bin/env bash
# scripts/download_leipzig.sh
# -----------------------------------------------------------------------
# Reproducible download of Leipzig 1M news corpora used in ph-project
# Phase 1 (KWIC extraction).
#
# Leipzig corpora
# ---------------
# Pinned corpus IDs (1M-sentence news, year 2020):
#   eng_news_2020_1M  — English
#   rus_news_2020_1M  — Russian
#   spa_news_2020_1M  — Spanish
#
# URL pattern:
#   https://downloads.wortschatz-leipzig.de/corpora/<ID>.tar.gz
#
# Files produced
# --------------
# Tarballs (kept for re-extract-without-re-download):
#   data/leipzig/_downloads/eng_news_2020_1M.tar.gz  (~200 MB)
#   data/leipzig/_downloads/rus_news_2020_1M.tar.gz  (~200 MB)
#   data/leipzig/_downloads/spa_news_2020_1M.tar.gz  (~200 MB)
#
# Extracted sentences files:
#   data/leipzig/en/eng_news_2020_1M-sentences.txt
#   data/leipzig/ru/rus_news_2020_1M-sentences.txt
#   data/leipzig/es/spa_news_2020_1M-sentences.txt
#
# SHA-256 verification
# --------------------
# Expected SHA-256 values are stored in EXPECTED_SHA256 below.
# If a value is still a TODO placeholder this script computes and PRINTS
# the actual hash of the downloaded tarball rather than silently accepting
# it.  Populate the values after the first successful download to pin them.
#
# Usage
# -----
#   cd ~/ph-project
#   bash scripts/download_leipzig.sh
#
# Environment variables
# ---------------------
#   LEIPZIG_DATA_DIR  — override the data root (default: data/leipzig
#                       relative to this script's parent directory)
#   SKIP_SHA          — set to "1" to skip sha256 verification entirely
#
# -----------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${LEIPZIG_DATA_DIR:-$REPO_ROOT/data/leipzig}"
SKIP_SHA="${SKIP_SHA:-0}"

DOWNLOADS_DIR="$DATA_DIR/_downloads"

mkdir -p "$DOWNLOADS_DIR"

# -----------------------------------------------------------------------
# Corpus definitions
# -----------------------------------------------------------------------
# Each entry maps: CORPUS_ID -> language dir (en/ru/es)
# Array indices must stay in sync.
# -----------------------------------------------------------------------

# SYNC NOTE: CORPUS_IDS must stay in sync with CORPUS_SOURCE_ID in
# phase1_kwic/extract.py (Python dict, canonical source of truth).
# If you add or rename a corpus here, update that dict too.
CORPUS_IDS=(
    "eng_news_2020_1M"
    "rus_news_2020_1M"
    "spa_news_2020_1M"
)

LANG_DIRS=(
    "en"
    "ru"
    "es"
)

BASE_URL="https://downloads.wortschatz-leipzig.de/corpora"

# -----------------------------------------------------------------------
# SHA-256 expected values.
# Set to "TODO" until confirmed.  Script will compute and print if TODO.
# Populate these after the first run.
# -----------------------------------------------------------------------
declare -A EXPECTED_SHA256=(
    ["eng_news_2020_1M"]="TODO"
    ["rus_news_2020_1M"]="TODO"
    ["spa_news_2020_1M"]="TODO"
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
        # EXPECTED_SHA256 is keyed by corpus ID (e.g., eng_news_2020_1M),
        # not by tarball filename — strip .tar.gz so the printed hint is
        # copy-pasteable into the array.
        local key="${basename%.tar.gz}"
        echo ""
        echo "  *** SHA-256 PLACEHOLDER for $basename ***"
        echo "  Actual sha256: $actual"
        echo "  Record this in scripts/download_leipzig.sh EXPECTED_SHA256:"
        echo "  [\"$key\"]=\"$actual\""
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

# -----------------------------------------------------------------------
# download_corpus CORPUS_ID LANG_DIR
#
# Downloads <CORPUS_ID>.tar.gz into _downloads/ (idempotent: skips if
# tarball already present), then extracts <CORPUS_ID>-sentences.txt to
# data/leipzig/<LANG_DIR>/ (idempotent: skips if sentences file already
# exists and is non-empty).  The tarball is retained in _downloads/ so
# re-extract is possible without re-downloading.
# -----------------------------------------------------------------------

download_corpus() {
    local corpus_id="$1"
    local lang_dir="$2"

    local tarball_url="$BASE_URL/${corpus_id}.tar.gz"
    local tarball_dest="$DOWNLOADS_DIR/${corpus_id}.tar.gz"
    local sentences_dir="$DATA_DIR/$lang_dir"
    local sentences_file="$sentences_dir/${corpus_id}-sentences.txt"

    mkdir -p "$sentences_dir"

    echo ""
    echo "--- $corpus_id ($lang_dir) ---"

    # ------------------------------------------------------------------
    # Step 1: Download tarball (idempotent — skip if already present).
    # ------------------------------------------------------------------
    if [[ -f "$tarball_dest" ]]; then
        echo "  Tarball already present: $tarball_dest"
        verify_or_print "$tarball_dest" "${EXPECTED_SHA256[$corpus_id]}"
    else
        local tarball_tmp="$tarball_dest.tmp"

        # Clean up any leftover .tmp on error.
        trap 'rm -f "$tarball_tmp"' ERR RETURN

        echo "  Downloading: $tarball_url"
        echo "  -> $tarball_dest"
        if command -v wget &>/dev/null; then
            wget --progress=bar:force -O "$tarball_tmp" "$tarball_url"
        else
            curl -L --progress-bar -o "$tarball_tmp" "$tarball_url"
        fi

        mv "$tarball_tmp" "$tarball_dest"
        trap - ERR RETURN

        verify_or_print "$tarball_dest" "${EXPECTED_SHA256[$corpus_id]}"
    fi

    # ------------------------------------------------------------------
    # Step 2: Extract sentences file (idempotent — skip if already
    # present and non-empty).
    # ------------------------------------------------------------------
    if [[ -f "$sentences_file" && -s "$sentences_file" ]]; then
        echo "  Sentences file already present: $sentences_file"
        local line_count
        line_count="$(wc -l < "$sentences_file")"
        local byte_count
        byte_count="$(wc -c < "$sentences_file")"
        echo "  Size: $byte_count bytes, $line_count lines"
        return 0
    fi

    echo "  Extracting ${corpus_id}-sentences.txt from tarball..."
    local sentences_tmp="$sentences_file.tmp"
    trap 'rm -f "$sentences_tmp"' ERR RETURN

    # tar member path inside the archive: <corpus_id>/<corpus_id>-sentences.txt
    # Use --wildcards in case the top-level dir name varies slightly.
    # Extract to stdout and write atomically via .tmp.
    tar -xOf "$tarball_dest" \
        --wildcards "*/${corpus_id}-sentences.txt" \
        > "$sentences_tmp"

    mv "$sentences_tmp" "$sentences_file"
    trap - ERR RETURN

    local line_count
    line_count="$(wc -l < "$sentences_file")"
    local byte_count
    byte_count="$(wc -c < "$sentences_file")"
    echo "  Extracted: $sentences_file"
    echo "  Size: $byte_count bytes, $line_count lines"
}

# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

echo ""
echo "=== Leipzig 1M news corpora download ==="
echo "Data root: $DATA_DIR"

for i in "${!CORPUS_IDS[@]}"; do
    download_corpus "${CORPUS_IDS[$i]}" "${LANG_DIRS[$i]}"
done

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

echo ""
echo "=== Download complete ==="
echo ""
echo "Tarballs in $DOWNLOADS_DIR:"
ls -lh "$DOWNLOADS_DIR" 2>/dev/null || echo "  (empty)"

echo ""
echo "Sentences files:"
for i in "${!CORPUS_IDS[@]}"; do
    local_file="$DATA_DIR/${LANG_DIRS[$i]}/${CORPUS_IDS[$i]}-sentences.txt"
    if [[ -f "$local_file" ]]; then
        local_size="$(wc -c < "$local_file")"
        local_lines="$(wc -l < "$local_file")"
        echo "  ${LANG_DIRS[$i]}: $local_file ($local_size bytes, $local_lines lines)"
    else
        echo "  ${LANG_DIRS[$i]}: NOT FOUND — $local_file"
    fi
done

echo ""
echo "If any SHA-256 values above say 'PLACEHOLDER', update EXPECTED_SHA256"
echo "in scripts/download_leipzig.sh with the printed values and commit."
