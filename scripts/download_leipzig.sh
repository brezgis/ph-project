#!/usr/bin/env bash
# scripts/download_leipzig.sh
# -----------------------------------------------------------------------
# Reproducible download of multi-year Leipzig 1M news corpora used in
# ph-project Phase 1 (KWIC extraction).
#
# Corpus set: years {2019, 2020, 2023} × langs {en, ru, es} = 9 corpora.
# Each year contributes ~1M sentences; the full set is ~3M per language.
# Year coverage is symmetric across all three languages (the only set with
# full overlap on Leipzig), keeping cross-linguistic comparisons clean.
#
# Leipzig corpora
# ---------------
# Pinned corpus IDs (1M-sentence news):
#   eng_news_2019_1M  — English 2019
#   eng_news_2020_1M  — English 2020
#   eng_news_2023_1M  — English 2023
#   rus_news_2019_1M  — Russian 2019
#   rus_news_2020_1M  — Russian 2020
#   rus_news_2023_1M  — Russian 2023
#   spa_news_2019_1M  — Spanish 2019
#   spa_news_2020_1M  — Spanish 2020
#   spa_news_2023_1M  — Spanish 2023
#
# URL pattern:
#   https://downloads.wortschatz-leipzig.de/corpora/<ID>.tar.gz
#
# Files produced
# --------------
# Tarballs (kept for re-extract-without-re-download):
#   data/leipzig/_downloads/<ID>.tar.gz  (~200 MB each, ~1.8 GB total)
#
# Extracted sentences files:
#   data/leipzig/en/eng_news_{2019,2020,2023}_1M-sentences.txt
#   data/leipzig/ru/rus_news_{2019,2020,2023}_1M-sentences.txt
#   data/leipzig/es/spa_news_{2019,2020,2023}_1M-sentences.txt
#
# SHA-256 verification
# --------------------
# Expected SHA-256 values are stored in EXPECTED_SHA256 below.
# If a value is still a TODO placeholder this script computes and PRINTS
# the actual hash of the downloaded tarball rather than silently accepting
# it.  Populate the values after the first successful download to pin them.
# All 9 hashes are currently pinned; TODO handling is retained so a future
# corpus addition can be verified before its hash is captured.
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

# SYNC NOTE: CORPUS_IDS must stay in sync with CORPUS_SOURCE_IDS in
# phase1_kwic/extract.py (Python dict, canonical source of truth).
# If you add or rename a corpus here, update that dict too.
CORPUS_IDS=(
    "eng_news_2019_1M"
    "eng_news_2020_1M"
    "eng_news_2023_1M"
    "rus_news_2019_1M"
    "rus_news_2020_1M"
    "rus_news_2023_1M"
    "spa_news_2019_1M"
    "spa_news_2020_1M"
    "spa_news_2023_1M"
)

LANG_DIRS=(
    "en"
    "en"
    "en"
    "ru"
    "ru"
    "ru"
    "es"
    "es"
    "es"
)

BASE_URL="https://downloads.wortschatz-leipzig.de/corpora"

# -----------------------------------------------------------------------
# SHA-256 expected values.
# Set to "TODO" until confirmed.  Script will compute and print if TODO.
# Populate these after the first run.
# -----------------------------------------------------------------------
declare -A EXPECTED_SHA256=(
    # 2019 corpora
    ["eng_news_2019_1M"]="6270ec2e22248e7f261505c5b7c26b55cea12cd4ef84e90f269f8eab4dc48ede"
    ["rus_news_2019_1M"]="93fed13d5d32bf491149223eac593fbc9fecb4a88e73905455e418a4bdd20012"
    ["spa_news_2019_1M"]="e58887b9e766c433f6a6c5ae2d87e892609f0595157b23f4ede0290c808dcfbc"
    # 2020 corpora
    ["eng_news_2020_1M"]="be782eb82690415241d623fd2448dfd3fc68102ac1ce971107cb130420abbb41"
    ["rus_news_2020_1M"]="f522a9cccc1d63a5f2ccf11a47e144dd5abd1c840e8ccfb90c249630aaad4657"
    ["spa_news_2020_1M"]="89fb1319f53b341466c065152467cb1bd3789a3ed9aa143807b1952503ef1d50"
    # 2023 corpora
    ["eng_news_2023_1M"]="c8a5a5e72897aa5e367b0319c1884831c02aaf29bf81342de31ca1b1cc8f3e4c"
    ["rus_news_2023_1M"]="9e09e5298f4a2a2dffed14d00478bbe61f02e53647986150879520c877e8f76d"
    ["spa_news_2023_1M"]="6a720452204673e44e16e129c49a9815291b6664e1585763b25adcf5c1d8b25b"
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
echo "=== Leipzig multi-year 1M news corpora download (2019 / 2020 / 2023) ==="
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
