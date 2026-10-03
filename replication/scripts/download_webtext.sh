#!/usr/bin/env bash
# Download the WebText (human) and GPT-2 Small (machine) splits from OpenAI's
# public gpt-2-output-dataset mirror. Files land in data/raw/ relative to this
# script's parent directory. Re-runnable: skips files already present.
#
# Total download size is roughly 600 MB. Train splits dominate (~250 MB each).
# If you only need a quick first pass, you can comment out the train lines and
# subsample from valid/test in prepare_csv.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/../data/raw"
mkdir -p "${RAW_DIR}"

BASE_URL="https://openaipublic.azureedge.net/gpt-2/output-dataset/v1"

FILES=(
    "webtext.train.jsonl"
    "webtext.valid.jsonl"
    "webtext.test.jsonl"
    "small-117M.train.jsonl"
    "small-117M.valid.jsonl"
    "small-117M.test.jsonl"
)

for f in "${FILES[@]}"; do
    target="${RAW_DIR}/${f}"
    if [[ -s "${target}" ]]; then
        echo "skip ${f} (already present)"
        continue
    fi
    echo "fetch ${f}"
    curl --fail --location --show-error --silent \
        --output "${target}.partial" \
        "${BASE_URL}/${f}"
    mv "${target}.partial" "${target}"
done

echo "done. files in ${RAW_DIR}:"
ls -lh "${RAW_DIR}"
