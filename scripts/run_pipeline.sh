#!/usr/bin/env bash
# Run the whole analysis, stage by stage. Each stage can be run on its own:
#
#   bash scripts/run_pipeline.sh                 # everything, in order
#   bash scripts/run_pipeline.sh attention ripser   # just those stages
#
# Stages: kwic, attention, ripser, distances, stats, baselines
#   kwic       KWIC extraction from the Leipzig corpora (needs scripts/download_leipzig.sh)
#   attention  mBERT attention + threshold features, per language (GPU)
#   ripser     barcodes, ripser and template features, per language (GPU)
#   distances  pairwise persistence-diagram distances (CPU, about an hour per metric at 8 jobs)
#   stats      permutation tests, controls and figures
#   baselines  fastText baselines (needs scripts/download_fasttext.sh)
#
# Executed notebooks are written to logs/executed/<notebook>_<lang>.ipynb;
# the Russian run of each attention notebook is also saved back in place.
# N_JOBS (default 4) caps parallelism for the CPU-heavy stages.
set -euo pipefail

cd "$(dirname "$0")/.."
LANGS=(en es ru)
N_JOBS="${N_JOBS:-4}"
EXEC_DIR=logs/executed
mkdir -p "$EXEC_DIR"

run_nb() {  # run_nb <notebook path> [lang]; extra env passes through
    local nb="$1" lang="${2:-}" name
    name="$(basename "$nb" .ipynb)${lang:+_$lang}"
    echo "=== $(date +%H:%M:%S) $name"
    PH_LANG="${lang:-ru}" PH_WORKERS="$N_JOBS" uv run jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
        --output-dir "$EXEC_DIR" --output "$name" "$nb"
}

stage_kwic() {
    for lang in "${LANGS[@]}"; do
        uv run python scripts/extract_kwic.py --lang "$lang" --domain color
    done
    run_nb notebooks/phase1_kwic_validation.ipynb
}

stage_attention() {
    for lang in "${LANGS[@]}"; do run_nb notebooks/mbert_attention_thresholds.ipynb "$lang"; done
    cp "$EXEC_DIR/mbert_attention_thresholds_ru.ipynb" notebooks/mbert_attention_thresholds.ipynb
}

stage_ripser() {
    for lang in "${LANGS[@]}"; do run_nb notebooks/mbert_attention_ripser.ipynb "$lang"; done
    cp "$EXEC_DIR/mbert_attention_ripser_ru.ipynb" notebooks/mbert_attention_ripser.ipynb
}

stage_distances() {
    uv run python scripts/compute_diagram_distances.py --n-jobs "$N_JOBS"
}

stage_stats() {
    run_nb notebooks/phase3_diagram_distances.ipynb
    run_nb notebooks/phase3_comparison.ipynb
    cp "$EXEC_DIR/phase3_diagram_distances.ipynb" "$EXEC_DIR/phase3_comparison.ipynb" notebooks/
    for metric in wasserstein bottleneck; do
        for dim in 0 1; do
            uv run python scripts/within_language_null.py --metric "$metric" --dim "$dim"
        done
    done
    uv run python scripts/within_language_null_figure.py
    uv run python scripts/length_matched_control.py --metric wasserstein
    uv run python scripts/length_matched_control.py --metric bottleneck
    uv run python scripts/kushnareva_sum_bars_per_lang.py
    uv run python scripts/kushnareva_sum_bars_per_color.py
}

stage_baselines() {
    for nb in baseline_A_descriptive baseline_B_topological baseline_C_aligned baseline_comparison; do
        run_nb "notebooks/$nb.ipynb"
        cp "$EXEC_DIR/$nb.ipynb" notebooks/
    done
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(kwic attention ripser distances stats baselines)
for s in "${stages[@]}"; do
    echo "##### stage: $s ($(date '+%F %H:%M'))"
    "stage_$s"
done
echo "##### done ($(date '+%F %H:%M'))"
