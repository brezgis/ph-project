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
DOMAINS=(color adjective)   # adjective = the control set (Dixon's non-color core adjectives)
N_JOBS="${N_JOBS:-4}"
EXEC_DIR=logs/executed
mkdir -p "$EXEC_DIR"

run_nb() {  # run_nb <notebook path> [lang] [domain]
    local nb="$1" lang="${2:-}" domain="${3:-color}" name
    name="$(basename "$nb" .ipynb)${lang:+_$lang}"
    if [ "$domain" != color ]; then name="${name}_$domain"; fi
    echo "=== $(date +%H:%M:%S) $name"
    PH_LANG="${lang:-ru}" PH_DOMAIN="$domain" PH_WORKERS="$N_JOBS" uv run jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
        --output-dir "$EXEC_DIR" --output "$name" "$nb"
}

stage_kwic() {
    for domain in "${DOMAINS[@]}"; do
        for lang in "${LANGS[@]}"; do
            uv run python scripts/extract_kwic.py --lang "$lang" --domain "$domain"
        done
    done
    run_nb notebooks/phase1_kwic_validation.ipynb
}

stage_attention() {
    for domain in "${DOMAINS[@]}"; do
        for lang in "${LANGS[@]}"; do run_nb notebooks/mbert_attention_thresholds.ipynb "$lang" "$domain"; done
    done
    cp "$EXEC_DIR/mbert_attention_thresholds_ru.ipynb" notebooks/mbert_attention_thresholds.ipynb
}

stage_ripser() {
    for domain in "${DOMAINS[@]}"; do
        for lang in "${LANGS[@]}"; do run_nb notebooks/mbert_attention_ripser.ipynb "$lang" "$domain"; done
    done
    cp "$EXEC_DIR/mbert_attention_ripser_ru.ipynb" notebooks/mbert_attention_ripser.ipynb
}

stage_distances() {
    for domain in "${DOMAINS[@]}"; do
        uv run python scripts/compute_diagram_distances.py --domain "$domain" --n-jobs "$N_JOBS"
    done
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
    for domain in color adjective; do
        for metric in wasserstein bottleneck; do
            uv run python scripts/language_effect.py --domain "$domain" --metric "$metric"
        done
    done
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
