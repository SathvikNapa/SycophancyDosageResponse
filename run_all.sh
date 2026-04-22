#!/usr/bin/env bash
# run_all.sh — End-to-end pipeline: baseline -> sycophancy -> zip results.
#
# Usage:
#   bash run_all.sh [OPTIONS]
#
# Options:
#   --model          Model key from config.MODELS      (default: GPT5_4Nano)
#   --n_attempts     Samples per question              (default: 10)
 #   --n_per_cat      Questions per MMLU-Pro category   (default: 30)
#   --n_syco_samples Repeated sycophancy runs per bin  (default: 5)
#   --n_bins         Number of entropy bins            (default: 5)
#   --bin_strategy   uniform or quantile               (default: quantile)
#
# Examples:
#   bash run_all.sh
#   bash run_all.sh --model ClaudeSonnet
#   bash run_all.sh --model llama3 --n_attempts 10 --n_syco_samples 3 --bin_strategy uniform

set -euo pipefail

# Defaults
MODEL="GPT5_4Nano"
N_ATTEMPTS=30`
N_PER_CAT=30
N_SYCO_SAMPLES=5
N_BINS=5
BIN_STRATEGY="quantile"

# Parse named args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --n_attempts)     N_ATTEMPTS="$2";     shift 2 ;;
        --n_per_cat)      N_PER_CAT="$2";      shift 2 ;;
        --n_syco_samples) N_SYCO_SAMPLES="$2"; shift 2 ;;
        --n_bins)         N_BINS="$2";         shift 2 ;;
        --bin_strategy)   BIN_STRATEGY="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "######################################"
echo "#  STEP 1 — Baseline experiment      #"
echo "######################################"
bash "$SCRIPT_DIR/run_baseline.sh" \
    --model        "$MODEL"        \
    --n_attempts   "$N_ATTEMPTS"   \
    --n_per_cat    "$N_PER_CAT"    \
    --n_bins       "$N_BINS"       \
    --bin_strategy "$BIN_STRATEGY"

echo ""
echo "######################################"
echo "#  STEP 2 — Sycophancy experiment    #"
echo "######################################"
bash "$SCRIPT_DIR/run_sycophancy.sh" \
    --model          "$MODEL"        \
    --n_syco_samples "$N_SYCO_SAMPLES" \
    --n_bins         "$N_BINS"       \
    --bin_strategy   "$BIN_STRATEGY"

echo ""
echo "######################################"
echo "#  STEP 3 — Package results          #"
echo "######################################"
bash "$SCRIPT_DIR/package_results.sh" --model "$MODEL"

echo ""
echo "All done. Archive: results_${MODEL}.zip"
