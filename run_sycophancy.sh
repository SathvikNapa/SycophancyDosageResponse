#!/usr/bin/env bash
# run_sycophancy.sh — Run the entropy-binned sycophancy pressure experiment.
#
# Usage:
#   bash run_sycophancy.sh [OPTIONS]
#
# Options:
#   --model          Model key from config.MODELS      (default: GPT5_4Nano)
#   --n_syco_samples Repeated runs per bin             (default: 5)
#   --concurrency    Max concurrent API calls          (default: 5)
#   --timeout_s      Per-call timeout in seconds       (default: 120)
#   --base_seed      Base random seed                  (default: 777)
#   --n_bins         Number of entropy bins            (default: 5)
#   --bin_strategy   uniform or quantile               (default: quantile)
#
# Prerequisite:
#   run_baseline.sh must have been run first for the same --model.
#
# Examples:
#   bash run_sycophancy.sh
#   bash run_sycophancy.sh --model ClaudeSonnet --n_syco_samples 5
#   bash run_sycophancy.sh --model llama3 --concurrency 3 --timeout_s 180

set -euo pipefail

# Defaults
MODEL="GPT5_4Nano"
N_SYCO_SAMPLES=5
CONCURRENCY=5
TIMEOUT_S=120
BASE_SEED=777
N_BINS=5
BIN_STRATEGY="quantile"

# Parse named args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --n_syco_samples) N_SYCO_SAMPLES="$2"; shift 2 ;;
        --concurrency)    CONCURRENCY="$2";    shift 2 ;;
        --timeout_s)      TIMEOUT_S="$2";      shift 2 ;;
        --base_seed)      BASE_SEED="$2";      shift 2 ;;
        --n_bins)         N_BINS="$2";         shift 2 ;;
        --bin_strategy)   BIN_STRATEGY="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE_PKL="$SCRIPT_DIR/experiment_out/$MODEL/base_experiment_metadata.pkl"
if [ ! -f "$BASELINE_PKL" ]; then
    echo "ERROR: Baseline metadata not found at $BASELINE_PKL"
    echo "       Run: bash run_baseline.sh --model $MODEL"
    exit 1
fi

echo "============================================"
echo "  Sycophancy Experiment"
echo "  --model          $MODEL"
echo "  --n_syco_samples $N_SYCO_SAMPLES"
echo "  --concurrency    $CONCURRENCY"
echo "  --timeout_s      $TIMEOUT_S"
echo "  --base_seed      $BASE_SEED"
echo "  --n_bins         $N_BINS"
echo "  --bin_strategy   $BIN_STRATEGY"
echo "============================================"

python "$SCRIPT_DIR/run_sycophancy.py" \
    --model          "$MODEL"          \
    --n_syco_samples "$N_SYCO_SAMPLES" \
    --concurrency    "$CONCURRENCY"    \
    --timeout_s      "$TIMEOUT_S"      \
    --base_seed      "$BASE_SEED"      \
    --n_bins         "$N_BINS"         \
    --bin_strategy   "$BIN_STRATEGY"

echo ""
echo "Sycophancy complete. Results in: experiment_out/$MODEL/entropy_bin/"
