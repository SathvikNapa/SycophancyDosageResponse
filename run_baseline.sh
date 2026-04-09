#!/usr/bin/env bash
# run_baseline.sh — Run the baseline uncertainty experiment.
#
# Usage:
#   bash run_baseline.sh [OPTIONS]
#
# Options:
#   --model          Model key from config.MODELS      (default: GPT5_4Nano)
#   --n_attempts     Samples per question              (default: 10)
#   --n_per_cat      Questions per MMLU-Pro category   (default: 30)
#   --seed           Random seed for dataset sampling  (default: 42)
#   --max_concurrent Max concurrent API requests       (default: 10)
#   --n_bins         Number of entropy bins            (default: 5)
#   --bin_strategy   uniform or quantile               (default: quantile)
#
# Examples:
#   bash run_baseline.sh
#   bash run_baseline.sh --model ClaudeSonnet --n_attempts 10
#   bash run_baseline.sh --model llama3 --bin_strategy uniform --n_bins 5

set -euo pipefail

# Defaults
MODEL="GPT5_4Nano"
N_ATTEMPTS=10
N_PER_CAT=30
SEED=42
MAX_CONCURRENT=10
N_BINS=5
BIN_STRATEGY="quantile"

# Parse named args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="$2";          shift 2 ;;
        --n_attempts)     N_ATTEMPTS="$2";     shift 2 ;;
        --n_per_cat)      N_PER_CAT="$2";      shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --n_bins)         N_BINS="$2";         shift 2 ;;
        --bin_strategy)   BIN_STRATEGY="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  Baseline Experiment"
echo "  --model          $MODEL"
echo "  --n_attempts     $N_ATTEMPTS"
echo "  --n_per_cat      $N_PER_CAT"
echo "  --seed           $SEED"
echo "  --max_concurrent $MAX_CONCURRENT"
echo "  --n_bins         $N_BINS"
echo "  --bin_strategy   $BIN_STRATEGY"
echo "============================================"

python "$SCRIPT_DIR/run_baseline.py" \
    --model          "$MODEL"        \
    --n_attempts     "$N_ATTEMPTS"   \
    --n_per_cat      "$N_PER_CAT"    \
    --seed           "$SEED"         \
    --max_concurrent "$MAX_CONCURRENT" \
    --n_bins         "$N_BINS"       \
    --bin_strategy   "$BIN_STRATEGY"

echo ""
echo "Baseline complete. Results in: experiment_out/$MODEL/"
