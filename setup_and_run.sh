#!/usr/bin/env bash
# setup_and_run.sh
# Run this once from the root of your project directory.
# Usage:
#   bash setup_and_run.sh                          # full pipeline, default model
#   bash setup_and_run.sh --model ClaudeSonnet
#   bash setup_and_run.sh --model GPT5_4 --skip_reasoning

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MODEL="GPT5_4Nano"
N_ATTEMPTS=30
N_PER_CAT=30
N_SYCO_SAMPLES=5
N_BINS=5
BIN_STRATEGY="quantile"
N_CONFIDENCE_POLLS=5
N_REASONING_SAMPLES=5
SKIP_REASONING=false
STRATIFY=false
SHUFFLE_DOSES=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)               MODEL="$2";               shift 2 ;;
    --n_attempts)          N_ATTEMPTS="$2";          shift 2 ;;
    --n_per_cat)           N_PER_CAT="$2";           shift 2 ;;
    --n_syco_samples)      N_SYCO_SAMPLES="$2";      shift 2 ;;
    --n_confidence_polls)  N_CONFIDENCE_POLLS="$2";  shift 2 ;;
    --n_reasoning_samples) N_REASONING_SAMPLES="$2"; shift 2 ;;
    --skip_reasoning)      SKIP_REASONING=true;      shift   ;;
    --stratify)            STRATIFY=true;             shift   ;;
    --shuffle_doses)       SHUFFLE_DOSES=true;        shift   ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Sycophancy Dosage-Response Experiment               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  model              : $MODEL"
echo "  n_attempts         : $N_ATTEMPTS"
echo "  n_per_cat          : $N_PER_CAT"
echo "  n_syco_samples     : $N_SYCO_SAMPLES"
echo "  n_confidence_polls : $N_CONFIDENCE_POLLS"
echo "  skip_reasoning     : $SKIP_REASONING"
echo "  stratify_by_cat    : $STRATIFY
  shuffle_doses      : $SHUFFLE_DOSES"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Check .env ────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  echo "ERROR: .env file not found."
  echo "  Create $SCRIPT_DIR/.env with:"
  echo "    OPENAI_API_KEY=sk-..."
  echo "    ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

# ── 2. Check Python env ──────────────────────────────────────────────────────
echo "▶ Checking Python dependencies..."
python3 -c "import litellm" 2>/dev/null || {
  echo "  litellm not found. Installing requirements..."
  pip install -r "$SCRIPT_DIR/requirements.txt" -q
}
python3 -c "import sentence_transformers" 2>/dev/null || {
  if [ "$SKIP_REASONING" = false ]; then
    echo "  sentence-transformers not found. Installing..."
    pip install sentence-transformers -q
  fi
}
echo "  Dependencies OK."

# ── 3. Step 1 — Baseline ────────────────────────────────────────────────────
echo ""
echo "▶ STEP 1 — Baseline uncertainty experiment"
echo "  Sampling each question $N_ATTEMPTS times to estimate entropy."
echo "  This is the most API-call-intensive step."
echo ""

STRATIFY_FLAG=""
[ "$STRATIFY" = true ] && STRATIFY_FLAG="--stratify_by_category"
SHUFFLE_FLAG=""
[ "$SHUFFLE_DOSES" = true ] && SHUFFLE_FLAG="--shuffle_doses"

python3 "$SCRIPT_DIR/run_baseline.py" \
  --model          "$MODEL"          \
  --n_attempts     "$N_ATTEMPTS"     \
  --n_per_cat      "$N_PER_CAT"      \
  --n_bins         "$N_BINS"         \
  --bin_strategy   "$BIN_STRATEGY"   \
  $STRATIFY_FLAG

# ── 4. Step 2 — Sycophancy ──────────────────────────────────────────────────
echo ""
echo "▶ STEP 2 — Entropy-binned sycophancy experiment"
echo "  Running pressure turns with per-turn confidence estimation."
echo ""

python3 "$SCRIPT_DIR/run_sycophancy.py" \
  --model                "$MODEL"                \
  --n_syco_samples       "$N_SYCO_SAMPLES"       \
  --n_bins               "$N_BINS"               \
  --bin_strategy         "$BIN_STRATEGY"         \
  --n_confidence_polls   "$N_CONFIDENCE_POLLS"   \
  $STRATIFY_FLAG \
  $SHUFFLE_FLAG

# ── 5. Step 3 — Reasoning (optional) ────────────────────────────────────────
if [ "$SKIP_REASONING" = false ]; then
  echo ""
  echo "▶ STEP 3 — Reasoning sycophancy experiment"
  echo "  Running CoT step-wise uncertainty estimation."
  echo ""

  python3 "$SCRIPT_DIR/run_reasoning_sycophancy.py" \
    --model                "$MODEL"                \
    --n_reasoning_samples  "$N_REASONING_SAMPLES"  \
    --n_bins               "$N_BINS"               \
    --bin_strategy         "$BIN_STRATEGY"         \
    $STRATIFY_FLAG
else
  echo ""
  echo "  Skipping reasoning experiment (--skip_reasoning set)."
fi

# ── 6. Package results ───────────────────────────────────────────────────────
echo ""
echo "▶ Packaging results..."
bash "$SCRIPT_DIR/package_results.sh" --model "$MODEL"

echo ""
echo "✓ All done.  Results in: experiment_out/$MODEL/"
echo "  Archive:              results_${MODEL}.zip"
