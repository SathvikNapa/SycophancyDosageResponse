#!/usr/bin/env bash
# package_results.sh — Zip all experiment outputs for a given model.
#
# Usage:
#   bash package_results.sh [OPTIONS]
#
# Options:
#   --model   Model key matching an experiment_out/<MODEL>/ directory  (default: GPT5_4Nano)
#
# Produces: results_<MODEL>.zip in the project root.
# Contents:
#   experiment_out/<MODEL>/base_experiment_metadata.pkl
#   experiment_out/<MODEL>/entropy_bin/bin_*_repeated.pkl
#   experiment_out/<MODEL>/uncertainty_buckets/*.pkl   (if present)
#
# Examples:
#   bash package_results.sh
#   bash package_results.sh --model ClaudeSonnet

set -euo pipefail

MODEL="GPT5_4Nano"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/experiment_out/$MODEL"
ARCHIVE="$SCRIPT_DIR/results_${MODEL}.zip"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Output directory not found: $SRC_DIR"
    exit 1
fi

[ -f "$ARCHIVE" ] && rm "$ARCHIVE"

echo "Packaging results for model: $MODEL"
echo "Source:  $SRC_DIR"
echo "Archive: $ARCHIVE"

cd "$SCRIPT_DIR"
zip -r "$ARCHIVE" "experiment_out/$MODEL"

echo ""
echo "Contents:"
unzip -l "$ARCHIVE"
echo ""
echo "Archive size: $(du -sh "$ARCHIVE" | cut -f1)"
echo "Done -> $ARCHIVE"
