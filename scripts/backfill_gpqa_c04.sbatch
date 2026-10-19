#!/bin/bash
#SBATCH --job-name=gpqa_backfill
#SBATCH --partition=cpu
#SBATCH --nodelist=c04
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/gpqa_backfill_%j.log

cd /home/snapaug1/SycophancyDosageResponse
echo "host=$(hostname) start=$(date)"
.venv-grid/bin/python -u scripts/backfill_gpqa_validated.py --concurrency 24
echo "exit=$? end=$(date)"
