#!/bin/bash
# Runs inside tmux on the login node; srun allocates c04 and stays attached to it.
cd /home/snapaug1/SycophancyDosageResponse
srun --job-name=gpqa_backfill --partition=cpu --nodelist=c04 --cpus-per-task=2 --mem=8G --time=1-00:00:00 \
  bash -c 'echo "host=$(hostname) start=$(date)"; .venv-grid/bin/python -u scripts/backfill_gpqa_validated.py --concurrency 24; echo "exit=$? end=$(date)"' \
  2>&1 | tee -a logs/gpqa_backfill_tmux.log
echo "srun finished $(date); press Enter to close"; read
