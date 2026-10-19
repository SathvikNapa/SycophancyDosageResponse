#!/bin/bash
# Runs inside tmux on the login node; srun allocates c04 and stays attached to it.
cd /home/snapaug1/SycophancyDosageResponse
srun --job-name=gpqa_repair --partition=cpu --nodelist=c04 --cpus-per-task=2 --mem=8G --time=1-00:00:00 \
  bash -c 'echo "host=$(hostname) start=$(date)"; .venv-grid/bin/python -u scripts/repair_gpqa_backfill.py --phase all --concurrency 8 --timeout_s 300; echo "exit=$? end=$(date)"' \
  2>&1 | tee -a logs/gpqa_repair_tmux.log
echo "srun finished $(date); press Enter to close"; read
