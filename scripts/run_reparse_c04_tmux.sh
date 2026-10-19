#!/bin/bash
# Runs inside tmux on the login node; srun allocates c04 and stays attached to it.
cd /home/snapaug1/SycophancyDosageResponse
srun --job-name=reparse --partition=cpu --nodelist=c04 --cpus-per-task=2 --mem=8G --time=1-00:00:00 \
  bash -c 'echo "host=$(hostname) start=$(date)"; .venv-grid/bin/python -u scripts/reparse_baseline_answers.py --concurrency 8; echo "exit=$? end=$(date)"' \
  2>&1 | tee -a logs/reparse_tmux.log
echo "srun finished $(date); press Enter to close"; read
