#!/bin/bash
# Runs inside tmux on the login node. Both stages run on c04 via srun and resume from
# experiment_out/_cot_fill/state.pkl, so either can be rerun safely.
cd /home/snapaug1/SycophancyDosageResponse
LOG=logs/cot_fill_tmux.log
echo "=== stage 1: Anthropic Batches ($(date))" | tee -a $LOG
srun --jobid=1815234 --overlap bash -c 'echo "host=$(hostname)"; .venv-grid/bin/python -u scripts/fill_cot_batch.py gen --budget 128' 2>&1 | tee -a $LOG
echo "=== stage 2: gateway gen + calibration + merge ($(date))" | tee -a $LOG
srun --job-name=cot_fill --partition=cpu --nodelist=c04 --cpus-per-task=2 --mem=12G --time=7-00:00:00 \
  bash -c 'echo "host=$(hostname)"; .venv-grid/bin/python -u scripts/fill_cot_batch.py all --budget 128 --gw_concurrency 6; echo "exit=$? end=$(date)"' 2>&1 | tee -a $LOG
echo "fill finished $(date); press Enter to close"; read
