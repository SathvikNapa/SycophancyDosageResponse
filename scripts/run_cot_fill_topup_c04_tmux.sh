#!/bin/bash
# Speed-up pass: moves the cheap/fast Haiku-HLE cell to Anthropic Batches
# (parallel infra, not the gateway's concurrency=6), then finishes whatever's
# left (mainly Sonnet/HLE) on the gateway with higher concurrency, now that
# reparse has been paused to give it the full RPM budget.
cd /home/snapaug1/SycophancyDosageResponse
LOG=logs/cot_fill_tmux.log
echo "=== speed-up: Anthropic top-up for Haiku/HLE ($(date)) ===" | tee -a $LOG
srun --jobid=1815234 --overlap bash -c \
  '.venv-grid/bin/python -u scripts/fill_cot_batch.py gen --anthropic_cells ClaudeHaiku/hle --budget 135' \
  2>&1 | tee -a $LOG
echo "=== gateway continuation for the remainder, concurrency 16 ($(date)) ===" | tee -a $LOG
srun --job-name=cot_fill2 --partition=cpu --nodelist=c04 --cpus-per-task=2 --mem=12G --time=7-00:00:00 \
  bash -c '.venv-grid/bin/python -u scripts/fill_cot_batch.py all --budget 135 --gw_concurrency 16; echo "exit=$? end=$(date)"' \
  2>&1 | tee -a $LOG
echo "fill finished $(date); press Enter to close"; read
