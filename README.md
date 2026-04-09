# Sycophancy Dosage-Response Experiment

Measures whether LLM sycophancy susceptibility is moderated by
per-question uncertainty, estimated via entropy over repeated samples.

## Project structure

```
config.py              — Model registry, prompt templates, all defaults
data.py                — MMLU-Pro loading and balanced sampling
generator.py           — ResponseGenerator (LiteLLM async wrapper)
entropy.py             — Entropy computation and KBins-based binning
sycophancy.py          — Conversation runner and repeated-sample aggregator
sycophancy_dosage.py   — Pressure turn templates (T1-T6)

run_baseline.py        — CLI: baseline uncertainty experiment
run_sycophancy.py      — CLI: entropy-binned sycophancy experiment

run_baseline.sh        — Shell wrapper for run_baseline.py
run_sycophancy.sh      — Shell wrapper for run_sycophancy.py
run_all.sh             — End-to-end: baseline -> sycophancy -> zip
package_results.sh     — Zip experiment_out/<MODEL>/ into results_<MODEL>.zip

requirements.txt
```

## Setup

```bash
uv sync
cp .env.example .env
```

`.env` should contain:
```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Running

### Full pipeline (recommended)
```bash
bash run_all.sh --model GPT5_4Nano --n_attempts 10 --n_per_cat 30 --n_syco_samples 5 --n_bins 5 --bin_strategy quantile

bash run_all.sh
bash run_all.sh --model ClaudeSonnet
bash run_all.sh --model llama3 --bin_strategy uniform
```

### Step by step
```bash
# 1. Baseline — sample each question N times, compute uncertainty + entropy
bash run_baseline.sh \
    --model GPT5_4Nano \
    --n_attempts 10 \
    --n_per_cat 30 \
    --seed 42 \
    --max_concurrent 10 \
    --n_bins 5 \
    --bin_strategy quantile

# 2. Sycophancy — run pressure experiment per entropy bin
bash run_sycophancy.sh \
    --model GPT5_4Nano \
    --n_syco_samples 5 \
    --concurrency 5 \
    --timeout_s 120 \
    --base_seed 777 \
    --n_bins 5 \
    --bin_strategy quantile

# 3. Package results
bash package_results.sh --model GPT5_4Nano
```

## Outputs

```
experiment_out/
└── <MODEL>/
    ├── base_experiment_metadata.pkl    # list of per-question dicts with
    │                                   # answers_generated, uncertainty, entropy, …
    └── entropy_bin/
        ├── bin_0_repeated.pkl          # aggregated sycophancy results for bin 0
        ├── bin_1_repeated.pkl          # …
        ├── bin_2_repeated.pkl
        ├── bin_3_repeated.pkl
        └── bin_4_repeated.pkl

results_<MODEL>.zip                     # produced by package_results.sh
```

Each `bin_N_repeated.pkl` is a list of dicts with:

| Field | Description |
|---|---|
| `query` | Question text |
| `gold_answer` | Correct option letter |
| `first_wrong_turn_per_run` | List of first-flip turn per run (never = n_doses+1) |
| `flip_rate` | Fraction of runs where model flipped |
| `max_fwt` | Max first-wrong-turn across runs |
| `median_fwt` | Median first-wrong-turn |
| `mean_fwt` | Mean first-wrong-turn |
| `raw_runs` | Full per-turn details for every run |

## Dose statements

Experiment with `sycophancy_dosage.py` to replace T1-T6 with other pressure templates.

