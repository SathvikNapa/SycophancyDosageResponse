# Reproducibility

This repository contains the code that produces every table, figure, and numerical claim in the paper *Understanding Sycophancy via Adversarial Pressure and Uncertainty Dynamics*. The paper source is managed separately (Overleaf) and not tracked here.

## Setup

```bash
uv sync           # installs from pyproject.toml + uv.lock
cp .env.example .env  # add API keys for the target models and the external calibrator
```

Required env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

## Pipeline order

The paper's analyses read from `experiment_out/<MODEL>/<DATASET>/...` (not in git). To regenerate that tree from scratch:

1. **Baseline entropy sampling** — `run_baseline.py` produces `base_experiment_metadata.pkl` for each (model, dataset). Runs the target model $K = 25$ times per question.
2. **Isotonic calibration** — `run_calibration.py` fits the entropy→hardness map per (model, dataset), producing `calibration.pkl`.
3. **Multi-turn pressure runs** — `run_sycophancy.py` produces the letter-only pressure trajectories used for RQ1–RQ3.
4. **Single-turn pressure runs** — `run_single_turn_pressure.py` produces the single-dose per-category cells used in the appendix.
5. **CoT + external calibrator (RQ4)** — `run_reasoning_calibrated.py` produces the per-step calibrated reasoning trajectories in `reasoning_calibrated_bin/`.
6. **Ensemble calibration (Experiment 5)** — `run_ensemble_calibration.py` produces the ECE numbers used in `fig:ensemble_ece`.
7. **Parser fix (Claude Sonnet HLE)** — `convert_reasoning_to_sycophancy.py` reruns the FINAL-ANSWER regex on Sonnet HLE traces (see Methods § HLE stress test).

## Paper artifact → producing script

| Artifact in paper | Producing script | Read from |
|---|---|---|
| `tab:coverage`, `tab:entropy_dist` | `build_analysis_dfs.py`, `entropy_analysis.py` | `experiment_out/*/base_experiment_metadata.pkl` |
| `tab:rq1`, RQ1 flip contrast | `entropy_analysis.py`, `run_rq_analysis_v2.py` | `experiment_out/*/entropy_bin/*_repeated.pkl` |
| `tab:rq2_combined`, `fig:rq2_coefs`, `fig:rq2_interaction`, `fig:rq2_empirical_interaction`, `tab:rq2_ci` | `run_rq_analysis_v2.py`, `run_separate_regressions.py` | Same as RQ1 |
| `fig:heterogeneity`, "Hierarchical (meta-analytic) treatment" paragraph | `hierarchical_analysis.py` | Analysis A per-model estimates (from `run_rq_analysis_v2.py`) |
| `tab:rq3`, `fig:rq3combined` | `run_rq_analysis_v2.py` | Same as RQ1 |
| `tab:rq4_mmlu`, `tab:rq4_gpqa`, `tab:rq4_hle`, `fig:rq4combined` | `run_reasoning_calibrated.py`, `reasoning_analysis.py` | `experiment_out/*/reasoning_calibrated_bin/bin_*_summary.pkl` |
| `tab:hle_consensus` (appendix) | `progress_report.py` (`build_consensus_df`) | `experiment_out/*/hle/reasoning_calibrated_bin/bin_*_cross_turn.pkl` |
| `fig:ensemble_ece`, ECE numbers | `run_ensemble_calibration.py` | `experiment_out/*/ensemble_calibration/` |
| `tab:cal_overall`, `tab:cal_turn` (calibrator validation appendix) | `calibration_analysis.py` | `experiment_out/*/reasoning_calibrated_bin/bin_*_reasoning.pkl` |
| `tab:single_turn` (appendix) | `run_single_turn_pressure.py` | `experiment_out/*/single_turn_pressure/` |
| Cross-dataset comparisons | `run_combined_datasets_analysis.py` | All of the above |

## Model versions

API calls were issued between May and June 2026 against provider default snapshots:

- Anthropic: `claude-4-5-haiku`, `claude-4-6-sonnet`
- OpenAI: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` (the mini model is also the external calibrator $\mathcal{C}$)
- Google: `gemini-3.5-flash`

## Hyperparameters

All numeric hyperparameters used in the paper are listed in Appendix "Implementation Hyperparameters" of the paper draft. The defaults in `config.py` match those values.

## Notes on non-reproducibility

- LLM sampling at temperature 1 is not bitwise deterministic even with a fixed seed; the entropy signal is robust to this in aggregate but individual `(question, sample)` outputs will differ across runs.
- `experiment_out/` snapshots used in the submission are archived separately; contact the authors for access.
