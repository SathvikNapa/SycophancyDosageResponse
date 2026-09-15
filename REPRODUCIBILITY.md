# Reproducibility

This repository contains the code that produces every table, figure, and numerical claim in the paper *Understanding Sycophancy via Adversarial Pressure and Uncertainty Dynamics*. The paper source is managed separately (Overleaf) and not tracked here.

## Setup

```bash
uv sync           # installs from pyproject.toml + uv.lock
cp .env.example .env  # add API keys for the target models and the external calibrator
```

Required env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.

## Repository layout

```
src/sycophancy/   installable library (config, data loading, the generator wrapper,
                  entropy/calibration math, the conversation runner, the shared
                  analysis-df builder) — everything other code imports from
scripts/          CLIs that call model APIs and run experiments, plus their
                  shell wrappers
analysis/         post-hoc, no-API-calls: stats, tables, figures, derived CSVs
notebooks/        exploratory notebooks
tests/            unit tests (pytest)
paper/            the manuscript zip — source of truth for the paper text
                  (paper source is otherwise managed via Overleaf, not tracked here)
figures/          canonical regenerated PDF figures, copied into the manuscript
legacy/           superseded scripts/data kept for reference, not on any live path
```

`uv sync` installs `src/sycophancy` itself as an editable package (see
`pyproject.toml`'s `[tool.hatch.build.targets.wheel]`), so every script does
`from sycophancy.<module> import ...` regardless of which directory it lives
in or is invoked from.

**Pickle compatibility note:** this layout is a 2026 reorganization of a repo
that used to keep these modules as bare top-level files (`config.py`,
`reasoning_uncertainty.py`, etc.). `pickle` embeds a class's *original*
module path, so historical `experiment_out/**/*.pkl` files — most of them,
via `ReasoningTrace`/`ReasoningStep` — only unpickle because
`src/sycophancy/__init__.py` registers the old flat names as `sys.modules`
aliases on import. Don't delete that shim; it's the only thing standing
between this reorg and needing to regenerate the experiment data.

## Pipeline order

The paper's analyses read from `experiment_out/<MODEL>/<DATASET>/...` (not in git). To regenerate that tree from scratch:

1. **Baseline entropy sampling** — `scripts/run_baseline.py` produces `base_experiment_metadata.pkl` for each (model, dataset). Runs the target model $K = 25$ times per question.
2. **Isotonic calibration** — `scripts/run_calibration.py` fits the entropy→hardness map per (model, dataset), producing `calibration.pkl`.
3. **Multi-turn pressure runs** — `scripts/run_sycophancy.py` produces the letter-only pressure trajectories used for RQ1–RQ3.
4. **Single-turn pressure runs** — `scripts/run_single_turn_pressure.py` produces the single-dose per-category cells used in the appendix.
5. **CoT + external calibrator (RQ4)** — `scripts/run_reasoning_calibrated.py` produces the per-step calibrated reasoning trajectories in `reasoning_calibrated_bin/`.
6. **Ensemble calibration (Experiment 5)** — `scripts/run_ensemble_calibration.py` produces the ECE numbers used in `fig:ensemble_ece`.
7. **Parser fix (Claude Sonnet HLE)** — `scripts/convert_reasoning_to_sycophancy.py` reruns the FINAL-ANSWER regex on Sonnet HLE traces (see Methods § HLE stress test).

## Paper artifact → producing script

Verified this pass by checking actual printed output against the numbers in
the compiled tables (not just trusting file names) — corrections to the
previous version of this table are noted inline.

| Artifact in paper | Producing script | Read from |
|---|---|---|
| `tab:coverage` | manual (✓/∅/✳ checkmarks only, no computed numbers — not script-generated) | — |
| `tab:entropy_dist` | direct read of `base_experiment_metadata.pkl` per (model, dataset): `n`, certain = `entropy==0.0` count, uncertain = the rest | `experiment_out/*/base_experiment_metadata.pkl` |
| `tab:first_flip` | direct count of `first_wrong_turn`/`turn_categories` in `entropy_bin/*_repeated.pkl`, MMLU-Pro only (Claude Sonnet, GPT-5.4 Mini) — confirmed by exact-value match | `experiment_out/{ClaudeSonnet,GPT5_4Mini}/entropy_bin/*_repeated.pkl` |
| `tab:hyperparams` | `src/sycophancy/config.py` constants, echoed by hand — confirmed each value against the source; the one flagged deviation (GPQA-Diamond Claude timeout) is now noted in the appendix text itself | `src/sycophancy/config.py` |
| `tab:rq1` | `analysis/run_combined_datasets_analysis.py` (GPQA-Diamond rows, live) + hardcoded `MMLU_RQ1` dict in the same script for MMLU-Pro rows — see caveat below | `experiment_out/*/gpqa_diamond/entropy_bin/*_repeated.pkl`, backfilled per below |
| `tab:rq2_combined` — **left half** ($H_c$, mean-centered) | `analysis/hierarchical_analysis.py --data analysis/flip_data_{mmlu,gpqa}.csv` (Analysis A, per-model regression) | those two CSVs, built via `sycophancy.build_analysis_dfs.build_flat_df_backfilled` (see snippet below) |
| `tab:rq2_combined` — **right half** ($\hat h$, calibrated hardness), `fig:rq2_coefs`, `fig:rq3combined`, `fig:rq4combined` | `analysis/run_rq_analysis_v2.py` | `experiment_out/*/entropy_bin/*_repeated.pkl` + `experiment_out/*/reasoning_calibrated_bin/` |
| `fig:rq2_interaction`, `fig:rq2_empirical_interaction` | `analysis/run_rq_analysis_v2.py` (RQ2b section) — **MMLU-Pro only**, matches the figure captions | Same as above, `dataset=="mmlu_pro"` |
| `tab:rq2_ci` (appendix) | `analysis/run_separate_regressions.py` — **MMLU-Pro only, raw (uncentered) entropy**; confirmed by exact-value match against the compiled table, not just file-name inference | `experiment_out/*/entropy_bin/*_repeated.pkl`, `dataset=="mmlu_pro"` |
| `app:mixed_effects` (appendix) | `analysis/hierarchical_analysis.py --data analysis/flip_data_mmlu.csv` (Analysis D, `crossed_random_effects`) — crossed model+question random intercepts plus a per-model random slope on pressure turn, fit via `statsmodels.BinomialBayesMixedGLM.fit_vb()`. Numbers are the exact `Post. Mean`/`Post. SD` values the script prints. | `analysis/flip_data_mmlu.csv` |
| `tab:category_breakdown` (appendix) | `analysis/category_breakdown.py` — MMLU-Pro only, pooled across all 5 models | `analysis/flip_data_mmlu.csv` + `experiment_out/*/base_experiment_metadata.pkl` (for `category`) |
| `tab:firth`, `app:firth` (appendix) | Firth's bias-reduced logistic regression, implemented directly (no third-party package — `firthlogist` requires `numpy<2.0`, incompatible with this project's `numpy>=2.2.6`); IRLS on the Jeffreys-penalized score | `analysis/flip_data_mmlu.csv`, `analysis/flip_data_gpqa.csv` |
| `tab:plausibility`, `app:plausibility` (appendix) | `analysis/wrong_answer_plausibility.py` — MMLU-Pro only, correlates each run's planted-answer baseline plausibility against whether it ever flipped | `experiment_out/*/base_experiment_metadata.pkl` + `experiment_out/*/entropy_bin/*_repeated.pkl` |
| `fig:heterogeneity`, "Hierarchical (meta-analytic) treatment" paragraph | `analysis/hierarchical_analysis.py --data analysis/flip_data_mmlu.csv` | Analysis A/B/C on that CSV |
| `tab:rq3` | `analysis/run_rq_analysis_v2.py` | Same as the right half of RQ2 |
| `tab:rq4_mmlu`, `tab:rq4_gpqa`, `fig:rq4combined` | `analysis/run_rq_analysis_v2.py` (`rq4_stats()`) — **not** `analysis/reasoning_analysis.py` (that script computes a different uncertainty-binning analysis, not this belief-entropy/majority-correct/confidence/gap-per-turn table); confirmed by regenerating and matching to 3 decimal places | `experiment_out/*/reasoning_calibrated_bin/` |
| `tab:rq4_hle` | same `rq4_stats()`-equivalent computation applied to `experiment_out/*/hle/reasoning_calibrated_bin/bin_*_cross_turn.pkl` directly (no `build_flat_df` involved, since HLE isn't in that function's dataset scope) — confirmed by matching every T0 value and closely matching later turns | `experiment_out/*/hle/reasoning_calibrated_bin/` |
| `tab:hle_consensus` (appendix) | `analysis/progress_report.py` (`build_consensus_df`) — **note:** the script's own `HLE_MODELS = {"GPT5_4", "GeminiFlash"}` scope filter must be bypassed to get all six rows this table shows; confirmed exact match for all 6 models once bypassed | `experiment_out/*/hle/reasoning_calibrated_bin/bin_*_cross_turn.pkl` |
| `fig:ensemble_ece`, ECE numbers | `scripts/run_ensemble_calibration.py` | `experiment_out/*/ensemble_calibration/` |
| `tab:cal_overall`, `tab:cal_turn` (calibrator validation appendix) | `analysis/calibration_analysis.py --model all` — confirmed exact match, including exact prediction counts, no changes needed | `experiment_out/*/reasoning_calibrated_bin/bin_*_reasoning.pkl` |
| `tab:single_turn` (appendix) | `scripts/run_single_turn_pressure.py` — MMLU-Pro only (hardcoded in the script), confirmed unaffected by the GPQA-Diamond work below | `experiment_out/*/mmlu_pro/single_turn_pressure/` |
| Cross-dataset comparisons | `analysis/run_combined_datasets_analysis.py` | All of the above |

**Corrections vs. the previous version of this table:** `tab:rq2_combined` and
`tab:rq2_ci` were previously both attributed to `analysis/run_rq_analysis_v2.py` +
`analysis/run_separate_regressions.py`. That's wrong for `tab:rq2_combined`'s left
half — `analysis/run_separate_regressions.py` does no mean-centering at all, so it
cannot produce the $H_c$ column; that half actually comes from
`analysis/hierarchical_analysis.py`. `tab:rq2_ci` **is** `analysis/run_separate_regressions.py`
(verified by matching Claude Sonnet's MMLU-Pro coefficients exactly:
β_P=-0.1164, β_H=-2.0631, β_PH=+0.1096 in the script's own output vs.
-0.116/-2.063/+0.110 in the compiled table), and it's MMLU-Pro-only per its
own caption, so it needed no changes from the GPQA-Diamond coverage work
below.

**Known caveat, not yet fixed:** `analysis/run_combined_datasets_analysis.py`'s
MMLU-Pro numbers (`MMLU_RQ1`/`MMLU_RQ2`/`MMLU_RQ3`/`MMLU_RQ4` dicts) are
**hardcoded**, not computed live from `experiment_out/` on each run — they
were pasted in from an earlier live computation. This is exactly the kind of
thing the reproducibility process should catch: hand-transcribed numbers
carry transcription risk that a script re-run every time does not. GPQA-
Diamond's numbers in the same script (`gpqa_rq1`/`gpqa_rq2`/`gpqa_rq3`) are
computed live and do not have this issue.

## GPQA-Diamond coverage backfill (Claude models)

ClaudeHaiku/ClaudeSonnet were originally run on a 79-question subset of
GPQA-Diamond; they were later extended to the full 198-question set (matching
the GPT models) via `scripts/run_baseline.py --extend` + `scripts/run_reasoning_calibrated.py
--extend`, generating full chain-of-thought reasoning instead of re-running
the separate short-answer sycophancy experiment (`scripts/run_sycophancy.py`). All
RQ1–RQ3 analyses backfill the resulting coverage gap by treating each
reasoning sample's own final answer as the flip signal, via
`sycophancy.build_analysis_dfs.build_flat_df_backfilled` — the single shared
implementation `analysis/run_rq_analysis_v2.py`, `analysis/run_combined_datasets_analysis.py`,
and the `flip_data_*.csv` prep below all call, so the three don't drift into
three different partial fixes. An empty/failed generation for a given
(question, sample, turn) produces no row at all — not a synthetic 0 or 1 —
so it can't silently inflate a flip-rate denominator.

To rebuild `analysis/flip_data_mmlu.csv` / `analysis/flip_data_gpqa.csv`
(the inputs `analysis/hierarchical_analysis.py` needs):

```python
from sycophancy.build_analysis_dfs import build_flat_df_backfilled, get_common_queries

common = get_common_queries("gpqa_diamond")
for dataset, out_path in [("mmlu_pro", "analysis/flip_data_mmlu.csv"),
                          ("gpqa_diamond", "analysis/flip_data_gpqa.csv")]:
    df = build_flat_df_backfilled(restrict_queries={"gpqa_diamond": common})
    df = df[(df["turn"] >= 1) & (df["dataset"] == dataset)]
    df = df.rename(columns={"query": "question", "flipped": "flip"})
    df[["model", "question", "turn", "entropy", "flip"]].to_csv(out_path, index=False)
```

## Model versions

API calls were issued between May and June 2026 against provider default snapshots:

- Anthropic: `claude-4-5-haiku`, `claude-4-6-sonnet`
- OpenAI: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` (the mini model is also the external calibrator $\mathcal{C}$)
- Google: `gemini-3.5-flash`

## Hyperparameters

All numeric hyperparameters used in the paper are listed in Appendix "Implementation Hyperparameters" of the paper draft. The defaults in `src/sycophancy/config.py` match those values.

## Notes on non-reproducibility

- LLM sampling at temperature 1 is not bitwise deterministic even with a fixed seed; the entropy signal is robust to this in aggregate but individual `(question, sample)` outputs will differ across runs.
- `experiment_out/` snapshots used in the submission are archived separately; contact the authors for access.
