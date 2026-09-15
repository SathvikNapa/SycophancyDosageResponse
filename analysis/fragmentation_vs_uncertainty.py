"""
fragmentation_vs_uncertainty.py
--------------------------------
Reviewer question: does fragmentation (losing consensus across repeated
CoT samples) actually track per-question baseline uncertainty, or is it
just a property of benchmark difficulty (i.e. "harder benchmark -> more
fragmentation" regardless of any individual question's own entropy)?

Uses data already collected -- no new API calls. Joins each T0-correct
question's baseline entropy (base_experiment_metadata.pkl) against
whether that question ever lost consensus in 1..6 (reasoning_calibrated_bin
cross_turn pkls, same lost_consensus definition as progress_report.py's
build_consensus_df / tab:hle_consensus), pooled across all three
benchmarks and all six models.

Non-obvious bug this script works around: reasoning_calibrated_bin's
cross_turn pkls store `query` truncated to 80 characters, while
base_experiment_metadata.pkl stores the full query text. Joining on
`query[:80]` recovers a full match (verified: 239/239 on Claude
Haiku/HLE, and the resulting per-model HLE n's/lost-consensus rates
reproduce tab:hle_consensus exactly).

Usage:
    uv run python analysis/fragmentation_vs_uncertainty.py
"""

from __future__ import annotations

import glob
import os
import pickle

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

EXPERIMENT_OUT = "experiment_out"
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "GeminiFlash"]
DATASETS = ["mmlu_pro", "gpqa_diamond", "hle"]
QUERY_TRUNC = 80  # cross_turn pkl's query field is truncated to this length


def _find_reasoning_cross_turn_dir(model: str, dataset: str) -> str | None:
    if dataset == "mmlu_pro":
        candidates = [
            os.path.join(EXPERIMENT_OUT, model, "reasoning_calibrated_bin"),
            os.path.join(EXPERIMENT_OUT, model, "mmlu_pro", "reasoning_calibrated_bin"),
        ]
    else:
        candidates = [os.path.join(EXPERIMENT_OUT, model, dataset, "reasoning_calibrated_bin")]
    for c in candidates:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "bin_*_cross_turn.pkl")):
            return c
    return None


def _find_baseline_meta(model: str, dataset: str) -> str | None:
    for c in (
        os.path.join(EXPERIMENT_OUT, model, dataset, "base_experiment_metadata.pkl"),
        os.path.join(EXPERIMENT_OUT, model, "base_experiment_metadata.pkl"),
    ):
        if os.path.exists(c):
            return c
    return None


def build_dataset() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            rdir = _find_reasoning_cross_turn_dir(model, dataset)
            bmeta = _find_baseline_meta(model, dataset)
            if not rdir or not bmeta:
                continue
            with open(bmeta, "rb") as f:
                meta = pickle.load(f)
            entropy_map = {m["query"][:QUERY_TRUNC]: m.get("entropy") for m in meta}

            all_rows = []
            for pkl in sorted(glob.glob(os.path.join(rdir, "bin_*_cross_turn.pkl"))):
                with open(pkl, "rb") as f:
                    all_rows.extend(pickle.load(f))
            if not all_rows:
                continue
            df = pd.DataFrame(all_rows)
            for q, g in df.groupby("query"):
                last_per_turn = g.sort_values(["turn", "step"]).drop_duplicates("turn", keep="last")
                turn_to_belief = dict(zip(last_per_turn["turn"], last_per_turn["majority_belief"]))
                turn_to_correct = dict(zip(last_per_turn["turn"], last_per_turn["majority_is_correct"]))
                t0_belief = turn_to_belief.get(0)
                t0_correct = bool(turn_to_correct.get(0, False)) and pd.notna(t0_belief)
                if not t0_correct:
                    continue
                lost = any(pd.isna(turn_to_belief.get(t)) for t in range(1, 7))
                entropy = entropy_map.get(q)
                if entropy is None:
                    continue
                rows.append({
                    "model": model, "dataset": dataset, "query": q,
                    "entropy": entropy, "lost_consensus": int(lost),
                })
    return pd.DataFrame(rows)


def main() -> None:
    df = build_dataset()
    print(f"Total T0-correct (model, dataset, query) rows: {len(df)}\n")

    print("=" * 72)
    print("Lost-consensus rate by dataset")
    print("=" * 72)
    print(df.groupby("dataset")["lost_consensus"].agg(["count", "mean"]))
    print()

    print("=" * 72)
    print("Per-dataset Spearman: baseline entropy vs. ever-lost-consensus")
    print("=" * 72)
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        if sub["entropy"].nunique() < 2:
            continue
        rho, p = stats.spearmanr(sub["entropy"], sub["lost_consensus"])
        print(f"  {ds:15s} n={len(sub):5d}  rho={rho:+.3f}  p={p:.4g}")

    rho, p = stats.spearmanr(df["entropy"], df["lost_consensus"])
    print(f"\nNaive pooled Spearman (no model/dataset control): rho={rho:+.3f}  p={p:.4g}")

    print()
    print("=" * 72)
    print("Pooled logistic regression: lost_consensus ~ entropy + model + dataset")
    print("=" * 72)
    model = smf.logit("lost_consensus ~ entropy + C(model) + C(dataset)", data=df).fit(disp=0)
    print(f"  beta(entropy) = {model.params['entropy']:+.4f}"
          f"  (SE={model.bse['entropy']:.4f}, p={model.pvalues['entropy']:.4g})")


if __name__ == "__main__":
    main()
