"""
persona_pilot_analysis.py
--------------------------
Quick read-out of the source-credibility persona pilot
(scripts/run_persona_pilot.py). Not a rigorous power analysis -- a fast
look at whether persona framing moves flip rate at all, across the two
pilot models (ClaudeSonnet, GPT5_4Nano) and two question strata
(confident: entropy==0; mid: entropy<0, T0-correct).

The "confident" stratum's baseline (no persona) arm isn't in
persona_pilot/results.pkl -- it's the existing free single_turn_pressure
run, reused here by filtering to the same questions and the same two doses
(T1/T6) the pilot uses.

Usage:
    uv run python analysis/persona_pilot_analysis.py
"""

from __future__ import annotations

import pickle

import pandas as pd
import statsmodels.formula.api as smf

MODELS = ["ClaudeSonnet", "GPT5_4Nano"]
EXPERIMENT_OUT = "experiment_out"


def load_pilot(model: str) -> pd.DataFrame:
    with open(f"{EXPERIMENT_OUT}/{model}/persona_pilot/results.pkl", "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res)
    df = df[df["flipped"].notna()].copy()
    df["flipped"] = df["flipped"].astype(int)
    return df


def load_confident_baseline(model: str, confident_queries: set[str]) -> pd.DataFrame:
    """Free reuse: existing no-persona single_turn_pressure results, filtered
    to the pilot's confident-stratum question set and its two doses."""
    with open(f"{EXPERIMENT_OUT}/{model}/single_turn_pressure/results.pkl", "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res)
    df = df[
        df["query"].isin(confident_queries)
        & df["dose_category"].isin(["personal_disagreement", "crowd_consensus"])
        & df["flipped"].notna()
    ].copy()
    df["flipped"] = df["flipped"].astype(int)
    df["model"] = model
    df["dataset"] = "mmlu_pro"
    df["stratum"] = "confident"
    df["persona"] = "baseline"
    return df[["model", "dataset", "stratum", "persona", "dose_category", "query", "flipped"]]


def main() -> None:
    frames = []
    for model in MODELS:
        pilot = load_pilot(model)
        frames.append(pilot[["model", "dataset", "stratum", "persona", "dose_category", "query", "flipped"]])
        confident_queries = set(pilot.loc[pilot["stratum"] == "confident", "query"])
        frames.append(load_confident_baseline(model, confident_queries))

    df = pd.concat(frames, ignore_index=True)
    print(f"Total valid trials: {len(df)}  (models={df['model'].unique().tolist()})")

    print("\n=== Flip rate by persona, pooled across both models ===")
    for stratum in ["confident", "mid", None]:
        sub = df if stratum is None else df[df["stratum"] == stratum]
        label = "ALL STRATA" if stratum is None else stratum
        print(f"\n-- {label} --")
        g = sub.groupby("persona")["flipped"].agg(["sum", "count"])
        g["rate"] = g["sum"] / g["count"]
        # order: baseline, novice, expert, adversarial
        order = [p for p in ["baseline", "novice", "expert", "adversarial"] if p in g.index]
        print(g.loc[order].to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== Flip rate by persona x model, pooled across strata ===")
    g = df.groupby(["model", "persona"])["flipped"].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== Pooled logistic regression: flipped ~ persona + model + stratum ===")
    print("(baseline persona, ClaudeSonnet, confident stratum = reference level)")
    model_fit = smf.logit(
        "flipped ~ C(persona, Treatment('baseline')) + C(model) + C(stratum)",
        data=df,
    ).fit(disp=0)
    print(model_fit.summary())

    print("\n=== Odds ratios (exp(coef)) ===")
    print(model_fit.params.apply(lambda b: f"{2.718281828**b:.2f}x"))


if __name__ == "__main__":
    main()
