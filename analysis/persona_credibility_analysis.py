"""
persona_credibility_analysis.py
---------------------------------
Confirmatory analysis for scripts/run_persona_credibility.py, feeding
paper/active_version/sections/appendix.tex's app:persona and
tables/persona_credibility.tex.

Unlike analysis/persona_pilot_analysis.py, baseline is a fresh arm for both
strata here (no free-data reuse), all six pressure categories are present
(not just T1/T6), and each cell has R=4 repetitions -- so real Wilson CIs
and a proper per-model + pooled logistic regression are meaningful here in
a way they weren't for the single-draw pilot.

Usage:
    uv run python analysis/persona_credibility_analysis.py
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

MODELS = ["ClaudeSonnet", "GPT5_4Mini"]
PERSONA_ORDER = ["baseline", "novice", "expert", "adversarial"]
EXPERIMENT_OUT = "experiment_out"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval, percentages. Kept identical to
    analysis/category_breakdown.py's implementation for consistency with
    the rest of the paper's reported CIs."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p * 100, max(0, (c - m) * 100), min(100, (c + m) * 100)


def load_model(model: str) -> pd.DataFrame:
    with open(f"{EXPERIMENT_OUT}/{model}/persona_credibility/results.pkl", "rb") as f:
        res = pickle.load(f)
    df = pd.DataFrame(res)
    n_total = len(df)
    df = df[df["flipped"].notna()].copy()
    df["flipped"] = df["flipped"].astype(int)
    n_err = n_total - len(df)
    print(f"{model}: {len(df)}/{n_total} valid ({n_err} errors, {n_err / n_total * 100:.1f}%)")
    return df


def print_flip_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols)["flipped"].agg(["sum", "count"])
    g["rate"], g["ci_lo"], g["ci_hi"] = zip(*[wilson(s, n) for s, n in zip(g["sum"], g["count"])])
    print(g.to_string(float_format=lambda x: f"{x:.1f}"))
    return g


def fit_logit(df: pd.DataFrame, formula: str, label: str):
    print(f"\n=== {label} ===")
    fit = smf.logit(formula, data=df).fit(disp=0)
    print(fit.summary())
    print("\nOdds ratios (95% CI):")
    conf = fit.conf_int()
    for name in fit.params.index:
        or_, lo, hi = np.exp(fit.params[name]), np.exp(conf.loc[name, 0]), np.exp(conf.loc[name, 1])
        p = fit.pvalues[name]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:55s} OR={or_:6.3f}  [{lo:6.3f}, {hi:6.3f}]  p={p:.4g} {star}")
    return fit


def main() -> None:
    frames = []
    for model in MODELS:
        df = load_model(model)
        df["model"] = model
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal valid trials: {len(df)}")

    print("\n" + "=" * 70)
    print("Flip rate by persona, pooled across models/strata/categories")
    print("=" * 70)
    df["persona"] = pd.Categorical(df["persona"], categories=PERSONA_ORDER, ordered=True)
    print_flip_table(df, ["persona"])

    print("\n" + "=" * 70)
    print("Flip rate by model x persona, pooled across strata/categories")
    print("=" * 70)
    print_flip_table(df, ["model", "persona"])

    print("\n" + "=" * 70)
    print("Flip rate by model x stratum x persona")
    print("=" * 70)
    print_flip_table(df, ["model", "stratum", "persona"])

    print("\n" + "=" * 70)
    print("Flip rate by persona x dose category, pooled across models")
    print("=" * 70)
    print_flip_table(df, ["dose_category", "persona"])

    formula = "flipped ~ C(persona, Treatment('baseline')) + C(stratum) + C(dose_category)"
    for model in MODELS:
        fit_logit(df[df["model"] == model], formula, f"{model}: persona effect (stratum + category controlled)")

    fit_logit(
        df,
        "flipped ~ C(persona, Treatment('baseline')) + C(model) + C(stratum) + C(dose_category)",
        "Pooled (model fixed effects): persona effect",
    )

    print("\n" + "=" * 70)
    print("LaTeX row values for tables/persona_credibility.tex (copy in manually)")
    print("=" * 70)
    for model in MODELS:
        sub = df[df["model"] == model]
        rates = sub.groupby("persona")["flipped"].mean() * 100
        print(f"{model}: " + " & ".join(f"{rates.get(p, float('nan')):.1f}" for p in PERSONA_ORDER))


if __name__ == "__main__":
    main()
