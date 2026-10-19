"""
wrong_answer_plausibility_v2.py — Rebuilds app:plausibility on the unified
CoT pipeline (paper_pipeline.py), for all 5 Claude/GPT models on MMLU-Pro.

The original analysis/wrong_answer_plausibility.py reads flips from
entropy_bin (the direct-answer protocol) and baseline letters from
base_experiment_metadata.pkl (pre-correction, full response text for many
Claude samples -- its own regex only matches a bare leading letter, so it
silently dropped most of those samples rather than computing anything
wrong with them). Both are the exact things replaced everywhere else in
this paper: Claude's direct-answer MMLU-Pro protocol was abandoned for CoT,
and the baseline-entropy extraction bug is fixed in the reparsed metadata.
This script uses the same canonical CoT dialogues, planted-letter
reconstruction, and reparsed baseline as the rest of the paper, rather than
leaving this one appendix on the retired protocol.

p_hat_q(a-dagger): proportion of a question's 25 reparsed baseline samples
that independently produced a-dagger (the planted wrong letter for that
dialogue), before any pressure was applied -- same definition as the
original, on corrected letters.
ever_flipped: from paper_pipeline.dialogues()'s ever_flip column (CoT,
canonical R=3 runs), among T0-correct, complete dialogues.

Usage: python analysis/wrong_answer_plausibility_v2.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp
from sycophancy.build_analysis_dfs import reconstruct_cot_wrong_letters

MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]


def baseline_prob_map(model):
    """query -> {letter: p_hat} from the 25 reparsed baseline samples."""
    d = pp.cell_dir(model, "mmlu_pro")
    path = os.path.join(d, "base_experiment_metadata_reparsed.pkl")
    if not os.path.exists(path):
        path = os.path.join(d, "base_experiment_metadata.pkl")
    meta = pickle.load(open(path, "rb"))
    out = {}
    for item in meta:
        letters = [a.strip() for a in item["answers_generated"] if isinstance(a, str) and len(a.strip()) == 1]
        n = len(letters)
        if n == 0:
            continue
        counts = {}
        for l in letters:
            counts[l] = counts.get(l, 0) + 1
        out[item["query"]] = {l: c / n for l, c in counts.items()}
    return out


def build_rows(model, pool):
    df, _ = pp.load_cell(model, "mmlu_pro", pool)
    d, _ = pp.dialogues(df)
    d = d[d["complete"] & d["t0_correct"]].copy()
    prob_map = baseline_prob_map(model)
    wrong_letters = reconstruct_cot_wrong_letters(model, "mmlu_pro")
    rows = []
    for _, row in d.iterrows():
        a_dagger = wrong_letters.get(row["query"])
        if a_dagger is None:
            continue
        p_hat = prob_map.get(row["query"], {}).get(a_dagger, 0.0)
        rows.append({"model": model, "query": row["query"], "a_dagger_p_hat": p_hat,
                     "ever_flipped": int(row["ever_flip"])})
    return rows


def main():
    pool = pp.canonical_pool("mmlu_pro")
    all_rows = []
    for model in MODELS:
        all_rows.extend(build_rows(model, pool))
    df = pd.DataFrame(all_rows)
    print(f"Total (question, dialogue) pairs: {len(df):,}\n")

    per_model = []
    for model in MODELS:
        sub = df[df["model"] == model]
        if sub["a_dagger_p_hat"].nunique() < 2:
            print(f"  {model:15s} insufficient variation, skipped")
            continue
        rho, p = stats.spearmanr(sub["a_dagger_p_hat"], sub["ever_flipped"])
        per_model.append({"model": model, "n": len(sub), "rho": rho, "p": p})
        print(f"  {model:15s} n={len(sub):6,}  rho={rho:+.3f}  p={p:.4g}")

    pooled_naive = stats.spearmanr(df["a_dagger_p_hat"], df["ever_flipped"])
    print(f"\nNaive pooled (no model fixed effects): n={len(df):,}  "
          f"rho={pooled_naive.correlation:+.3f}  p={pooled_naive.pvalue:.4g}")

    import statsmodels.formula.api as smf
    model_res = smf.logit("ever_flipped ~ a_dagger_p_hat + C(model)", data=df).fit(disp=0)
    beta = model_res.params["a_dagger_p_hat"]
    se = model_res.bse["a_dagger_p_hat"]
    pv = model_res.pvalues["a_dagger_p_hat"]
    print(f"Pooled logistic (model fixed effects): beta={beta:+.3f} (SE={se:.3f}, p={pv:.4g})")

    out = pd.DataFrame(per_model)
    out.to_csv(os.path.join(pp.OUT, "plausibility.csv"), index=False)
    pd.Series({"pooled_naive_rho": pooled_naive.correlation, "pooled_naive_p": pooled_naive.pvalue,
              "beta": beta, "se": se, "p": pv, "n_total": len(df)}).to_csv(
        os.path.join(pp.OUT, "plausibility_pooled.csv"))

    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"\textbf{Model} & $\boldsymbol{n}$ & \makecell{\textbf{Spearman}\\$\boldsymbol{\rho}$} & $\boldsymbol{p}$ \\",
             r"\midrule"]
    for r in per_model:
        p_s = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        lines.append(f"{pp.LABELS[r['model']]} & {r['n']:,} & ${r['rho']:+.3f}$ & {p_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(pp.OUT, "tables", "plausibility.tex")
    open(tex_path, "w").write("\n".join(lines))
    print(f"\nwrote {tex_path}")


if __name__ == "__main__":
    main()
