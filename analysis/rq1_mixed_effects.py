"""
rq1_mixed_effects.py — Extends rq2_appendix.mixed_effects_glm() (fit only on
MMLU-Pro) to GPQA-Diamond and HLE, for the pooled RQ1 uncertainty main effect
(beta_H) requested for the appendix: a single hierarchical model accounting
for model, question, and repeated turn observations, rather than per-cell
Holm-corrected tests alone.

Same model spec, same 5 Claude/GPT models (GeminiFlash excluded, matching
the original's scope), same crossed random effects:
    logit P(flip) = b0 + u_m + v_q + w_m*turn + bP*turn + bH*Hc + bPH*turn*Hc
MMLU-Pro's fit already exists in analysis/paper_out/rq2_mixed_effects.pkl
and is NOT re-run here; this script only adds gpqa_diamond and hle.

Usage: python analysis/rq1_mixed_effects.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]


def fit_one(dataset):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    pool = pp.canonical_pool(dataset)
    rows = []
    for model in MODELS:
        df, _ = pp.load_cell(model, dataset, pool)
        if df is None:
            continue
        d, _ = pp.dialogues(df)
        d = d[d["complete"] & d["t0_correct"] & d["entropy"].notna()].copy()
        if d.empty:
            continue
        d["Hc"] = d["entropy"] - d["entropy"].mean()
        for _, row in d.iterrows():
            last = int(row["first_flip"]) if row["ever_flip"] else pp.N_TURNS - 1
            for t in range(1, last + 1):
                rows.append({"model": model, "query": row["query"], "turn": float(t), "Hc": row["Hc"],
                            "flip": int(row["ever_flip"] and t == last)})
    person_turns = pd.DataFrame(rows)
    print(f"{dataset}: {len(person_turns)} person-turn rows, "
          f"{person_turns['model'].nunique()} models, {person_turns['query'].nunique()} questions", flush=True)

    vc_formulas = {"model_intercept": "0 + C(model)", "question_intercept": "0 + C(query)",
                   "model_slope": "0 + C(model):turn"}
    try:
        model_fit = BinomialBayesMixedGLM.from_formula(
            "flip ~ turn + Hc + turn:Hc", vc_formulas, person_turns, vcp_p=0.5)
        result = model_fit.fit_vb()
        converged = True
    except Exception as e:
        print(f"{dataset}: mixed-effects GLM failed to fit: {type(e).__name__}: {e}", flush=True)
        return None

    fe_mean = dict(zip(model_fit.fep_names, result.fe_mean))
    fe_sd = dict(zip(model_fit.fep_names, result.fe_sd))
    vcp_mean = dict(zip(model_fit.vcp_names, result.vcp_mean))
    out = {
        "dataset": dataset,
        "beta_PH": fe_mean.get("turn:Hc"), "beta_PH_sd": fe_sd.get("turn:Hc"),
        "beta_P": fe_mean.get("turn"), "beta_P_sd": fe_sd.get("turn"),
        "beta_H": fe_mean.get("Hc"), "beta_H_sd": fe_sd.get("Hc"),
        "n_rows": len(person_turns), "n_questions": person_turns["query"].nunique(),
        "converged": converged,
        "sigma_model_intercept": float(np.exp(vcp_mean.get("model_intercept", np.nan))),
        "sigma_question_intercept": float(np.exp(vcp_mean.get("question_intercept", np.nan))),
        "sigma_model_slope": float(np.exp(vcp_mean.get("model_slope", np.nan))),
    }
    print(f"{dataset} result:", out, flush=True)
    return out


def main():
    results = [pickle.load(open(os.path.join(pp.OUT, "rq2_mixed_effects.pkl"), "rb"))]
    results[0]["dataset"] = "mmlu_pro"
    for dataset in ["gpqa_diamond", "hle"]:
        r = fit_one(dataset)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    out_path = os.path.join(pp.OUT, "rq1_mixed_effects_all.csv")
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(df[["dataset", "beta_H", "beta_H_sd", "beta_P", "beta_P_sd", "beta_PH", "beta_PH_sd", "n_rows", "n_questions"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
