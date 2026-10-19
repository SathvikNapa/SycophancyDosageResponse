"""
rq3_classification_sensitivity.py — Classification sensitivity (NAACL plan
Experiment 3.4) for the RQ3 resister/capitulator/flat.n.s. labels.

The RQ3 label is a direct statistical threshold rule (slope sign + p<0.05
from a cluster-robust logit), not an unsupervised clustering algorithm, so
pairwise co-clustering frequency / adjusted Rand index don't directly
apply (there is no partition to compare across perturbations -- each
cell's label depends only on that cell's own data). The appropriate
translations, used here instead:
  1. Threshold sensitivity: does any cell's label change at p<0.01 or
     p<0.10 instead of the paper's p<0.05?
  2. Leave-one-model-out sensitivity for the RQ2 pooled (DerSimonian-Laird)
     interaction estimate, since that one IS a genuine pooled statistic
     across models where single-model influence is a real question.

Pure reanalysis of analysis/paper_out/results.csv, no new computation.

Usage: python analysis/rq3_classification_sensitivity.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp


def label_at(slope, p, alpha):
    if pd.isna(slope) or pd.isna(p):
        return "flat/n.s."
    if slope < 0 and p < alpha:
        return "resister"
    if slope > 0 and p < alpha:
        return "capitulator"
    return "flat/n.s."


def threshold_sensitivity():
    df = pd.read_csv(os.path.join(pp.OUT, "results.csv"))
    cols = {}
    for alpha in (0.01, 0.05, 0.10):
        col = f"label_a{alpha:.2f}"
        df[col] = df.apply(lambda r: label_at(r["slope"], r["slope_p"], alpha), axis=1)
        cols[alpha] = col
    changed = df[df[cols[0.01]] != df[cols[0.10]]]
    print("Threshold sensitivity (p<0.01 vs p<0.05 vs p<0.10):")
    print(df[["model", "dataset", "slope", "slope_p", cols[0.01], cols[0.05], cols[0.10]]]
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n{len(changed)} of {len(df)} cells have a DIFFERENT label between alpha=0.01 and alpha=0.10:")
    if len(changed):
        print(changed[["model", "dataset", "slope_p", cols[0.01], cols[0.10]]].to_string())
    df.to_csv(os.path.join(pp.OUT, "rq3_threshold_sensitivity.csv"), index=False)


def loo_pooled_interaction():
    """Leave-one-model-out sensitivity for RQ2's pooled beta_PH per dataset."""
    df = pd.read_csv(os.path.join(pp.OUT, "results.csv"))
    rows = []
    for dataset in pp.DATASETS:
        sub = df[(df.dataset == dataset) & df["bPH"].notna() & df["sePH"].notna()]
        if sub.empty:
            continue
        models = sub["model"].tolist()
        for excluded in models:
            kept = sub[sub["model"] != excluded]
            est, lo, hi, p, tau2 = pp.dersimonian_laird(kept["bPH"].values, kept["sePH"].values)
            rows.append({"dataset": dataset, "excluded_model": excluded, "beta_PH_loo": est,
                         "lo": lo, "hi": hi, "p": p, "n_kept": len(kept)})
    out = pd.DataFrame(rows)
    if out.empty:
        print("\nLOO pooled-interaction check: beta_PH_se column not in results.csv, skipped "
              "(per-model SEs needed; check paper_pipeline.rq2() output columns).")
        return
    print("\nLeave-one-model-out sensitivity, pooled RQ2 interaction (beta_PH):")
    print(out.to_string(float_format=lambda x: f"{x:.4f}"))
    out.to_csv(os.path.join(pp.OUT, "rq3_loo_pooled_interaction.csv"), index=False)


if __name__ == "__main__":
    threshold_sensitivity()
    loo_pooled_interaction()
