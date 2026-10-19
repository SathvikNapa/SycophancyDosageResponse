"""
rq1_matched80_subsample.py — Cross-cutting check #2 (NAACL plan): compares
matched 80-question subsamples (GeminiFlash/HLE's actual eligible n_q, not
an artificial restriction -- see rq1_mixed_effects.py's n_questions column
and the session note on why it's 80) against each model/dataset cell's own
full-size RQ1 result, to see whether n=80 is big enough to reach a similar
conclusion. This is NOT a remedy for selection bias (the plan is explicit
about that) -- it only asks whether a sample this size is underpowered
relative to what's otherwise available.

For every (model, dataset) cell with n_q > 80, draws 500 random 80-question
subsamples (without replacement, matching an actual single smaller dataset
rather than a bootstrap resample), recomputes rq1()'s delta/CI/p on each,
and reports how often the full-data conclusion (sign, significance) would
have survived at n=80.

Usage: python analysis/rq1_matched80_subsample.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

N_SUBSAMPLES = 500
N_SUB = 80


def subsample_rq1(d, rng, n_sub, n_reps):
    queries = d["query"].unique()
    if len(queries) < n_sub:
        return None
    deltas, ps = [], []
    for _ in range(n_reps):
        chosen = rng.choice(queries, n_sub, replace=False)
        sub = d[d["query"].isin(chosen)]
        r = pp.rq1(sub, n_boot=1, rng=rng)  # n_boot=1: skip the inner CI bootstrap, we only need delta/p here
        deltas.append(r["delta"])
        ps.append(r["p"])
    return np.array(deltas), np.array(ps)


def main():
    rng = np.random.default_rng(0)
    full = pd.read_csv(os.path.join(pp.OUT, "results.csv"))
    rows = []
    for dataset in pp.DATASETS:
        pool = pp.canonical_pool(dataset)
        for model in pp.MODELS:
            df, _ = pp.load_cell(model, dataset, pool)
            if df is None:
                continue
            d, _ = pp.dialogues(df)
            d = d[d["complete"] & d["t0_correct"] & d["entropy"].notna()]
            n_q = d["query"].nunique()
            if n_q <= N_SUB:
                continue  # this cell IS already <=80 (or close); nothing to subsample from
            out = subsample_rq1(d, rng, N_SUB, N_SUBSAMPLES)
            if out is None:
                continue
            deltas, ps = out
            full_row = full[(full.model == model) & (full.dataset == dataset)]
            full_delta = full_row["delta"].iloc[0] if len(full_row) else np.nan
            full_sig = (full_row["p"].iloc[0] < 0.05) if len(full_row) and pd.notna(full_row["p"].iloc[0]) else False
            same_sign = np.mean(np.sign(deltas) == np.sign(full_delta)) if pd.notna(full_delta) else np.nan
            sig_at_80 = np.mean(ps < 0.05)
            rows.append({
                "model": model, "dataset": dataset, "n_q_full": n_q,
                "full_delta": full_delta, "full_significant": full_sig,
                "subsample_delta_mean": np.nanmean(deltas), "subsample_delta_std": np.nanstd(deltas),
                "pct_same_sign_as_full": 100 * same_sign, "pct_significant_at_n80": 100 * sig_at_80,
            })
            print(f"{model:15s} {dataset:15s} n_q_full={n_q:4d}  full_delta={full_delta:+.1f}  "
                  f"n80_delta_mean={np.nanmean(deltas):+.1f} (sd={np.nanstd(deltas):.1f})  "
                  f"same_sign={100*same_sign:.0f}%  sig_at_n80={100*sig_at_80:.0f}%", flush=True)

    out = pd.DataFrame(rows)
    out_path = os.path.join(pp.OUT, "rq1_matched80_subsample.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
