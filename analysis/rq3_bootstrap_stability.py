"""
rq3_bootstrap_stability.py — Question-level bootstrap stability of the
resister/capitulator/flat-n.s. classification used throughout RQ3
(paper_pipeline.rq3()'s own rule: cluster-robust logit of wrong~turn,
label = resister if slope<0 & p<0.05, capitulator if slope>0 & p<0.05,
else flat/n.s.), per NAACL plan Experiment 3.1.

Resamples unique question IDs with replacement, preserving every R=3
replicate and turn for each sampled question (same unit rq3() itself
operates on). Each resampled draw of a question is given a fresh
synthetic cluster id before refitting cluster-robust SEs -- otherwise a
query drawn twice would silently collapse into one cluster inside that
replicate's SE computation, understating its p-value and inflating
apparent classification stability. This is what the plan means by
"refit clustering in each replicate."

n_boot=2000 per (model, dataset) cell, matching the plan's own number.

Usage: python analysis/rq3_bootstrap_stability.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

N_BOOT = 2000


def label_from_slope(slope, p):
    if pd.isna(slope) or pd.isna(p):
        return "flat/n.s."
    if slope < 0 and p < 0.05:
        return "resister"
    if slope > 0 and p < 0.05:
        return "capitulator"
    return "flat/n.s."


def cell_long(model, dataset, pool):
    df, _ = pp.load_cell(model, dataset, pool)
    if df is None:
        return None
    d, w = pp.dialogues(df)
    keep = d[d["complete"] & d["t0_correct"]].set_index(["query", "run"]).index
    w = w.loc[w.index.isin(keep)]
    long = w[list(range(1, pp.N_TURNS))].stack().rename("wrong").reset_index().rename(columns={"level_2": "turn"})
    long.columns = ["query", "run", "turn", "wrong"]
    long["wrong"] = long["wrong"].astype(int)
    return long


def bootstrap_cell(long, rng, n_boot):
    queries = long["query"].unique()
    by_query = {q: g for q, g in long.groupby("query")}
    labels = []
    for _ in range(n_boot):
        draws = rng.choice(queries, len(queries))
        parts = []
        for i, q in enumerate(draws):
            g = by_query[q].copy()
            g["query"] = f"{q}__{i}"  # fresh synthetic cluster id per draw
            parts.append(g)
        resampled = pd.concat(parts, ignore_index=True)
        if resampled["wrong"].nunique() < 2:
            labels.append("flat/n.s.")
            continue
        try:
            fit = pp.cluster_logit("wrong ~ turn", resampled, "query")
            labels.append(label_from_slope(fit.params["turn"], fit.pvalues["turn"]))
        except Exception:
            labels.append("flat/n.s.")
    return labels


def run_cell(args):
    model, dataset, seed = args
    pool = pp.canonical_pool(dataset)
    long = cell_long(model, dataset, pool)
    if long is None or long.empty:
        return None
    rng = np.random.default_rng(seed)
    labels = bootstrap_cell(long, rng, N_BOOT)
    counts = pd.Series(labels).value_counts(normalize=True) * 100
    orig = pp.rq3(*pp.dialogues(pp.load_cell(model, dataset, pool)[0]))
    row = {"model": model, "dataset": dataset, "orig_label": orig["label"],
           "n_boot": len(labels)}
    for lab in ("resister", "capitulator", "flat/n.s."):
        row[f"pct_{lab.replace('/', '_')}"] = counts.get(lab, 0.0)
    pct_flat = row["pct_flat_n.s."]
    print(f"done {model}/{dataset}: orig={orig['label']}  "
          f"resister={row['pct_resister']:.1f}%  capitulator={row['pct_capitulator']:.1f}%  "
          f"flat={pct_flat:.1f}%", flush=True)
    return row


def main():
    import multiprocessing as mp

    cells = [(model, dataset, i) for i, (dataset, model) in
             enumerate((d, m) for d in pp.DATASETS for m in pp.MODELS)]
    n_workers = min(len(cells), mp.cpu_count())
    print(f"running {len(cells)} cells across {n_workers} parallel workers", flush=True)

    with mp.Pool(n_workers) as p:
        results = p.map(run_cell, cells)

    rows = [r for r in results if r is not None]
    out = pd.DataFrame(rows)
    out_path = os.path.join(pp.OUT, "rq3_bootstrap_stability.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(out.to_string(float_format=lambda x: f"{x:.1f}"))
    print(out.to_string(float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
