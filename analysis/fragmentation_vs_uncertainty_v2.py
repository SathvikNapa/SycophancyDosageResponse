"""
fragmentation_vs_uncertainty_v2.py — Rebuilds app:fragmentation on the
final corrected/complete R=3 CoT pipeline (paper_pipeline.py), the same
data and lost-consensus definition as the main-text fragmentation
paragraph and paper_pipeline.fragmentation(), instead of the original
fragmentation_vs_uncertainty.py's ad-hoc truncated-query join against
unreparsed base_experiment_metadata.pkl on whatever bin files happened to
be present (no canonical-pool or R=3 restriction).

Per-query unit: for each (model, dataset, query) with a T0 majority-correct
answer among the canonical R=3 reps, lost_consensus = whether the R=3
majority answer disappears (no answer with >=2/3 agreement) at any later
turn before the dialogue is otherwise incomplete. Joined against reparsed
baseline entropy. Same six targets as the main-text fragmentation claim
(5 Claude/GPT models + Gemini 3.5 Flash).

Usage: python analysis/fragmentation_vs_uncertainty_v2.py
"""
import os
import sys
from collections import Counter

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp


def per_query_rows(model, dataset, pool):
    df, reparsed = pp.load_cell(model, dataset, pool)
    if df is None:
        return []
    ent_map, _ = pp.load_baseline(model, dataset)
    g = df[df["answer"].notna()].groupby(["query", "turn"])["answer"].apply(list)
    rows = []
    for q in df["query"].unique():
        t0 = g.get((q, 0))
        if not t0 or len(t0) < pp.R:
            continue
        maj = Counter(t0).most_common(1)[0]
        gold = df.loc[df["query"] == q, "gold"].iloc[0]
        if maj[1] < 2 or maj[0] != gold:
            continue
        entropy = ent_map.get(q)
        if entropy is None:
            continue
        lost = False
        for t in range(1, pp.N_TURNS):
            a = g.get((q, t))
            if a is None or len(a) < pp.R:
                break
            if Counter(a).most_common(1)[0][1] < 2:
                lost = True
                break
        rows.append({"model": model, "dataset": dataset, "query": q,
                     "entropy": entropy, "lost_consensus": int(lost)})
    return rows


def main():
    rows = []
    for dataset in pp.DATASETS:
        pool = pp.canonical_pool(dataset)
        for model in pp.MODELS:
            rows.extend(per_query_rows(model, dataset, pool))
    df = pd.DataFrame(rows)
    print(f"Total T0-correct (model, dataset, query) rows: {len(df)}\n")

    print(df.groupby("dataset")["lost_consensus"].agg(["count", "mean"]))
    print()

    per_ds = {}
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        if sub["entropy"].nunique() < 2:
            print(f"  {ds:15s} insufficient entropy variation, skipped")
            continue
        rho, p = stats.spearmanr(sub["entropy"], sub["lost_consensus"])
        per_ds[ds] = (rho, p, len(sub))
        print(f"  {ds:15s} n={len(sub):5d}  rho={rho:+.3f}  p={p:.4g}")

    rho_pool, p_pool = stats.spearmanr(df["entropy"], df["lost_consensus"])
    print(f"\nNaive pooled Spearman: rho={rho_pool:+.3f}  p={p_pool:.4g}")

    model_res = smf.logit("lost_consensus ~ entropy + C(model) + C(dataset)", data=df).fit(disp=0)
    beta = model_res.params["entropy"]
    se = model_res.bse["entropy"]
    pv = model_res.pvalues["entropy"]
    print(f"\nPooled logistic (model+dataset FE): beta(entropy)={beta:+.4f} (SE={se:.4f}, p={pv:.4g})")
    hle_coef = model_res.params.get("C(dataset)[T.hle]", float("nan"))
    hle_p = model_res.pvalues.get("C(dataset)[T.hle]", float("nan"))
    print(f"HLE dataset FE (rel. to baseline level): {hle_coef:+.3f} (p={hle_p:.4g})")

    out = {"n_total": len(df), "beta": beta, "se": se, "p": pv,
           "rho_pool": rho_pool, "p_pool": p_pool, "hle_coef": hle_coef, "hle_p": hle_p}
    for ds, (rho, p, n) in per_ds.items():
        out[f"rho_{ds}"] = rho
        out[f"p_{ds}"] = p
        out[f"n_{ds}"] = n
    pd.Series(out).to_csv(os.path.join(pp.OUT, "fragmentation_vs_uncertainty.csv"))
    print(f"\nwrote {os.path.join(pp.OUT, 'fragmentation_vs_uncertainty.csv')}")


if __name__ == "__main__":
    main()
