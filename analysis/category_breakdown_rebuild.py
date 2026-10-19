"""
category_breakdown_rebuild.py — Rebuilds app:category_breakdown (Subject-
Category Robustness, MMLU-Pro) using paper_pipeline.py's RQ1 definition
(ever-flip among T0-correct dialogues, cluster bootstrap CI, cluster-robust
logit test), pooled across all 5 Claude/GPT models within each subject
group, rather than the retired pooled-turn-level Mann-Whitney test.

STEM = biology, chemistry, computer science, engineering, math, physics.
Humanities & Social = business, economics, health, history, law,
philosophy, psychology, other. Same grouping the original used.

Usage: python analysis/category_breakdown_rebuild.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

STEM = {"biology", "chemistry", "computer science", "engineering", "math", "physics"}
HUMANITIES = {"business", "economics", "health", "history", "law", "philosophy", "psychology", "other"}
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]


def category_map():
    path = os.path.join(pp.cell_dir("ClaudeHaiku", "mmlu_pro"), "base_experiment_metadata.pkl")
    meta = pickle.load(open(path, "rb"))
    return {m["query"]: m["category"] for m in meta}


def pooled_dialogues():
    pool = pp.canonical_pool("mmlu_pro")
    cat = category_map()
    frames = []
    for model in MODELS:
        df, _ = pp.load_cell(model, "mmlu_pro", pool)
        d, _ = pp.dialogues(df)
        d = d[d["complete"] & d["t0_correct"] & d["entropy"].notna()].copy()
        d["model"] = model
        d["cluster"] = model + "||" + d["query"]
        d["group"] = d["query"].map(cat).map(lambda c: "STEM" if c in STEM else "Humanities & Social")
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def group_stats(d, n_boot, rng):
    c, u = d[d["certain"]], d[~d["certain"]]
    cert, _, _ = pp.wilson(int(c["ever_flip"].sum()), len(c))
    unc, _, _ = pp.wilson(int(u["ever_flip"].sum()), len(u))
    by_cluster = {k: g for k, g in d.groupby("cluster")}
    clusters = np.array(list(by_cluster))
    diffs = []
    for _ in range(n_boot):
        s = pd.concat([by_cluster[k] for k in rng.choice(clusters, len(clusters))])
        cc, uu = s[s["certain"]], s[~s["certain"]]
        if len(cc) and len(uu):
            diffs.append(100 * (uu["ever_flip"].mean() - cc["ever_flip"].mean()))
    lo, hi = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)) if diffs else (np.nan, np.nan)
    fit = pp.cluster_logit("ever_flip ~ unc", d.assign(ever_flip=d["ever_flip"].astype(int),
                                                       unc=(~d["certain"]).astype(int)), "cluster")
    p = fit.pvalues["unc"] / 2 if fit.params["unc"] > 0 else 1 - fit.pvalues["unc"] / 2
    return dict(n_q=d["query"].nunique(), n_dlg=len(d), cert=cert, unc=unc,
                delta=unc - cert, lo=lo, hi=hi, p=p)


def main():
    rng = np.random.default_rng(0)
    d = pooled_dialogues()
    rows = {g: group_stats(d[d.group == g], 500, rng) for g in ("STEM", "Humanities & Social")}
    out = pd.DataFrame(rows).T
    out.to_csv(os.path.join(pp.OUT, "category_breakdown.csv"))
    print(out.to_string(), flush=True)

    lines = [r"\begin{tabular}{@{}lccc c@{}}", r"\toprule",
             r"\textbf{Group} & \textbf{Certain (\%)} & \textbf{Uncertain (\%)} & "
             r"$\boldsymbol{\Delta}_{\mathrm{pp}}$ [95\% CI] & $\boldsymbol{p}$ \\", r"\midrule"]
    for g, r in out.iterrows():
        p_s = "$<$0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        g_tex = g.replace("&", r"\&")  # literal "&" in a group name must be
                                       # escaped, not left as a column separator
        lines.append(f"{g_tex} & {r['cert']:.1f} & {r['unc']:.1f} & "
                     f"${r['delta']:+.1f}$ [${r['lo']:+.1f}$, ${r['hi']:+.1f}$] & {p_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(pp.OUT, "tables", "category_breakdown.tex")
    open(tex_path, "w").write("\n".join(lines))
    print(f"wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
