"""
single_turn_entropy_correlation.py — Joins the single-turn pressure
robustness check's flip data (experiment_out/<model>/single_turn_pressure/
results_n25.pkl, R=25 reps x 6 pressure categories per T0-correct query)
against reparsed baseline entropy (paper_pipeline.load_baseline), to
support app:single_turn's Spearman-correlation claim.

No committed script previously did this join: run_single_turn_pressure.py
loads entropy only to print summary statistics at generation time, and
does not persist it onto the saved results or compute this correlation.
This script also uses reparsed (not raw) baseline entropy, consistent
with every other entropy-based analysis in the final pipeline.

Per-query flip rate pools all 6 pressure categories x 25 reps (150
one-turn trials per query, same quantity as tab:single_turn's "All"
column, computed per query instead of averaged over queries).

Usage: python analysis/single_turn_entropy_correlation.py
"""
import os
import pickle
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "GeminiFlash"]


def load_baseline_any_layout(model):
    """paper_pipeline.load_baseline assumes MMLU-Pro metadata lives at the
    model root (true for the other 5 models); GeminiFlash instead keeps it
    under a mmlu_pro/ subdir, same as every other dataset for that model."""
    ent_map, reparsed = pp.load_baseline(model, "mmlu_pro")
    if ent_map:
        return ent_map, reparsed
    d = os.path.join("experiment_out", model, "mmlu_pro")
    for name, is_reparsed in (("base_experiment_metadata_reparsed.pkl", True),
                              ("base_experiment_metadata.pkl", False)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            meta = pickle.load(open(p, "rb"))
            return {m["query"]: m["entropy"] for m in meta}, is_reparsed
    return {}, False


def main():
    rows = []
    for model in MODELS:
        path = os.path.join("experiment_out", model, "single_turn_pressure", "results_n25.pkl")
        df = pd.DataFrame(pickle.load(open(path, "rb")))
        df = df[df["flipped"].notna()].copy()
        df["flipped"] = df["flipped"].astype(bool)
        ent_map, reparsed = load_baseline_any_layout(model)
        per_q = df.groupby("query")["flipped"].mean()
        ent = per_q.index.map(ent_map.get)
        valid = ent.notna()
        if valid.sum() < 2:
            print(f"  {model:15s} insufficient entropy coverage, skipped")
            continue
        rho, p = stats.spearmanr(ent[valid], per_q[valid])
        rows.append({"model": model, "n_q": int(valid.sum()), "rho": rho, "p": p, "reparsed": reparsed})
        print(f"  {model:15s} n_q={int(valid.sum()):4d}  rho={rho:+.3f}  p={p:.4g}  reparsed={reparsed}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(pp.OUT, "single_turn_entropy.csv"), index=False)
    n_sig = int((out["p"] < 0.05).sum())
    print(f"\nrho range: [{out['rho'].min():+.3f}, {out['rho'].max():+.3f}]")
    print(f"significant at p<0.05: {n_sig} of {len(out)} models")
    print(f"wrote {os.path.join(pp.OUT, 'single_turn_entropy.csv')}")


if __name__ == "__main__":
    main()
