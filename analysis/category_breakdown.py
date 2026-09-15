"""
category_breakdown.py
----------------------
Reviewer-requested robustness check (NAACL/Stanford review, RQ1 section):
"Could you report per-subject or per-category effects (e.g., humanities
vs. STEM) on MMLU-Pro, to test whether uncertainty-driven flips are
concentrated in specific areas?"

Uses data already collected — no new API calls. Joins each question's
MMLU-Pro category (stored in base_experiment_metadata.pkl, identical
across models since all five target models see the same 420-question
MMLU-Pro set) onto the existing flip_data_mmlu.csv, then reruns the
paper's own RQ1 methodology (Wilson CIs + one-sided Mann-Whitney,
uncertain > certain; see run_combined_datasets_analysis.py) separately
within each subject group, pooling across all five models.

STEM / Humanities & Social grouping (fixed here, not tuned to the
result — report this list verbatim if asked how the split was made):
  STEM                  = biology, chemistry, computer science,
                          engineering, math, physics
  Humanities & Social   = business, economics, health, history, law,
                          other, philosophy, psychology

Usage:
    uv run python analysis/category_breakdown.py
"""

from __future__ import annotations

import glob
import pickle

import numpy as np
import pandas as pd
from scipy import stats

EXPERIMENT_OUT = "experiment_out"
FLIP_DATA = "analysis/flip_data_mmlu.csv"

STEM = {"biology", "chemistry", "computer science", "engineering", "math", "physics"}
HUMANITIES_SOCIAL = {"business", "economics", "health", "history", "law", "other",
                      "philosophy", "psychology"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p * 100, max(0, (c - m) * 100), min(100, (c + m) * 100)


def load_category_map() -> dict[str, str]:
    """query -> category, from any one model's MMLU-Pro baseline metadata
    (the question set is identical across all five target models)."""
    for pkl in sorted(glob.glob(f"{EXPERIMENT_OUT}/*/base_experiment_metadata.pkl")):
        with open(pkl, "rb") as f:
            meta = pickle.load(f)
        cat_map = {m["query"]: m.get("category") for m in meta if m.get("category")}
        if cat_map:
            return cat_map
    raise FileNotFoundError("No base_experiment_metadata.pkl with a 'category' field found.")


def group_for(category: str) -> str:
    if category in STEM:
        return "STEM"
    if category in HUMANITIES_SOCIAL:
        return "Humanities & Social"
    return "Unclassified"


def main() -> None:
    df = pd.read_csv(FLIP_DATA)  # columns: model, question, turn, entropy, flip
    df = df[df["turn"] >= 1].copy()

    cat_map = load_category_map()
    df["category"] = df["question"].map(cat_map)
    n_unmapped = df["category"].isna().sum()
    if n_unmapped:
        print(f"WARNING: {n_unmapped} rows have no category match (dropped).")
    df = df.dropna(subset=["category"])
    df["group"] = df["category"].map(group_for)

    df["certain"] = (df["entropy"] == 0.0).astype(int)

    print("=" * 72)
    print("Per-category flip-rate contrast (certain vs. uncertain), pooled "
          "across all 5 models, MMLU-Pro, turns >= 1")
    print("=" * 72)

    n_questions_by_cat = (df.drop_duplicates("question")["category"].value_counts())
    print("\nQuestions per category (of 420 total):")
    print(n_questions_by_cat.to_string())

    rows = []
    for group in ["STEM", "Humanities & Social"]:
        g = df[df["group"] == group]
        cert, unct = g[g["certain"] == 1], g[g["certain"] == 0]
        cp, clo, chi = wilson(int(cert["flip"].sum()), len(cert))
        up, ulo, uhi = wilson(int(unct["flip"].sum()), len(unct))
        _, pv = stats.mannwhitneyu(unct["flip"], cert["flip"], alternative="greater") \
            if len(unct) > 0 and len(cert) > 0 else (0, np.nan)
        n_q = g["question"].nunique()
        rows.append({
            "group": group, "n_questions": n_q,
            "certain_n": len(cert), "certain_flip_pct": cp, "certain_lo": clo, "certain_hi": chi,
            "uncertain_n": len(unct), "uncertain_flip_pct": up, "uncertain_lo": ulo, "uncertain_hi": uhi,
            "delta_pp": up - cp, "p_one_sided": pv,
        })
        print(f"\n{group} ({n_q} questions):")
        print(f"  Certain:   {cp:5.1f}% [{clo:.1f}, {chi:.1f}]  (n={len(cert)})")
        print(f"  Uncertain: {up:5.1f}% [{ulo:.1f}, {uhi:.1f}]  (n={len(unct)})")
        print(f"  Delta:     {up - cp:+5.1f}pp   Mann-Whitney (uncertain > certain) p={pv:.4g}")

    out = pd.DataFrame(rows)
    out_path = "analysis/results/category_breakdown_mmlu.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
