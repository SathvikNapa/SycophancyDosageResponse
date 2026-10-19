"""
rq4_paired_transitions.py — Paired confidence changes by answer-transition
type (NAACL plan Experiment 4.2), on the 3 headline cells.

For every dialogue (query, run), compares T0's (correct, confidence)
against every later turn's (correct, confidence), classifies each pair
into one of four transitions (correct->correct, correct->incorrect,
incorrect->correct, incorrect->incorrect), and reports mean
delta-confidence = conf(T_t) - conf(T0) per transition.

Uses orig_self_conf (the original GPT-5.4-Mini-based calibration already
used throughout the paper, unaffected by any of this session's Haiku work)
as the primary column, with haiku_final_conf alongside as a cross-check --
flagged explicitly as partial: the Claude Haiku batch had a max_tokens
truncation bug (fixed only in a smoke test, not re-run at scale per the
user's decision to stop spending on 4.1), so haiku_final_conf is missing
for ~43% of traces and the missingness is not random (shorter-preamble
responses completed, longer ones didn't) -- reported for whatever signal
it carries, not as a clean replication.

Usage: python analysis/rq4_paired_transitions.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp


def transition_label(t0_correct, tt_correct):
    if pd.isna(t0_correct) or pd.isna(tt_correct):
        return None
    if t0_correct and tt_correct:
        return "correct->correct"
    if t0_correct and not tt_correct:
        return "correct->incorrect"
    if not t0_correct and tt_correct:
        return "incorrect->correct"
    return "incorrect->incorrect"


def main():
    df = pd.read_pickle(os.path.join(pp.OUT, "calibration_haiku_results.pkl"))
    n_total = len(df)
    n_haiku = df["haiku_final_conf"].notna().sum()
    print(f"orig_self_conf coverage: {df['orig_self_conf'].notna().sum():,}/{n_total:,}")
    print(f"haiku_final_conf coverage: {n_haiku:,}/{n_total:,} "
          f"({100*n_haiku/n_total:.1f}% -- partial due to the truncation bug, see docstring)\n")

    rows = []
    for (model, dataset, query, run), g in df.groupby(["model", "dataset", "query", "run"]):
        t0 = g[g.turn == 0]
        if t0.empty:
            continue
        t0 = t0.iloc[0]
        for _, tt in g[g.turn > 0].iterrows():
            label = transition_label(t0["correct"], tt["correct"])
            if label is None:
                continue
            d_orig = (tt["orig_self_conf"] - t0["orig_self_conf"]
                      if pd.notna(tt["orig_self_conf"]) and pd.notna(t0["orig_self_conf"]) else np.nan)
            d_haiku = (tt["haiku_final_conf"] - t0["haiku_final_conf"]
                       if pd.notna(tt["haiku_final_conf"]) and pd.notna(t0["haiku_final_conf"]) else np.nan)
            rows.append({"model": model, "dataset": dataset, "turn": tt["turn"],
                         "transition": label, "delta_orig": d_orig, "delta_haiku": d_haiku})

    pt = pd.DataFrame(rows)
    print("Mean delta-confidence by transition type (pooled across headline cells):")
    g = pt.groupby("transition").agg(
        n=("delta_orig", "size"),
        mean_delta_orig=("delta_orig", "mean"), sd_delta_orig=("delta_orig", "std"),
        n_haiku=("delta_haiku", "count"),
        mean_delta_haiku=("delta_haiku", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nPer (model, dataset), correct->incorrect transitions specifically (the key RQ4 case):")
    c2i = pt[pt.transition == "correct->incorrect"]
    g2 = c2i.groupby(["model", "dataset"]).agg(
        n=("delta_orig", "size"), mean_delta_orig=("delta_orig", "mean"),
        n_haiku=("delta_haiku", "count"), mean_delta_haiku=("delta_haiku", "mean"),
    )
    print(g2.to_string(float_format=lambda x: f"{x:.2f}"))

    out_path = os.path.join(pp.OUT, "rq4_paired_transitions.csv")
    pt.to_csv(out_path, index=False)
    g.to_csv(os.path.join(pp.OUT, "rq4_paired_transitions_summary.csv"))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
