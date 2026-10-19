"""
calibration_haiku_extract.py — Re-extracts the headline-cell traces at
per-step granularity (not the whole-trace text used by the earlier
plausibility/judge scripts), to replicate calibrate_trace()/calibrate_step()
(src/sycophancy/calibrator.py) with Claude Haiku as the calibrator model
instead of GPT-5.4 Mini, for direct comparison against the existing
tr.steps[-1].confidence values already used throughout the paper.

Faithful to the original: options formatted as "\n".join(options) (a
List[str] of "A. text" items from base_experiment_metadata.pkl, exactly
as scripts/run_reasoning_calibrated.py passes them to calibrate_trace),
not reformatted into a dict-style listing.

Usage: python analysis/calibration_haiku_extract.py
"""
import glob
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

HEADLINE_CELLS = [("ClaudeHaiku", "gpqa_diamond"), ("ClaudeHaiku", "hle"), ("ClaudeSonnet", "hle")]


def options_map(model, dataset):
    d = pp.cell_dir(model, dataset)
    meta = pickle.load(open(os.path.join(d, "base_experiment_metadata.pkl"), "rb"))
    return {m["query"]: m["options"] for m in meta}


def extract_cell(model, dataset, pool):
    paths = sorted(glob.glob(os.path.join(pp.cell_dir(model, dataset), "reasoning_calibrated_bin", "bin_*_reasoning.pkl")))
    opts_map = options_map(model, dataset)
    rows = []
    for p in paths:
        trajs = pickle.load(open(p, "rb"))
        for traj in trajs:
            if traj.query not in pool:
                continue
            opts = opts_map.get(traj.query, [])
            for t, turn in enumerate(traj.raw_traces[:pp.N_TURNS]):
                for r, tr in enumerate(turn[:pp.R]):
                    if not tr.steps:
                        continue
                    rows.append({
                        "model": model, "dataset": dataset, "query": traj.query, "options": opts,
                        "gold": traj.gold_answer, "run": r, "turn": t,
                        "step_texts": [s.step_text for s in tr.steps],
                        "orig_self_conf": tr.steps[-1].confidence,
                        "final_answer": tr.final_answer,
                        "correct": (tr.final_answer == traj.gold_answer) if tr.final_answer else None,
                    })
    return rows


def main():
    all_rows = []
    for model, dataset in HEADLINE_CELLS:
        pool = pp.canonical_pool(dataset)
        rows = extract_cell(model, dataset, pool)
        print(f"{model:15s} {dataset:15s} {len(rows):6,} traces", flush=True)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).reset_index(drop=True)
    df["trace_id"] = df.index
    n_steps = df["step_texts"].apply(len)
    print(f"\ntotal traces: {len(df):,}")
    print(f"total step-calls: {n_steps.sum():,}  (mean {n_steps.mean():.2f} steps/trace, max {n_steps.max()})")

    out_path = os.path.join(pp.OUT, "calibration_haiku_headline.pkl")
    df.to_pickle(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
