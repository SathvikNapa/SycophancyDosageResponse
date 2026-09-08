"""
convert_reasoning_to_sycophancy.py
-----------------------------------
Reconstructs bin_*_repeated.pkl (run_sycophancy format) from
bin_*_reasoning.pkl (run_reasoning_sycophancy format).

The reasoning pipeline stores UncertaintyTrajectory objects whose raw_traces
field contains [turn][run] ReasoningTrace objects. Each ReasoningTrace has the
final_answer the model gave at that turn, so we can reconstruct every field
that build_flat_df() needs:

    is_wrong, is_correct, first_wrong_turn, turn_categories, parsed_turns

Fields that run_sycophancy.py writes but reasoning traces do not store
(wrong_answer_text, wrong_answer_letter, prompt, seed) are filled with None —
they are not read by build_flat_df() or any downstream analysis.

Usage
-----
# Convert all models / datasets that have reasoning_bin but no entropy_bin:
    python convert_reasoning_to_sycophancy.py

# Convert a specific model + dataset:
    python convert_reasoning_to_sycophancy.py --model ClaudeHaiku --dataset gpqa_diamond

# Dry run (show what would be written, write nothing):
    python convert_reasoning_to_sycophancy.py --dry_run
"""

from __future__ import annotations

import argparse
import os
import pickle
from glob import glob
from typing import Optional

import numpy as np

EXPERIMENT_OUT = "experiment_out"
DATASET_SUBDIRS = {"gpqa_diamond", "aime_2025"}

TURN_CATEGORIES = [
    None,
    "personal_disagreement",
    "personal_certainty",
    "social_proof",
    "expert_authority",
    "personal_accusatory",
    "crowd_consensus",
]


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def trajectory_to_repeated_question(traj) -> dict:
    """
    Converts one UncertaintyTrajectory to the question-dict format that
    run_sycophancy.py writes into bin_*_repeated.pkl.
    """
    query      = traj.query or ""
    gold       = (traj.gold_answer or "").strip().upper()
    raw_traces = traj.raw_traces  # [turn_idx][run_idx] -> ReasoningTrace

    n_turns = len(raw_traces)
    n_runs  = len(raw_traces[0]) if raw_traces else 0

    turn_cats = TURN_CATEGORIES[:n_turns]

    raw_runs = []
    for run_idx in range(n_runs):
        parsed_turns      = []
        is_wrong          = []
        is_correct        = []
        turn_confidences  = []
        first_wrong_turn  = n_turns  # sentinel: never flipped

        for turn_idx in range(n_turns):
            trace = raw_traces[turn_idx][run_idx]
            ans   = (trace.final_answer or "").strip().upper()
            wrong = int(bool(ans) and bool(gold) and ans != gold)

            parsed_turns.append(ans)
            is_wrong.append(wrong)
            is_correct.append(1 - wrong)
            turn_confidences.append(float(trace.final_confidence or 0.0))

            if wrong and first_wrong_turn == n_turns:
                first_wrong_turn = turn_idx

        first_flip_cat = (
            turn_cats[first_wrong_turn]
            if first_wrong_turn < n_turns
            else None
        )

        raw_runs.append({
            "query":               query,
            "gold_answer":         gold,
            "wrong_answer_text":   None,
            "wrong_answer_letter": None,
            "seed":                None,
            "timeout_s":           None,
            "prompt":              None,
            "n_turns":             n_turns,
            "raw_turns":           [raw_traces[t][run_idx].raw_text for t in range(n_turns)],
            "parsed_turns":        parsed_turns,
            "is_wrong":            is_wrong,
            "is_correct":          is_correct,
            "turn_categories":     turn_cats,
            "turn_confidences":    turn_confidences,
            "first_wrong_turn":    first_wrong_turn,
            "first_flip_category": first_flip_cat,
            "errors":              [],
        })

    # ── Question-level aggregates (mirrors run_sycophancy.py) ────────────────
    all_fwt      = [r["first_wrong_turn"] for r in raw_runs]
    flip_rate_by_run = [
        sum(r["is_wrong"]) / len(r["is_wrong"]) if r["is_wrong"] else 0.0
        for r in raw_runs
    ]

    cat_counts: dict = {}
    for r in raw_runs:
        cat = r["first_flip_category"]
        if cat is not None:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    return {
        "query":                    query,
        "gold_answer":              gold,
        "raw_runs":                 raw_runs,
        "first_wrong_turn_per_run": all_fwt,
        "flip_rate":                float(np.mean(flip_rate_by_run)),
        "max_fwt":                  int(max(all_fwt)) if all_fwt else n_turns,
        "median_fwt":               float(np.median(all_fwt)) if all_fwt else float(n_turns),
        "mean_fwt":                 float(np.mean(all_fwt)) if all_fwt else float(n_turns),
        "mean_turn_confidences":    [
            float(np.mean([r["turn_confidences"][t] for r in raw_runs]))
            for t in range(n_turns)
        ],
        "first_flip_category_counts": cat_counts,
        "flip_rate_by_category":    {},
        "mean_conf_at_first_flip_by_category": {},
    }


def convert_bin(reasoning_pkl: str, out_pkl: str, dry_run: bool) -> int:
    """Converts one bin_*_reasoning.pkl -> bin_*_repeated.pkl. Returns question count."""
    with open(reasoning_pkl, "rb") as f:
        trajectories = pickle.load(f)

    questions = [trajectory_to_repeated_question(t) for t in trajectories]

    if dry_run:
        print(f"  [DRY RUN] Would write {len(questions)} questions -> {out_pkl}")
    else:
        os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
        with open(out_pkl, "wb") as f:
            pickle.dump(questions, f)
        print(f"  Wrote {len(questions)} questions -> {out_pkl}")

    return len(questions)


# ---------------------------------------------------------------------------
# Discovery: find all reasoning_bin dirs that lack entropy_bin
# ---------------------------------------------------------------------------

def iter_conversion_targets(
    experiment_dir: str,
    model_filter:   Optional[str],
    dataset_filter: Optional[str],
):
    """
    Yields (model, dataset_label, reasoning_bin_dir, entropy_bin_dir) for
    every model/dataset that has reasoning_bin pkl files.
    """
    for model in sorted(os.listdir(experiment_dir)):
        model_dir = os.path.join(experiment_dir, model)
        if not os.path.isdir(model_dir):
            continue
        if model_filter and model != model_filter:
            continue

        # mmlu_pro (root-level reasoning_bin)
        if not dataset_filter or dataset_filter == "mmlu_pro":
            rbin = os.path.join(model_dir, "reasoning_bin")
            ebin = os.path.join(model_dir, "entropy_bin")
            if glob(os.path.join(rbin, "bin_*_reasoning.pkl")):
                yield model, "mmlu_pro", rbin, ebin

        # dataset subdirs
        for subdir in DATASET_SUBDIRS:
            if dataset_filter and dataset_filter not in (subdir, "all"):
                continue
            sub_dir  = os.path.join(model_dir, subdir)
            rbin     = os.path.join(sub_dir, "reasoning_bin")
            ebin     = os.path.join(sub_dir, "entropy_bin")
            if glob(os.path.join(rbin, "bin_*_reasoning.pkl")):
                yield model, subdir, rbin, ebin


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Convert reasoning bin pkls to sycophancy repeated format.")
    p.add_argument("--model",       type=str, default=None, help="Limit to one model key")
    p.add_argument("--dataset",     type=str, default=None,
                   help="Limit to one dataset: mmlu_pro | gpqa_diamond | aime_2025 | all")
    p.add_argument("--out_dir",     type=str, default=EXPERIMENT_OUT)
    p.add_argument("--dry_run",     action="store_true",
                   help="Print what would be written without writing anything")
    p.add_argument("--overwrite",   action="store_true",
                   help="Overwrite existing entropy_bin pkls (default: skip)")
    args = p.parse_args()

    total_written = 0
    total_skipped = 0

    for model, dataset, rbin_dir, ebin_dir in iter_conversion_targets(
        args.out_dir, args.model, args.dataset
    ):
        reasoning_pkls = sorted(glob(os.path.join(rbin_dir, "bin_*_reasoning.pkl")))
        print(f"\n{'='*60}")
        print(f"Model={model}  Dataset={dataset}")
        print(f"  reasoning_bin : {rbin_dir}")
        print(f"  entropy_bin   : {ebin_dir}")
        print(f"  Found {len(reasoning_pkls)} reasoning pkl(s)")

        for rpkl in reasoning_pkls:
            bin_id   = os.path.basename(rpkl).split("_")[1]
            out_pkl  = os.path.join(ebin_dir, f"bin_{bin_id}_repeated.pkl")

            if os.path.exists(out_pkl) and not args.overwrite:
                print(f"  SKIP bin {bin_id}: {out_pkl} already exists (use --overwrite to replace)")
                total_skipped += 1
                continue

            n = convert_bin(rpkl, out_pkl, dry_run=args.dry_run)
            total_written += 1

    print(f"\nDone. Written={total_written}  Skipped={total_skipped}")


if __name__ == "__main__":
    main()
