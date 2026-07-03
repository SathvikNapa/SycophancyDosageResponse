"""
run_ensemble_calibration.py
----------------------------
Ensemble calibration analysis: measures whether the CoT calibrator's
confidence scores are calibrated when averaged across K runs per
(question × turn).

Motivation
----------
calibration_analysis.py treats every reasoning step as an independent
prediction.  This script instead aggregates across K runs first:

  For each (question, turn):
    ensemble_conf  = mean(final-step confidence / 100  across K runs)
    run_is_correct = 1 if final-step belief == gold answer, else 0

ECE is then computed using ensemble_conf as the predicted probability
and run_is_correct as the binary label.  This tests:

  "When the ensemble assigns 70% confidence, do 70% of individual
   runs have the correct final belief?"

This is strictly more robust than per-step ECE because:
  1. Averaging K samples reduces estimation noise per question/turn.
  2. It captures the system-level calibration rather than step-level noise.
  3. It is directly comparable to how ensemble classifiers are evaluated.

Three outputs per model+dataset
--------------------------------
  1. reliability_curve_ensemble.png  — reliability diagram (ensemble vs per-step)
  2. ece_by_turn_ensemble.png        — ECE, accuracy, confidence by turn T0-T6
  3. ensemble_calibration_metrics.csv — numbers for every model × dataset

Usage
-----
  python run_ensemble_calibration.py                        # all models
  python run_ensemble_calibration.py --model ClaudeSonnet
  python run_ensemble_calibration.py --model all --n_bins 15
"""

from __future__ import annotations

import argparse
import os
import pickle
from glob import glob
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from reasoning_uncertainty import UncertaintyTrajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_OUT = "experiment_out"

DATASET_SUBDIRS = ["gpqa_diamond", "hle", "aime_2025"]

MODEL_LABELS = {
    "ClaudeHaiku":  "Claude Haiku",
    "ClaudeSonnet": "Claude Sonnet",
    "GPT5_4":       "GPT-5.4",
    "GPT5_4Mini":   "GPT-5.4 Mini",
    "GPT5_4Nano":   "GPT-5.4 Nano",
    "GeminiFlash":  "Gemini Flash",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_pkl_dir(path: str) -> List[UncertaintyTrajectory]:
    trajs: List[UncertaintyTrajectory] = []
    if not os.path.isdir(path):
        return trajs
    for fpath in sorted(glob(os.path.join(path, "bin_*_reasoning.pkl"))):
        with open(fpath, "rb") as f:
            trajs.extend(pickle.load(f))
    return trajs


def load_all_trajectories(
    model: str,
    base_dir: str = EXPERIMENT_OUT,
) -> dict[str, List[UncertaintyTrajectory]]:
    """Returns {dataset_name: [trajectories]}."""
    result: dict[str, List[UncertaintyTrajectory]] = {}

    # MMLU-Pro (root level)
    root_trajs = _load_pkl_dir(
        os.path.join(base_dir, model, "reasoning_calibrated_bin")
    )
    if root_trajs:
        result["mmlu_pro"] = root_trajs

    # Subdatasets
    for sub in DATASET_SUBDIRS:
        sub_trajs = _load_pkl_dir(
            os.path.join(base_dir, model, sub, "reasoning_calibrated_bin")
        )
        if sub_trajs:
            result[sub] = sub_trajs

    return result


def available_models(base_dir: str = EXPERIMENT_OUT) -> List[str]:
    models = []
    for model in sorted(os.listdir(base_dir)):
        model_dir = os.path.join(base_dir, model)
        if not os.path.isdir(model_dir):
            continue
        # Has any reasoning_calibrated_bin with pkl files
        for subpath in [
            os.path.join(model_dir, "reasoning_calibrated_bin"),
            *[os.path.join(model_dir, s, "reasoning_calibrated_bin") for s in DATASET_SUBDIRS],
        ]:
            if os.path.isdir(subpath) and glob(os.path.join(subpath, "bin_*_reasoning.pkl")):
                models.append(model)
                break
    return models


# ---------------------------------------------------------------------------
# Ensemble extraction
# ---------------------------------------------------------------------------

def extract_ensemble_predictions(
    trajectories: List[UncertaintyTrajectory],
) -> pd.DataFrame:
    """
    For each (question, turn) compute:
      - ensemble_conf: mean final-step calibrator confidence across K runs
      - run_is_correct: per-run binary correctness (1 = belief == gold)
      - n_runs: number of valid runs contributing to this cell

    Returns a DataFrame with one row per (question, turn, run).
    Each row has ensemble_conf (same for all runs of that question×turn)
    and run_is_correct (individual run label).
    """
    rows = []
    for traj in trajectories:
        gold = traj.gold_answer
        if gold is None:
            continue

        for t_idx, turn_runs in enumerate(traj.raw_traces):
            # Collect final-step confidence and correctness for each run
            run_confs = []
            run_correct = []
            for trace in turn_runs:
                if not trace.steps:
                    continue
                final = trace.steps[-1]
                if final.confidence is None or final.current_belief is None:
                    continue
                run_confs.append(final.confidence / 100.0)
                run_correct.append(
                    int(final.current_belief.strip().upper() == gold.strip().upper())
                )

            if not run_confs:
                continue

            ensemble_conf = float(np.mean(run_confs))
            # Emit one row per run (so N = questions × turns × K)
            for correct in run_correct:
                rows.append({
                    "query":          traj.query,
                    "gold":           gold,
                    "turn":           t_idx,
                    "ensemble_conf":  ensemble_conf,
                    "run_is_correct": correct,
                    "n_runs":         len(run_correct),
                })

    return pd.DataFrame(rows)


def extract_question_turn_predictions(
    trajectories: List[UncertaintyTrajectory],
) -> pd.DataFrame:
    """
    Aggregated view: one row per (question, turn).
      - ensemble_conf: mean final-step confidence across K runs
      - frac_correct: fraction of K runs with correct final belief
      - majority_correct: 1 if majority belief == gold
    """
    rows = []
    for traj in trajectories:
        gold = traj.gold_answer
        if gold is None:
            continue

        for t_idx, turn_runs in enumerate(traj.raw_traces):
            run_confs = []
            run_correct = []
            for trace in turn_runs:
                if not trace.steps:
                    continue
                final = trace.steps[-1]
                if final.confidence is None or final.current_belief is None:
                    continue
                run_confs.append(final.confidence / 100.0)
                run_correct.append(
                    int(final.current_belief.strip().upper() == gold.strip().upper())
                )

            if not run_confs:
                continue

            rows.append({
                "query":            traj.query,
                "gold":             gold,
                "turn":             t_idx,
                "ensemble_conf":    float(np.mean(run_confs)),
                "frac_correct":     float(np.mean(run_correct)),
                "majority_correct": int(np.mean(run_correct) >= 0.5),
                "n_runs":           len(run_correct),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def compute_ece(
    confs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard equal-width binned ECE.
    Returns (ece, bin_centres, bin_accs, bin_counts).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres, bin_accs, bin_counts = [], [], []
    N = len(confs)
    ece_val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs <= hi)
        if mask.sum() == 0:
            continue
        c = float(confs[mask].mean())
        a = float(labels[mask].mean())
        cnt = int(mask.sum())
        ece_val += cnt / N * abs(a - c)
        bin_centres.append(c)
        bin_accs.append(a)
        bin_counts.append(cnt)
    return (
        float(ece_val),
        np.array(bin_centres),
        np.array(bin_accs),
        np.array(bin_counts),
    )


def brier_score(confs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((confs - labels) ** 2))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_reliability_ax(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.10)
    ax.set_xlabel("Mean ensemble confidence")
    ax.set_ylabel("Fraction correct")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)


def plot_reliability(
    df_run: pd.DataFrame,
    out_path: str,
    model: str,
    dataset: str,
    n_bins: int,
) -> None:
    """Reliability diagram: all turns combined and T0 only."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    label = MODEL_LABELS.get(model, model)
    fig.suptitle(
        f"Ensemble calibration reliability - {label} ({dataset})",
        fontsize=12, fontweight="bold",
    )

    subsets = [
        (df_run, "All turns"),
        (df_run[df_run["turn"] == 0], "T0 (baseline only)"),
    ]
    for ax, (sub, title) in zip(axes, subsets):
        _setup_reliability_ax(ax)
        if sub.empty:
            ax.set_title(f"{title} - no data")
            continue

        confs  = sub["ensemble_conf"].values
        labels = sub["run_is_correct"].values
        ece_v, bcs, baccs, bcnts = compute_ece(confs, labels, n_bins)
        bs = brier_score(confs, labels)

        ax.bar(bcs, baccs, width=0.07, alpha=0.3, color="steelblue", align="center")
        ax.plot(bcs, baccs, "o-", color="steelblue", label="Ensemble", zorder=3)
        for cx, ac, cnt in zip(bcs, baccs, bcnts):
            ax.text(cx, ac + 0.015, f"{cnt}", ha="center", fontsize=5.5, color="steelblue")

        ax.set_title(
            f"{title}  (n={len(sub):,})\n"
            f"ECE={ece_v:.3f}  Brier={bs:.3f}  "
            f"Acc={labels.mean():.1%}  MeanConf={confs.mean():.1%}",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ece_by_turn(
    df_run: pd.DataFrame,
    out_path: str,
    model: str,
    dataset: str,
    n_bins: int,
) -> None:
    """ECE, accuracy, and confidence vs. pressure turn."""
    turns = sorted(df_run["turn"].unique())
    if len(turns) < 2:
        return

    ece_vals, acc_vals, conf_vals = [], [], []
    for t in turns:
        sub = df_run[df_run["turn"] == t]
        if sub.empty:
            ece_vals.append(np.nan); acc_vals.append(np.nan); conf_vals.append(np.nan)
            continue
        e, *_ = compute_ece(sub["ensemble_conf"].values, sub["run_is_correct"].values, n_bins)
        ece_vals.append(e)
        acc_vals.append(float(sub["run_is_correct"].mean()))
        conf_vals.append(float(sub["ensemble_conf"].mean()))

    label = MODEL_LABELS.get(model, model)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(turns, ece_vals,  "o-", color="tomato",    label="ECE (↓ better)")
    ax.plot(turns, acc_vals,  "s-", color="steelblue", label="Accuracy")
    ax.plot(turns, conf_vals, "^-", color="orange",    label="Mean confidence")
    ax.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Turn (0 = baseline, 1-6 = pressure)")
    ax.set_ylabel("Value")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title(
        f"Ensemble calibration by turn - {label} ({dataset})",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_model_summary(
    summary_rows: list[dict],
    out_path: str,
    n_bins: int,
) -> None:
    """
    Three-panel summary across models × datasets.
    Left: overall ECE by model (bars).
    Centre: ECE at T0 vs. T6 (grouped bars).
    Right: accuracy vs. mean confidence scatter.
    """
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return

    # Aggregate to model level (mean across datasets weighted by n)
    def wavg(grp, col, wt="n_runs"):
        return np.average(grp[col], weights=grp[wt])

    model_summary = (
        df.groupby("model")
        .apply(lambda g: pd.Series({
            "ece_overall":  wavg(g, "ece_overall"),
            "ece_t0":       wavg(g[g["turn"] == 0], "ece", wt="n") if not g[g["turn"] == 0].empty else np.nan,
            "ece_t6":       wavg(g[g["turn"] == 6], "ece", wt="n") if not g[g["turn"] == 6].empty else np.nan,
            "acc":          wavg(g, "acc"),
            "conf":         wavg(g, "conf"),
            "n_runs":       g["n_runs"].sum(),
        }), include_groups=False)
        .reset_index()
    )
    model_summary["label"] = model_summary["model"].map(
        lambda m: MODEL_LABELS.get(m, m)
    )
    model_summary = model_summary.sort_values("ece_overall")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cross-model ensemble calibration summary", fontsize=13, fontweight="bold")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    x = np.arange(len(model_summary))
    labels_x = model_summary["label"].tolist()

    # Panel 1: overall ECE bar chart
    ax = axes[0]
    bars = ax.bar(x, model_summary["ece_overall"], color=colors[:len(x)], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels_x, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ECE (↓ better)")
    ax.set_title("Overall ensemble ECE", fontsize=11, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: ECE at T0 vs T6
    ax = axes[1]
    w = 0.35
    ax.bar(x - w/2, model_summary["ece_t0"], width=w, label="T0 (baseline)", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, model_summary["ece_t6"], width=w, label="T6 (max pressure)", color="tomato", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels_x, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("ECE")
    ax.set_title("ECE: baseline vs. max pressure", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: accuracy vs confidence scatter
    ax = axes[2]
    for i, row in model_summary.iterrows():
        ax.scatter(row["conf"], row["acc"], s=100, color=colors[i % len(colors)], zorder=3)
        ax.annotate(row["label"], (row["conf"], row["acc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)
    lo = min(model_summary[["acc", "conf"]].min()) - 0.02
    hi = max(model_summary[["acc", "conf"]].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="Perfect")
    ax.set_xlabel("Mean ensemble confidence")
    ax.set_ylabel("Accuracy")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("Accuracy vs. mean confidence", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved cross-model summary: {out_path}")


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

def run_model_dataset(
    model: str,
    dataset: str,
    trajs: List[UncertaintyTrajectory],
    n_bins: int,
    base_dir: str,
) -> list[dict]:
    """
    Runs ensemble calibration for one (model, dataset) pair.
    Returns a list of per-turn metric dicts for the summary CSV.
    """
    print(f"  Dataset: {dataset}  ({len(trajs)} trajectories)")

    df_run = extract_ensemble_predictions(trajs)
    if df_run.empty:
        print("    No valid ensemble predictions — skipping.")
        return []

    out_dir = os.path.join(base_dir, model, dataset, "ensemble_calibration") \
              if dataset != "mmlu_pro" \
              else os.path.join(base_dir, model, "ensemble_calibration")
    os.makedirs(out_dir, exist_ok=True)

    plot_reliability(
        df_run,
        os.path.join(out_dir, "reliability_curve_ensemble.png"),
        model, dataset, n_bins,
    )
    plot_ece_by_turn(
        df_run,
        os.path.join(out_dir, "ece_by_turn_ensemble.png"),
        model, dataset, n_bins,
    )

    # Per-turn metrics for summary
    summary_rows = []
    turns = sorted(df_run["turn"].unique())
    overall_e, *_ = compute_ece(
        df_run["ensemble_conf"].values, df_run["run_is_correct"].values, n_bins
    )

    for t in turns:
        sub = df_run[df_run["turn"] == t]
        e, *_ = compute_ece(sub["ensemble_conf"].values, sub["run_is_correct"].values, n_bins)
        summary_rows.append({
            "model":        model,
            "dataset":      dataset,
            "turn":         t,
            "ece":          e,
            "acc":          float(sub["run_is_correct"].mean()),
            "conf":         float(sub["ensemble_conf"].mean()),
            "n":            len(sub),
            "ece_overall":  overall_e,
            "n_runs":       len(sub),
        })

    # Save per-model CSV
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "ensemble_calibration_metrics.csv"), index=False
    )
    print(f"    ECE overall={overall_e:.4f}  "
          f"acc={df_run['run_is_correct'].mean():.1%}  "
          f"conf={df_run['ensemble_conf'].mean():.1%}  "
          f"n={len(df_run):,}")

    return summary_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble calibration analysis")
    p.add_argument("--model",   type=str, default="all",
                   help="Model key or 'all'")
    p.add_argument("--out_dir", type=str, default=EXPERIMENT_OUT)
    p.add_argument("--n_bins",  type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    models = available_models(args.out_dir) if args.model == "all" else [args.model]

    if not models:
        print("No models with reasoning_calibrated_bin data found.")
        return

    print(f"Models to analyse: {models}\n")

    all_summary: list[dict] = []

    for model in models:
        label = MODEL_LABELS.get(model, model)
        print(f"\n{'='*60}")
        print(f"Model: {label}")
        dataset_trajs = load_all_trajectories(model, args.out_dir)
        if not dataset_trajs:
            print("  No data — skipping.")
            continue

        for dataset, trajs in dataset_trajs.items():
            rows = run_model_dataset(model, dataset, trajs, args.n_bins, args.out_dir)
            all_summary.extend(rows)

    if not all_summary:
        print("\nNo results to summarise.")
        return

    # Global summary CSV
    summary_df = pd.DataFrame(all_summary)
    summary_csv = os.path.join(args.out_dir, "ensemble_calibration_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved global summary: {summary_csv}")

    # Print summary table
    print("\n--- Ensemble ECE summary (all turns) ---")
    print(f"{'Model':<18} {'Dataset':<14} {'ECE':>8} {'Acc':>8} {'Conf':>8} {'n':>8}")
    print("-" * 64)
    for row in sorted(all_summary, key=lambda r: (r["model"], r["dataset"], r["turn"])):
        if row["turn"] == 0:  # Only print T0 row in summary (show degradation separately)
            continue
    seen = set()
    for row in sorted(all_summary, key=lambda r: (r["model"], r["dataset"])):
        key = (row["model"], row["dataset"])
        if key in seen:
            continue
        seen.add(key)
        print(f"{row['model']:<18} {row['dataset']:<14} {row['ece_overall']:>8.4f} "
              f"{row['acc']:>8.1%} {row['conf']:>8.1%} {row['n']:>8,}")

    # Cross-model summary plot (mmlu_pro only for comparability)
    plot_cross_model_summary(
        all_summary,
        out_path=os.path.join(args.out_dir, "ensemble_calibration_cross_model.png"),
        n_bins=args.n_bins,
    )

    print("\nDone. Per-model plots in <out_dir>/<model>/ensemble_calibration/")


if __name__ == "__main__":
    main()
