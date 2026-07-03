"""
calibration_analysis.py — Calibration evaluation for the reasoning uncertainty model.

Checks whether the calibrator's per-step confidence scores are calibrated
against actual correctness, i.e. P(correct | confidence ≈ c) ≈ c.

Three analyses
--------------
1. Reliability curve + ECE  — all steps pooled, then final step only
2. Turn-level calibration    — ECE at T0 vs T1..TK (does social pressure degrade calibration?)
3. Entropy as a signal       — does belief_entropy / cluster_entropy track accuracy?

Usage
-----
  python calibration_analysis.py --model ClaudeSonnet
  python calibration_analysis.py --model all            # all available models

Outputs
-------
  experiment_out/<MODEL>/calibration/
    reliability_curve.png
    turn_calibration.png
    entropy_accuracy.png
    calibration_metrics.pkl
    calibration_metrics.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from reasoning_uncertainty import ReasoningTrace, UncertaintyTrajectory


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

@dataclass
class StepPred:
    confidence: float    # 0–1
    is_correct: bool     # belief == gold_answer
    turn_idx:   int
    step_idx:   int
    is_final:   bool     # True for the last step of a trace


def extract_predictions(trajectories: List[UncertaintyTrajectory]) -> List[StepPred]:
    """
    Flatten all per-step calibrator predictions across questions, turns, and runs.

    Each prediction: calibrator assigned BELIEF and CONFIDENCE to a reasoning
    step; is_correct = (calibrated belief == gold answer).
    """
    preds: List[StepPred] = []
    for traj in trajectories:
        gold = traj.gold_answer
        if gold is None:
            continue
        for t_idx, turn_traces in enumerate(traj.raw_traces):
            for trace in turn_traces:
                n_steps = len(trace.steps)
                for s_pos, step in enumerate(trace.steps):
                    if step.confidence is None or step.current_belief is None:
                        continue
                    preds.append(StepPred(
                        confidence=step.confidence / 100.0,
                        is_correct=(step.current_belief.upper() == gold.upper()),
                        turn_idx=t_idx,
                        step_idx=step.step_index,
                        is_final=(s_pos == n_steps - 1),
                    ))
    return preds


def load_trajectories_for_model(
    model: str,
    base_dir: str = "experiment_out",
    subdir: str = "reasoning_calibrated_bin",
) -> List[UncertaintyTrajectory]:
    """Load all bin PKL files for a given model."""
    path = os.path.join(base_dir, model, subdir)
    if not os.path.isdir(path):
        return []
    trajs: List[UncertaintyTrajectory] = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith("_reasoning.pkl"):
            with open(os.path.join(path, fname), "rb") as f:
                trajs.extend(pickle.load(f))
    return trajs


def available_models(base_dir: str = "experiment_out") -> List[str]:
    models = []
    for model in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, model, "reasoning_calibrated_bin")
        if os.path.isdir(path) and any(f.endswith("_reasoning.pkl") for f in os.listdir(path)):
            models.append(model)
    return models


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def reliability_curve(
    preds: List[StepPred],
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_centres, mean_accuracy, bin_counts).
    Only non-empty bins are included.
    """
    confs   = np.array([p.confidence for p in preds])
    correct = np.array([float(p.is_correct) for p in preds])

    edges   = np.linspace(0, 1, n_bins + 1)
    centres, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        counts.append(int(mask.sum()))

    return np.array(centres), np.array(accs), np.array(counts)


def ece(preds: List[StepPred], n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    if not preds:
        return float("nan")
    confs   = np.array([p.confidence for p in preds])
    correct = np.array([float(p.is_correct) for p in preds])
    N = len(preds)
    edges = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        err += mask.sum() / N * abs(correct[mask].mean() - confs[mask].mean())
    return float(err)


def brier_score(preds: List[StepPred]) -> float:
    if not preds:
        return float("nan")
    confs   = np.array([p.confidence for p in preds])
    correct = np.array([float(p.is_correct) for p in preds])
    return float(np.mean((confs - correct) ** 2))


def mean_confidence(preds: List[StepPred]) -> float:
    if not preds:
        return float("nan")
    return float(np.mean([p.confidence for p in preds]))


def accuracy(preds: List[StepPred]) -> float:
    if not preds:
        return float("nan")
    return float(np.mean([p.is_correct for p in preds]))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

TURN_COLORS = plt.cm.RdYlGn_r  # green (T0) → red (high pressure)


def _reliability_ax(
    ax: plt.Axes,
    centres: np.ndarray,
    accs: np.ndarray,
    counts: np.ndarray,
    label: str,
    color: str,
    alpha: float = 0.85,
) -> None:
    """Draw bars and confidence line on a reliability axis."""
    bar_w = 0.08
    ax.bar(centres, accs, width=bar_w, alpha=0.35, color=color, align="center", zorder=2)
    ax.plot(centres, accs, "o-", color=color, label=label, alpha=alpha, zorder=3)

    # annotate counts on bars
    for cx, acc, cnt in zip(centres, accs, counts):
        ax.text(cx, acc + 0.015, f"{cnt}", ha="center", va="bottom", fontsize=5.5, color=color)


def setup_reliability_ax(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.10)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Fraction correct (accuracy)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Plot 1 — Reliability diagram (all steps vs. final step)
# ---------------------------------------------------------------------------

def plot_reliability(
    all_preds:   List[StepPred],
    out_path:    str,
    model:       str,
    n_bins:      int = 10,
) -> None:
    final_preds = [p for p in all_preds if p.is_final]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    fig.suptitle(f"Calibration reliability — {model}", fontsize=13, fontweight="bold")

    for ax, preds, title_suffix in zip(
        axes,
        [all_preds, final_preds],
        ["All steps", "Final step only"],
    ):
        setup_reliability_ax(ax)
        centres, accs, counts = reliability_curve(preds, n_bins=n_bins)
        _reliability_ax(ax, centres, accs, counts, label="Calibrator", color="steelblue")

        ece_val = ece(preds, n_bins=n_bins)
        bs      = brier_score(preds)
        acc     = accuracy(preds)
        conf    = mean_confidence(preds)
        ax.set_title(
            f"{title_suffix}  (n={len(preds):,})\n"
            f"ECE={ece_val:.3f}  Brier={bs:.3f}  Acc={acc:.2%}  MeanConf={conf:.2%}",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2 — ECE by turn (calibration drift under pressure)
# ---------------------------------------------------------------------------

def plot_turn_calibration(
    all_preds: List[StepPred],
    out_path:  str,
    model:     str,
    n_bins:    int = 10,
) -> None:
    turns = sorted(set(p.turn_idx for p in all_preds))
    if len(turns) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Calibration by turn (pressure dose) — {model}", fontsize=13, fontweight="bold")

    # Left: reliability curves per turn
    ax = axes[0]
    setup_reliability_ax(ax)
    ax.set_title("Reliability curve by turn", fontsize=10)

    turn_ece  = []
    turn_acc  = []
    turn_conf = []

    for t in turns:
        tpreds = [p for p in all_preds if p.turn_idx == t]
        c, a, cnt = reliability_curve(tpreds, n_bins=n_bins)
        color = TURN_COLORS(t / max(turns))
        label = f"T{t} baseline" if t == 0 else f"T{t} pressure"
        _reliability_ax(ax, c, a, cnt, label=label, color=color, alpha=0.7)
        turn_ece.append(ece(tpreds, n_bins=n_bins))
        turn_acc.append(accuracy(tpreds))
        turn_conf.append(mean_confidence(tpreds))

    ax.legend(fontsize=6.5, ncol=2)

    # Right: ECE / accuracy / confidence vs. turn
    ax2 = axes[1]
    ax2.plot(turns, turn_ece,  "o-", color="tomato",    label="ECE")
    ax2.plot(turns, turn_acc,  "s-", color="steelblue", label="Accuracy")
    ax2.plot(turns, turn_conf, "^-", color="orange",    label="Mean confidence")
    ax2.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Turn (0 = baseline, 1+ = pressure doses)")
    ax2.set_ylabel("Value")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_title("ECE / accuracy / confidence vs. turn", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3 — Entropy as a calibration signal
# ---------------------------------------------------------------------------

def plot_entropy_vs_accuracy(
    trajectories: List[UncertaintyTrajectory],
    out_path:     str,
    model:        str,
    n_bins:       int = 10,
) -> None:
    """
    Treat –belief_entropy (high = certain) as a confidence proxy.
    Plot whether it correctly ranks accuracy.
    """
    rows = []
    for traj in trajectories:
        gold = traj.gold_answer
        if gold is None:
            continue
        for t_idx, turn_steps in enumerate(traj.turn_trajectories):
            for su in turn_steps:
                if su.majority_belief is None:
                    continue
                rows.append({
                    "neg_bel_H":  -su.belief_entropy,
                    "cl_H":       su.cluster_entropy,
                    "spread":     su.semantic_spread,
                    "correct":    float(su.majority_belief.upper() == gold.upper()),
                    "turn":       t_idx,
                })

    if not rows:
        return

    # neg_bel_H = -belief_entropy_stored = true Shannon entropy (≥ 0; 0 = certain)
    # cl_H = cluster entropy (≥ 0; 0 = certain)
    # spread = semantic spread (≥ 0; 0 = certain)
    bel_H   = np.array([r["neg_bel_H"] for r in rows])
    cl_H    = np.array([r["cl_H"]      for r in rows])
    spread  = np.array([r["spread"]    for r in rows])
    correct = np.array([r["correct"]   for r in rows])

    # Normalise to [0, 1] where 1 = maximally certain (all signals ≥ 0, so negate first).
    def norm01(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Entropy signals vs. majority accuracy — {model}", fontsize=13, fontweight="bold")

    for ax, signal, name in zip(
        axes,
        [norm01(-bel_H), norm01(-cl_H), norm01(-spread)],
        ["Belief entropy certainty (0=uncertain → 1=certain)",
         "Cluster entropy certainty (0=uncertain → 1=certain)",
         "Semantic spread certainty (0=uncertain → 1=certain)"],
    ):
        # Bin by signal
        edges = np.linspace(signal.min(), signal.max(), n_bins + 1)
        bin_sigs, bin_accs, bin_cnts = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (signal >= lo) & (signal <= hi)
            if mask.sum() == 0:
                continue
            bin_sigs.append(signal[mask].mean())
            bin_accs.append(correct[mask].mean())
            bin_cnts.append(int(mask.sum()))

        ax.plot(bin_sigs, bin_accs, "o-", color="mediumorchid")
        ax.bar(bin_sigs, bin_accs, width=0.08, alpha=0.3, color="mediumorchid", align="center")
        ax.plot([0, 1], [correct.mean(), correct.mean()], "k--", lw=1, alpha=0.4, label=f"Mean acc={correct.mean():.2%}")
        for bx, ba, bc in zip(bin_sigs, bin_accs, bin_cnts):
            ax.text(bx, ba + 0.012, f"{bc}", ha="center", fontsize=5.5, color="purple")

        ax.set_xlabel(f"Normalised signal (→ more certain)")
        ax.set_ylabel("Fraction majority correct")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 1.1)
        ax.set_title(name, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    model:          str
    n_predictions:  int
    overall_ece:    float
    final_step_ece: float
    overall_brier:  float
    overall_acc:    float
    overall_conf:   float
    ece_by_turn:    List[float]
    acc_by_turn:    List[float]
    conf_by_turn:   List[float]


def build_report(model: str, all_preds: List[StepPred], n_bins: int = 10) -> CalibrationReport:
    turns = sorted(set(p.turn_idx for p in all_preds))
    ece_by_turn  = [ece([p for p in all_preds if p.turn_idx == t], n_bins) for t in turns]
    acc_by_turn  = [accuracy([p for p in all_preds if p.turn_idx == t]) for t in turns]
    conf_by_turn = [mean_confidence([p for p in all_preds if p.turn_idx == t]) for t in turns]
    final_preds  = [p for p in all_preds if p.is_final]

    return CalibrationReport(
        model=model,
        n_predictions=len(all_preds),
        overall_ece=ece(all_preds, n_bins),
        final_step_ece=ece(final_preds, n_bins),
        overall_brier=brier_score(all_preds),
        overall_acc=accuracy(all_preds),
        overall_conf=mean_confidence(all_preds),
        ece_by_turn=ece_by_turn,
        acc_by_turn=acc_by_turn,
        conf_by_turn=conf_by_turn,
    )


def print_report(r: CalibrationReport) -> None:
    print(f"\n{'─'*60}")
    print(f"  Model           : {r.model}")
    print(f"  N predictions   : {r.n_predictions:,}")
    print(f"  Overall ECE     : {r.overall_ece:.4f}")
    print(f"  Final-step ECE  : {r.final_step_ece:.4f}")
    print(f"  Brier score     : {r.overall_brier:.4f}")
    print(f"  Accuracy        : {r.overall_acc:.2%}")
    print(f"  Mean confidence : {r.overall_conf:.2%}")
    print(f"  ECE by turn     : {[f'{e:.3f}' for e in r.ece_by_turn]}")
    print(f"  Acc by turn     : {[f'{a:.2%}' for a in r.acc_by_turn]}")
    print(f"  Conf by turn    : {[f'{c:.2%}' for c in r.conf_by_turn]}")


def save_metrics(report: CalibrationReport, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "calibration_metrics.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(report, f)

    csv_path = os.path.join(out_dir, "calibration_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        f.write(f"model,{report.model}\n")
        f.write(f"n_predictions,{report.n_predictions}\n")
        f.write(f"overall_ece,{report.overall_ece:.6f}\n")
        f.write(f"final_step_ece,{report.final_step_ece:.6f}\n")
        f.write(f"overall_brier,{report.overall_brier:.6f}\n")
        f.write(f"overall_acc,{report.overall_acc:.6f}\n")
        f.write(f"overall_conf,{report.overall_conf:.6f}\n")
        for t, (e, a, c) in enumerate(zip(report.ece_by_turn, report.acc_by_turn, report.conf_by_turn)):
            f.write(f"ece_turn_{t},{e:.6f}\n")
            f.write(f"acc_turn_{t},{a:.6f}\n")
            f.write(f"conf_turn_{t},{c:.6f}\n")
    print(f"  Saved metrics: {pkl_path}, {csv_path}")


# ---------------------------------------------------------------------------
# Cross-model summary plot
# ---------------------------------------------------------------------------

def plot_cross_model(
    reports: List[CalibrationReport],
    out_path: str,
) -> None:
    if len(reports) < 2:
        return
    models   = [r.model for r in reports]
    ece_vals = [r.overall_ece for r in reports]
    acc_vals = [r.overall_acc for r in reports]
    conf_vals= [r.overall_conf for r in reports]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Cross-model calibration comparison", fontsize=13, fontweight="bold")

    # ECE bar chart
    ax = axes[0]
    bars = ax.bar(x, ece_vals, color="steelblue", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("ECE (lower = better)")
    ax.set_title("Overall ECE by model")
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Accuracy vs confidence scatter
    ax2 = axes[1]
    ax2.scatter(conf_vals, acc_vals, s=90, zorder=3)
    for m, cv, av in zip(models, conf_vals, acc_vals):
        ax2.annotate(m, (cv, av), textcoords="offset points", xytext=(5, 3), fontsize=8)
    lo = min(min(conf_vals), min(acc_vals)) - 0.02
    hi = max(max(conf_vals), max(acc_vals)) + 0.02
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4, label="Perfect calibration")
    ax2.set_xlabel("Mean confidence")
    ax2.set_ylabel("Accuracy")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_title("Accuracy vs. mean confidence")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved cross-model plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_model(model: str, args: argparse.Namespace) -> Optional[CalibrationReport]:
    print(f"\n{'='*60}")
    print(f"Analysing calibration for: {model}")

    trajs = load_trajectories_for_model(model, base_dir=args.out_dir)
    if not trajs:
        print(f"  No reasoning trajectories found — skipping.")
        return None

    preds = extract_predictions(trajs)
    if not preds:
        print("  No predictions with confidence values — skipping.")
        return None

    out_dir = os.path.join(args.out_dir, model, "calibration")

    plot_reliability(preds, os.path.join(out_dir, "reliability_curve.png"), model, n_bins=args.n_bins)
    plot_turn_calibration(preds, os.path.join(out_dir, "turn_calibration.png"), model, n_bins=args.n_bins)
    plot_entropy_vs_accuracy(trajs, os.path.join(out_dir, "entropy_accuracy.png"), model, n_bins=args.n_bins)

    report = build_report(model, preds, n_bins=args.n_bins)
    print_report(report)
    save_metrics(report, out_dir)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration analysis for reasoning uncertainty.")
    p.add_argument("--model",   type=str, default="all",
                   help="Model key (e.g. ClaudeSonnet) or 'all' for every available model.")
    p.add_argument("--out_dir", type=str, default="experiment_out")
    p.add_argument("--n_bins",  type=int, default=10,
                   help="Number of confidence bins for reliability curves.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.model == "all":
        models = available_models(args.out_dir)
        if not models:
            print("No models with calibrated reasoning data found.")
            return
        print(f"Found models: {models}")
    else:
        models = [args.model]

    reports = []
    for model in models:
        r = run_for_model(model, args)
        if r is not None:
            reports.append(r)

    if len(reports) >= 2:
        plot_cross_model(
            reports,
            out_path=os.path.join(args.out_dir, "cross_model_calibration.png"),
        )

    print(f"\nDone. Plots and metrics in {args.out_dir}/<model>/calibration/")


if __name__ == "__main__":
    main()
