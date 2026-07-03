"""
reasoning_analysis.py — Binning-free analysis of CoT reasoning sycophancy.

Uses reasoning_calibrated_bin/bin_X_summary.pkl (Claude Haiku + Sonnet only).

Outcome: first_flip_at_turn — binary.
  1 on the FIRST turn where the model's majority belief flips from correct to
  incorrect. Subsequent wrong turns are excluded so each question contributes
  at most one flip event. Questions where the model was wrong at baseline
  (turn 0) are dropped.

Uncertainty predictors (run as separate models):
  belief_entropy   — Shannon entropy of the model's own stated beliefs across
                     reasoning runs at the last step of the baseline turn.
  cluster_entropy  — Semantic entropy (embedding-cluster diversity) at the same
                     point; captures meaning-level uncertainty beyond exact-match.

Three approaches applied to each predictor:
  Approach 2  — Certain / Uncertain split + quantile bins on uncertain remainder
  Approach 3a — Logistic regression: first_flip ~ pressure * entropy + C(model)
  Approach 3b — Mixed-effects LPM:   first_flip ~ pressure * entropy,
                groups = model_name  (linear probability model via mixedlm)
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = ["ClaudeHaiku", "ClaudeSonnet"]
MODEL_LABELS = {"ClaudeHaiku": "Claude Haiku", "ClaudeSonnet": "Claude Sonnet"}
DATA_DIR = "experiment_out"
OUT_DIR = "experiment_out"
N_UNCERTAIN_BINS = 3
ENTROPY_PREDICTORS = ["belief_entropy", "cluster_entropy"]
PREDICTOR_LABELS = {
    "belief_entropy": "Belief entropy (self-reported)",
    "cluster_entropy": "Cluster entropy (calibrator)",
}
# belief_entropy: 0 = certain, <0 = uncertain  → uncertain if < 0
# cluster_entropy: 0 = certain, >0 = uncertain → uncertain if > 0
UNCERTAIN_DIRECTION = {
    "belief_entropy": "negative",   # uncertain < 0
    "cluster_entropy": "positive",  # uncertain > 0
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_summary_dfs(model: str) -> pd.DataFrame:
    bin_dir = Path(DATA_DIR) / model / "reasoning_calibrated_bin"
    frames = []
    for f in sorted(bin_dir.glob("bin_*_summary.pkl")):
        with open(f, "rb") as fh:
            frames.append(pd.DataFrame(pickle.load(fh)))
    df = pd.concat(frames, ignore_index=True)
    df["model_name"] = model
    df["model_label"] = MODEL_LABELS[model]
    return df


def build_reasoning_df() -> pd.DataFrame:
    """
    One row per (model, question, pressure_turn).

    For each question:
      - baseline_belief_entropy / baseline_cluster_entropy: value at the last
        reasoning step of turn 0.
      - baseline_correct: whether majority_is_correct at turn 0 last step.
      - flip: 1 only on the FIRST turn where majority flips to incorrect.
        Subsequent incorrect turns are 0 (not counted again).
        Questions where baseline is wrong are excluded entirely.
    """
    all_rows = []

    for model in MODELS:
        df = _load_summary_dfs(model)

        # Near-zero cluster_entropy placeholder → treat as 0
        df["cluster_entropy"] = df["cluster_entropy"].apply(
            lambda x: 0.0 if abs(x) < 1e-6 else x
        )

        for query, qdf in df.groupby("query"):
            # Last step per turn
            last = (
                qdf.sort_values("step")
                .groupby("turn")
                .last()
                .reset_index()
            )
            last = last.sort_values("turn")

            # Baseline must exist
            baseline = last[last["turn"] == 0]
            if baseline.empty:
                continue

            b = baseline.iloc[0]
            if not b["majority_is_correct"]:
                continue  # wrong at baseline → skip

            base_be = float(b["belief_entropy"])
            base_ce = float(b["cluster_entropy"])

            # Determine first flip turn
            pressure_turns = last[last["turn"] > 0].sort_values("turn")
            first_flip_found = False
            for _, row in pressure_turns.iterrows():
                is_wrong = not row["majority_is_correct"]
                first_flip = int(is_wrong and not first_flip_found)
                if is_wrong and not first_flip_found:
                    first_flip_found = True

                all_rows.append(
                    {
                        "model_name": model,
                        "model_label": MODEL_LABELS[model],
                        "query": query,
                        "pressure_turn": int(row["turn"]),
                        "flip": first_flip,
                        "baseline_belief_entropy": base_be,
                        "baseline_cluster_entropy": base_ce,
                    }
                )

    df_out = pd.DataFrame(all_rows)
    print(f"Reasoning long-format rows : {len(df_out):,}")
    print(f"Questions                  : {df_out['query'].nunique()}")
    print(f"Models                     : {df_out['model_name'].nunique()}")
    print(
        f"Overall first-flip rate    : "
        f"{df_out.groupby('query')['flip'].max().mean():.3f}"
    )
    return df_out


# ---------------------------------------------------------------------------
# Approach 2 — Certain / Uncertain split
# ---------------------------------------------------------------------------

CERTAIN_THRESHOLD = 0.0


def _assign_bin(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    col = f"baseline_{predictor}"
    df = df.copy()
    df["certainty_bin"] = "Certain"

    direction = UNCERTAIN_DIRECTION[predictor]
    unc_mask = df[col] < 0 if direction == "negative" else df[col] > 0
    if unc_mask.sum() == 0:
        return df

    unc_vals = df.loc[unc_mask, col].values
    edges = np.unique(
        np.percentile(unc_vals, np.linspace(0, 100, N_UNCERTAIN_BINS + 1))
    )
    actual_bins = len(edges) - 1
    bin_ids = np.clip(
        np.searchsorted(edges[1:], unc_vals, side="right"), 0, actual_bins - 1
    )
    df.loc[unc_mask, "certainty_bin"] = [f"Uncertain-{b}" for b in bin_ids]
    return df


def run_approach2(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("APPROACH 2 — Certain / Uncertain split (by first-flip turn)")
    print("=" * 70)

    fig, axes = plt.subplots(
        len(ENTROPY_PREDICTORS), len(MODELS),
        figsize=(6 * len(MODELS), 4 * len(ENTROPY_PREDICTORS)),
        sharey="row",
    )

    for row_i, predictor in enumerate(ENTROPY_PREDICTORS):
        df2 = _assign_bin(df, predictor)
        bin_order = ["Certain"] + [f"Uncertain-{i}" for i in range(N_UNCERTAIN_BINS)]
        bin_order = [b for b in bin_order if b in df2["certainty_bin"].unique()]

        print(f"\n--- {PREDICTOR_LABELS[predictor]} ---")
        pivot = (
            df2.groupby(["certainty_bin", "pressure_turn"])["flip"]
            .mean()
            .unstack("pressure_turn")
            .reindex(bin_order)
        )
        print(pivot.round(3).to_string())

        for col_i, model in enumerate(MODELS):
            ax = axes[row_i][col_i]
            sub = df2[df2["model_name"] == model]
            for bin_name in bin_order:
                g = sub[sub["certainty_bin"] == bin_name]
                if g.empty:
                    continue
                rates = g.groupby("pressure_turn")["flip"].mean()
                ent_vals = g[f"baseline_{predictor}"]
                label = f"{bin_name}\n[{ent_vals.min():.2f}, {ent_vals.max():.2f}]"
                ax.plot(rates.index, rates.values, marker="o", label=label)
            ax.set_title(f"{MODEL_LABELS[model]}\n({PREDICTOR_LABELS[predictor]})")
            ax.set_xlabel("Pressure turn")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_xticks(range(1, 7))
            ax.legend(fontsize=7)
            if col_i == 0:
                ax.set_ylabel("P(first flip)")

    fig.suptitle(
        "Approach 2: First-flip rate by certainty bin × pressure turn", fontsize=13
    )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "reasoning_approach2.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Approach 3a — Logistic regression
# ---------------------------------------------------------------------------

def run_approach3a(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("APPROACH 3a — Logistic regression (continuous entropy)")
    print("=" * 70)

    cat_order = ["ClaudeHaiku", "ClaudeSonnet"]

    for predictor in ENTROPY_PREDICTORS:
        col = f"baseline_{predictor}"
        df3 = df.copy()
        df3["entropy"] = df3[col]
        df3["model_name"] = pd.Categorical(df3["model_name"], categories=cat_order)

        formula = "flip ~ pressure_turn * entropy + C(model_name)"
        print(f"\n[{PREDICTOR_LABELS[predictor]}]  {formula}")
        print(f"N = {len(df3):,}")

        nonzero_frac = (df3["entropy"] != 0).mean()
        if nonzero_frac < 0.05:
            print(f"  WARNING: {100*(1-nonzero_frac):.1f}% of entropy values are 0 "
                  f"— regression likely degenerate, skipping.")
            continue

        result = smf.logit(formula, data=df3).fit(disp=False, maxiter=300)

        or_df = pd.concat(
            [
                np.exp(result.params).rename("OR"),
                np.exp(result.conf_int()).rename(columns={0: "CI_low", 1: "CI_high"}),
                result.pvalues.rename("p"),
            ],
            axis=1,
        )
        print(or_df.round(4).to_string())

        # Marginal slopes at each pressure turn
        b_e = result.params.get("entropy", 0)
        b_ix = result.params.get("pressure_turn:entropy", 0)
        print("\nMarginal log-odds slope on entropy per pressure turn:")
        for t in range(1, 7):
            slope = b_e + b_ix * t
            print(f"  Turn {t}: {slope:+.4f}  (OR = {np.exp(slope):.4f})")

    # Visualise predicted P(first flip) for both predictors side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    pressure_vals = np.arange(1, 7)

    for ax, predictor in zip(axes, ENTROPY_PREDICTORS):
        col = f"baseline_{predictor}"
        df3 = df.copy()
        df3["entropy"] = df3[col]
        df3["model_name"] = pd.Categorical(df3["model_name"], categories=cat_order)
        result = smf.logit(
            "flip ~ pressure_turn * entropy + C(model_name)", data=df3
        ).fit(disp=False, maxiter=300)

        ent_range = np.linspace(df3["entropy"].min(), 0, 80)
        colors = {"ClaudeHaiku": "#1f77b4", "ClaudeSonnet": "#ff7f0e"}
        styles = {1: "-", 2: "--", 3: ":", 4: "-.", 5: (0,(3,1,1,1)), 6: (0,(5,1))}

        for model in cat_order:
            for t in [1, 3, 6]:
                grid = pd.DataFrame(
                    {
                        "entropy": ent_range,
                        "pressure_turn": t,
                        "model_name": pd.Categorical(
                            [model] * len(ent_range), categories=cat_order
                        ),
                    }
                )
                pred = result.predict(grid)
                ax.plot(
                    ent_range, pred,
                    color=colors[model],
                    linestyle=styles[t],
                    label=f"{MODEL_LABELS[model]} T{t}",
                    alpha=0.85,
                )

        ax.set_xlabel(f"{PREDICTOR_LABELS[predictor]}\n(more negative → more uncertain)")
        ax.set_ylabel("P(first flip)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(f"Logistic: {predictor}")
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Approach 3a: Predicted P(first flip) — logistic regression", fontsize=13
    )
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "reasoning_approach3a_logistic.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Approach 3b — Mixed-effects LPM
# ---------------------------------------------------------------------------

def run_approach3b(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("APPROACH 3b — Mixed-effects model (model_name as random intercept)")
    print("=" * 70)
    print("NOTE: mixedlm uses a linear link (LPM). Predictions outside [0,1]")
    print("are possible; interpret coefficients as probability differences.\n")

    for predictor in ENTROPY_PREDICTORS:
        col = f"baseline_{predictor}"
        df3 = df.copy()
        df3["entropy"] = df3[col]

        formula = "flip ~ pressure_turn * entropy"
        print(f"\n[{PREDICTOR_LABELS[predictor]}]  {formula}  |  groups = model_name")
        print(f"N = {len(df3):,}")

        nonzero_frac = (df3["entropy"] != 0).mean()
        if nonzero_frac < 0.05:
            print(f"  WARNING: {100*(1-nonzero_frac):.1f}% of entropy values are 0 "
                  f"— skipping degenerate model.")
            continue

        result = smf.mixedlm(formula, data=df3, groups=df3["model_name"]).fit(
            reml=False, disp=False
        )
        print(result.summary())

        # Random effects
        re = result.random_effects
        print("\nRandom intercepts per model:")
        for model, vals in re.items():
            v = float(vals.iloc[0]) if hasattr(vals, "iloc") else float(vals)
            print(f"  {MODEL_LABELS.get(model, model)}: {v:+.4f}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    df = build_reasoning_df()

    print("\nBaseline entropy distributions (questions × model):")
    for predictor in ENTROPY_PREDICTORS:
        col = f"baseline_{predictor}"
        by_model = (
            df.drop_duplicates(["model_name", "query"])
            .groupby("model_name")[col]
            .describe()[["mean", "std", "min", "max"]]
        )
        print(f"\n{PREDICTOR_LABELS[predictor]}:\n{by_model.round(3).to_string()}")

    run_approach2(df)
    run_approach3a(df)
    run_approach3b(df)


if __name__ == "__main__":
    main()
