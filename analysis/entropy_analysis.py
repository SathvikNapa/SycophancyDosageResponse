"""
entropy_analysis.py — Two alternative entropy-based sycophancy analyses for GPT models.

Approach 2: Certain/Uncertain split
    - Items with entropy == 0 go into a dedicated "Certain" stratum.
    - The remaining "Uncertain" items are quantile-binned into 3 bins.
    - Flip rates are reported per stratum × pressure level.

Approach 3: Logistic regression (continuous entropy)
    - One row per (model, question, run, pressure_turn).
    - flip ~ pressure_level * entropy + C(model_name)
    - Odds ratios and a predicted-probability heatmap are reported.

Only uses MMLU-Pro data from GPT5_4, GPT5_4Mini, GPT5_4Nano.
"""

from __future__ import annotations

import os
import pickle
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = ["GPT5_4Nano", "GPT5_4Mini", "GPT5_4", "ClaudeHaiku", "ClaudeSonnet"]
MODEL_LABELS = {
    "GPT5_4Nano": "GPT-5.4 Nano",
    "GPT5_4Mini": "GPT-5.4 Mini",
    "GPT5_4": "GPT-5.4",
    "ClaudeHaiku": "Claude Haiku",
    "ClaudeSonnet": "Claude Sonnet",
}
PRESSURE_LABELS = {
    "personal_disagreement": "T1: Personal disagree",
    "personal_certainty": "T2: Personal certainty",
    "social_proof": "T3: Social proof",
    "expert_authority": "T4: Expert authority",
    "personal_accusatory": "T5: Personal accuse",
    "crowd_consensus": "T6: Crowd consensus",
}
N_UNCERTAIN_BINS = 3
OUT_DIR = "v1_results/experiment_out"
BASE_DIR = "experiment_out"  # current run has entropy + category in base metadata

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_base(model: str) -> dict[str, dict]:
    path = os.path.join(BASE_DIR, model, "base_experiment_metadata.pkl")
    with open(path, "rb") as f:
        items = pickle.load(f)
    return {it["query"]: it for it in items}


def _load_bin_items(model: str) -> list[dict]:
    base = os.path.join(OUT_DIR, model, "entropy_bin")
    out: list[dict] = []
    for fname in sorted(os.listdir(base)):
        if fname.endswith("_repeated.pkl"):
            with open(os.path.join(base, fname), "rb") as f:
                out.extend(pickle.load(f))
    return out


def build_long_df() -> pd.DataFrame:
    """
    One row per (model, question, run, pressure_turn).
    Only includes runs where the model initially answered correctly (turn 0).
    """
    rows: list[dict] = []
    for model in MODELS:
        base_meta = _load_base(model)
        bin_items = _load_bin_items(model)

        for item in bin_items:
            query = item["query"]
            meta = base_meta.get(query, {})
            entropy = meta.get("entropy", 0.0)
            category = meta.get("category", "unknown")

            for run_idx, run in enumerate(item["raw_runs"]):
                if run["is_correct"][0] != 1:
                    continue

                is_wrong = run["is_wrong"]  # [0/1, turn0..turn6]
                # turn_categories absent in v1 data
                turn_cats = run.get("turn_categories", [None] * len(is_wrong))

                for turn in range(1, len(is_wrong)):
                    rows.append(
                        {
                            "model_name": model,
                            "model_label": MODEL_LABELS[model],
                            "query": query,
                            "category": category,
                            "entropy": float(entropy),
                            "run_idx": run_idx,
                            "pressure_level": turn,
                            "pressure_category": turn_cats[turn] if turn < len(turn_cats) else None,
                            "flip": int(is_wrong[turn]),
                        }
                    )

    df = pd.DataFrame(rows)
    print(f"Long-format rows: {len(df):,}  "
          f"(questions: {df['query'].nunique()}, "
          f"models: {df['model_name'].nunique()})")
    return df


# ---------------------------------------------------------------------------
# Approach 2 — Certain / Uncertain split + quantile bins on remainder
# ---------------------------------------------------------------------------

CERTAIN_THRESHOLD = 0.0   # entropy == 0 → Certain; < 0 → Uncertain


def assign_certainty_bin(df: pd.DataFrame, n_uncertain_bins: int = N_UNCERTAIN_BINS) -> pd.DataFrame:
    """
    Adds a 'certainty_bin' column:
      'Certain'       — entropy == 0 (model always gave the same answer)
      'Uncertain-0/1/2' — quantile bins on entropy < 0 (0 = least uncertain)
    """
    df = df.copy()
    df["certainty_bin"] = "Certain"

    uncertain_mask = df["entropy"] < CERTAIN_THRESHOLD
    if uncertain_mask.sum() == 0:
        return df

    unc = df.loc[uncertain_mask, "entropy"]
    quantiles = np.linspace(0, 100, n_uncertain_bins + 1)
    edges = np.unique(np.percentile(unc, quantiles))
    actual_bins = len(edges) - 1

    if actual_bins < n_uncertain_bins:
        print(f"  NOTE: only {actual_bins} distinct uncertain quantile edges found")

    bin_ids = np.searchsorted(edges[1:], unc.values, side="right")
    bin_ids = np.clip(bin_ids, 0, actual_bins - 1)

    df.loc[uncertain_mask, "certainty_bin"] = [
        f"Uncertain-{b}" for b in bin_ids
    ]
    return df


def _entropy_range_label(df: pd.DataFrame, bin_name: str) -> str:
    sub = df.loc[df["certainty_bin"] == bin_name, "entropy"]
    if sub.empty:
        return bin_name
    return f"{bin_name}\n[{sub.min():.3f}, {sub.max():.3f}]"


def run_approach2(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("APPROACH 2 — Certain / Uncertain split")
    print("=" * 70)

    df2 = assign_certainty_bin(df)

    # --- summary table ---
    bin_order = ["Certain"] + [f"Uncertain-{i}" for i in range(N_UNCERTAIN_BINS)]
    bin_order = [b for b in bin_order if b in df2["certainty_bin"].unique()]

    summary = (
        df2.groupby(["certainty_bin", "pressure_level"])["flip"]
        .agg(flip_rate="mean", n="count")
        .reset_index()
    )

    print("\nFlip rate by certainty bin × pressure level (all models pooled):\n")
    pivot = summary.pivot(
        index="certainty_bin", columns="pressure_level", values="flip_rate"
    ).reindex(bin_order)
    print(pivot.round(3).to_string())

    n_pivot = summary.pivot(
        index="certainty_bin", columns="pressure_level", values="n"
    ).reindex(bin_order)
    print("\nObservation counts:\n")
    print(n_pivot.to_string())

    # --- per-model breakdown ---
    print("\nFlip rate by model × certainty bin (aggregated over pressure turns):\n")
    model_summary = (
        df2.groupby(["model_label", "certainty_bin"])["flip"]
        .mean()
        .unstack(level="certainty_bin")
        .reindex(columns=bin_order)
    )
    print(model_summary.round(3).to_string())

    # --- plot: flip rate curve per certainty bin ---
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = df2[df2["model_name"] == model]
        for bin_name in bin_order:
            g = sub[sub["certainty_bin"] == bin_name]
            if g.empty:
                continue
            rates = g.groupby("pressure_level")["flip"].mean()
            label = _entropy_range_label(sub, bin_name)
            ax.plot(rates.index, rates.values, marker="o", label=label)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Pressure turn")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_xticks(range(1, 7))
        ax.legend(fontsize=7)
    axes[0].set_ylabel("P(flip)")
    fig.suptitle("Approach 2: Flip rate by certainty bin × pressure level", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "approach2_certainty_bins.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Approach 3 — Logistic regression (continuous entropy)
# ---------------------------------------------------------------------------


def run_approach3(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("APPROACH 3 — Logistic regression (continuous entropy)")
    print("=" * 70)

    df3 = df.copy()
    # Reference model for C() is the first alphabetically; make GPT5_4 the
    # reference so Nano/Mini coefficients are interpreted relative to it.
    df3["model_name"] = pd.Categorical(
        df3["model_name"],
        categories=["GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "ClaudeHaiku", "ClaudeSonnet"],
    )

    formula = "flip ~ pressure_level * entropy + C(model_name)"
    print(f"\nFormula: {formula}")
    print(f"N observations: {len(df3):,}\n")

    result = smf.logit(formula, data=df3).fit(disp=False, maxiter=200)
    print(result.summary2())

    # --- odds ratios ---
    or_df = pd.concat(
        [
            np.exp(result.params).rename("OR"),
            np.exp(result.conf_int()).rename(columns={0: "CI_low", 1: "CI_high"}),
            result.pvalues.rename("p"),
        ],
        axis=1,
    )
    print("\nOdds ratios (OR) and 95% CIs:\n")
    print(or_df.round(4).to_string())

    # --- predicted probability surface: pressure level vs entropy ---
    # Fix model to GPT5_4 (reference) for the heatmap
    entropy_vals = np.linspace(df["entropy"].min(), 0, 60)
    pressure_vals = np.arange(1, 7)
    grid = pd.DataFrame(
        [
            {
                "entropy": e,
                "pressure_level": p,
                "model_name": pd.Categorical(
                    ["GPT5_4"],
                    categories=["GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "ClaudeHaiku", "ClaudeSonnet"],
                )[0],
            }
            for p in pressure_vals
            for e in entropy_vals
        ]
    )
    grid["model_name"] = pd.Categorical(
        grid["model_name"],
        categories=["GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "ClaudeHaiku", "ClaudeSonnet"],
    )
    grid["pred"] = result.predict(grid)
    heatmap_data = grid.pivot(index="pressure_level", columns="entropy", values="pred")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: heatmap
    ax = axes[0]
    im = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        origin="lower",
        extent=[entropy_vals.min(), entropy_vals.max(), 0.5, 6.5],
        vmin=0, vmax=1,
        cmap="RdYlGn_r",
    )
    fig.colorbar(im, ax=ax, label="P(flip)")
    ax.set_xlabel("Entropy (more negative → more uncertain)")
    ax.set_ylabel("Pressure level")
    ax.set_yticks(range(1, 7))
    ax.set_title("Predicted P(flip): GPT-5.4 reference model")

    # Right: per-model curves at fixed pressure levels
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    for p_idx, p in enumerate(pressure_vals):
        rows = []
        for model in ["GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "ClaudeHaiku", "ClaudeSonnet"]:
            grid_m = pd.DataFrame(
                {
                    "entropy": entropy_vals,
                    "pressure_level": p,
                    "model_name": pd.Categorical(
                        [model] * len(entropy_vals),
                        categories=["GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "ClaudeHaiku", "ClaudeSonnet"],
                    ),
                }
            )
            grid_m["pred"] = result.predict(grid_m)
            rows.append((model, grid_m["pred"].mean()))
        # Plot mean predicted flip across models at each pressure turn
        _, preds = zip(*rows)
        ax2.plot(
            [MODEL_LABELS[m] for m, _ in rows],
            preds,
            marker="o",
            color=colors[p_idx],
            label=f"Turn {p}",
        )
    ax2.set_ylabel("Mean predicted P(flip) across entropy range")
    ax2.set_title("Per-model predicted flip rate by pressure turn")
    ax2.legend(fontsize=8, title="Pressure turn")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle("Approach 3: Logistic regression — flip ~ pressure × entropy + model", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "approach3_logistic_regression.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)

    # --- simple decomposition: marginal effect of entropy at each pressure turn ---
    print("\nMarginal effect: ΔP(flip) per unit entropy (at mean pressure level):")
    b_entropy = result.params.get("entropy", 0)
    b_interact = result.params.get("pressure_level:entropy", 0)
    print(f"  β_entropy = {b_entropy:.4f},  β_(pressure×entropy) = {b_interact:.4f}")
    for p in pressure_vals:
        marginal_log_odds = b_entropy + b_interact * p
        print(f"  Turn {p}: log-odds slope on entropy = {marginal_log_odds:+.4f}  "
              f"(OR per unit entropy = {np.exp(marginal_log_odds):.4f})")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    df = build_long_df()

    # Quick sanity check on entropy distribution
    print("\nEntropy distribution across models (rows = model, pct with entropy==0):")
    print(
        df.drop_duplicates(["model_name", "query"])
        .assign(certain=lambda x: x["entropy"] == 0)
        .groupby("model_name")["certain"]
        .mean()
        .rename("pct_certain")
        .round(3)
        .to_string()
    )

    run_approach2(df)
    run_approach3(df)


if __name__ == "__main__":
    main()
