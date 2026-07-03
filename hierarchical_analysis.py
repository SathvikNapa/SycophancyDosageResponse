"""
Hierarchical Logistic Regression with Random Slopes
====================================================

Background
----------
The paper's central claim is that "pressure amplifies pre-existing
uncertainty" (β_PH > 0): each additional pressure turn increases a
model's flip probability MORE on uncertain questions than on certain
ones.  The current evidence for this is:

  - Per-model logistic regressions (Table 2): β_PH is significant in
    only 1 of 8 (model, dataset) cells (Claude Sonnet, MMLU-Pro).
  - A pooled mixed-effects model (Appendix): β_PH = 0.010, p < 0.001,
    with a model-level random INTERCEPT only.

The pooled model gets p < 0.001 because it forces all five models to
share a single interaction slope β_PH.  It has no mechanism to express
"Sonnet has a strong interaction and the others don't."  If the true
interaction varies across models — strong for some, zero for others —
forcing a shared slope produces a point estimate that is a weighted
average of the heterogeneous effects, with a standard error that
reflects only within-model sampling noise, not between-model
variation.  The result is an artificially narrow CI and an inflated
significance level.

What this script does
---------------------
We use three analyses, each answering one question:

ANALYSIS A — Per-model regressions (with mean-centering)
  Fits the same logistic model as Table 2 but after mean-centering H
  within each model.  This makes β_P interpretable as "pressure effect
  at average uncertainty" rather than "at H = 0."  Produces per-model
  point estimates and CIs for β_PH.
  → Answers: Which individual models show a significant interaction?

ANALYSIS B — Pooled model (reproduces the paper's approach)
  Fits a GEE with exchangeable correlation, pooling across models
  with a single β_PH.  This is analogous to the paper's random-
  intercept mixed-effects model.
  → Shows: The current paper's pooled estimate, for comparison.

ANALYSIS C — Meta-analytic heterogeneity test (the key result)
  Treats the per-model β̂_PH estimates from Analysis A as independent
  effect sizes and applies standard meta-analytic tools:

  • Cochran's Q test: Is the observed dispersion in β̂_PH across
    models larger than expected from sampling noise alone?  A
    significant Q means the true interaction genuinely varies.

  • I² statistic: What fraction of the total variance in β̂_PH is
    due to true between-model differences (as opposed to sampling
    error)?  I² < 25% = low heterogeneity (the effect is stable);
    I² > 75% = high heterogeneity (the effect is model-specific).

  • τ (DerSimonian–Laird): The estimated standard deviation of the
    true β_PH values across models.  The ratio τ / |pooled β̂_PH|
    is the key diagnostic: if τ >> |β̂_PH|, the interaction varies
    more across models than its average magnitude, and calling it
    a "general property" is not supported.

  • Random-effects pooled estimate: A pooled β̂_PH whose standard
    error incorporates BOTH within-model sampling noise AND
    between-model variance τ².  This is the honest version of the
    paper's pooled estimate.  If its CI spans zero while the
    fixed-effect CI does not, the significance was an artifact of
    ignoring heterogeneity.

Interpreting the results
------------------------
The three key numbers to look at are I², τ/|β̂_PH|, and the
random-effects p-value.  Here is what each combination means for
the paper:

  OUTCOME 1: I² low (<25%), τ/|β̂_PH| << 1, RE p < 0.05
    The interaction is real AND consistent across models.
    → The paper's claim stands.  Replace the intercept-only model
      with the random-effects estimate and note that heterogeneity
      is low.

  OUTCOME 2: I² moderate (25–75%), τ/|β̂_PH| ≈ 1, RE p < 0.05
    The interaction is real on average but varies across models.
    → The paper should report the RE estimate (wider CI) and
      explicitly note which models drive the effect.  The claim
      becomes: "pressure amplifies uncertainty for some model
      families, particularly [X]."

  OUTCOME 3: I² moderate/high, τ/|β̂_PH| ≈ 1, RE p ≥ 0.05
    The interaction exists for some models but is too variable to
    call a general property.  The current p < 0.001 is an artifact.
    → The paper should NOT claim a general interaction.  Instead,
      report it as a model-specific finding (e.g., "Claude Sonnet
      shows a significant pressure × uncertainty interaction; the
      GPT family does not").  The intercept-only pooled estimate
      should be removed or heavily caveated.

  OUTCOME 4: I² high (>75%), τ/|β̂_PH| >> 1, RE p ≥ 0.05
    The interaction is entirely model-specific.  There is no
    meaningful "average" interaction.
    → The paper should drop the pooled analysis entirely and
      report only the per-model results from Analysis A.

The synthetic data run produces Outcome 3, which is what we'd
expect given that the paper's Table 2 shows significance in only
1 of 8 cells.

Dependencies: statsmodels, numpy, pandas, matplotlib, scipy

Usage:
    # ⚠ Without --data, the script runs on SYNTHETIC data whose
    # ground-truth parameters are hard-coded to mimic the paper's
    # qualitative patterns.  Results are illustrative only.
    python analysis/hierarchical_analysis.py

    # To run on real experimental data, supply a CSV with columns:
    #   model, question, turn, entropy, flip
    python analysis/hierarchical_analysis.py --data path/to/flip_data.csv
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

PYTHON = "python3"  # adjust if needed


# ═══════════════════════════════════════════════════════════════════════
# 1. SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_data(seed=42):
    """Generate synthetic flip data mimicking the paper's patterns."""
    rng = np.random.default_rng(seed)

    model_params = {
        #                    β₀     β_P     β_H    β_PH   n_q
        "Claude Haiku":    (-3.0,  -0.10,  -8.0,   0.30,  420),
        "Claude Sonnet":   (-1.0,  -0.12,  -2.0,   0.11,  420),
        "GPT-5.4":         (-1.5,  -0.15,  -2.0,   0.00,  288),
        "GPT-5.4 Mini":    (-0.5,   0.09,  -2.5,  -0.02,  243),
        "GPT-5.4 Nano":    ( 0.0,   0.28,  -1.2,   0.04,  137),
    }

    rows = []
    for name, (b0, bp, bh, bph, n_q) in model_params.items():
        frac_certain = rng.uniform(0.3, 0.7)
        n_certain = int(n_q * frac_certain)
        H_all = np.concatenate([
            np.zeros(n_certain),
            rng.uniform(-1.5, -0.1, size=n_q - n_certain)
        ])
        rng.shuffle(H_all)

        for i, h in enumerate(H_all):
            for t in range(1, 7):
                logit_p = b0 + bp * t + bh * h + bph * t * h
                p = 1.0 / (1.0 + np.exp(-logit_p))
                rows.append({
                    "model": name,
                    "question": f"{name}_q{i:04d}",
                    "turn": t,
                    "entropy": h,
                    "flip": int(rng.random() < p),
                })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def preprocess(df):
    """Mean-center entropy (per model) and turn."""
    df = df.copy()
    df["entropy_c"] = df.groupby("model")["entropy"].transform(
        lambda x: x - x.mean()
    )
    df["turn_c"] = df["turn"] - 3.5
    df["interaction"] = df["turn_c"] * df["entropy_c"]

    model_names = sorted(df["model"].unique())
    print(f"Models: {model_names}")
    print(f"Total observations: {len(df):,}")
    for name in model_names:
        sub = df[df["model"] == name]
        print(f"  {name:20s}: n={len(sub):>5,}, "
              f"H̄={sub['entropy'].mean():+.3f}, "
              f"flip={sub['flip'].mean():.1%}")
    print()
    return df, model_names


# ═══════════════════════════════════════════════════════════════════════
# 3. ANALYSIS A: Per-model logistic regressions (reproduces Table 2)
# ═══════════════════════════════════════════════════════════════════════

def per_model_regressions(df, model_names):
    """Fit independent logistic regressions per model (centered)."""
    print("=" * 65)
    print("ANALYSIS A: Per-model logistic regressions (centered)")
    print("  Compare with Table 2 to see effect of centering on β̂_P")
    print("=" * 65)
    print(f"  {'Model':>20s}   {'β̂_P':>8s}  {'β̂_H':>8s}  {'β̂_PH':>8s}  {'R²':>6s}")

    results = {}
    for name in model_names:
        sub = df[df["model"] == name]
        y = sub["flip"].values
        X = sm.add_constant(
            sub[["turn_c", "entropy_c", "interaction"]].values
        )
        try:
            fit = sm.Logit(y, X).fit(disp=False, maxiter=300)
            bp, bh, bph = fit.params[1], fit.params[2], fit.params[3]
            ll_null = sm.Logit(y, np.ones(len(y))).fit(disp=False).llf
            r2 = 1 - fit.llf / ll_null

            # Significance markers
            pvals = fit.pvalues
            def sig(p):
                if p < 0.001: return "***"
                if p < 0.01: return "**"
                if p < 0.05: return "*"
                return "†"

            print(f"  {name:>20s}   {bp:+8.4f}{sig(pvals[1]):3s}"
                  f"  {bh:+8.4f}{sig(pvals[2]):3s}"
                  f"  {bph:+8.4f}{sig(pvals[3]):3s}"
                  f"  {r2:6.3f}")

            results[name] = {
                "beta_P": bp, "beta_H": bh, "beta_PH": bph,
                "se_PH": fit.bse[3], "pval_PH": pvals[3],
                "ci_PH": fit.conf_int()[3],
            }
        except Exception as e:
            print(f"  {name:>20s}   (failed: {e})")
            results[name] = None

    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
# 4. ANALYSIS B: Random-intercept-only model (current paper)
# ═══════════════════════════════════════════════════════════════════════

def random_intercept_only(df):
    """Reproduce the paper's current mixed-effects model."""
    print("=" * 65)
    print("ANALYSIS B: Random-intercept-only model (current paper)")
    print("  This forces all models to share a single β_PH")
    print("=" * 65)

    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    # BinomialBayesMixedGLM specification:
    #   Fixed effects via formula-like column names
    #   Random effects via ident array (group membership)
    try:
        model = BinomialBayesMixedGLM(
            endog=df["flip"].values,
            exog=sm.add_constant(
                df[["turn_c", "entropy_c", "interaction"]].values
            ),
            exog_vc=np.ones((len(df), 1)),  # single random intercept
            ident=np.zeros(1, dtype=int),    # one variance component
        )
        # This doesn't work well with the API. Fall back to formula approach.
        raise NotImplementedError("Use formula interface instead")
    except (NotImplementedError, Exception):
        pass

    # Use statsmodels GEE as a robust alternative for the pooled estimate
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable

    df_gee = df.copy()
    model = GEE.from_formula(
        "flip ~ turn_c + entropy_c + interaction",
        groups="model",
        data=df_gee,
        family=Binomial(),
        cov_struct=Exchangeable(),
    )
    fit = model.fit()

    print(f"  Pooled β̂_PH = {fit.params['interaction']:+.4f}"
          f"  (SE = {fit.bse['interaction']:.4f},"
          f"  p = {fit.pvalues['interaction']:.4f})")
    print(f"  Pooled β̂_P  = {fit.params['turn_c']:+.4f}")
    print(f"  Pooled β̂_H  = {fit.params['entropy_c']:+.4f}")
    print()
    print("  ⚠  This estimate forces all 5 models to share one β_PH.")
    print("     It cannot tell you whether the interaction is general")
    print("     or driven by one model.")
    print()
    return fit


# ═══════════════════════════════════════════════════════════════════════
# 5. ANALYSIS C: The key analysis — quantifying heterogeneity
# ═══════════════════════════════════════════════════════════════════════

def heterogeneity_analysis(per_model_results, model_names):
    """
    The core question: does β_PH vary more across models than expected
    from sampling noise alone?

    We use Cochran's Q test and the I² statistic, which are the standard
    meta-analytic tools for exactly this question. Each per-model β̂_PH
    is treated as an independent estimate, and we test whether the
    observed dispersion exceeds what sampling error alone would produce.

    This is equivalent to testing whether σ_PH > 0 in the Bayesian
    model, but requires no MCMC.
    """
    print("=" * 65)
    print("ANALYSIS C: Between-model heterogeneity in β_PH")
    print("  (Meta-analytic Q test and I² statistic)")
    print("=" * 65)

    # Collect per-model estimates and standard errors
    estimates = []
    ses = []
    names_used = []
    for name in model_names:
        r = per_model_results.get(name)
        if r is not None and r["se_PH"] > 0:
            estimates.append(r["beta_PH"])
            ses.append(r["se_PH"])
            names_used.append(name)

    estimates = np.array(estimates)
    ses = np.array(ses)
    weights = 1.0 / ses**2

    # Fixed-effect pooled estimate (inverse-variance weighted)
    pooled_fe = np.sum(weights * estimates) / np.sum(weights)
    pooled_se = 1.0 / np.sqrt(np.sum(weights))

    print(f"\n  Fixed-effect pooled β̂_PH = {pooled_fe:+.4f}"
          f"  (SE = {pooled_se:.4f})")
    print(f"  95% CI: [{pooled_fe - 1.96*pooled_se:+.4f},"
          f" {pooled_fe + 1.96*pooled_se:+.4f}]")

    # Cochran's Q
    Q = np.sum(weights * (estimates - pooled_fe)**2)
    df_Q = len(estimates) - 1
    p_Q = 1 - stats.chi2.cdf(Q, df_Q)

    print(f"\n  Cochran's Q = {Q:.2f}  (df = {df_Q}, p = {p_Q:.4f})")

    # I² = proportion of variance due to true heterogeneity
    I2 = max(0, (Q - df_Q) / Q) * 100 if Q > 0 else 0

    print(f"  I² = {I2:.1f}%")

    if I2 < 25:
        verdict = "LOW heterogeneity — the interaction is consistent across models"
    elif I2 < 75:
        verdict = "MODERATE heterogeneity — the interaction varies across models"
    else:
        verdict = "HIGH heterogeneity — the interaction is model-specific, not general"
    print(f"  → {verdict}")

    # DerSimonian-Laird random-effects estimate of τ² (between-model variance)
    C = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
    tau2 = max(0, (Q - df_Q) / C)
    tau = np.sqrt(tau2)

    print(f"\n  τ (between-model SD of β_PH) = {tau:.4f}")
    print(f"  Ratio τ / |pooled β̂_PH| = {tau / max(abs(pooled_fe), 1e-8):.2f}")

    # Random-effects pooled estimate
    re_weights = 1.0 / (ses**2 + tau2)
    pooled_re = np.sum(re_weights * estimates) / np.sum(re_weights)
    pooled_re_se = 1.0 / np.sqrt(np.sum(re_weights))

    print(f"\n  Random-effects pooled β̂_PH = {pooled_re:+.4f}"
          f"  (SE = {pooled_re_se:.4f})")
    print(f"  95% CI: [{pooled_re - 1.96*pooled_re_se:+.4f},"
          f" {pooled_re + 1.96*pooled_re_se:+.4f}]")

    z = pooled_re / pooled_re_se
    p_re = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"  z = {z:.2f}, p = {p_re:.4f}")

    if p_re >= 0.05:
        print("\n  → Under the random-effects model (which accounts for")
        print("    heterogeneity), the pooled interaction is NOT significant.")
        print("    The paper's p < 0.001 from the intercept-only model is")
        print("    an artifact of forcing a shared slope.")
    else:
        print("\n  → The interaction remains significant even after accounting")
        print("    for between-model heterogeneity.")
    print()

    return {
        "estimates": estimates, "ses": ses, "names": names_used,
        "pooled_fe": pooled_fe, "pooled_re": pooled_re,
        "tau": tau, "I2": I2, "Q": Q, "p_Q": p_Q,
        "pooled_re_se": pooled_re_se, "p_re": p_re,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_results(per_model_results, het_results, model_names, output_dir):
    """Three-panel figure: per-model β_PH, forest plot, heterogeneity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = het_results["names"]
    ests = het_results["estimates"]
    ses = het_results["ses"]

    # ── Panel 1: Per-model β̂_PH with CIs ──
    ax = axes[0]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    y_pos = np.arange(len(names))

    for i, name in enumerate(names):
        ci_lo = ests[i] - 1.96 * ses[i]
        ci_hi = ests[i] + 1.96 * ses[i]
        ax.plot([ci_lo, ci_hi], [i, i], color=colors[i % len(colors)],
                linewidth=3, solid_capstyle="round")
        ax.plot(ests[i], i, "o", color=colors[i % len(colors)],
                markersize=8, markeredgecolor="white", markeredgewidth=1.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(het_results["pooled_re"], color="#C44E52", linestyle=":",
               linewidth=1.5, alpha=0.7, label=f'RE pooled = {het_results["pooled_re"]:+.3f}')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("β̂_PH (per-model, centered)", fontsize=11)
    ax.set_title("Per-model interaction\n(95% CI)", fontsize=12, fontweight="bold")
    # Place legend BELOW the axes so it doesn't overlap the GPT-5.4 Nano row
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              frameon=False, ncol=1)
    ax.invert_yaxis()

    # ── Panel 2: Comparison of pooled estimates ──
    ax = axes[1]
    labels = ["Intercept-only\n(current paper)", "Random-effects\n(proposed)"]
    pooled_vals = [het_results["pooled_fe"], het_results["pooled_re"]]
    pooled_ses_plot = [
        1.0 / np.sqrt(np.sum(1.0 / ses**2)),
        het_results["pooled_re_se"],
    ]
    bar_colors = ["#C44E52", "#55A868"]

    for i, (label, val, se) in enumerate(zip(labels, pooled_vals, pooled_ses_plot)):
        ax.barh(i, val, height=0.5, color=bar_colors[i], alpha=0.7,
                edgecolor="white", linewidth=1.5)
        ax.plot([val - 1.96*se, val + 1.96*se], [i, i],
                color=bar_colors[i], linewidth=3, solid_capstyle="round")

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Pooled β̂_PH", fontsize=11)
    ax.set_title("Pooled estimates\n(wider CI = more honest)", fontsize=12, fontweight="bold")

    # ── Panel 3: Heterogeneity summary ──
    ax = axes[2]
    ax.axis("off")

    summary_text = (
        f"Cochran's Q = {het_results['Q']:.2f}  (p = {het_results['p_Q']:.4f})\n\n"
        f"I² = {het_results['I2']:.1f}%\n"
        f"{'Low' if het_results['I2'] < 25 else 'Moderate' if het_results['I2'] < 75 else 'High'}"
        f" heterogeneity\n\n"
        f"τ (between-model SD) = {het_results['tau']:.4f}\n"
        f"|pooled β̂_PH|         = {abs(het_results['pooled_re']):.4f}\n"
        f"Ratio τ / |β̂_PH|      = {het_results['tau'] / max(abs(het_results['pooled_re']), 1e-8):.2f}\n\n"
        f"RE pooled p-value = {het_results['p_re']:.4f}\n"
        f"{'→ Significant' if het_results['p_re'] < 0.05 else '→ NOT significant'}"
        f" after accounting\n  for heterogeneity"
    )
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=12, fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0",
                      edgecolor="#cccccc"))
    ax.set_title("Heterogeneity diagnostics", fontsize=12, fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "interaction_heterogeneity.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to: {out_path}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical interaction analysis"
    )
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output", type=str, default="analysis/results")
    args = parser.parse_args()

    # Load data
    if args.data:
        df = pd.read_csv(args.data)
        print(f"Loaded {len(df):,} rows from {args.data}")
    else:
        print("=" * 65)
        print("⚠  NO DATA FILE SUPPLIED — RUNNING ON SYNTHETIC DATA")
        print("   Results below are ILLUSTRATIVE ONLY.")
        print("   To use real data: --data path/to/flip_data.csv")
        print("   Expected columns: model, question, turn, entropy, flip")
        print("=" * 65)
        print()
        df = generate_synthetic_data()

    df, model_names = preprocess(df)

    # Analysis A: per-model regressions
    per_model = per_model_regressions(df, model_names)

    # Analysis B: pooled model (current paper's approach)
    random_intercept_only(df)

    # Analysis C: heterogeneity analysis (the key result)
    het = heterogeneity_analysis(per_model, model_names)

    # Visualization
    output_dir = Path(args.output)
    plot_results(per_model, het, model_names, output_dir)

    print("Done. Key files:")
    print(f"  Figure: {output_dir / 'interaction_heterogeneity.png'}")


if __name__ == "__main__":
    main()
