"""
rq2_appendix.py — Rebuilds the "Full RQ2 Coefficient Table and Hardness
Robustness Check" appendix (tab:rq2_combined, fig:rq2_interaction,
fig:heterogeneity) on the final corrected/complete dataset.

Calibrated hardness h-hat: isotonic regression of the baseline per-question
error rate on |H| (sycophancy.entropy.fit_isotonic_calibration -- reused
unchanged), fit per (model, dataset) on the reparsed baseline metadata when
available. This is the same regressor scripts/run_calibration.py produces,
refit here because that script's calibration.pkl sidecars were fit on the
pre-correction base_experiment_metadata.pkl and are now stale for the
affected Claude/GPT5_4Nano cells.

The H_c and h-hat regressions both use paper_pipeline.py's discrete-time
survival parameterization (first-flip framing), for consistency with the
main-text RQ2 -- not the retired pooled-turns logistic the original appendix
used -- so "robustness check" means "same model class, alternate uncertainty
measure," not a different statistical test entirely.

Usage: python analysis/rq2_appendix.py
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp
from sycophancy.entropy import compute_entropy, fit_isotonic_calibration

OUT = pp.OUT
DATASETS = ["mmlu_pro", "gpqa_diamond", "hle"]
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]


def models_for(dataset):
    """HLE additionally has Gemini 3.5 Flash coverage (no MMLU-Pro/GPQA-Diamond
    data for that model), matching the main-text pooled RQ2 analysis and every
    other HLE-scoped table in this paper."""
    return MODELS + ["GeminiFlash"] if dataset == "hle" else MODELS


def fit_hardness(model, dataset):
    """Returns {query: calibrated_prob in [0,1]} for this cell, fit on the
    reparsed baseline metadata (falls back to original if absent)."""
    d = pp.cell_dir(model, dataset)
    for name in ("base_experiment_metadata_reparsed.pkl", "base_experiment_metadata.pkl"):
        path = os.path.join(d, name)
        if os.path.exists(path):
            items = pickle.load(open(path, "rb"))
            break
    else:
        return {}
    for item in items:
        if "entropy" not in item:
            item["entropy"] = compute_entropy(item.get("answers_generated", []))
    reg = fit_isotonic_calibration(items)
    X = np.array([-item["entropy"] for item in items])
    probs = reg.predict(X)
    return {item["query"]: float(p) for item, p in zip(items, probs)}


def rq2_generic(d, col, center):
    """paper_pipeline.rq2(), parameterized by uncertainty column; `center`
    mean-centers it (done for H, not for h-hat, which already has a
    meaningful zero -- matching the appendix's original model spec)."""
    d = d[d["complete"] & d["t0_correct"] & d[col].notna()].copy()
    d["U"] = d[col] - d[col].mean() if center else d[col]
    rows = []
    for _, row in d.iterrows():
        last = int(row["first_flip"]) if row["ever_flip"] else pp.N_TURNS - 1
        for t in range(1, last + 1):
            rows.append({"query": row["query"], "turn": t, "U": row["U"],
                         "flip": int(row["ever_flip"] and t == last)})
    person_turns = pd.DataFrame(rows)
    r = {k: np.nan for k in ("bP", "seP", "pP", "bU", "seU", "pU", "bPU", "sePU", "pPU")}
    r["separation"] = False
    if len(person_turns) < 30 or person_turns["flip"].nunique() < 2 or person_turns["U"].std() == 0:
        return r
    try:
        fit = pp.cluster_logit("flip ~ turn + U + turn:U", person_turns, "query")
        for k, name in (("P", "turn"), ("U", "U"), ("PU", "turn:U")):
            r[f"b{k}"], r[f"se{k}"], r[f"p{k}"] = fit.params[name], fit.bse[name], fit.pvalues[name]
        r["separation"] = bool(np.any(np.abs(fit.params) > 10))
    except Exception:
        r["separation"] = True
    return r


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else r"^\dagger"


def fmt_cell(b, p):
    if pd.isna(b):
        return None
    sign = "+" if b >= 0 else "-"
    return f"${sign}${abs(b):.3f}$^{{{stars(p)}}}$" if stars(p) != r"^\dagger" else f"${sign}${abs(b):.3f}$^\\dagger$"


def main():
    records = []
    for dataset in DATASETS:
        pool = pp.canonical_pool(dataset)
        for model in models_for(dataset):
            df, _ = pp.load_cell(model, dataset, pool)
            if df is None:
                continue
            d, _ = pp.dialogues(df)
            hmap = fit_hardness(model, dataset)
            d["hardness"] = d["query"].map(hmap)
            rH = rq2_generic(d, "entropy", center=True)
            rU = rq2_generic(d, "hardness", center=False)
            records.append({"dataset": dataset, "model": model, **{f"H_{k}": v for k, v in rH.items()},
                            **{f"U_{k}": v for k, v in rU.items()}})
            print(f"done {model}/{dataset}: H sep={rH['separation']} U sep={rU['separation']}", flush=True)
    res = pd.DataFrame(records)
    res.to_csv(os.path.join(OUT, "rq2_appendix.csv"), index=False)

    # ---- table ----
    lines = [r"\begin{tabular}{ll ccc ccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{\textbf{$H_c$-based (main)}} & \multicolumn{3}{c}{\textbf{$\hat h$-based (robustness)}} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"\textbf{Dataset} & \textbf{Model} & $\hat\beta_P$ & $\hat\beta_H$ & $\hat\beta_{PH}$ & $\hat\beta_P$ & $\hat\beta_{\hat h}$ & $\hat\beta_{P\hat h}$ \\",
             r"\midrule"]
    for i, dataset in enumerate(DATASETS):
        sub = res[res.dataset == dataset]
        ds_models = models_for(dataset)
        lines.append(r"\multirow{%d}{*}{%s}" % (len(ds_models), pp.LABELS[dataset]))
        for j, model in enumerate(ds_models):
            row = sub[sub.model == model]
            prefix = "  &" if j else " &"
            if row.empty:
                lines.append(f"{prefix} {pp.LABELS[model]} & \\multicolumn{{6}}{{c}}{{--- (no data)}} \\\\")
                continue
            r = row.iloc[0]
            hc = [fmt_cell(r[f"H_b{k}"], r[f"H_p{k}"]) for k in ("P", "U", "PU")]
            uc = [fmt_cell(r[f"U_b{k}"], r[f"U_p{k}"]) for k in ("P", "U", "PU")]
            h_ok = all(hc) and not bool(r.get("H_separation", False))
            u_ok = all(uc) and not bool(r.get("U_separation", False))
            left = " & ".join(hc) if h_ok else r"\multicolumn{3}{c}{--- \tiny{(separation)}}"
            right = " & ".join(uc) if u_ok else r"\multicolumn{3}{c}{--- \tiny{($\hat h$ collapsed)}}"
            lines.append(f"{prefix} {pp.LABELS[model]} & {left} & {right} \\\\")
        if i < len(DATASETS) - 1:
            lines += [r"\addlinespace", r"\midrule"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(OUT, "tables", "rq2_combined.tex")
    open(tex_path, "w").write("\n".join(lines))
    print(f"wrote {tex_path}")

    # ---- fig: predicted flip probability at two hardness levels, MMLU-Pro ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # figsize/margins tuned for how this actually renders: it sits in a figure*
    # (full double-column width, ~6.75in on the page) with 5 panels, so each one
    # prints at only ~1.3in -- the default 5% x-margin around a narrow T=1..6
    # range wasted a visible fraction of that. Tight xlim + small wspace lets
    # the curves fill each panel instead of floating in padding.
    fig, axes = plt.subplots(1, len(MODELS), figsize=(19, 4.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    turns = np.arange(1, 7)
    mmlu = res[res.dataset == "mmlu_pro"].set_index("model")
    for ax, model in zip(axes, MODELS):
        if model not in mmlu.index:
            continue
        r = mmlu.loc[model]
        b0 = 0.0  # intercept not stored/needed: plot the *shape* (relative log-odds), not absolute level
        for H_val, style, lbl in ((0.0, "-", "$H=0$"), (-1.5, "--", "$H=-1.5$")):
            if pd.isna(r.H_bP):
                continue
            logit = r.H_bP * turns + r.H_bU * H_val + r.H_bPU * turns * H_val
            prob = 1 / (1 + np.exp(-logit))
            ax.plot(turns, prob, style, color="#7C3AED", label=lbl, linewidth=2)
        ax.set_title(pp.LABELS[model], fontsize=13)
        ax.set_xlabel("$T$", fontsize=11)
        ax.set_xlim(0.85, 6.15)
        ax.set_xticks(range(1, 7))
        ax.margins(x=0)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Predicted $P(\\text{flip})$ (shape only)", fontsize=11)
    axes[0].legend(fontsize=10, loc="upper left")
    fig.suptitle("Predicted flip probability shape across $T_1$--$T_6$ at two $H$ levels (MMLU-Pro)\n"
                  "Curves show the fitted interaction's effect on slope, not an absolute baseline rate.",
                  fontsize=11)
    fig.tight_layout()
    fig_path = "paper/active_version/sections/fig/rq2_interaction.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")

    # ---- fig: heterogeneity (per-model MMLU-Pro CIs + pooled) ----
    pooled = pd.read_csv(os.path.join(OUT, "rq2_pooled.csv")).set_index("dataset")
    mmlu_pool = pooled.loc["mmlu_pro"] if "mmlu_pro" in pooled.index else None
    full = pd.read_csv(os.path.join(OUT, "results.csv"))
    mm = full[full.dataset == "mmlu_pro"].set_index("model")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    ax = axes[0]
    ys = np.arange(len(MODELS))
    for i, model in enumerate(MODELS):
        if model not in mm.index:
            continue
        r = mm.loc[model]
        if pd.isna(r.bPH) or bool(r.separation):
            continue
        lo, hi = r.bPH - 1.96 * r.sePH, r.bPH + 1.96 * r.sePH
        ax.errorbar(r.bPH, ys[i], xerr=[[r.bPH - lo], [hi - r.bPH]], fmt="o", color="#7C3AED", capsize=3)
    ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([pp.LABELS[m] for m in MODELS], fontsize=9)
    ax.set_xlabel(r"$\hat\beta_{PH}$"); ax.set_title("Per-model, MMLU-Pro")

    ax = axes[1]
    if mmlu_pool is not None:
        ax.errorbar(mmlu_pool.beta_PH, 0, xerr=[[mmlu_pool.beta_PH - mmlu_pool.lo], [mmlu_pool.hi - mmlu_pool.beta_PH]],
                    fmt="s", color="#DC2626", capsize=4, markersize=8,
                    label=f"pooled (k={int(mmlu_pool.k)}): {mmlu_pool.beta_PH:+.3f}, p={mmlu_pool.p:.3f}")
        ax.axvline(0, color="#9CA3AF", linestyle="--", linewidth=0.8)
        ax.set_yticks([]); ax.set_xlabel(r"$\hat\beta_{PH}$ (pooled)"); ax.legend(fontsize=8)
        ax.set_title("Random-effects pooled, MMLU-Pro")
    fig.suptitle("RQ2 interaction heterogeneity, final corrected dataset", fontsize=11)
    fig.tight_layout()
    fig_path = "paper/active_version/sections/fig/interaction_heterogeneity.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


def raw_h_ci_table():
    """app:logreg_ci's tables/rq2_ci.tex -- same survival-model fit as the main
    table, with H not mean-centered (bH and bPH are shift-invariant, so only
    bP differs from the main table; recomputed directly via rq2_generic rather
    than algebraically derived, to avoid a hand-arithmetic error). Cluster-
    robust (by question) SEs throughout, not naive MLE/inverse-Hessian SEs --
    the original caption's stated method; updated to describe what this
    refit actually uses."""
    pool = pp.canonical_pool("mmlu_pro")
    rows = []
    for model in MODELS:
        df, _ = pp.load_cell(model, "mmlu_pro", pool)
        d, _ = pp.dialogues(df)
        r = rq2_generic(d, "entropy", center=False)
        rows.append({"model": model, **r})
    res = pd.DataFrame(rows).set_index("model")
    res.to_csv(os.path.join(OUT, "rq2_raw_h_ci.csv"))

    lines = [r"\begin{tabular}{llll}", r"\toprule",
             r"\textbf{Model} & $\hat\beta_P$ \tiny{[CI]} & $\hat\beta_H$ \tiny{[CI]} & $\hat\beta_{PH}$ \tiny{[CI]} \\",
             r"\midrule"]
    for model in MODELS:
        r = res.loc[model]
        if pd.isna(r.bP) or bool(r.separation):
            lines.append(f"{pp.LABELS[model]} & \\multicolumn{{3}}{{c}}{{--- \\tiny{{(separation)}}}} \\\\")
            continue
        cells = []
        for k in ("P", "U", "PU"):
            b, se = r[f"b{k}"], r[f"se{k}"]
            lo, hi = b - 1.96 * se, b + 1.96 * se
            sign = "+" if b >= 0 else "-"
            cells.append(f"${sign}${abs(b):.3f} \\tiny{{[${'+' if lo>=0 else '-'}${abs(lo):.3f}, "
                         f"${'+' if hi>=0 else '-'}${abs(hi):.3f}]}}")
        lines.append(f"{pp.LABELS[model]} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(OUT, "tables", "rq2_ci.tex")
    open(tex_path, "w").write("\n".join(lines))
    print(f"wrote {tex_path}")
    return res


def mixed_effects_glm():
    """app:mixed_effects -- crossed-random-effects binomial GLM on MMLU-Pro,
    pooling all 5 Claude/GPT models, on the same survival-style person-period
    rows as the rest of RQ2 (one row per dialogue-turn at risk, flip=1 at the
    first-flip turn only), rather than the retired pooled-all-turns rows:
      logit P(flip) = b0 + u_m + v_q + w_m*turn + bP*turn + bH*Hc + bPH*turn*Hc
    u_m, v_q: crossed random intercepts for model and question.
    w_m: per-model random slope on pressure turn.
    Fit via statsmodels BinomialBayesMixedGLM (variational Bayes), matching
    the original's estimator choice.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    pool = pp.canonical_pool("mmlu_pro")
    rows = []
    for model in MODELS:
        df, _ = pp.load_cell(model, "mmlu_pro", pool)
        d, _ = pp.dialogues(df)
        d = d[d["complete"] & d["t0_correct"] & d["entropy"].notna()].copy()
        d["Hc"] = d["entropy"] - d["entropy"].mean()
        for _, row in d.iterrows():
            last = int(row["first_flip"]) if row["ever_flip"] else pp.N_TURNS - 1
            for t in range(1, last + 1):
                rows.append({"model": model, "query": row["query"], "turn": float(t), "Hc": row["Hc"],
                            "flip": int(row["ever_flip"] and t == last)})
    person_turns = pd.DataFrame(rows)
    print(f"mixed-effects: {len(person_turns)} person-turn rows, "
          f"{person_turns['model'].nunique()} models, {person_turns['query'].nunique()} questions "
          f"(shared across models, MMLU-Pro)", flush=True)

    # Three independent variance components, matching the paper's model spec:
    #   u_m  (per-model intercept)         -> "0 + C(model)"
    #   v_q  (per-question intercept)      -> "0 + C(query)" -- questions are the
    #        same 420 across all 5 models, so this is a true crossed (not nested) effect
    #   w_m  (per-model slope on turn)     -> "0 + C(model):turn"
    # vcp_p=0.5 per the statsmodels docstring's own stated recommendation for
    # Binomial-family numerical stability (default 1 can make the posterior
    # mode hard to find).
    vc_formulas = {"model_intercept": "0 + C(model)", "question_intercept": "0 + C(query)",
                   "model_slope": "0 + C(model):turn"}
    try:
        model_fit = BinomialBayesMixedGLM.from_formula(
            "flip ~ turn + Hc + turn:Hc", vc_formulas, person_turns, vcp_p=0.5)
        result = model_fit.fit_vb()
        converged = True
    except Exception as e:
        print(f"mixed-effects GLM failed to fit: {type(e).__name__}: {e}", flush=True)
        return None

    fe_mean = dict(zip(model_fit.fep_names, result.fe_mean))
    fe_sd = dict(zip(model_fit.fep_names, result.fe_sd))
    vcp_mean = dict(zip(model_fit.vcp_names, result.vcp_mean))
    vcp_sd = dict(zip(model_fit.vcp_names, result.vcp_sd))
    out = {
        "beta_PH": fe_mean.get("turn:Hc"), "beta_PH_sd": fe_sd.get("turn:Hc"),
        "beta_P": fe_mean.get("turn"), "beta_P_sd": fe_sd.get("turn"),
        "beta_H": fe_mean.get("Hc"), "beta_H_sd": fe_sd.get("Hc"),
        "n_rows": len(person_turns), "n_questions": person_turns["query"].nunique(),
        "converged": converged,
        # vcp_mean/sd are log(SD) of each random effect per the docstring
        # ("prior standard deviation of log(s)"); exponentiate for the SD itself.
        "sigma_model_intercept": float(np.exp(vcp_mean.get("model_intercept", np.nan))),
        "sigma_question_intercept": float(np.exp(vcp_mean.get("question_intercept", np.nan))),
        "sigma_model_slope": float(np.exp(vcp_mean.get("model_slope", np.nan))),
    }
    print("mixed-effects result:", out, flush=True)
    pickle.dump(out, open(os.path.join(OUT, "rq2_mixed_effects.pkl"), "wb"))
    return out


if __name__ == "__main__":
    main()
    raw_h_ci_table()
    mixed_effects_glm()
