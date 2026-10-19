"""
rq3_rebuild.py — Regenerates tables/rq3.tex and fig/rq3_combined.pdf from
analysis/paper_out/results.csv (paper_pipeline.py's cluster-robust logit
trend, not the retired plain-OLS-on-6-points fit).

Unit note: the original table reports slopes in "pp/turn" from an OLS trend
on the 6 raw per-turn percentages. paper_pipeline.py's rq3() instead fits a
cluster-robust (by question) logistic trend (wrong ~ turn) on person-turn
data, for the same rigor as RQ1/RQ2 -- but that coefficient is on the
log-odds scale, not pp/turn, and the two are not numerically comparable.
This script reports the *average marginal effect* (AME) in pp/turn --
beta_logit * mean(p*(1-p)) * 100, evaluated at each cell's mean T1-T6 wrong
rate -- as the interpretable, caption-compatible number, while keeping the
cluster-robust p-value (not an OLS p-value on 6 points, which the original
could not cluster by question at all) for significance.

Usage: python analysis/rq3_rebuild.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

DATASETS = [("mmlu_pro", "MMLU-Pro", 5), ("gpqa_diamond", "GPQA-Diam.", 5), ("hle", "HLE", 6)]
MODELS_5 = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]
MODELS_HLE = MODELS_5 + ["GeminiFlash"]


def stars(p):
    if pd.isna(p):
        return r"\dagger"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else r"\dagger"


def ame_pp_per_turn(row):
    """Average marginal effect of the logit slope, in percentage points per
    turn, evaluated at the cell's own mean T1-T6 wrong rate (as a fraction)."""
    if pd.isna(row["slope"]):
        return np.nan
    pbar = np.nanmean([row[f"T{t}"] for t in range(1, 7)]) / 100.0
    return row["slope"] * pbar * (1 - pbar) * 100


def build_table(res):
    lines = [r"\noindent", r"\begin{minipage}{\columnwidth}", r"\centering", "",
             r"\captionof{table}{", "Per-turn flip-rate trend, cluster-robust (by question) logistic fit,",
             "reported as the average marginal effect (AME) in pp/turn at each cell's mean",
             "wrong rate (\\autoref{sec:methods:stats}). Negative AME indicates resistance,",
             "positive indicates capitulation; significance stars are from the underlying",
             "cluster-robust logit slope, not the AME approximation itself.",
             "Non-significant classifications are marked $\\dagger$.",
             "Full trajectories appear in \\autoref{fig:rq3combined}.", "}",
             r"\label{tab:rq3}", "", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
             r"\renewcommand{\arraystretch}{1.04}", "",
             r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"\textbf{Dataset}", r"& \textbf{Model}", r"& \textbf{AME \tiny{(pp/turn)}}",
             r"& \textbf{Group} \\", r"\midrule", ""]
    for i, (ds, label, n) in enumerate(DATASETS):
        sub = res[res.dataset == ds].set_index("model")
        models = MODELS_HLE if ds == "hle" else MODELS_5
        lines.append(r"\multirow{%d}{*}{%s}" % (n, label))
        for model in models:
            r = sub.loc[model]
            mlabel = pp.LABELS[model]
            if pd.isna(r.slope):
                lines.append(f"  & {mlabel}")
                lines.append(r"      & \multicolumn{2}{c}{$\varnothing$ \;\scriptsize(no reliable slope)} \\")
                continue
            ame = ame_pp_per_turn(r)
            sign = "+" if ame >= 0 else "-"
            st = stars(r.slope_p)
            group = "Resister" if r.slope < 0 else "Capitulator"
            group_tag = group if st != r"\dagger" else f"{group}$^\\dagger$"
            lines.append(f"  & {mlabel} & ${sign}${abs(ame):.2f}$^{{{st}}}$ & {group_tag} \\\\")
        if i < len(DATASETS) - 1:
            lines += [r"\addlinespace", r"\midrule"]
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{2pt}", "",
              r"{\footnotesize", r"\emph{Significance (cluster-robust logit slope):}",
              r"$^{***}p{<}0.001$,", r"$^{**}p{<}0.01$,", r"$^{*}p{<}0.05$,", r"$^\dagger p{\geq}0.05$.\\",
              r"\emph{Reproducibility:}", r"\texttt{analysis/rq3\_rebuild.py}", "}", "",
              r"\end{minipage}"]
    path = "paper/active_version/tables/rq3.tex"
    open(path, "w").write("\n".join(lines))
    print(f"wrote {path}")


def build_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    turns = np.arange(1, 7)
    colors = {"ClaudeHaiku": "#DC2626", "ClaudeSonnet": "#EA580C", "GPT5_4": "#2563EB",
              "GPT5_4Mini": "#7C3AED", "GPT5_4Nano": "#059669", "GeminiFlash": "#CA8A04"}
    for ax, (ds, label, n) in zip(axes, DATASETS):
        sub = res[res.dataset == ds].set_index("model")
        models = MODELS_HLE if ds == "hle" else MODELS_5
        for model in models:
            if model not in sub.index:
                continue
            r = sub.loc[model]
            y = np.array([r[f"T{t}"] for t in turns])
            lo = np.array([r[f"T{t}_lo"] for t in turns])
            hi = np.array([r[f"T{t}_hi"] for t in turns])
            if np.all(np.isnan(y)):
                continue
            ax.plot(turns, y, "-o", color=colors[model], label=pp.LABELS[model], markersize=4)
            ax.fill_between(turns, lo, hi, color=colors[model], alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("Pressure turn $T$")
        ax.set_ylim(0, 102)
        ax.set_xticks(turns)
    axes[0].set_ylabel("Flip rate (%)")
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("RQ3: per-turn flip-rate trajectories (95% Wilson CI bands)", fontsize=12)
    fig.tight_layout()
    path = "paper/active_version/sections/fig/rq3_combined.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    res = pd.read_csv("analysis/paper_out/results.csv")
    build_table(res)
    build_figure(res)


if __name__ == "__main__":
    main()
