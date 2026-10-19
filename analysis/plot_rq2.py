"""
plot_rq2.py — RQ2 coefficient plot (beta_P, beta_H, beta_PH per model x dataset)
from analysis/paper_out/results.csv, replacing paper/active_version/sections/fig/rq2_coefs.pdf.

Small-multiples grid: one row per coefficient (Pressure / Entropy / Interaction),
one column per dataset. beta_P, beta_H and beta_PH live on genuinely different
scales and mean different things, so earlier versions that overlaid all three on
one shared (symlog) x-axis per model produced a cramped, hard-to-read tick mess
at the zero crossing. Splitting them into their own row gives each coefficient a
plain linear axis sized to its own data, which is far more legible than a single
symlog axis spanning three unrelated quantities.

Usage: python analysis/plot_rq2.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_pipeline as pp

matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"

OUT_PATH = "paper/active_version/sections/fig/rq2_coefs.pdf"
DATASETS = ["mmlu_pro", "gpqa_diamond", "hle"]
DATASET_LABELS = {"mmlu_pro": "MMLU-Pro", "gpqa_diamond": "GPQA-Diamond", "hle": "HLE"}
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "GeminiFlash"]
# Fixed categorical order (never cycled/re-picked per panel): muted navy, maroon, dark teal
# -- subdued print/journal tones rather than bright web-dashboard hues.
ROWS = [
    ("bP", "seP", r"Pressure ($\hat{\beta}_P$)", "#2C4A6E"),
    ("bH", "seH", r"Entropy ($\hat{\beta}_H$)", "#8C3A3A"),
    ("bPH", "sePH", r"Interaction ($\hat{\beta}_{PH}$)", "#2F6B5E"),
]
MUTED_GRAY = "#9CA3AF"


def row_xlim(res, b_col, se_col):
    ok = res[~res["separation"].fillna(False)]
    sub = ok[[b_col, se_col]].dropna()
    lo = (sub[b_col] - 1.96 * sub[se_col]).min()
    hi = (sub[b_col] + 1.96 * sub[se_col]).max()
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def main():
    res = pd.read_csv("analysis/paper_out/results.csv").set_index(["dataset", "model"])
    y_pos = np.arange(len(MODELS))[::-1]  # top row = first model, matching reading order

    fig, axes = plt.subplots(len(ROWS), len(DATASETS), figsize=(16, 11), sharey="row")

    for r, (b_col, se_col, row_label, color) in enumerate(ROWS):
        xlim = row_xlim(res, b_col, se_col)
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            ax.axvline(0, color=MUTED_GRAY, linewidth=0.9, linestyle="--", zorder=1)
            for i, model in enumerate(MODELS):
                key = (ds, model)
                if key not in res.index:
                    continue
                row = res.loc[key]
                b, se = row.get(b_col), row.get(se_col)
                separated = bool(row.get("separation", False))
                if pd.isna(b) or pd.isna(se) or separated:
                    ax.text(0.03, y_pos[i], "excluded" if separated else "n/a",
                             transform=ax.get_yaxis_transform(), va="center",
                             fontsize=9, color=MUTED_GRAY, style="italic")
                    continue
                lo, hi = b - 1.96 * se, b + 1.96 * se
                sig = (lo > 0) or (hi < 0)
                ax.errorbar(b, y_pos[i], xerr=[[b - lo], [hi - b]],
                            fmt="o", color=color, alpha=0.95 if sig else 0.4,
                            capsize=3, markersize=7, linewidth=1.6,
                            markeredgewidth=0)
            ax.set_xlim(*xlim)
            ax.xaxis.grid(True, alpha=0.4, linestyle=":")
            ax.yaxis.grid(False)
            ax.tick_params(axis="x", labelsize=11)
            if r == 0:
                ax.set_title(DATASET_LABELS[ds], fontsize=15, fontweight="bold")
            if r == len(ROWS) - 1:
                ax.set_xlabel("Coefficient value", fontsize=12)
            if c == 0:
                ax.set_yticks(y_pos)
                ax.set_yticklabels([pp.LABELS[m] for m in MODELS], fontsize=12.5)
            else:
                ax.tick_params(axis="y", left=False)

        axes[r, 0].annotate(row_label, xy=(-0.42, 0.5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=14,
                             fontweight="bold", color=color)

    sig_handle = plt.Line2D([], [], marker="o", color="#4B5563", linestyle="", markersize=7,
                             label="95% CI excludes zero")
    nonsig_handle = plt.Line2D([], [], marker="o", color="#4B5563", linestyle="", markersize=7,
                                alpha=0.4, label="95% CI includes zero")
    fig.legend(handles=[sig_handle, nonsig_handle], loc="upper center",
               bbox_to_anchor=(0.52, 1.015), ncol=2, framealpha=0.9,
               edgecolor="#ccc", fontsize=12)
    fig.suptitle("RQ2: discrete-time survival coefficients, cluster-robust SEs",
                  fontsize=16, y=1.06, fontweight="bold")
    fig.tight_layout(rect=[0.03, 0, 1, 1])
    fig.subplots_adjust(wspace=0.12, hspace=0.3)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
