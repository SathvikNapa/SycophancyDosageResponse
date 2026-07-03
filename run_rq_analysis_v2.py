"""
run_rq_analysis_v2.py
---------------------
Runs RQ1-RQ4 analyses for MMLU-Pro and GPQA-Diamond (79 common questions).
Saves paper-ready v2 plots to experiment_out/rq_plots/
and writes updated LaTeX tables to paper_tables_v2.tex.
"""

from __future__ import annotations

import os
import pickle
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from build_analysis_dfs import build_flat_df, get_common_queries

# ── Constants ────────────────────────────────────────────────────────────────

PLOT_DIR   = "experiment_out/rq_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_ORDER   = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]
MODEL_LABELS  = ["Claude\nHaiku", "Claude\nSonnet", "GPT-5.4", "GPT-5.4\nMini", "GPT-5.4\nNano"]
DATASET_ORDER = ["mmlu_pro", "gpqa_diamond"]
DATASET_LABELS = {"mmlu_pro": "MMLU-Pro", "gpqa_diamond": "GPQA-Diamond"}

RESISTER_COLOR    = "#2563EB"
CAPITULATOR_COLOR = "#DC2626"
CERTAIN_COLOR     = "#6B7280"
UNCERTAIN_COLOR   = "#F59E0B"

RESISTER_MODELS    = {"ClaudeHaiku", "ClaudeSonnet", "GPT5_4"}
CAPITULATOR_MODELS = {"GPT5_4Mini", "GPT5_4Nano"}
PENDING_MODELS     = {"GeminiFlash"}

HARDNESS_BINS   = [("Easy",   0.0, 1/3),
                   ("Medium", 1/3, 2/3),
                   ("Hard",   2/3, 1.001)]
HARDNESS_COLORS = {"Easy": "#22C55E", "Medium": "#F59E0B", "Hard": "#EF4444"}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})

# ── Helpers ──────────────────────────────────────────────────────────────────

def hardness_label(p):
    if p < 1/3:  return "Easy"
    if p < 2/3:  return "Medium"
    return "Hard"

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return p*100, max(0,(c-m)*100), min(100,(c+m)*100)

def load_gpqa_reasoning_df(
    experiment_dir="experiment_out",
    models=MODEL_ORDER,
    common_queries=None,
):
    """Loads calibrated cross-turn rows from experiment_out/<model>/gpqa_diamond/."""
    rows = []
    for model in models:
        d = os.path.join(experiment_dir, model, "gpqa_diamond", "reasoning_calibrated_bin")
        for pkl in sorted(glob.glob(os.path.join(d, "bin_*_cross_turn.pkl"))):
            bin_idx = int(os.path.basename(pkl).split("_")[1])
            with open(pkl, "rb") as f:
                cross = pickle.load(f)
            for ct in cross:
                q = ct.get("query","")
                if common_queries is not None and q not in common_queries:
                    continue
                rows.append({
                    "model":                         model,
                    "dataset":                       "gpqa_diamond",
                    "bin_idx":                       bin_idx,
                    "query":                         q,
                    "gold_answer":                   ct.get("gold_answer"),
                    "step":                          ct.get("step"),
                    "turn":                          ct.get("turn"),
                    "belief_entropy":                ct.get("belief_entropy"),
                    "majority_is_correct":           ct.get("majority_is_correct"),
                    "mean_self_reported_confidence": ct.get("mean_self_reported_confidence"),
                    "cluster_drift":                 ct.get("cluster_drift"),
                })
    return pd.DataFrame(rows)


# ── Load data ────────────────────────────────────────────────────────────────

print("Loading flat df...")
common_gpqa = get_common_queries("gpqa_diamond")
df = build_flat_df(restrict_queries={"gpqa_diamond": common_gpqa})
pressure = df[df["turn"] >= 1].copy()

print("Loading MMLU reasoning df...")
from build_analysis_dfs import build_reasoning_df, REASONING_TRACE
mmlu_rdf = build_reasoning_df(reasoning_dir=REASONING_TRACE)
mmlu_cal = mmlu_rdf[mmlu_rdf["trace_type"]=="reasoning_calibrated_bin"].copy()
mmlu_cal["dataset"] = "mmlu_pro"

print("Loading GPQA reasoning df...")
gpqa_cal = load_gpqa_reasoning_df(common_queries=common_gpqa)

REASONING_COLS = ["belief_entropy","majority_is_correct","mean_self_reported_confidence"]
for col in REASONING_COLS:
    if col in mmlu_cal.columns:
        mmlu_cal[col] = pd.to_numeric(mmlu_cal[col], errors="coerce")
    if not gpqa_cal.empty and col in gpqa_cal.columns:
        gpqa_cal[col] = pd.to_numeric(gpqa_cal[col], errors="coerce")
    elif not gpqa_cal.empty:
        gpqa_cal[col] = np.nan

cal = pd.concat([mmlu_cal, gpqa_cal], ignore_index=True)

print(f"Flat df: {len(df)} rows  |  Reasoning df: {len(cal)} rows")
print()

# ═══════════════════════════════════════════════════════════════════════════
# RQ1 — Easy / Medium / Hard flip rates
# ═══════════════════════════════════════════════════════════════════════════

print("=== RQ1 ===")
rq1_rows = []
for ds in DATASET_ORDER:
    sub = pressure[pressure["dataset"]==ds].dropna(subset=["calibrated_prob"])
    for model in MODEL_ORDER:
        m = sub[sub["model"]==model]
        if len(m) == 0:
            continue
        groups = {lbl: m[(m["calibrated_prob"] >= lo) & (m["calibrated_prob"] < hi)]
                  for lbl, lo, hi in HARDNESS_BINS}
        for label, grp in groups.items():
            n = len(grp); k = int(grp["flipped"].sum())
            p_val, ci_lo, ci_hi = wilson(k, n)
            rq1_rows.append(dict(dataset=ds, model=model, hardness=label,
                                 n=n, k=k, flip_pct=p_val, ci_lo=ci_lo, ci_hi=ci_hi))
        group_vals = [groups[lbl]["flipped"].values for lbl, _, _ in HARDNESS_BINS
                      if len(groups[lbl]) > 0]
        if len(group_vals) >= 2:
            _, kw_p = stats.kruskal(*group_vals)
        else:
            kw_p = np.nan
        pcts = {lbl: wilson(int(groups[lbl]["flipped"].sum()), len(groups[lbl]))[0]
                for lbl, _, _ in HARDNESS_BINS}
        print(f"  {ds} {model}: easy={pcts['Easy']:.1f}%  med={pcts['Medium']:.1f}%  "
              f"hard={pcts['Hard']:.1f}%  KW p={kw_p:.4f}")

rq1_df = pd.DataFrame(rq1_rows)

# -- Distribution of questions per hardness bin --
qdf = df.drop_duplicates(subset=["model","dataset","query"]).dropna(subset=["calibrated_prob"]).copy()
qdf["hardness"] = qdf["calibrated_prob"].apply(hardness_label)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, ds in zip(axes, DATASET_ORDER):
    sub = qdf[qdf["dataset"]==ds]
    x = np.arange(len(MODEL_ORDER))
    bottom = np.zeros(len(MODEL_ORDER))
    for lbl, _, _ in HARDNESS_BINS:
        counts = [len(sub[(sub["model"]==m) & (sub["hardness"]==lbl)]) for m in MODEL_ORDER]
        ax.bar(x, counts, bottom=bottom, label=lbl, color=HARDNESS_COLORS[lbl],
               alpha=0.88, edgecolor="white", linewidth=0.5)
        for xi, (c, bot) in enumerate(zip(counts, bottom)):
            if c > 0:
                ax.text(xi, bot + c/2, str(c), ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottom = bottom + np.array(counts, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_ylabel("Unique questions")
    ax.set_title(DATASET_LABELS[ds])
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.xaxis.grid(False)
    ax.legend(loc="upper right", framealpha=0.9)

fig.suptitle("Question Distribution by Calibrated Hardness Bin (Easy: p<1/3, Medium: 1/3-2/3, Hard: p>=2/3)",
             fontsize=10, y=1.01)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "rq1_hardness_distribution.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# -- Flip rate bar chart (Easy / Medium / Hard) --
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
w = 0.22
offsets = np.array([-w, 0.0, w])
for ax, ds in zip(axes, DATASET_ORDER):
    sub = rq1_df[rq1_df["dataset"]==ds]
    x = np.arange(len(MODEL_ORDER))
    for (label, _, _), offset in zip(HARDNESS_BINS, offsets):
        color = HARDNESS_COLORS[label]
        vals = [sub[(sub["model"]==m) & (sub["hardness"]==label)]["flip_pct"].values for m in MODEL_ORDER]
        lo   = [sub[(sub["model"]==m) & (sub["hardness"]==label)]["ci_lo"].values for m in MODEL_ORDER]
        hi   = [sub[(sub["model"]==m) & (sub["hardness"]==label)]["ci_hi"].values for m in MODEL_ORDER]
        ys  = [v[0] if len(v) else np.nan for v in vals]
        los = [v[0] if len(v) else 0.0 for v in lo]
        his = [v[0] if len(v) else 0.0 for v in hi]
        bar_ys = [y if not np.isnan(y) else 0.0 for y in ys]
        err = [[max(0.0, y-l) if not np.isnan(y) else 0.0 for y,l in zip(ys,los)],
               [max(0.0, h-y) if not np.isnan(y) else 0.0 for y,h in zip(ys,his)]]
        ax.bar(x + offset, bar_ys, w, label=label, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(x + offset, bar_ys, yerr=err, fmt="none",
                    color="#374151", capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_ylabel("Flip rate (%)")
    ax.set_title(DATASET_LABELS[ds])
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.xaxis.grid(False); ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.9)

fig.suptitle("RQ1: Flip Rate by Calibrated Hardness (95% Wilson CI)", fontsize=12, y=1.01)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "rq1_uncertainty_flip_rate_v2.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ═══════════════════════════════════════════════════════════════════════════
# RQ2 — Logistic regression per model x dataset
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== RQ2 ===")
try:
    from statsmodels.formula.api import logit
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("  WARNING: statsmodels not available")

rq2_rows = []
for ds in DATASET_ORDER:
    sub = pressure[pressure["dataset"]==ds].copy()
    sub = sub.dropna(subset=["calibrated_prob","flipped"])
    for model in MODEL_ORDER:
        m = sub[sub["model"]==model].copy()
        if len(m) < 50 or not HAS_SM:
            continue
        try:
            mod = logit("flipped ~ turn + calibrated_prob + turn:calibrated_prob", data=m).fit(disp=0)
            b, p, ci = mod.params, mod.pvalues, mod.conf_int()
            rq2_rows.append(dict(
                dataset=ds, model=model,
                b_pressure=b["turn"], p_pressure=p["turn"],
                ci_p_lo=ci.loc["turn",0], ci_p_hi=ci.loc["turn",1],
                b_hardness=b["calibrated_prob"], p_hardness=p["calibrated_prob"],
                ci_h_lo=ci.loc["calibrated_prob",0], ci_h_hi=ci.loc["calibrated_prob",1],
                b_inter=b["turn:calibrated_prob"], p_inter=p["turn:calibrated_prob"],
                ci_i_lo=ci.loc["turn:calibrated_prob",0], ci_i_hi=ci.loc["turn:calibrated_prob",1],
                r2=1 - mod.llf/mod.llnull, n=len(m),
            ))
            print(f"  {ds} {model}: b_P={b['turn']:.3f}({'***' if p['turn']<0.001 else 'ns'})  "
                  f"b_H={b['calibrated_prob']:.3f}({'***' if p['calibrated_prob']<0.001 else 'ns'})  "
                  f"R2={1-mod.llf/mod.llnull:.3f}  n={len(m)}")
        except Exception as e:
            print(f"  {ds} {model}: ERROR {e}")

rq2_df = pd.DataFrame(rq2_rows)

# Plot RQ2 v2 — coefficient plot, one panel per dataset
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
coef_info = [("b_pressure","ci_p_lo","ci_p_hi","p_pressure","Pressure dose ($\\hat{\\beta}_P$)","#4B5563"),
             ("b_hardness", "ci_h_lo","ci_h_hi","p_hardness","Hardness prob ($\\hat{\\beta}_H$)", "#DC2626"),
             ("b_inter",    "ci_i_lo","ci_i_hi","p_inter",   "Interaction ($\\hat{\\beta}_{PH}$)","#7C3AED")]

for ax, ds in zip(axes, DATASET_ORDER):
    sub = rq2_df[rq2_df["dataset"]==ds]
    y_positions = np.arange(len(MODEL_ORDER))
    offset_step = 0.22
    handles = []
    for ci_idx, (b_col, lo_col, hi_col, p_col, label, color) in enumerate(coef_info):
        offsets = (ci_idx - 1) * offset_step
        ys, xerr_lo, xerr_hi, alphas = [], [], [], []
        for model in MODEL_ORDER:
            row = sub[sub["model"]==model]
            if row.empty:
                ys.append(np.nan); xerr_lo.append(0); xerr_hi.append(0); alphas.append(0.3)
            else:
                bv = row[b_col].values[0]
                ys.append(bv)
                xerr_lo.append(bv - row[lo_col].values[0])
                xerr_hi.append(row[hi_col].values[0] - bv)
                sig = row[p_col].values[0] < 0.05
                alphas.append(0.9 if sig else 0.35)
        for i, (y, xlo, xhi, alp) in enumerate(zip(ys, xerr_lo, xerr_hi, alphas)):
            if np.isnan(y):
                continue
            h = ax.errorbar(y, y_positions[i]+offsets, xerr=[[xlo],[xhi]],
                            fmt="o", color=color, alpha=alp,
                            capsize=3, markersize=5, linewidth=1.2,
                            label=label if i == 0 else "")
            if i == 0:
                handles.append(h)
    ax.axvline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_xlabel("Coefficient value")
    ax.set_title(DATASET_LABELS[ds])
    ax.xaxis.grid(True, alpha=0.3, linestyle=":")
    ax.yaxis.grid(False)

axes[0].legend(handles=handles, loc="upper center",
               bbox_to_anchor=(1.05, -0.18), ncol=3,
               framealpha=0.9, edgecolor="#ccc")
fig.suptitle("RQ2: Logistic Regression Coefficients: calibrated hardness (faded = p >= 0.05)", fontsize=12, y=1.02)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "rq2_coefficients_v2.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ═══════════════════════════════════════════════════════════════════════════
# RQ3 — Pressure curves T1-T6
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== RQ3 ===")
rq3_rows = []
for ds in DATASET_ORDER:
    sub = pressure[pressure["dataset"]==ds]
    for model in MODEL_ORDER:
        m = sub[sub["model"]==model]
        for t in range(1, 7):
            grp = m[m["turn"]==t]
            if len(grp) == 0:
                continue
            pv, lo, hi = wilson(int(grp["flipped"].sum()), len(grp))
            rq3_rows.append(dict(dataset=ds, model=model, turn=t,
                                 flip_pct=pv, ci_lo=lo, ci_hi=hi, n=len(grp)))
        xs = np.arange(1,7)
        ys = [wilson(int(m[m["turn"]==t]["flipped"].sum()), len(m[m["turn"]==t]))[0]
              for t in xs if len(m[m["turn"]==t]) > 0]
        if len(ys) >= 2:
            slope, _, _, pval, _ = stats.linregress(xs[:len(ys)], ys)
            group = "Resister" if model in RESISTER_MODELS else ("Capitulator" if model not in PENDING_MODELS else "TBD")
            print(f"  {ds} {model}: slope={slope:.3f} pp/turn p={pval:.4f} [{group}]  "
                  f"T1={ys[0]:.1f}% T6={ys[-1]:.1f}%")

rq3_df = pd.DataFrame(rq3_rows)

# Plot RQ3 v2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
for ax, ds in zip(axes, DATASET_ORDER):
    sub = rq3_df[rq3_df["dataset"]==ds]
    for model in MODEL_ORDER:
        m = sub[sub["model"]==model].sort_values("turn")
        if m.empty:
            continue
        color = RESISTER_COLOR if model in RESISTER_MODELS else CAPITULATOR_COLOR
        ls    = "-" if model in RESISTER_MODELS else "--"
        label = f"{model.replace('GPT5_4','GPT-5.4').replace('Claude','Claude ')}"
        ax.plot(m["turn"], m["flip_pct"], color=color, linestyle=ls,
                marker="o", markersize=4, linewidth=1.6, label=label)
        ax.fill_between(m["turn"], m["ci_lo"], m["ci_hi"],
                        color=color, alpha=0.12)
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"T{t}" for t in range(1, 7)])
    ax.set_xlabel("Pressure dose")
    ax.set_ylabel("Flip rate (%)")
    ax.set_title(DATASET_LABELS[ds])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
    ax.xaxis.grid(False); ax.yaxis.grid(True, alpha=0.3, linestyle=":")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
all_h = {l: h for l, h in zip(labels+l2, handles+h2)}
fig.legend(all_h.values(), all_h.keys(), loc="lower center",
           bbox_to_anchor=(0.5, -0.12), ncol=5, framealpha=0.9, edgecolor="#ccc",
           fontsize=8.5)
from matplotlib.lines import Line2D
leg_extra = [Line2D([0],[0],color=RESISTER_COLOR,lw=2,label="Resister"),
             Line2D([0],[0],color=CAPITULATOR_COLOR,lw=2,linestyle="--",label="Capitulator")]
axes[1].legend(handles=leg_extra, loc="upper right", fontsize=8.5, framealpha=0.9)

fig.suptitle("RQ3: Flip Rate Across Pressure Doses T1-T6 (95% CI shaded)", fontsize=12, y=1.02)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "rq3_pressure_curves_v2.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ═══════════════════════════════════════════════════════════════════════════
# RQ4 — Reasoning trajectory (calibrated)
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== RQ4 ===")

def rq4_stats(rdf):
    """Returns df with mean belief_entropy, majority_is_correct, conf (0-1) per model x turn."""
    rdf = rdf.copy()
    rdf["conf_01"] = pd.to_numeric(rdf["mean_self_reported_confidence"], errors="coerce") / 100.0
    rdf["mc"]      = pd.to_numeric(rdf["majority_is_correct"], errors="coerce")
    rdf["be"]      = pd.to_numeric(rdf["belief_entropy"],       errors="coerce")
    last = rdf.sort_values("step").groupby(["model","dataset","query","turn"]).last().reset_index()
    out = last.groupby(["model","dataset","turn"]).agg(
        be_mean  =("be",      "mean"),
        be_se    =("be",      lambda x: x.sem()),
        mc_mean  =("mc",      "mean"),
        mc_se    =("mc",      lambda x: x.sem()),
        conf_mean=("conf_01", "mean"),
        conf_se  =("conf_01", lambda x: x.sem()),
    ).reset_index()
    out["ovconf"] = out["conf_mean"] - out["mc_mean"]
    return out

rq4 = rq4_stats(cal)

for ds in DATASET_ORDER:
    for model in MODEL_ORDER:
        sub = rq4[(rq4["dataset"]==ds)&(rq4["model"]==model)]
        if sub.empty: continue
        t0 = sub[sub["turn"]==0]
        t6 = sub[sub["turn"]==6]
        if t0.empty or t6.empty: continue
        print(f"  {ds} {model}: T0 mc={t0['mc_mean'].values[0]:.3f} conf={t0['conf_mean'].values[0]:.3f} gap={t0['ovconf'].values[0]:.3f} | "
              f"T6 mc={t6['mc_mean'].values[0]:.3f} conf={t6['conf_mean'].values[0]:.3f} gap={t6['ovconf'].values[0]:.3f}")

# Plot RQ4 v2 — 3 metrics x 2 datasets
z = 1.96
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharey="row")

metric_cfg = [
    ("be_mean",   "be_se",   "Belief Entropy",         None),
    ("mc_mean",   "mc_se",   "Majority Correct Rate",  None),
    ("ovconf",    None,      "Overconfidence Gap\n(Calibrated Conf. - Correct Rate)", None),
]

for col, ds in enumerate(DATASET_ORDER):
    rq4_ds = rq4[rq4["dataset"]==ds]
    for row, (mean_col, se_col, ylabel, _) in enumerate(metric_cfg):
        ax = axes[row][col]
        for model in MODEL_ORDER:
            m = rq4_ds[rq4_ds["model"]==model].sort_values("turn")
            if m.empty: continue
            color = RESISTER_COLOR if model in RESISTER_MODELS else CAPITULATOR_COLOR
            ls    = "-" if model in RESISTER_MODELS else "--"
            label = model.replace("GPT5_4","GPT-5.4").replace("Claude","Claude ")
            ys = m[mean_col].values
            ax.plot(m["turn"], ys, color=color, linestyle=ls,
                    marker="o", markersize=3.5, linewidth=1.5, label=label)
            if se_col and se_col in m.columns:
                se = m[se_col].values
                ax.fill_between(m["turn"], ys - z*se, ys + z*se,
                                color=color, alpha=0.1)
        ax.set_xticks(range(0, 7))
        ax.set_xticklabels([f"T{t}" for t in range(0, 7)], fontsize=8)
        ax.xaxis.grid(False); ax.yaxis.grid(True, alpha=0.3, linestyle=":")
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 0:
            ax.set_title(DATASET_LABELS[ds], fontsize=11)
        if row == 2:
            ax.set_xlabel("Pressure turn")

handles, labels_leg = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower center",
           bbox_to_anchor=(0.5, -0.04), ncol=5,
           framealpha=0.9, edgecolor="#ccc", fontsize=9)
fig.suptitle("RQ4: CoT Reasoning Trajectory Under Pressure (Calibrated)", fontsize=13, y=1.01)
fig.tight_layout()
out = os.path.join(PLOT_DIR, "rq4_trajectory_v2.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {out}")

# ═══════════════════════════════════════════════════════════════════════════
# LaTeX tables v2
# ═══════════════════════════════════════════════════════════════════════════

print("\nWriting paper_tables_v2.tex ...")

def sig_star(p):
    if np.isnan(p): return r"$^\dagger$"
    if p < 0.001: return r"$^{***}$"
    if p < 0.01:  return r"$^{**}$"
    if p < 0.05:  return r"$^{*}$"
    return r"$^\dagger$"

def fmt_pct(p, lo, hi):
    return f"{p:.1f} [{lo:.1f},\\,{hi:.1f}]"

MNAME = {"ClaudeHaiku":"ClaudeHaiku","ClaudeSonnet":"ClaudeSonnet",
         "GPT5_4":"GPT-5.4","GPT5_4Mini":"GPT-5.4 Mini","GPT5_4Nano":"GPT-5.4 Nano",
         "GeminiFlash":"Gemini 3.5 Flash"}

lines = []
def L(s=""): lines.append(s)

# ── RQ1 table ────────────────────────────────────────────────────────────────
L(r"\begin{table*}[t]")
L(r"\centering")
L(r"\caption{RQ1: Flip rate (\%) by calibrated hardness bin under pressure turns T1--T6.")
L(r"  Easy: $p{<}1/3$; Medium: $1/3{\leq}p{<}2/3$; Hard: $p{\geq}2/3$.")
L(r"  Wilson 95\% CIs in brackets. $p$ from Kruskal-Wallis across three groups.}")
L(r"\label{tab:rq1v2}")
L(r"\small")
L(r"\begin{tabular}{llccccc}")
L(r"\toprule")
L(r"\textbf{Dataset} & \textbf{Model} & \textbf{Easy (\%)} & \textbf{Medium (\%)} & \textbf{Hard (\%)} & $\Delta_{E{\to}H}$ & $p$ \\")
L(r"\midrule")

for ds in DATASET_ORDER:
    ds_label = DATASET_LABELS[ds]
    first = True
    nrows = sum(1 for m in MODEL_ORDER if not rq1_df[(rq1_df["dataset"]==ds)&(rq1_df["model"]==m)].empty)
    for model in MODEL_ORDER:
        rows = {lbl: rq1_df[(rq1_df["dataset"]==ds)&(rq1_df["model"]==model)&(rq1_df["hardness"]==lbl)]
                for lbl, _, _ in HARDNESS_BINS}
        if all(r.empty for r in rows.values()):
            continue
        def bin_str(lbl):
            r = rows[lbl]
            return fmt_pct(*r[["flip_pct","ci_lo","ci_hi"]].values[0]) if not r.empty else "-"
        easy_pct  = rows["Easy"]["flip_pct"].values[0]  if not rows["Easy"].empty  else None
        hard_pct  = rows["Hard"]["flip_pct"].values[0]  if not rows["Hard"].empty  else None
        delta = f"{hard_pct - easy_pct:+.1f}" if (easy_pct is not None and hard_pct is not None) else "-"
        sub_p = pressure[(pressure["dataset"]==ds)&(pressure["model"]==model)].dropna(subset=["calibrated_prob"])
        group_vals = [sub_p[(sub_p["calibrated_prob"] >= lo) & (sub_p["calibrated_prob"] < hi)]["flipped"].values
                      for _, lo, hi in HARDNESS_BINS]
        group_vals = [g for g in group_vals if len(g) > 0]
        if len(group_vals) >= 2:
            _, kw_p = stats.kruskal(*group_vals)
            p_str = r"$<$0.001" if kw_p < 0.001 else f"{kw_p:.3f}"
        else:
            p_str = "-"
        ds_col = f"\\multirow{{{nrows}}}{{*}}{{{ds_label}}}" if first else ""
        L(f"{ds_col} & {MNAME[model]} & {bin_str('Easy')} & {bin_str('Medium')} & {bin_str('Hard')} & {delta} & {p_str} \\\\")
        first = False
    L(r"\addlinespace")

L(r"\bottomrule")
L(r"\end{tabular}")
L(r"\end{table*}")
L()

# ── RQ2 table ────────────────────────────────────────────────────────────────
L(r"\begin{table*}[t]")
L(r"\centering")
L(r"\caption{RQ2: Logistic regression predicting answer flip.")
L(r"  Model: flip $\sim \beta_P \cdot \text{dose} + \beta_H \cdot \text{hardness} + \beta_{PH} \cdot \text{dose}\times\text{hardness}$.")
L(r"  Hardness = isotonic-calibrated error probability $\in[0,1]$.")
L(r"  $^{***}p{<}0.001$, $^{**}p{<}0.01$, $^{*}p{<}0.05$, $^\dagger p{\geq}0.05$. $R^2$ = McFadden.}")
L(r"\label{tab:rq2v2}")
L(r"\small")
L(r"\begin{tabular}{llccccc}")
L(r"\toprule")
L(r"\textbf{Dataset} & \textbf{Model} & $\hat\beta_P$ & $\hat\beta_H$ & $\hat\beta_{PH}$ & $R^2$ & $n$ \\")
L(r"\midrule")

for ds in DATASET_ORDER:
    ds_label = DATASET_LABELS[ds]
    sub = rq2_df[rq2_df["dataset"]==ds]
    first = True
    nrows = len(sub)
    for model in MODEL_ORDER:
        row = sub[sub["model"]==model]
        if row.empty:
            ds_col = f"\\multirow{{{nrows}}}{{*}}{{{ds_label}}}" if first else ""
            L(f"{ds_col} & {MNAME[model]} & --- & --- & --- & --- & --- \\\\")
            first = False
            continue
        bp = row["b_pressure"].values[0]; pp = row["p_pressure"].values[0]
        be = row["b_hardness"].values[0]; pe = row["p_hardness"].values[0]
        bi = row["b_inter"].values[0];    pi = row["p_inter"].values[0]
        r2 = row["r2"].values[0];         n  = int(row["n"].values[0])
        ds_col = f"\\multirow{{{nrows}}}{{*}}{{{ds_label}}}" if first else ""
        L(f"{ds_col} & {MNAME[model]} & "
          f"${bp:+.3f}${sig_star(pp)} & "
          f"${be:+.3f}${sig_star(pe)} & "
          f"${bi:+.3f}${sig_star(pi)} & "
          f"{r2:.3f} & {n:,} \\\\")
        first = False
    L(r"\addlinespace")

L(r"\bottomrule")
L(r"\end{tabular}")
L(r"{\footnotesize Sign of $\hat\beta_P$: negative = resister (flip rate falls with dose); positive = capitulator.}")
L(r"\end{table*}")
L()

# ── RQ3 table ────────────────────────────────────────────────────────────────
L(r"\begin{table*}[t]")
L(r"\centering")
L(r"\caption{RQ3: Flip rate (\%) at selected pressure doses (MMLU-Pro and GPQA-Diamond).")
L(r"  T2, T4, T5 omitted; see Figure~\ref{fig:rq3v2}. Slope = pp/turn across T1--T6.")
L(r"  $^{***}p{<}0.001$, $^{**}p{<}0.01$, $^{*}p{<}0.05$, $^\dagger p{\geq}0.05$.}")
L(r"\label{tab:rq3v2}")
L(r"\small")
L(r"\setlength{\tabcolsep}{4pt}")
L(r"\begin{tabular}{llccccc}")
L(r"\toprule")
L(r"\textbf{Dataset} & \textbf{Model} & \textbf{T1} & \textbf{T3} & \textbf{T6} & \textbf{Slope} & \textbf{Group} \\")
L(r"\midrule")

for ds in DATASET_ORDER:
    ds_label = DATASET_LABELS[ds]
    sub = rq3_df[rq3_df["dataset"]==ds]
    first = True
    nrows = sum(1 for m in MODEL_ORDER if not sub[sub["model"]==m].empty)
    for model in MODEL_ORDER:
        m = sub[sub["model"]==model]
        if m.empty: continue
        def gv(t): return m[m["turn"]==t]["flip_pct"].values
        t1 = gv(1); t3 = gv(3); t6 = gv(6)
        t1s = f"{t1[0]:.1f}" if len(t1) else "---"
        t3s = f"{t3[0]:.1f}" if len(t3) else "---"
        t6s = f"{t6[0]:.1f}" if len(t6) else "---"
        xs = np.arange(1,7)
        ys = [m[m["turn"]==t]["flip_pct"].values[0] for t in xs if len(m[m["turn"]==t])>0]
        if len(ys)>=2:
            sl, _, _, pv, _ = stats.linregress(xs[:len(ys)], ys)
            sign = "+" if sl >= 0 else ""
            sl_str = f"${sign}{sl:.2f}${sig_star(pv)}"
        else:
            sl_str = "---"
        group = "Resister" if model in RESISTER_MODELS else ("Capitulator" if model not in PENDING_MODELS else "TBD")
        ds_col = f"\\multirow{{{nrows}}}{{*}}{{{ds_label}}}" if first else ""
        L(f"{ds_col} & {MNAME[model]} & {t1s} & {t3s} & {t6s} & {sl_str} & {group} \\\\")
        first = False
    L(r"\addlinespace")

L(r"\bottomrule")
L(r"\end{tabular}")
L(r"\end{table*}")
L()

# ── RQ4 table ────────────────────────────────────────────────────────────────
L(r"\begin{table*}[t]")
L(r"\centering")
L(r"\caption{RQ4: CoT reasoning trajectory at baseline (T0) and max pressure (T6).")
L(r"  \textit{Bel.\ Ent.} = belief entropy (0 = consistent; more negative = more spread).")
L(r"  \textit{Maj.\ Corr.} = fraction of runs whose majority belief is correct.")
L(r"  \textit{Conf.} = mean calibrated confidence (0--1 scale).")
L(r"  \textit{Gap} = Conf.\ $-$ Maj.\ Corr.\ (overconfidence).}")
L(r"\label{tab:rq4v2}")
L(r"\small")
L(r"\begin{tabular}{lllcccc}")
L(r"\toprule")
L(r"\textbf{Dataset} & \textbf{Model} & \textbf{Turn} & \textbf{Bel.\ Ent.} & \textbf{Maj.\ Corr.} & \textbf{Conf.} & \textbf{Gap} \\")
L(r"\midrule")

for ds in DATASET_ORDER:
    ds_label = DATASET_LABELS[ds]
    sub = rq4[rq4["dataset"]==ds]
    nrows = sum(1 for m in MODEL_ORDER if not sub[sub["model"]==m].empty) * 2
    first = True
    for model in MODEL_ORDER:
        m = sub[sub["model"]==model]
        if m.empty: continue
        for turn in [0, 6]:
            t = m[m["turn"]==turn]
            if t.empty:
                L(f"& {MNAME[model]} & T{turn} & --- & --- & --- & --- \\\\")
                continue
            be   = t["be_mean"].values[0]
            mc   = t["mc_mean"].values[0]
            conf = t["conf_mean"].values[0]
            gap  = t["ovconf"].values[0]
            ds_col = f"\\multirow{{{nrows}}}{{*}}{{{ds_label}}}" if (first and turn==0) else ""
            mr     = f"\\multirow{{2}}{{*}}{{{MNAME[model]}}}" if turn==0 else ""
            L(f"{ds_col} & {mr} & T{turn} & ${be:.3f}$ & {mc:.3f} & {conf:.3f} & {gap:.3f} \\\\")
        first = False
        L(r"\addlinespace")

L(r"\bottomrule")
L(r"\end{tabular}")
L(r"\end{table*}")

tex = "\n".join(lines)
with open("paper_tables_v2.tex", "w") as f:
    f.write("% paper_tables_v2.tex — MMLU-Pro + GPQA-Diamond (79 common questions)\n")
    f.write("% Preamble: \\usepackage{booktabs,multirow,graphicx}\n\n")
    f.write(tex)

print("  Saved paper_tables_v2.tex")
print("\nDone.")
