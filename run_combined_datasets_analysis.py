"""
run_combined_datasets_analysis.py
----------------------------------
Plots RQ1-RQ4 with MMLU-Pro and GPQA-Diamond merged into single panels.

  Color   = model identity
  Solid + circle marker  = MMLU-Pro
  Dashed + square marker = GPQA-Diamond

Outputs: experiment_out/rq_plots/rq{1-4}_combined_datasets.png
"""

from __future__ import annotations
import os, pickle, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.formula.api import logit

PLOT_DIR = "experiment_out/rq_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_ORDER  = ["ClaudeHaiku","ClaudeSonnet","GPT5_4","GPT5_4Mini","GPT5_4Nano"]
MODEL_LABELS = {
    "ClaudeHaiku":  "Claude\nHaiku",
    "ClaudeSonnet": "Claude\nSonnet",
    "GPT5_4":       "GPT-5.4",
    "GPT5_4Mini":   "GPT-5.4\nMini",
    "GPT5_4Nano":   "GPT-5.4\nNano",
}
MNAME = {
    "ClaudeHaiku": "Claude Haiku", "ClaudeSonnet": "Claude Sonnet",
    "GPT5_4": "GPT-5.4", "GPT5_4Mini": "GPT-5.4 Mini", "GPT5_4Nano": "GPT-5.4 Nano",
}
RESISTER_MODELS    = {"ClaudeHaiku","ClaudeSonnet","GPT5_4"}
CAPITULATOR_MODELS = {"GPT5_4Mini","GPT5_4Nano"}

MODEL_COLORS = {
    "ClaudeHaiku":  "#4C72B0",
    "ClaudeSonnet": "#55A868",
    "GPT5_4":       "#C44E52",
    "GPT5_4Mini":   "#DD8452",
    "GPT5_4Nano":   "#8172B2",
}

CERTAIN_COLOR   = "#6B7280"
UNCERTAIN_COLOR = "#F59E0B"

DS_MARKER  = {"mmlu_pro": "o", "gpqa_diamond": "s"}
DS_LS      = {"mmlu_pro": "-", "gpqa_diamond": "--"}
DS_LABEL   = {"mmlu_pro": "MMLU-Pro", "gpqa_diamond": "GPQA-Diamond"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

def wilson(k, n, z=1.96):
    if n == 0: return 0., 0., 0.
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d; m = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return p*100, max(0,(c-m)*100), min(100,(c+m)*100)

def sig_star(p):
    if pd.isna(p):  return r"$^\dagger$"
    if p < 0.001:   return r"$^{***}$"
    if p < 0.01:    return r"$^{**}$"
    if p < 0.05:    return r"$^{*}$"
    return r"$^\dagger$"

# ═══════════════════════════════════════════════════════════════════════════
# MMLU-Pro hardcoded results
# ═══════════════════════════════════════════════════════════════════════════

MMLU_RQ1 = {
    "ClaudeHaiku":  (0.0,  0.0, 0.1,  28.2, 27.1, 29.3, 28.2, 0.0),
    "ClaudeSonnet": (10.2, 9.6,10.8,  71.1, 69.5, 72.7, 60.9, 0.0),
    "GPT5_4":       (16.4,15.6,17.2,  35.2, 32.1, 38.4, 18.8, 0.0),
    "GPT5_4Mini":   (52.1,50.9,53.4,  79.5, 77.0, 81.7, 27.4, 0.0),
    "GPT5_4Nano":   (65.4,63.9,66.9,  76.1, 71.4, 80.2, 10.7, 0.0),
}
MMLU_RQ2 = {
    "ClaudeHaiku":  ( 1.378,0.4956,-2.586, 5.342,-25.116,0.0,-29.847,-20.386, 0.484,0.4440,-0.756,1.725, 0.687,12600),
    "ClaudeSonnet": (-0.116,0.0,  -0.152,-0.080, -2.063,0.0, -2.234, -1.892, 0.110,0.0,    0.071,0.148, 0.368,12600),
    "GPT5_4":       (-0.150,0.0,  -0.186,-0.114, -1.979,0.0, -2.530, -1.429,-0.004,0.9586,-0.148,0.140, 0.038, 8640),
    "GPT5_4Mini":   ( 0.086,0.0,   0.057, 0.115, -2.541,0.0, -3.304, -1.779,-0.030,0.7752,-0.235,0.175, 0.035, 7290),
    "GPT5_4Nano":   ( 0.279,0.0,   0.237, 0.320, -1.193,0.1230,-2.709,0.323, 0.038,0.8620,-0.389,0.464, 0.039, 4110),
}
MMLU_RQ3 = {
    "ClaudeHaiku":  {1:15.5,2:14.8,3:14.7,4:14.7,5:12.8,6:11.7},
    "ClaudeSonnet": {1:29.0,2:27.2,3:27.0,4:26.1,5:24.5,6:18.7},
    "GPT5_4":       {1:23.5,2:22.2,3:19.2,4:16.2,5:15.3,6:13.3},
    "GPT5_4Mini":   {1:45.8,2:57.2,3:58.4,4:58.8,5:59.3,6:58.8},
    "GPT5_4Nano":   {1:42.5,2:62.3,3:69.3,4:73.1,5:74.9,6:76.1},
}
MMLU_RQ3_CI = {
    "ClaudeHaiku":  {1:(14.0,17.1),2:(13.4,16.4),3:(13.3,16.3),4:(13.2,16.2),5:(11.4,14.3),6:(10.4,13.1)},
    "ClaudeSonnet": {1:(27.1,31.0),2:(25.3,29.1),3:(25.1,28.9),4:(24.3,28.0),5:(22.7,26.4),6:(17.1,20.4)},
    "GPT5_4":       {1:(21.4,25.8),2:(20.2,24.4),3:(17.2,21.3),4:(14.4,18.2),5:(13.5,17.2),6:(11.7,15.2)},
    "GPT5_4Mini":   {1:(43.1,48.7),2:(54.4,60.0),3:(55.6,61.1),4:(56.1,61.6),5:(56.5,62.0),6:(56.1,61.6)},
    "GPT5_4Nano":   {1:(38.8,46.2),2:(58.6,65.9),3:(65.8,72.7),4:(69.7,76.3),5:(71.5,78.0),6:(72.7,79.1)},
}
MMLU_RQ4 = {
    "ClaudeHaiku":  {0:(-0.033,0.752,0.918,0.167), 6:(-0.184,0.356,0.909,0.553)},
    "ClaudeSonnet": {0:(-0.007,0.876,0.930,0.054), 6:(-0.162,0.504,0.919,0.416)},
    "GPT5_4":       {0:(-0.031,0.821,0.956,0.135), 6:(-0.216,0.570,0.923,0.353)},
    "GPT5_4Mini":   {0:(-0.077,0.761,0.946,0.185), 6:(-0.174,0.451,0.928,0.477)},
    "GPT5_4Nano":   {0:(-0.093,0.666,0.920,0.254), 6:(-0.112,0.396,0.915,0.519)},
}

# ═══════════════════════════════════════════════════════════════════════════
# GPQA-Diamond fresh computation
# ═══════════════════════════════════════════════════════════════════════════

def _inline_entropy(answers):
    if not answers: return 0.0
    from collections import Counter
    c = Counter(answers); t = len(answers)
    p = np.array([v/t for v in c.values()])
    return float((p * np.log(p)).sum())

def load_entropy_map(model):
    path = f"experiment_out/{model}/gpqa_diamond/base_experiment_metadata.pkl"
    try:
        with open(path, "rb") as f: meta = pickle.load(f)
        out = {}
        for item in meta:
            ent = item.get("entropy") or _inline_entropy(item.get("answers_generated", []))
            out[item["query"]] = {"entropy": ent, "uncertainty": item.get("uncertainty")}
        return out
    except: return {}

entropy_maps = {m: load_entropy_map(m) for m in MODEL_ORDER}
print("GPQA entropy map sizes:", {m: len(v) for m, v in entropy_maps.items()})

# Common queries
common_queries = None
for m in MODEL_ORDER:
    try:
        with open(f"experiment_out/{m}/gpqa_diamond/base_experiment_metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        qs = {item["query"] for item in meta}
        common_queries = qs if common_queries is None else common_queries & qs
    except: pass
print(f"Common GPQA queries: {len(common_queries) if common_queries else 0}")

# Load GPQA sycophancy
gpqa_rows = []
for model in MODEL_ORDER:
    em = entropy_maps[model]
    for pkl in sorted(glob.glob(f"experiment_out/{model}/gpqa_diamond/entropy_bin/bin_*_repeated.pkl")):
        with open(pkl, "rb") as f: questions = pickle.load(f)
        for q in questions:
            query = q["query"]
            if common_queries and query not in common_queries: continue
            ev = em.get(query, {})
            for run_idx, run in enumerate(q.get("raw_runs", [])):
                iw = run.get("is_wrong", []); tc = run.get("turn_categories", [None]*len(iw))
                fwt = run.get("first_wrong_turn", len(iw))
                for turn, (wrong, cat) in enumerate(zip(iw, tc)):
                    gpqa_rows.append({"model": model, "query": query,
                                      "entropy": ev.get("entropy"),
                                      "turn": turn, "flipped": int(wrong),
                                      "first_wrong_turn": fwt})

gpqa_df = pd.DataFrame(gpqa_rows)
gpqa_df["certain"] = (gpqa_df["entropy"] == 0.0).astype("Int64")
gpqa_df.loc[gpqa_df["entropy"].isna(), "certain"] = pd.NA
gpqa_pressure = gpqa_df[gpqa_df["turn"] >= 1].copy()

# Load GPQA reasoning
rrows = []
for model in MODEL_ORDER:
    for pkl in sorted(glob.glob(
            f"experiment_out/{model}/gpqa_diamond/reasoning_calibrated_bin/bin_*_cross_turn.pkl")):
        with open(pkl, "rb") as f: cross = pickle.load(f)
        for ct in cross:
            rrows.append({"model": model,
                "turn": ct.get("turn"), "step": ct.get("step"),
                "majority_is_correct": pd.to_numeric(ct.get("majority_is_correct"), errors="coerce"),
                "mean_conf": pd.to_numeric(ct.get("mean_self_reported_confidence"), errors="coerce") / 100.0,
                "belief_entropy": pd.to_numeric(ct.get("belief_entropy"), errors="coerce")})
gpqa_rdf = pd.DataFrame(rrows)
gpqa_rdf["gap"] = gpqa_rdf["mean_conf"] - gpqa_rdf["majority_is_correct"]
print(f"GPQA flat: {len(gpqa_df)} rows  GPQA reasoning: {len(gpqa_rdf)} rows")

# ── GPQA stats ────────────────────────────────────────────────────────────
MODELS_WITH_ENT = [m for m in MODEL_ORDER if entropy_maps[m]]

gpqa_rq1 = {}
for model in MODEL_ORDER:
    m = gpqa_pressure[gpqa_pressure["model"] == model]
    if entropy_maps[model]:
        cert = m[m["certain"] == 1]; unct = m[m["certain"] == 0]
        cp, clo, chi = wilson(int(cert["flipped"].sum()), len(cert))
        up, ulo, uhi = wilson(int(unct["flipped"].sum()), len(unct))
        _, pv = stats.mannwhitneyu(unct["flipped"], cert["flipped"], alternative="greater") \
            if len(unct) > 0 and len(cert) > 0 else (0, np.nan)
        gpqa_rq1[model] = (cp, clo, chi, up, ulo, uhi, up-cp, pv)
    else:
        ap, alo, ahi = wilson(int(m["flipped"].sum()), len(m))
        gpqa_rq1[model] = (np.nan, np.nan, np.nan, ap, alo, ahi, np.nan, np.nan)

gpqa_rq2 = {}
for model in MODELS_WITH_ENT:
    m = gpqa_pressure[(gpqa_pressure["model"] == model) & gpqa_pressure["entropy"].notna()].copy()
    if len(m) < 30: continue
    try:
        mod = logit("flipped ~ turn + entropy + turn:entropy", data=m).fit(disp=0)
        b, p, ci = mod.params, mod.pvalues, mod.conf_int()
        gpqa_rq2[model] = (b["turn"], p["turn"], ci.loc["turn",0], ci.loc["turn",1],
                           b["entropy"], p["entropy"], ci.loc["entropy",0], ci.loc["entropy",1],
                           b["turn:entropy"], p["turn:entropy"],
                           ci.loc["turn:entropy",0], ci.loc["turn:entropy",1],
                           1-mod.llf/mod.llnull, len(m))
    except: pass

gpqa_rq3 = {}; gpqa_rq3_ci = {}
for model in MODEL_ORDER:
    m = gpqa_pressure[gpqa_pressure["model"] == model]
    gpqa_rq3[model] = {}; gpqa_rq3_ci[model] = {}
    for t in range(1, 7):
        grp = m[m["turn"] == t]
        if len(grp) == 0: continue
        pv, lo, hi = wilson(int(grp["flipped"].sum()), len(grp))
        gpqa_rq3[model][t] = pv; gpqa_rq3_ci[model][t] = (lo, hi)

last = gpqa_rdf.sort_values("step").groupby(["model", "turn"]).last().reset_index()
gpqa_rq4_full = {}
for model in MODEL_ORDER:
    m = last[last["model"] == model]
    gpqa_rq4_full[model] = {
        int(row["turn"]): (row["belief_entropy"], row["majority_is_correct"],
                           row["mean_conf"], row["gap"])
        for _, row in m.iterrows()
    }

# ═══════════════════════════════════════════════════════════════════════════
# RQ1 — single panel, grouped bars (certain / uncertain × dataset)
# ═══════════════════════════════════════════════════════════════════════════
print("\nPlotting RQ1 combined_datasets...")

fig, ax = plt.subplots(figsize=(11, 4.8))
x = np.arange(len(MODEL_ORDER))
# 4 bars per model: MMLU-certain, MMLU-uncertain, GPQA-certain, GPQA-uncertain
bar_w = 0.18
offsets = [-1.5*bar_w, -0.5*bar_w, 0.5*bar_w, 1.5*bar_w]
configs = [
    (MMLU_RQ1,  "cert",   CERTAIN_COLOR,   "",   "MMLU-Pro Certain"),
    (MMLU_RQ1,  "unct",   UNCERTAIN_COLOR, "",   "MMLU-Pro Uncertain"),
    (gpqa_rq1,  "cert",   CERTAIN_COLOR,   "//", "GPQA-Diam. Certain"),
    (gpqa_rq1,  "unct",   UNCERTAIN_COLOR, "//", "GPQA-Diam. Uncertain"),
]

for off, (data, kind, color, hatch, label) in zip(offsets, configs):
    ys, los, his = [], [], []
    for model in MODEL_ORDER:
        d = data.get(model)
        if d is None:
            ys.append(0); los.append(0); his.append(0); continue
        if kind == "cert":
            yv, lv, hv = d[0], d[1], d[2]
        else:
            yv, lv, hv = d[3], d[4], d[5]
        if pd.isna(yv):
            ys.append(0); los.append(0); his.append(0)
        else:
            ys.append(yv); los.append(lv); his.append(hv)
    err = [[max(0, y-l) for y, l in zip(ys, los)],
           [max(0, h-y) for y, h in zip(ys, his)]]
    ax.bar(x + off, ys, bar_w, label=label, color=color, hatch=hatch,
           alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.errorbar(x + off, ys, yerr=err, fmt="none", color="#374151", capsize=2.5, linewidth=0.9)

# N/A annotations for GPQA models without entropy
for xi, model in enumerate(MODEL_ORDER):
    d = gpqa_rq1.get(model)
    if d and pd.isna(d[0]):
        ax.text(xi + 0.5*bar_w, 3, "N/A", ha="center", va="bottom",
                fontsize=7, color="#9CA3AF", style="italic")

ax.set_xticks(x)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
ax.set_ylabel("Flip rate (%)")
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.yaxis.grid(True, alpha=0.3, linestyle=":")
ax.xaxis.grid(False)

# Legend: color = certain/uncertain, hatch = dataset
leg_handles = [
    Patch(facecolor=CERTAIN_COLOR,   edgecolor="#555", label="Certain"),
    Patch(facecolor=UNCERTAIN_COLOR, edgecolor="#555", label="Uncertain"),
    Patch(facecolor="#ccc", hatch="",   edgecolor="#555", label="MMLU-Pro"),
    Patch(facecolor="#ccc", hatch="//", edgecolor="#555", label="GPQA-Diamond"),
]
ax.legend(handles=leg_handles, loc="upper left", framealpha=0.9, fontsize=8.5)
ax.set_title("RQ1: Flip Rate — Certain vs Uncertain (95% Wilson CI)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/rq1_combined_datasets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved rq1_combined_datasets.png")

# ═══════════════════════════════════════════════════════════════════════════
# RQ2 — single forest plot, filled=MMLU open=GPQA
# ═══════════════════════════════════════════════════════════════════════════
print("Plotting RQ2 combined_datasets...")

def build_rq2_df(data):
    rows = []
    for model, d in data.items():
        rows.append(dict(model=model, b_P=d[0], p_P=d[1], lo_P=d[2], hi_P=d[3],
                         b_E=d[4], p_E=d[5], lo_E=d[6], hi_E=d[7],
                         b_I=d[8], p_I=d[9], lo_I=d[10], hi_I=d[11]))
    return pd.DataFrame(rows)

rq2_mmlu = build_rq2_df(MMLU_RQ2)
rq2_gpqa = build_rq2_df(gpqa_rq2)

coef_cfg = [
    ("b_P", "lo_P", "hi_P", "p_P", r"Pressure ($\hat{\beta}_P$)"),
    ("b_E", "lo_E", "hi_E", "p_E", r"Entropy ($\hat{\beta}_E$)"),
    ("b_I", "lo_I", "hi_I", "p_I", r"Interaction ($\hat{\beta}_{PE}$)"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
y_pos = np.arange(len(MODEL_ORDER))
step = 0.18  # vertical offset between MMLU and GPQA for same model

for ax, (bc, lc, hc, pc, coef_label) in zip(axes, coef_cfg):
    for i, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        # MMLU-Pro: filled circle
        row_m = rq2_mmlu[rq2_mmlu["model"] == model]
        if not row_m.empty:
            bv = row_m[bc].values[0]; sig = row_m[pc].values[0] < 0.05
            lo = abs(bv - row_m[lc].values[0]); hi = abs(row_m[hc].values[0] - bv)
            ax.errorbar(bv, y_pos[i] + step/2, xerr=[[lo],[hi]],
                        fmt="o", color=color, alpha=0.9 if sig else 0.3,
                        capsize=3, markersize=6, linewidth=1.3,
                        label=MNAME[model] if ax is axes[0] else "")
        # GPQA-Diamond: open square
        row_g = rq2_gpqa[rq2_gpqa["model"] == model] if not rq2_gpqa.empty else pd.DataFrame()
        if not row_g.empty:
            bv = row_g[bc].values[0]; sig = row_g[pc].values[0] < 0.05
            lo = abs(bv - row_g[lc].values[0]); hi = abs(row_g[hc].values[0] - bv)
            ax.errorbar(bv, y_pos[i] - step/2, xerr=[[lo],[hi]],
                        fmt="s", color=color, alpha=0.9 if sig else 0.3,
                        capsize=3, markersize=6, linewidth=1.3, markerfacecolor="white",
                        markeredgewidth=1.5)
    ax.axvline(0, color="#9CA3AF", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_xlabel("Coefficient value")
    ax.set_title(coef_label, fontsize=10)
    ax.xaxis.grid(True, alpha=0.3, linestyle=":"); ax.yaxis.grid(False)

# Legends: model colors + dataset markers
model_handles = [Line2D([0],[0], color=MODEL_COLORS[m], marker="o", lw=0,
                         markersize=7, label=MNAME[m]) for m in MODEL_ORDER]
ds_handles = [
    Line2D([0],[0], color="#555", marker="o", lw=0, markersize=7, label="MMLU-Pro"),
    Line2D([0],[0], color="#555", marker="s", lw=0, markersize=7, markerfacecolor="white",
           markeredgewidth=1.5, label="GPQA-Diamond"),
]
axes[1].legend(handles=model_handles + ds_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), ncol=4, framealpha=0.9, fontsize=8.5)
fig.suptitle("RQ2: Logistic Regression Coefficients — Both Datasets (faded = p ≥ 0.05)",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/rq2_combined_datasets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved rq2_combined_datasets.png")

# ═══════════════════════════════════════════════════════════════════════════
# RQ3 — single panel, color=model, solid+circle=MMLU, dashed+square=GPQA
# ═══════════════════════════════════════════════════════════════════════════
print("Plotting RQ3 combined_datasets...")

fig, ax = plt.subplots(figsize=(9, 5))

for model in MODEL_ORDER:
    color = MODEL_COLORS[model]
    short = MNAME[model]

    # MMLU-Pro
    if model in MMLU_RQ3 and MMLU_RQ3[model]:
        turns = sorted(MMLU_RQ3[model].keys())
        ys = [MMLU_RQ3[model][t] for t in turns]
        ax.plot(turns, ys, color=color, linestyle="-", marker="o",
                markersize=5, linewidth=1.8, label=f"{short} (MMLU)")
        if model in MMLU_RQ3_CI:
            los = np.array([MMLU_RQ3_CI[model][t][0] for t in turns], dtype=float)
            his = np.array([MMLU_RQ3_CI[model][t][1] for t in turns], dtype=float)
            ax.fill_between(turns, los, his, color=color, alpha=0.10)

    # GPQA-Diamond
    if model in gpqa_rq3 and gpqa_rq3[model]:
        turns_g = sorted(gpqa_rq3[model].keys())
        ys_g = [gpqa_rq3[model][t] for t in turns_g]
        ax.plot(turns_g, ys_g, color=color, linestyle="--", marker="s",
                markersize=5, linewidth=1.8, label=f"{short} (GPQA)",
                markerfacecolor="white", markeredgewidth=1.5)
        if model in gpqa_rq3_ci:
            los_g = np.array([gpqa_rq3_ci[model][t][0] for t in turns_g
                              if t in gpqa_rq3_ci[model]], dtype=float)
            his_g = np.array([gpqa_rq3_ci[model][t][1] for t in turns_g
                              if t in gpqa_rq3_ci[model]], dtype=float)
            t_ci = [t for t in turns_g if t in gpqa_rq3_ci[model]]
            ax.fill_between(t_ci, los_g, his_g, color=color, alpha=0.07)

ax.set_xticks(range(1, 7))
ax.set_xticklabels([f"T{t}" for t in range(1, 7)])
ax.set_xlabel("Pressure dose")
ax.set_ylabel("Flip rate (%)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))
ax.yaxis.grid(True, alpha=0.3, linestyle=":")
ax.xaxis.grid(False)

# Dataset legend
ds_leg = [
    Line2D([0],[0], color="#555", linestyle="-",  marker="o", lw=1.8, markersize=5, label="MMLU-Pro"),
    Line2D([0],[0], color="#555", linestyle="--", marker="s", lw=1.8, markersize=5,
           markerfacecolor="white", markeredgewidth=1.3, label="GPQA-Diamond"),
]
model_leg = [Line2D([0],[0], color=MODEL_COLORS[m], lw=2.5, label=MNAME[m]) for m in MODEL_ORDER]
legend1 = ax.legend(handles=ds_leg, loc="upper right", fontsize=8.5, framealpha=0.9, title="Dataset")
ax.add_artist(legend1)
ax.legend(handles=model_leg, loc="upper left", fontsize=8.5, framealpha=0.9, title="Model")

ax.set_title("RQ3: Flip Rate Across Pressure Doses T1–T6 (95% CI shaded)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/rq3_combined_datasets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved rq3_combined_datasets.png")

# ═══════════════════════════════════════════════════════════════════════════
# RQ4 — 3 metric subplots, color=model, solid+circle=MMLU, dashed+square=GPQA
# ═══════════════════════════════════════════════════════════════════════════
print("Plotting RQ4 combined_datasets...")

metric_cfg = [
    ("mc",   "Majority Correct Rate"),
    ("conf", "Calibrated Confidence"),
    ("gap",  "Overconfidence Gap"),
]
metric_idx = {"mc": 1, "conf": 2, "gap": 3}  # tuple index in rq4 dicts

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (mc, ylabel) in zip(axes, metric_cfg):
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        short = MNAME[model]

        # MMLU-Pro (T0 and T6 only)
        if model in MMLU_RQ4 and MMLU_RQ4[model]:
            turns_m = sorted(MMLU_RQ4[model].keys())
            ys_m = [MMLU_RQ4[model][t][metric_idx[mc]] for t in turns_m]
            ax.plot(turns_m, ys_m, color=color, linestyle="-", marker="o",
                    markersize=5, linewidth=1.8, label=f"{short} (MMLU)")

        # GPQA-Diamond (full turn range)
        if model in gpqa_rq4_full and gpqa_rq4_full[model]:
            turns_g = sorted(gpqa_rq4_full[model].keys())
            ys_g = [gpqa_rq4_full[model][t][metric_idx[mc]] for t in turns_g]
            ax.plot(turns_g, ys_g, color=color, linestyle="--", marker="s",
                    markersize=5, linewidth=1.8, label=f"{short} (GPQA)",
                    markerfacecolor="white", markeredgewidth=1.5)

    all_turns = sorted({t for m in MMLU_RQ4.values() for t in m}
                       | {t for m in gpqa_rq4_full.values() for t in m})
    ax.set_xticks(all_turns)
    ax.set_xticklabels([f"T{t}" for t in all_turns], fontsize=8)
    ax.set_xlabel("Pressure turn")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, linestyle=":")
    ax.xaxis.grid(False)

# Single shared legend
ds_leg = [
    Line2D([0],[0], color="#555", linestyle="-",  marker="o", lw=1.8, markersize=5, label="MMLU-Pro"),
    Line2D([0],[0], color="#555", linestyle="--", marker="s", lw=1.8, markersize=5,
           markerfacecolor="white", markeredgewidth=1.3, label="GPQA-Diamond"),
]
model_leg = [Line2D([0],[0], color=MODEL_COLORS[m], lw=2.5, label=MNAME[m]) for m in MODEL_ORDER]
fig.legend(handles=model_leg + ds_leg, loc="lower center",
           bbox_to_anchor=(0.5, -0.08), ncol=7, framealpha=0.9, fontsize=8.5)
fig.suptitle("RQ4: CoT Reasoning Trajectory Under Pressure — Both Datasets", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(f"{PLOT_DIR}/rq4_combined_datasets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved rq4_combined_datasets.png")

print("\nDone.")
