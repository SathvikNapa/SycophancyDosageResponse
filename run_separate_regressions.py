"""
run_separate_regressions.py
---------------------------
Runs three logistic regressions per model, separately for each predictor:

  Model 1 — Pressure only:      flip ~ turn
  Model 2 — Entropy only:       flip ~ entropy
  Model 3 — Full interaction:   flip ~ turn + entropy + turn:entropy

Prints coefficients, odds ratios, 95% CIs, p-values, and McFadden R²
for every model × dataset combination where data is available.

Converts log-odds coefficients to plain-English probability shifts
(evaluated at the dataset's mean flip rate as baseline).
"""

from __future__ import annotations
import os, pickle, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.special import logit as logit_fn
from statsmodels.formula.api import logit
from collections import Counter

# ── Helpers ────────────────────────────────────────────────────────────────

def _inline_entropy(answers):
    if not answers: return 0.0
    c = Counter(answers); t = len(answers)
    p = np.array([v/t for v in c.values()])
    return float((p * np.log(p)).sum())

def load_entropy_map(path):
    try:
        with open(path, "rb") as f: meta = pickle.load(f)
        out = {}
        for item in meta:
            ent = item.get("entropy") or _inline_entropy(item.get("answers_generated", []))
            out[item["query"]] = {"entropy": ent}
        return out
    except: return {}

def load_sycophancy_df(model, dataset):
    """Loads pressure-turn rows for one model/dataset into a DataFrame."""
    if dataset == "mmlu_pro":
        bin_glob = f"experiment_out/{model}/entropy_bin/bin_*_repeated.pkl"
        emap     = load_entropy_map(f"experiment_out/{model}/base_experiment_metadata.pkl")
    else:
        bin_glob = f"experiment_out/{model}/{dataset}/entropy_bin/bin_*_repeated.pkl"
        emap     = load_entropy_map(f"experiment_out/{model}/{dataset}/base_experiment_metadata.pkl")

    rows = []
    for pkl in sorted(glob.glob(bin_glob)):
        try:
            with open(pkl, "rb") as f: questions = pickle.load(f)
        except: continue
        for q in questions:
            query = q["query"]
            ev = emap.get(query, {})
            for run in q.get("raw_runs", []):
                iw = run.get("is_wrong", [])
                for turn, wrong in enumerate(iw):
                    if turn == 0: continue          # skip baseline turn
                    rows.append({"turn": turn,
                                 "entropy": ev.get("entropy"),
                                 "flipped": int(wrong)})
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df[df["entropy"].notna()].copy()
    return df

def logreg(formula, df):
    """Fit logistic regression, return (params, pvalues, conf_int, r2, n)."""
    try:
        m = logit(formula, data=df).fit(disp=0, maxiter=200)
        r2 = 1 - m.llf / m.llnull
        return m.params, m.pvalues, m.conf_int(), r2, len(df)
    except Exception as e:
        return None, None, None, None, len(df)

def stars(p):
    if p is None or np.isnan(p): return "n/s"
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "n/s"

def prob_shift(beta, base_rate):
    """
    At a baseline flip probability of base_rate,
    how much does prob change for a +1 unit increase in predictor?
    """
    lo = logit_fn(base_rate / 100)
    hi = lo + beta
    return sigmoid(hi) * 100 - base_rate

def fmt_coef(name, b, lo, hi, p, base_rate):
    OR   = np.exp(b)
    dP   = prob_shift(b, base_rate)
    sign = "+" if dP >= 0 else ""
    return (f"      {name:<22} β={b:+.4f}  OR={OR:.3f}  "
            f"95%CI=[{np.exp(lo):.3f},{np.exp(hi):.3f}]  "
            f"p={p:.4f} {stars(p)}  "
            f"→ {sign}{dP:.1f}pp at base {base_rate:.1f}%")

# ── Main ───────────────────────────────────────────────────────────────────

MODEL_ORDER = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]
MNAME = {
    "ClaudeHaiku":  "Claude Haiku",
    "ClaudeSonnet": "Claude Sonnet",
    "GPT5_4":       "GPT-5.4",
    "GPT5_4Mini":   "GPT-5.4 Mini",
    "GPT5_4Nano":   "GPT-5.4 Nano",
}
DATASETS = [("mmlu_pro", "MMLU-Pro"), ("gpqa_diamond", "GPQA-Diamond")]

sep = "═" * 80

print(sep)
print("  SEPARATE LOGISTIC REGRESSIONS PER MODEL")
print("  Model 1: flip ~ turn          (pressure dose only)")
print("  Model 2: flip ~ entropy       (baseline uncertainty only)")
print("  Model 3: flip ~ turn + entropy + turn:entropy  (full interaction model)")
print()
print("  Coefficients are log-odds (β). Also shown:")
print("    OR  = odds ratio = exp(β)")
print("    95%CI = confidence interval on OR")
print("    →   = probability-point shift at that model's mean flip rate")
print(sep)

for dataset_key, dataset_label in DATASETS:
    print(f"\n{'─'*80}")
    print(f"  DATASET: {dataset_label}")
    print(f"{'─'*80}")

    for model in MODEL_ORDER:
        df = load_sycophancy_df(model, dataset_key)
        if df.empty:
            print(f"\n  [{MNAME[model]}]  no data available")
            continue

        n_total   = len(df)
        base_rate = df["flipped"].mean() * 100   # overall flip rate as baseline
        certain_n = (df["entropy"] == 0).sum()
        uncertain_n = (df["entropy"] < 0).sum()

        print(f"\n  ┌─ {MNAME[model]} ({'resister' if model in {'ClaudeHaiku','ClaudeSonnet','GPT5_4'} else 'capitulator'})")
        print(f"  │  n={n_total:,}  mean flip rate={base_rate:.1f}%  "
              f"certain rows={certain_n:,}  uncertain rows={uncertain_n:,}")

        # ── Model 1: pressure only ─────────────────────────────────────────
        params, pvals, ci, r2, _ = logreg("flipped ~ turn", df)
        print(f"  │")
        print(f"  │  MODEL 1 — Pressure (dose) only   McFadden R²={r2:.4f}")
        if params is not None:
            print(fmt_coef("turn (T1→T2→…→T6)", params["turn"],
                           ci.loc["turn",0], ci.loc["turn",1],
                           pvals["turn"], base_rate))
        else:
            print("      [fit failed]")

        # ── Model 2: entropy only ──────────────────────────────────────────
        params, pvals, ci, r2, _ = logreg("flipped ~ entropy", df)
        print(f"  │")
        print(f"  │  MODEL 2 — Entropy only            McFadden R²={r2:.4f}")
        if params is not None:
            print(fmt_coef("entropy (0→more-neg)", params["entropy"],
                           ci.loc["entropy",0], ci.loc["entropy",1],
                           pvals["entropy"], base_rate))
        else:
            print("      [fit failed]")

        # ── Model 3: full interaction ──────────────────────────────────────
        params, pvals, ci, r2, _ = logreg("flipped ~ turn + entropy + turn:entropy", df)
        print(f"  │")
        print(f"  │  MODEL 3 — Full (turn + entropy + turn×entropy)  McFadden R²={r2:.4f}")
        if params is not None:
            print(fmt_coef("turn",           params["turn"],
                           ci.loc["turn",0], ci.loc["turn",1],
                           pvals["turn"], base_rate))
            print(fmt_coef("entropy",        params["entropy"],
                           ci.loc["entropy",0], ci.loc["entropy",1],
                           pvals["entropy"], base_rate))
            print(fmt_coef("turn:entropy",   params["turn:entropy"],
                           ci.loc["turn:entropy",0], ci.loc["turn:entropy",1],
                           pvals["turn:entropy"], base_rate))
        else:
            print("      [fit failed]")

        print(f"  └{'─'*60}")

print(f"\n{sep}")
print("  LEGEND")
print("  β     : log-odds coefficient. +β raises flip probability, −β lowers it.")
print("  OR    : odds ratio = exp(β). OR>1 = more likely to flip, OR<1 = less.")
print("  →     : approximate probability-point (pp) change at that model's")
print("          baseline flip rate, for a +1 unit increase in the predictor.")
print("          (turn: one additional pressure turn; entropy: −1 unit more uncertain)")
print("  R²    : McFadden pseudo-R². ~0 = predictor explains little;")
print("          ~0.2 = moderate fit; ~0.4+ = good fit.")
print("  ***   : p<0.001  ** p<0.01  * p<0.05  n/s p≥0.05")
print(sep)
