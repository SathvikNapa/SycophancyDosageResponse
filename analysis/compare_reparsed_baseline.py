"""
compare_reparsed_baseline.py — RQ1-RQ3 (MMLU-Pro, GPQA-Diamond) and RQ4
(Claude Sonnet, GPQA-Diamond) computed twice with identical code:

  old : baseline entropy from base_experiment_metadata.pkl
  new : baseline entropy from base_experiment_metadata_reparsed.pkl
        (scripts/reparse_baseline_answers.py)

Row construction mirrors run_combined_datasets_analysis.py's GPQA path
(empty/failed generations produce no row; Claude GPQA gaps backfilled from
reasoning_calibrated_bin), applied to both datasets. Each RQ is reported for
all dialogues ("all") and for dialogues correct at T0 ("t0c"), since the
methods section restricts RQ1-RQ3 to the correct-at-T0 subset.

RQ4 averages each question's final-step values per turn (the per-question
mean), comparing Sonnet GPQA before the gap fill (earliest backups) with the
repaired files.

Writes analysis/compare_reparsed_baseline.csv and prints Markdown tables.
"""

import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import logit

import sycophancy.reasoning_uncertainty as ru
sys.modules["reasoning_uncertainty"] = ru  # pickle compatibility shim
from sycophancy.build_analysis_dfs import COT_FLIP_CELLS, backfill_flat_from_reasoning, reconstruct_cot_wrong_letters

warnings.filterwarnings("ignore")
EXP = "experiment_out"
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]
DATASETS = {"mmlu_pro": "", "gpqa_diamond": "gpqa_diamond"}
META = {"old": "base_experiment_metadata.pkl", "new": "base_experiment_metadata_reparsed.pkl"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p * 100, max(0, (c - m) * 100), min(100, (c + m) * 100)


def entropy_map(model, sub, which):
    path = os.path.join(EXP, model, sub, META[which])
    with open(path, "rb") as f:
        return {m["query"]: m["entropy"] for m in pickle.load(f)}


def load_rows(model, dataset, emap):
    sub = DATASETS[dataset]
    if (model, dataset) in COT_FLIP_CELLS:
        bf = backfill_flat_from_reasoning(model, dataset, set(emap),
                                          wrong_letters=reconstruct_cot_wrong_letters(model, dataset))
        df = bf[["query", "run_idx", "turn", "flipped", "correct"]].rename(columns={"run_idx": "run"})
        return _finish(df, emap)
    rows = []
    for pkl in sorted(glob.glob(os.path.join(EXP, model, sub, "entropy_bin", "bin_*_repeated.pkl"))):
        with open(pkl, "rb") as f:
            questions = pickle.load(f)
        for q in questions:
            for run_idx, run in enumerate(q.get("raw_runs", [])):
                iw = run.get("is_wrong", [])
                raw = run.get("raw_turns", [None] * len(iw))
                ic = run.get("is_correct", [0] * len(iw))
                for turn, wrong in enumerate(iw):
                    if turn >= len(raw) or not raw[turn]:
                        continue  # empty/failed generation — no observation
                    rows.append({"query": q["query"], "run": run_idx, "turn": turn,
                                 "flipped": int(wrong), "correct": int(ic[turn])})
    df = pd.DataFrame(rows)
    if dataset == "gpqa_diamond":
        have = set(df["query"]) if len(df) else set()
        gap = set(emap) - have
        if gap:
            bf = backfill_flat_from_reasoning(model, dataset, gap)
            if not bf.empty:
                bf = bf[["query", "run_idx", "turn", "flipped", "correct"]].rename(columns={"run_idx": "run"})
                bf["run"] = bf["run"] + 1000  # keep run ids distinct from entropy_bin runs
                df = pd.concat([df, bf], ignore_index=True)
    return _finish(df, emap)


def _finish(df, emap):
    df["entropy"] = df["query"].map(emap)
    df = df[df["entropy"].notna()].copy()
    df["certain"] = (df["entropy"] == 0.0).astype(int)
    t0 = df[df["turn"] == 0].set_index(["query", "run"])["correct"]
    df["t0_correct"] = df.set_index(["query", "run"]).index.map(lambda k: t0.get(k, 0) == 1)
    return df


def rq_stats(df):
    pr = df[df["turn"] >= 1]
    out = {"n_q": pr["query"].nunique(), "n_rows": len(pr),
           "n_certain_q": pr.loc[pr["certain"] == 1, "query"].nunique()}
    cert, unc = pr[pr["certain"] == 1], pr[pr["certain"] == 0]
    out["cert"] = wilson(int(cert["flipped"].sum()), len(cert))[0]
    out["unc"] = wilson(int(unc["flipped"].sum()), len(unc))[0]
    out["delta"] = out["unc"] - out["cert"]
    out["p_mwu"] = (stats.mannwhitneyu(unc["flipped"], cert["flipped"], alternative="greater")[1]
                    if len(cert) and len(unc) else np.nan)
    m = pr.copy()
    m["Hc"] = m["entropy"] - m["entropy"].mean()
    for k in ("bP", "pP", "bH", "pH", "bPH", "pPH"):
        out[k] = np.nan
    try:
        fit = logit("flipped ~ turn + Hc + turn:Hc", data=m).fit(disp=0, maxiter=200)
        out.update(bP=fit.params["turn"], pP=fit.pvalues["turn"], bH=fit.params["Hc"], pH=fit.pvalues["Hc"],
                   bPH=fit.params["turn:Hc"], pPH=fit.pvalues["turn:Hc"])
    except Exception:
        pass
    per_turn = pr.groupby("turn")["flipped"].mean() * 100
    for t in range(1, 7):
        out[f"T{t}"] = per_turn.get(t, np.nan)
    out["slope"] = (np.polyfit(per_turn.index, per_turn.values, 1)[0] if len(per_turn) >= 2 else np.nan)
    return out


def rq4_sonnet_gpqa():
    """Per-question final-step means per turn: before fill (earliest backups) vs repaired."""
    base = os.path.join(EXP, "ClaudeSonnet", "gpqa_diamond", "reasoning_calibrated_bin")
    before = {0: "_backup_before_gpqa_validated_fill_20261005_025242",
              1: "_backup_before_gpqa_validated_fill_20261005_030231"}
    res = {}
    for label, paths in {
        "before": [os.path.join(base, before[b], f"bin_{b}_cross_turn.pkl") for b in (0, 1)],
        "repaired": [os.path.join(base, f"bin_{b}_cross_turn.pkl") for b in (0, 1)],
    }.items():
        rows = []
        for p in paths:
            with open(p, "rb") as f:
                rows.extend(pickle.load(f))
        d = pd.DataFrame(rows)
        d["conf"] = pd.to_numeric(d["mean_self_reported_confidence"], errors="coerce") / 100
        d["corr"] = pd.to_numeric(d["majority_is_correct"], errors="coerce")
        d["bent"] = pd.to_numeric(d["belief_entropy"], errors="coerce")
        fin = d.sort_values("step").groupby(["query", "turn"]).last().reset_index()
        g = fin.groupby("turn").agg(corr=("corr", "mean"), conf=("conf", "mean"),
                                    bent=("bent", "mean"), n_q=("query", "nunique"),
                                    conf_present=("conf", lambda s: s.notna().sum()))
        res[label] = g
    return res


def main():
    records = []
    for which in ("old", "new"):
        for dataset, sub in DATASETS.items():
            for model in MODELS:
                if not os.path.exists(os.path.join(EXP, model, sub, META[which])):
                    print(f"missing {which} metadata for {model}/{dataset}", file=sys.stderr)
                    continue
                df = load_rows(model, dataset, entropy_map(model, sub, which))
                for subset, part in (("all", df), ("t0c", df[df["t0_correct"]])):
                    r = rq_stats(part)
                    r.update(version=which, dataset=dataset, model=model, subset=subset)
                    records.append(r)
    res = pd.DataFrame(records)
    res.to_csv("analysis/compare_reparsed_baseline.csv", index=False)

    def side(col, fmt="{:.1f}"):
        piv = res.pivot_table(index=["subset", "dataset", "model"], columns="version", values=col, aggfunc="first")
        return piv.apply(lambda r: f"{fmt.format(r.get('old', np.nan))} → {fmt.format(r.get('new', np.nan))}", axis=1)

    for subset in ("all", "t0c"):
        print(f"\n### RQ1–RQ3 ({'all dialogues' if subset == 'all' else 'correct-at-T0 dialogues'}) — old → new\n")
        tab = pd.DataFrame({
            "certain Qs": side("n_certain_q", "{:.0f}"),
            "certain %": side("cert"), "uncertain %": side("unc"), "Δ pp": side("delta"),
            "p(MWU)": side("p_mwu", "{:.3g}"),
            "β_H": side("bH", "{:.3f}"), "β_PH": side("bPH", "{:.3f}"), "p(β_PH)": side("pPH", "{:.3g}"),
            "β_P": side("bP", "{:.3f}"), "RQ3 slope pp/turn": side("slope", "{:+.2f}"),
        }).loc[subset]
        print(tab.to_string())

    print("\n### RQ4 — Claude Sonnet, GPQA-Diamond (per-question final-step means)\n")
    r4 = rq4_sonnet_gpqa()
    for label, g in r4.items():
        t0, t6 = g.loc[0], g.loc[6]
        print(f"{label:9}: maj-correct {t0['corr']:.3f} → {t6['corr']:.3f} ({(t6['corr'] - t0['corr']) * 100:+.1f} pp); "
              f"conf {t0['conf']:.3f} → {t6['conf']:.3f} ({(t6['conf'] - t0['conf']) * 100:+.1f} pp); "
              f"gap T6 {t6['conf'] - t6['corr']:.3f}; questions T0/T6 {int(t0['n_q'])}/{int(t6['n_q'])}; "
              f"conf present T6 {int(t6['conf_present'])}")
    pd.concat(r4, names=["version"]).to_csv("analysis/compare_reparsed_rq4_sonnet_gpqa.csv")


if __name__ == "__main__":
    main()
