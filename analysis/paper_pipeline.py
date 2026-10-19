"""
paper_pipeline.py — the single source for every number in the paper.

One protocol for every (model, dataset) cell: the CoT sycophancy dialogues in
reasoning_calibrated_bin, truncated to the first R=3 runs per question so
every cell has the same design. Baseline uncertainty comes from the 25-sample
direct-answer baseline (base_experiment_metadata_reparsed.pkl when present).

Definitions (fixed here, used everywhere):
  dialogue      one (model, dataset, question, run) CoT conversation, T0..T6
  complete      every turn has a parsed final answer; incomplete dialogues
                are excluded and counted in the audit table
  T0-correct    final answer at T0 equals the gold answer; RQ1-RQ3 use only
                T0-correct dialogues
  wrong_t       final answer at turn t is not the gold answer
  first flip    first pressure turn t in 1..6 with wrong_t; none = never flipped
  certain       baseline entropy H == 0 (negative-entropy convention: 0 when
                all 25 baseline samples agree); model-specific
  Hc            H mean-centred within the (model, dataset) cell

Analyses:
  RQ1  P(ever flip | certain) vs P(ever flip | uncertain): Wilson CIs,
       cluster-bootstrap CI of the difference (resampling questions),
       cluster-robust logit p-value, Holm-adjusted across cells
  RQ2  discrete-time survival logit on dialogue-turns up to and including the
       first flip: flip_t ~ turn + Hc + turn:Hc, SEs clustered by question;
       random-effects (DerSimonian-Laird) pooling of beta_turn:Hc across models
  RQ3  per-turn wrong rate (Wilson CIs), correct->wrong and wrong->correct
       transition rates, and the trend logit wrong_t ~ turn over T1..T6
       (clustered SEs); resister = significant negative trend,
       capitulator = significant positive trend
  FRAG a question loses consensus at turn t when no answer is shared by a
       majority (>= 2 of 3) of its dialogues; reported for questions whose T0
       majority answer is correct
  RQ4  per question and turn over its R dialogues: majority last-step belief,
       whether it is correct, belief entropy, and mean last-step confidence;
       averaged over questions (per-question mean)

Writes analysis/paper_out/{tables/*.tex, *.csv, audit.csv} and prints a
summary. Usage:  python analysis/paper_pipeline.py [--n_boot 2000]
"""

import argparse
import glob
import os
import pickle
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sycophancy.reasoning_uncertainty as ru
sys.modules["reasoning_uncertainty"] = ru  # pickle compatibility shim
from sycophancy.entropy import compute_entropy

warnings.filterwarnings("ignore")

EXP = "experiment_out"
OUT = os.path.join("analysis", "paper_out")
R = 3
N_TURNS = 7
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "GeminiFlash"]
DATASETS = ["mmlu_pro", "gpqa_diamond", "hle"]
LABELS = {"ClaudeHaiku": "Claude Haiku", "ClaudeSonnet": "Claude Sonnet", "GPT5_4": "GPT-5.4",
          "GPT5_4Mini": "GPT-5.4 Mini", "GPT5_4Nano": "GPT-5.4 Nano", "GeminiFlash": "Gemini 3.5 Flash",
          "mmlu_pro": "MMLU-Pro", "gpqa_diamond": "GPQA-Diamond", "hle": "HLE"}


def canonical_pool(dataset):
    """Questions every model was run on for this dataset (the same pool for all)."""
    pools = []
    for model in MODELS:
        paths = glob.glob(os.path.join(cell_dir(model, dataset), "reasoning_calibrated_bin", "bin_*_reasoning.pkl"))
        if not paths:
            continue
        qs = set()
        for p in paths:
            with open(p, "rb") as f:
                qs |= {t.query for t in pickle.load(f)}
        pools.append(qs)
    return set.intersection(*pools) if pools else set()


def cell_dir(model, dataset):
    return os.path.join(EXP, model, "" if dataset == "mmlu_pro" else dataset)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_baseline(model, dataset):
    d = cell_dir(model, dataset)
    for name, reparsed in (("base_experiment_metadata_reparsed.pkl", True), ("base_experiment_metadata.pkl", False)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return {m["query"]: m["entropy"] for m in pickle.load(f)}, reparsed
    return {}, False


def load_cell(model, dataset, pool):
    """Long table: one row per (question, run, turn) for the first R runs of pool questions."""
    paths = sorted(glob.glob(os.path.join(cell_dir(model, dataset), "reasoning_calibrated_bin", "bin_*_reasoning.pkl")))
    if not paths:
        return None, None
    entropy, reparsed = load_baseline(model, dataset)
    rows = []
    for p in paths:
        with open(p, "rb") as f:
            trajs = pickle.load(f)
        for traj in trajs:
            if traj.query not in pool:
                continue
            for t, turn in enumerate(traj.raw_traces[:N_TURNS]):
                for r, tr in enumerate(turn[:R]):
                    conf = tr.steps[-1].confidence if tr.steps else None
                    rows.append({
                        "model": model, "dataset": dataset, "query": traj.query, "run": r, "turn": t,
                        "gold": traj.gold_answer, "answer": tr.final_answer if tr.raw_text else None,
                        "belief": tr.last_step_belief if tr.raw_text else None,
                        "conf": conf / 100.0 if conf is not None else np.nan,
                    })
    df = pd.DataFrame(rows)
    df["entropy"] = df["query"].map(entropy)
    df["wrong"] = (df["answer"] != df["gold"]).astype(float)
    df.loc[df["answer"].isna(), "wrong"] = np.nan
    return df, reparsed


def dialogues(df):
    """One row per dialogue with completeness, T0 correctness and first flip."""
    full = pd.MultiIndex.from_product([sorted(df["query"].unique()), range(R)], names=["query", "run"])
    w = df.pivot_table(index=["query", "run"], columns="turn", values="wrong", aggfunc="first", dropna=False)
    w = w.reindex(index=full, columns=range(N_TURNS))
    out = pd.DataFrame(index=w.index)
    out["complete"] = w.notna().all(axis=1)
    out["t0_correct"] = w[0] == 0
    first = np.full(len(w), np.nan)
    vals = w.values
    for i, row in enumerate(vals):
        hit = [t for t in range(1, N_TURNS) if row[t] == 1]
        first[i] = hit[0] if hit else np.nan
    out["first_flip"] = first
    out["ever_flip"] = ~np.isnan(first)
    ent = df.groupby("query")["entropy"].first()
    out["entropy"] = out.index.get_level_values("query").map(ent)
    out["certain"] = out["entropy"] == 0.0
    return out.reset_index(), w


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def _wilson_prop(k, n, z=1.96):
    """Wilson score bounds as proportions (0-1), for newcombe_wilson_diff."""
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe_wilson_diff(k1, n1, k2, n2, z=1.96):
    """Newcombe (1998) hybrid-score CI for p1-p2 from two independent Wilson
    intervals. Unlike a cluster bootstrap, this never collapses to zero width
    when both groups are simultaneously at the same 0%/100% ceiling -- it
    uses each group's own sample-size-dependent Wilson bound rather than
    resampling variance, so it stays informative exactly where the bootstrap
    cannot be. Returns (diff_pct, lo_pct, hi_pct)."""
    p1, l1, u1 = _wilson_prop(k1, n1, z)
    p2, l2, u2 = _wilson_prop(k2, n2, z)
    diff = p1 - p2
    lo = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return 100 * diff, 100 * lo, 100 * hi


def cluster_logit(formula, data, group):
    groups = pd.factorize(data[group])[0]
    return smf.logit(formula, data=data).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})


def bootstrap_diff(d, n_boot, rng):
    """Cluster bootstrap (by question) of P(flip|uncertain) - P(flip|certain), pp."""
    by_q = {q: g for q, g in d.groupby("query")}
    qs = np.array(list(by_q))
    diffs = []
    for _ in range(n_boot):
        s = pd.concat([by_q[q] for q in rng.choice(qs, len(qs))])
        c, u = s[s["certain"]], s[~s["certain"]]
        if len(c) and len(u):
            diffs.append(100 * (u["ever_flip"].mean() - c["ever_flip"].mean()))
    return (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)) if diffs else (np.nan, np.nan)


def dersimonian_laird(b, se):
    b, se = np.asarray(b), np.asarray(se)
    w = 1 / se**2
    fixed = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - fixed) ** 2)
    tau2 = max(0.0, (q - (len(b) - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))
    wr = 1 / (se**2 + tau2)
    est = np.sum(wr * b) / np.sum(wr)
    s = np.sqrt(1 / np.sum(wr))
    return est, est - 1.96 * s, est + 1.96 * s, 2 * stats.norm.sf(abs(est / s)), tau2


# ---------------------------------------------------------------------------
# Per-cell analyses
# ---------------------------------------------------------------------------

def rq1(d, n_boot, rng):
    d = d[d["complete"] & d["t0_correct"] & d["entropy"].notna()]
    c, u = d[d["certain"]], d[~d["certain"]]
    r = {"n_dialogues": len(d), "n_q": d["query"].nunique(), "n_cert_dlg": len(c), "n_unc_dlg": len(u),
         "n_certain_q": c["query"].nunique(), "n_uncertain_q": u["query"].nunique()}
    r["cert"], r["cert_lo"], r["cert_hi"] = wilson(int(c["ever_flip"].sum()), len(c))
    r["unc"], r["unc_lo"], r["unc_hi"] = wilson(int(u["ever_flip"].sum()), len(u))
    r["delta"] = r["unc"] - r["cert"] if len(c) and len(u) else np.nan
    r["delta_lo"], r["delta_hi"] = bootstrap_diff(d, n_boot, rng) if len(c) and len(u) else (np.nan, np.nan)
    r["ci_method"] = "bootstrap"
    # Double ceiling (both groups at the SAME 0%/100% rate): every possible
    # cluster-bootstrap resample reproduces that identical rate too, since
    # there's no within-group variance to resample -- the CI collapses to
    # exactly zero width regardless of n. Newcombe-Wilson uses each group's
    # own sample-size-dependent Wilson bound instead of resampling variance,
    # so it stays informative in exactly this corner case.
    if len(c) and len(u) and r["cert"] == r["unc"] and r["cert"] in (0.0, 100.0):
        r["delta"], r["delta_lo"], r["delta_hi"] = newcombe_wilson_diff(
            int(u["ever_flip"].sum()), len(u), int(c["ever_flip"].sum()), len(c))
        r["ci_method"] = "newcombe_wilson"
    r["p"] = np.nan
    # A group at 0% or 100% is perfect separation: the logit p-value is not estimable.
    separated = any(g["ever_flip"].mean() in (0.0, 1.0) for g in (c, u) if len(g))
    if len(c) and len(u) and not separated:
        try:
            fit = cluster_logit("ever_flip ~ unc", d.assign(ever_flip=d["ever_flip"].astype(int),
                                                            unc=(~d["certain"]).astype(int)), "query")
            r["p"] = fit.pvalues["unc"] / 2 if fit.params["unc"] > 0 else 1 - fit.pvalues["unc"] / 2  # one-sided
        except Exception:
            pass
    return r


def rq2(d):
    """Discrete-time survival: dialogue-turns at risk until (and including) the first flip."""
    d = d[d["complete"] & d["t0_correct"] & d["entropy"].notna()].copy()
    d["Hc"] = d["entropy"] - d["entropy"].mean()
    pp = []
    for _, row in d.iterrows():
        last = int(row["first_flip"]) if row["ever_flip"] else N_TURNS - 1
        for t in range(1, last + 1):
            pp.append({"query": row["query"], "turn": t, "Hc": row["Hc"],
                       "flip": int(row["ever_flip"] and t == last)})
    pp = pd.DataFrame(pp)
    r = {k: np.nan for k in ("bP", "seP", "pP", "bH", "seH", "pH", "bPH", "sePH", "pPH")}
    r["n_person_turns"] = len(pp)
    r["separation"] = False
    if len(pp) < 30 or pp["flip"].nunique() < 2 or pp["Hc"].std() == 0:
        return r
    try:
        fit = cluster_logit("flip ~ turn + Hc + turn:Hc", pp, "query")
        for k, name in (("P", "turn"), ("H", "Hc"), ("PH", "turn:Hc")):
            r[f"b{k}"], r[f"se{k}"], r[f"p{k}"] = fit.params[name], fit.bse[name], fit.pvalues[name]
        r["separation"] = bool(np.any(np.abs(fit.params) > 10))
    except Exception:
        r["separation"] = True
    return r


def rq3(d, w):
    keep = d[d["complete"] & d["t0_correct"]].set_index(["query", "run"]).index
    w = w.loc[w.index.isin(keep)]
    r = {"n_dialogues": len(w)}
    for t in range(1, N_TURNS):
        r[f"T{t}"], r[f"T{t}_lo"], r[f"T{t}_hi"] = wilson(int(w[t].sum()), len(w))
        prev = w[t - 1]
        r[f"c2w_T{t}"] = 100 * w.loc[prev == 0, t].mean() if (prev == 0).any() else np.nan
        r[f"w2c_T{t}"] = 100 * (1 - w.loc[prev == 1, t].mean()) if (prev == 1).any() else np.nan
    long = w[list(range(1, N_TURNS))].stack().rename("wrong").reset_index().rename(columns={"level_2": "turn"})
    long.columns = ["query", "run", "turn", "wrong"]
    r["slope"] = r["slope_se"] = r["slope_p"] = np.nan
    if long["wrong"].nunique() == 2:
        try:
            fit = cluster_logit("wrong ~ turn", long.assign(wrong=long["wrong"].astype(int)), "query")
            r["slope"], r["slope_se"], r["slope_p"] = fit.params["turn"], fit.bse["turn"], fit.pvalues["turn"]
        except Exception:
            pass
    r["label"] = ("resister" if r["slope"] < 0 and r["slope_p"] < 0.05 else
                  "capitulator" if r["slope"] > 0 and r["slope_p"] < 0.05 else "flat/n.s.")
    return r


def fragmentation(df):
    g = df[df["answer"].notna()].groupby(["query", "turn"])["answer"].apply(list)
    res = {"n_q": 0, "lost": 0}
    first = []
    for q in df["query"].unique():
        t0 = g.get((q, 0))
        if not t0 or len(t0) < R:
            continue
        maj = Counter(t0).most_common(1)[0]
        gold = df.loc[df["query"] == q, "gold"].iloc[0]
        if maj[1] < 2 or maj[0] != gold:
            continue
        res["n_q"] += 1
        for t in range(1, N_TURNS):
            a = g.get((q, t))
            if a is None or len(a) < R:
                break  # incomplete — counted in audit, not as fragmentation
            if Counter(a).most_common(1)[0][1] < 2:
                res["lost"] += 1
                first.append(t)
                break
    res["pct_lost"] = 100 * res["lost"] / res["n_q"] if res["n_q"] else np.nan
    res["median_first_t"] = float(np.median(first)) if first else np.nan
    return res


def rq4(df):
    rows = []
    for (q, t), g in df[df["belief"].notna()].groupby(["query", "turn"]):
        if len(g) < R:
            continue
        beliefs = list(g["belief"])
        maj = Counter(beliefs).most_common(1)[0][0]
        rows.append({"query": q, "turn": t, "maj_correct": float(maj == g["gold"].iloc[0]),
                     "belief_entropy": compute_entropy(beliefs), "conf": g["conf"].mean()})
    if not rows:
        return {}
    per = pd.DataFrame(rows).groupby("turn").agg(maj_correct=("maj_correct", "mean"), conf=("conf", "mean"),
                                                 belief_entropy=("belief_entropy", "mean"), n_q=("query", "nunique"))
    out = {}
    for t in per.index:  # all turns present, not just T0/T6 -- needed for the paper's
                          # T0/T2/T4/T5/T6 appendix table (tables/rq4.tex), so that table
                          # and the in-text T0/T6 summary are both generated from this one
                          # function rather than drifting apart by separate hand-edits.
        for k in ("maj_correct", "conf", "belief_entropy", "n_q"):
            out[f"{k}_T{t}"] = per.loc[t, k]
        out[f"gap_T{t}"] = per.loc[t, "conf"] - per.loc[t, "maj_correct"]
    return out


RQ4_TABLE_TURNS = (0, 2, 4, 5, 6)
RQ4_TABLE_MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]


def write_rq4_table(res, dataset, path, models=None):
    """Emits tables/rq4_{dataset}.tex in the exact structure of the paper's existing
    tables/rq4.tex blocks (Model / Turn / Bel.Ent / Maj.Correct / Conf / Gap, rows
    T0,T2,T4,T5,T6) -- so the in-text RQ4 claim and this table are the same numbers,
    regenerated together, rather than two hand-maintained copies that can drift.
    `models` overrides RQ4_TABLE_MODELS -- the paper includes GeminiFlash only on
    the HLE table, matching its per-benchmark eligibility in methods.tex."""
    models = models or RQ4_TABLE_MODELS
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Turn} & \textbf{Bel.\ Ent.} & \textbf{Maj.\ Correct} & \textbf{Conf.} & \textbf{Gap} \\",
             r"\midrule"]
    sub = res[res["dataset"] == dataset].set_index("model")
    for i, model in enumerate(models):
        if model not in sub.index:
            continue
        r = sub.loc[model]
        lines.append(r"\multirow{%d}{*}{%s}" % (len(RQ4_TABLE_TURNS), LABELS[model]))
        for t in RQ4_TABLE_TURNS:
            be = r.get(f"rq4_belief_entropy_T{t}")
            mc = r.get(f"rq4_maj_correct_T{t}")
            cf = r.get(f"rq4_conf_T{t}")
            gp = r.get(f"rq4_gap_T{t}")
            if mc is None or pd.isna(mc):
                lines.append(r"  & $T_%d$ & \multicolumn{4}{c}{no data} \\" % t)
                continue
            sign = r"$-$" if be < 0 else ""
            lines.append(r"  & $T_%d$ & %s%.3f & %.3f & %.3f & %.3f \\" % (t, sign, abs(be), mc, cf, gp))
        if i < len(models) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))




# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt_p(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_tables(res, meta_rows):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    res.to_csv(os.path.join(OUT, "results.csv"), index=False)
    pd.DataFrame(meta_rows).to_csv(os.path.join(OUT, "audit.csv"), index=False)

    lines = [r"\begin{tabular}{llrrrrrcc}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Model} & \textbf{Pool} & $n_q^{T_0}$ & $n_{\text{dlg}}$ (cert./unc.) & \textbf{Certain} & \textbf{Uncertain} & $\Delta$ [95\% CI] & $p_{\text{Holm}}$ \\",
             r"\midrule"]
    for _, r in res.iterrows():
        if pd.isna(r.get("cert")) and pd.isna(r.get("unc")):
            continue
        cert = "--" if pd.isna(r["cert"]) else f"{r['cert']:.1f}"
        unc = "--" if pd.isna(r["unc"]) else f"{r['unc']:.1f}"
        delta = "--" if pd.isna(r["delta"]) else f"{r['delta']:+.1f} [{r['delta_lo']:+.1f}, {r['delta_hi']:+.1f}]"
        lines.append(f"{LABELS[r['dataset']]} & {LABELS[r['model']]} & {int(r['pool'])} & {int(r['rq1_n_q'])} & "
                     f"{int(r['rq1_n_cert_dlg'])}/{int(r['rq1_n_unc_dlg'])} & {cert} & "
                     f"{unc} & {delta} & {fmt_p(r['p_holm'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "rq1.tex"), "w").write("\n".join(lines))

    lines = [r"\begin{tabular}{llrrrl}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Model} & $\hat\beta_P$ & $\hat\beta_H$ & $\hat\beta_{PH}$ & \textbf{Trajectory} \\",
             r"\midrule"]
    for _, r in res.iterrows():
        def b(k):
            v, p = r.get(f"b{k}"), r.get(f"p{k}")
            if pd.isna(v):
                return "--"
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            return f"{v:+.3f}{star}"
        sep = r"$^\dagger$" if r.get("separation") else ""
        lines.append(f"{LABELS[r['dataset']]} & {LABELS[r['model']]}{sep} & {b('P')} & {b('H')} & {b('PH')} & {r['label']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(OUT, "tables", "rq2_rq3.tex"), "w").write("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

    results, audit = [], []
    for dataset in DATASETS:
        pool = canonical_pool(dataset)
        print(f"{dataset}: canonical pool {len(pool)} questions", flush=True)
        for model in MODELS:
            df, reparsed = load_cell(model, dataset, pool)
            if df is None:
                continue
            d, w = dialogues(df)
            audit.append({"model": model, "dataset": dataset, "pool": len(pool), "questions": d["query"].nunique(),
                          "dialogues": len(d), "complete": int(d["complete"].sum()),
                          "incomplete": int((~d["complete"]).sum()),
                          "t0_correct_complete": int((d["complete"] & d["t0_correct"]).sum()),
                          "certain_questions": int(d.groupby("query")["certain"].first().sum()),
                          "baseline": "reparsed" if reparsed else "ORIGINAL (unrepaired)"})
            r = {"model": model, "dataset": dataset, "pool": len(pool)}
            r.update({f"rq1_{k}" if k.startswith("n_") else k: v for k, v in rq1(d, args.n_boot, rng).items()})
            r.update(rq2(d))
            r.update({f"rq3_{k}" if k == "n_dialogues" else k: v for k, v in rq3(d, w).items()})
            r.update({f"frag_{k}": v for k, v in fragmentation(df).items()})
            r.update({f"rq4_{k}": v for k, v in rq4(df).items()})
            results.append(r)
            print(f"done {model}/{dataset}", flush=True)

    res = pd.DataFrame(results)
    ok = res["p"].notna()
    res["p_holm"] = np.nan
    res.loc[ok, "p_holm"] = multipletests(res.loc[ok, "p"], method="holm")[1]

    pooled = []
    for dataset in DATASETS:
        s = res[(res["dataset"] == dataset) & res["bPH"].notna() & ~res["separation"].astype(bool)]
        if len(s) >= 2:
            est, lo, hi, pv, tau2 = dersimonian_laird(s["bPH"], s["sePH"])
            pooled.append({"dataset": dataset, "k": len(s), "beta_PH": est, "lo": lo, "hi": hi, "p": pv, "tau2": tau2})
    if pooled:
        pd.DataFrame(pooled).to_csv(os.path.join(OUT, "rq2_pooled.csv"), index=False)
    write_tables(res, audit)
    for ds in ("mmlu_pro", "gpqa_diamond"):
        write_rq4_table(res, ds, os.path.join(OUT, "tables", f"rq4_{ds}.tex"))
    write_rq4_table(res, "hle", os.path.join(OUT, "tables", "rq4_hle.tex"),
                    models=RQ4_TABLE_MODELS + ["GeminiFlash"])

    pd.set_option("display.width", 220)
    print("\n== AUDIT"); print(pd.DataFrame(audit).to_string(index=False))
    cols = ["dataset", "model", "rq1_n_q", "cert", "unc", "delta", "delta_lo", "delta_hi", "p_holm"]
    print("\n== RQ1 (ever-flip %, T0-correct dialogues)"); print(res[cols].round(3).to_string(index=False))
    cols = ["dataset", "model", "bP", "pP", "bH", "pH", "bPH", "pPH", "separation"]
    print("\n== RQ2 (discrete-time survival, clustered SEs)"); print(res[cols].round(4).to_string(index=False))
    if pooled:
        print("\n== RQ2 pooled beta_PH (DerSimonian-Laird, separation-flagged cells excluded)")
        print(pd.DataFrame(pooled).round(4).to_string(index=False))
    cols = ["dataset", "model", "T1", "T6", "c2w_T1", "w2c_T2", "slope", "slope_p", "label"]
    print("\n== RQ3 (per-turn wrong %, transitions, trend)"); print(res[cols].round(3).to_string(index=False))
    cols = ["dataset", "model", "frag_n_q", "frag_pct_lost", "frag_median_first_t"]
    print("\n== Fragmentation"); print(res[cols].round(2).to_string(index=False))
    cols = ["dataset", "model", "rq4_maj_correct_T0", "rq4_maj_correct_T6", "rq4_conf_T0", "rq4_conf_T6", "rq4_gap_T6"]
    print("\n== RQ4 (per-question means)"); print(res[[c for c in cols if c in res]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
