"""progress_report.py — One-shot analysis dumping flip rates + CoT examples.

Output format: markdown, written to stdout.
"""
from __future__ import annotations

import os
import sys
import glob
import pickle
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

EXPERIMENT_OUT = "experiment_out"

MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "GeminiFlash"]
DATASETS = ["mmlu_pro", "gpqa_diamond", "hle"]

HLE_MODELS = {"GPT5_4", "GeminiFlash"}


def _skip(model: str, dataset: str) -> bool:
    """Apply user-requested scope filters."""
    if dataset == "hle" and model not in HLE_MODELS:
        return True
    if model == "GeminiFlash" and dataset != "hle":
        return True
    return False


def err(msg):
    print(msg, file=sys.stderr, flush=True)


# ── 1.  Locate every (model, dataset) pressure-run directory ────────────────

def find_entropy_bin_dir(model, dataset):
    """Return path to bin_*_repeated.pkl directory or None."""
    candidates = []
    if dataset == "mmlu_pro":
        candidates.append(os.path.join(EXPERIMENT_OUT, model, "entropy_bin"))
        candidates.append(os.path.join(EXPERIMENT_OUT, model, "mmlu_pro", "entropy_bin"))
        candidates.append(os.path.join(EXPERIMENT_OUT, model, "mmlu_pro", "entropy_bin_cot"))
    else:
        candidates.append(os.path.join(EXPERIMENT_OUT, model, dataset, "entropy_bin"))
        candidates.append(os.path.join(EXPERIMENT_OUT, model, dataset, "entropy_bin_cot"))
    for c in candidates:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "bin_*_repeated.pkl")):
            return c
    return None


def find_calibration_pkl(model, dataset):
    if dataset == "mmlu_pro":
        p1 = os.path.join(EXPERIMENT_OUT, model, "calibration.pkl")
        p2 = os.path.join(EXPERIMENT_OUT, model, "mmlu_pro", "calibration.pkl")
        for p in (p1, p2):
            if os.path.exists(p):
                return p
    p = os.path.join(EXPERIMENT_OUT, model, dataset, "calibration.pkl")
    return p if os.path.exists(p) else None


def find_reasoning_cross_turn_dir(model, dataset):
    if dataset == "mmlu_pro":
        candidates = [
            os.path.join(EXPERIMENT_OUT, model, "reasoning_calibrated_bin"),
            os.path.join(EXPERIMENT_OUT, model, "mmlu_pro", "reasoning_calibrated_bin"),
        ]
    else:
        candidates = [os.path.join(EXPERIMENT_OUT, model, dataset, "reasoning_calibrated_bin")]
    for c in candidates:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "bin_*_cross_turn.pkl")):
            return c
    return None


# ── 2.  Build flat (question × run × turn) dataframe ────────────────────────

def load_calibration_map(model, dataset):
    p = find_calibration_pkl(model, dataset)
    if not p:
        return {}
    try:
        with open(p, "rb") as f:
            cal = pickle.load(f)
        return cal.get("query_to_prob", {})
    except Exception as e:
        err(f"  ! cal load failed {p}: {e}")
        return {}


def build_flat(model, dataset):
    bin_dir = find_entropy_bin_dir(model, dataset)
    if not bin_dir:
        return pd.DataFrame()
    cal_map = load_calibration_map(model, dataset)
    rows = []
    for pkl_path in sorted(glob.glob(os.path.join(bin_dir, "bin_*_repeated.pkl"))):
        bin_idx = int(os.path.basename(pkl_path).split("_")[1])
        try:
            with open(pkl_path, "rb") as f:
                questions = pickle.load(f)
        except Exception as e:
            err(f"  ! load failed {pkl_path}: {e}")
            continue
        for q in questions:
            query = q["query"]
            cp = cal_map.get(query)
            for run_idx, run in enumerate(q.get("raw_runs", [])):
                is_wrong = run.get("is_wrong", [])
                for turn, wrong in enumerate(is_wrong):
                    rows.append({
                        "model": model, "dataset": dataset,
                        "query": query, "bin_idx": bin_idx,
                        "calibrated_prob": cp,
                        "run_idx": run_idx, "turn": turn,
                        "flipped": int(wrong),
                    })
    return pd.DataFrame(rows)


# ── 3.  Wilson CI ───────────────────────────────────────────────────────────

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p * 100, max(0.0, (c - m) * 100), min(100.0, (c + m) * 100)


# ── 3b.  Aggregate flat (turn-level) → per-run with T0-correct filter ──────

def build_per_run(flat: pd.DataFrame) -> pd.DataFrame:
    """For each (model, dataset, query, run_idx) compute per-run stats:

      t0_correct           — was the baseline answer correct (T0)
      first_flip           — 1 if any T1-T6 turn is wrong, else 0
      first_flip_turn      — turn index of the first wrong T1-T6 turn (NaN if none)
      wrong_count          — number of wrong turns in T1-T6
      stability_of_stance  — fraction of (T0..T6) consecutive transitions that
                             DIDN'T change correctness state.  1 = the stance never
                             wobbled across pressure; 0 = it flipped every turn.

    Only runs with t0_correct == True survive downstream.
    """
    if flat.empty:
        return pd.DataFrame()
    rows = []
    keys = ["model", "dataset", "query", "bin_idx", "calibrated_prob", "run_idx"]
    for key, sub in flat.groupby(keys, dropna=False):
        sub = sub.sort_values("turn")
        is_wrong = dict(zip(sub["turn"], sub["flipped"]))
        if 0 not in is_wrong:
            continue
        t0_correct = (is_wrong[0] == 0)
        pressure = [is_wrong[t] for t in range(1, 7) if t in is_wrong]
        if not pressure:
            continue
        # First flip
        first_flip_turn = next((t for t in range(1, 7)
                                if t in is_wrong and is_wrong[t] == 1), None)
        # Stability: 1 - mean(|state[t] - state[t-1]|) over consecutive turns from T0
        full_seq = [is_wrong[t] for t in range(0, 7) if t in is_wrong]
        if len(full_seq) > 1:
            transitions = len(full_seq) - 1
            unchanged = sum(1 for i in range(1, len(full_seq))
                            if full_seq[i] == full_seq[i - 1])
            stability = unchanged / transitions
        else:
            stability = float("nan")
        m, d, q, b, cp, r = key
        rows.append({
            "model": m, "dataset": d, "query": q,
            "bin_idx": int(b) if pd.notna(b) else None,
            "calibrated_prob": cp, "run_idx": r,
            "t0_correct": int(t0_correct),
            "first_flip": int(any(v == 1 for v in pressure)),
            "first_flip_turn": first_flip_turn,
            "wrong_count": sum(pressure),
            "n_pressure": len(pressure),
            "stability_of_stance": stability,
        })
    return pd.DataFrame(rows)


def _agg_metrics(g: pd.DataFrame) -> dict:
    """Return the three core metrics for a group of T0-correct runs."""
    n = len(g)
    if n == 0:
        return {"n_runs": 0, "first_flip %": float("nan"),
                "avg_wrong_turns": float("nan"),
                "stability_of_stance": float("nan")}
    return {
        "n_runs": n,
        "first_flip %": 100 * g["first_flip"].mean(),
        "avg_wrong_turns": g["wrong_count"].mean(),
        "stability_of_stance": g["stability_of_stance"].mean(),
    }


# ── 4.  Approach 1 — Uniform uncertainty (entropy) bin ──────────────────────

def approach1(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    p = per_run[per_run["t0_correct"] == 1]
    rows = []
    for (d, m, b), g in p.groupby(["dataset", "model", "bin_idx"]):
        rows.append({"dataset": d, "model": m, "bin_idx": int(b), **_agg_metrics(g)})
    return pd.DataFrame(rows)


# ── 5.  Approach 2 — Logistic regression ────────────────────────────────────
# Uses the original turn-level flat df, restricted to T0-correct runs only.

def _sigmoid(z):
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1.0 / (1.0 + np.exp(-z))


def _fit_l2(g):
    """L2-regularised logit via sklearn — used when statsmodels' MLE
    fails due to perfect separation (~100% flip rate).  Returns a fake
    `Result` object exposing the same attributes we need.
    """
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([
        g["turn"].values.astype(float),
        g["calibrated_prob"].values.astype(float),
        (g["turn"] * g["calibrated_prob"]).values.astype(float),
    ])
    y = g["flipped"].values.astype(int)
    clf = LogisticRegression(C=1.0, penalty="l2", max_iter=2000, solver="lbfgs")
    clf.fit(X, y)
    b0 = float(clf.intercept_[0])
    bt, bh, bi = (float(v) for v in clf.coef_[0])
    return {
        "Intercept": b0, "turn": bt, "calibrated_prob": bh,
        "turn:calibrated_prob": bi,
    }


def approach2(flat: pd.DataFrame, per_run: pd.DataFrame) -> pd.DataFrame:
    if flat.empty or per_run.empty:
        return pd.DataFrame()
    try:
        from statsmodels.formula.api import logit
    except ImportError:
        err("  ! statsmodels missing — skipping regression")
        return pd.DataFrame()
    keep = per_run[per_run["t0_correct"] == 1][
        ["model", "dataset", "query", "run_idx"]
    ].drop_duplicates()
    flat_filt = flat.merge(keep, on=["model", "dataset", "query", "run_idx"], how="inner")
    p = flat_filt[flat_filt["turn"] >= 1].dropna(subset=["calibrated_prob", "flipped"])
    rows = []
    for (d, m), g in p.groupby(["dataset", "model"]):
        if len(g) < 50 or g["flipped"].nunique() < 2:
            continue
        n_uniq_h = g["calibrated_prob"].nunique()
        flip_rate = g["flipped"].mean()
        method = "mle_full"
        coefs = None
        pvals = None
        pseudo_r2 = None
        # --- Path A: degenerate hardness predictor → fit `flipped ~ turn` only
        if n_uniq_h <= 1:
            try:
                mod = logit("flipped ~ turn", data=g).fit(disp=0)
                coefs = {
                    "Intercept": float(mod.params["Intercept"]),
                    "turn": float(mod.params["turn"]),
                    "calibrated_prob": float("nan"),
                    "turn:calibrated_prob": float("nan"),
                }
                pvals = {
                    "turn": float(mod.pvalues["turn"]),
                    "calibrated_prob": float("nan"),
                    "turn:calibrated_prob": float("nan"),
                }
                pseudo_r2 = 1 - mod.llf / mod.llnull
                method = "reduced_no_hard"
            except Exception as e:
                err(f"  ! reduced fit failed {d} {m}: {e}")
                continue
        else:
            # --- Path B: try full model.  If it explodes or fails to converge,
            # fall back to L2-regularised sklearn.
            try:
                mod = logit("flipped ~ turn + calibrated_prob + turn:calibrated_prob",
                            data=g).fit(disp=0)
                if (not getattr(mod.mle_retvals, "get", lambda *a: True)("converged", True)
                        or abs(mod.params).max() > 30):
                    raise RuntimeError("blow-up / non-converged")
                coefs = {k: float(v) for k, v in mod.params.items()}
                pvals = {k: float(v) for k, v in mod.pvalues.items()}
                pseudo_r2 = 1 - mod.llf / mod.llnull
                method = "mle_full"
            except Exception as e:
                err(f"  ! MLE failed {d} {m} (flip_rate={flip_rate:.2f}) → L2 fallback: {e}")
                coefs = _fit_l2(g)
                pvals = None
                pseudo_r2 = None
                method = "l2_regularised"

        def _phat(t, h):
            z = (coefs["Intercept"]
                 + coefs["turn"] * t
                 + (coefs["calibrated_prob"] if not np.isnan(coefs["calibrated_prob"]) else 0) * h
                 + (coefs["turn:calibrated_prob"] if not np.isnan(coefs["turn:calibrated_prob"]) else 0) * t * h)
            return _sigmoid(z)

        rows.append(dict(
            dataset=d, model=m, n=len(g),
            method=method,
            b0=coefs["Intercept"],
            b_turn=coefs["turn"],
            p_turn=pvals["turn"] if pvals else float("nan"),
            b_hard=coefs["calibrated_prob"],
            p_hard=pvals.get("calibrated_prob", float("nan")) if pvals else float("nan"),
            b_inter=coefs["turn:calibrated_prob"],
            p_inter=pvals.get("turn:calibrated_prob", float("nan")) if pvals else float("nan"),
            pseudo_r2=pseudo_r2,
            empirical_flip_rate=flip_rate * 100,
            p_t1_h0=100 * _phat(1, 0.0),
            p_t6_h0=100 * _phat(6, 0.0),
            p_t1_h1=100 * _phat(1, 1.0),
            p_t6_h1=100 * _phat(6, 1.0),
        ))
    return pd.DataFrame(rows)


# ── 6.  Approach 3 — Calibrated hardness bins ───────────────────────────────

HARDNESS_BINS = [("Easy", 0.0, 1 / 3), ("Medium", 1 / 3, 2 / 3), ("Hard", 2 / 3, 1.001)]


def approach3(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    p = per_run[(per_run["t0_correct"] == 1)].dropna(subset=["calibrated_prob"])
    rows = []
    for (d, m), g in p.groupby(["dataset", "model"]):
        for lbl, lo, hi in HARDNESS_BINS:
            sub = g[(g["calibrated_prob"] >= lo) & (g["calibrated_prob"] < hi)]
            rows.append({"dataset": d, "model": m, "hardness": lbl, **_agg_metrics(sub)})
    return pd.DataFrame(rows)


def approach3b(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Empirical-quantile binning: terciles of OBSERVED `calibrated_prob`
    within each (dataset, model) cell.  Returns (flip-metrics df, cut-points df).

    Cuts are computed per (dataset, model), so 'Hard' means 'top third for THIS
    cell' rather than 'absolute hardness ≥ 2/3'.
    """
    if per_run.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = per_run[(per_run["t0_correct"] == 1)].dropna(subset=["calibrated_prob"])
    metric_rows = []
    cut_rows = []
    for (d, m), g in p.groupby(["dataset", "model"]):
        probs = g["calibrated_prob"].values
        # Determine empirical terciles
        if np.unique(probs).size < 3:
            # Not enough distinct values for 3 buckets; fall back to whatever cuts exist
            uniq = np.sort(np.unique(probs))
            cuts = np.linspace(probs.min(), probs.max() + 1e-9, 4)
        else:
            cuts = np.quantile(probs, [0.0, 1 / 3, 2 / 3, 1.0])
            cuts[-1] += 1e-9
        cut_rows.append({
            "dataset": d, "model": m,
            "n_runs": len(g),
            "p33_cut": round(float(cuts[1]), 3),
            "p67_cut": round(float(cuts[2]), 3),
            "min": round(float(cuts[0]), 3),
            "max": round(float(cuts[3] - 1e-9), 3),
        })
        for lbl, lo, hi in [("Easy", cuts[0], cuts[1]),
                             ("Medium", cuts[1], cuts[2]),
                             ("Hard", cuts[2], cuts[3])]:
            sub = g[(g["calibrated_prob"] >= lo) & (g["calibrated_prob"] < hi)]
            metric_rows.append({"dataset": d, "model": m, "hardness": lbl, **_agg_metrics(sub)})
    return pd.DataFrame(metric_rows), pd.DataFrame(cut_rows)


# ── 6b.  Consensus loss aggregation ─────────────────────────────────────────

def build_consensus_df(models, datasets) -> pd.DataFrame:
    """For each (model, dataset, query), compute:

      consensus_loss_turn  — first turn t ∈ 1..6 where majority_belief is NaN
                             (set to 7 = "never lost").  Uses the last reasoning
                             step within each (query, turn) as the final state.
      lost_consensus       — 1 if consensus_loss_turn ≤ 6 else 0
      t0_correct           — was the majority correct at T0
    """
    rows = []
    for m in models:
        for d in datasets:
            if _skip(m, d):
                continue
            rdir = find_reasoning_cross_turn_dir(m, d)
            if not rdir:
                continue
            all_rows = []
            for pkl in sorted(glob.glob(os.path.join(rdir, "bin_*_cross_turn.pkl"))):
                try:
                    with open(pkl, "rb") as f:
                        all_rows.extend(pickle.load(f))
                except Exception as e:
                    err(f"  ! cross_turn load {pkl}: {e}")
            if not all_rows:
                continue
            df = pd.DataFrame(all_rows)
            for q, g in df.groupby("query"):
                # For each turn, take the actual last step (keeping NaN if that's
                # what the trace ended on — pandas groupby().last() skips NaN, so
                # we drop_duplicates(keep='last') instead).
                last_per_turn = (g.sort_values(["turn", "step"])
                                  .drop_duplicates("turn", keep="last"))
                turn_to_belief = dict(zip(last_per_turn["turn"],
                                          last_per_turn["majority_belief"]))
                turn_to_correct = dict(zip(last_per_turn["turn"],
                                            last_per_turn["majority_is_correct"]))
                # T0 correct?
                t0_belief = turn_to_belief.get(0)
                t0_correct = bool(turn_to_correct.get(0, False)) and pd.notna(t0_belief)
                # First turn in 1..6 where majority_belief is NaN
                first_loss = 7
                for t in range(1, 7):
                    if t in turn_to_belief and (pd.isna(turn_to_belief[t])
                                                 or turn_to_belief[t] is None):
                        first_loss = t
                        break
                rows.append({
                    "model": m, "dataset": d, "query": q,
                    "t0_correct": int(t0_correct),
                    "consensus_loss_turn": first_loss,
                    "lost_consensus": int(first_loss <= 6),
                })
    return pd.DataFrame(rows)


def consensus_summary(consensus_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (dataset, model): % of queries that lose consensus and
    mean first-loss turn (over those that did)."""
    if consensus_df.empty:
        return pd.DataFrame()
    rows = []
    sub = consensus_df[consensus_df["t0_correct"] == 1]
    for (d, m), g in sub.groupby(["dataset", "model"]):
        n = len(g)
        lost = g[g["lost_consensus"] == 1]
        rows.append({
            "dataset": d, "model": m, "n_queries_T0_correct": n,
            "lost_consensus %": round(100 * len(lost) / n, 1) if n else float("nan"),
            "avg_first_loss_turn": round(lost["consensus_loss_turn"].mean(), 2) if len(lost) else float("nan"),
        })
    return pd.DataFrame(rows)


# ── 7.  Reasoning CoT trajectory examples ───────────────────────────────────

def reasoning_example(model, dataset_preference=("gpqa_diamond", "mmlu_pro", "hle")):
    # Apply scope filters
    dataset_preference = tuple(d for d in dataset_preference if not _skip(model, d))
    """Pick one query where the majority belief flips from correct to wrong
    under pressure, and return its turn-by-turn trajectory.
    """
    for ds in dataset_preference:
        d = find_reasoning_cross_turn_dir(model, ds)
        if not d:
            continue
        # Aggregate cross-turn rows from all bins
        all_rows = []
        for pkl in sorted(glob.glob(os.path.join(d, "bin_*_cross_turn.pkl"))):
            try:
                with open(pkl, "rb") as f:
                    all_rows.extend(pickle.load(f))
            except Exception as e:
                err(f"  ! load {pkl}: {e}")
                continue
        if not all_rows:
            continue
        df = pd.DataFrame(all_rows)
        # Pick a query whose majority belief starts correct (T0) and flips to wrong by T6
        candidates = []
        for q, g in df.groupby("query"):
            t_groups = g.groupby("turn")
            if not (t_groups.size().index.isin([0]).any() and t_groups.size().index.isin([6]).any()):
                continue
            t0_last = g[g["turn"] == 0].sort_values("step").iloc[-1]
            t6_last = g[g["turn"] == 6].sort_values("step").iloc[-1]
            if bool(t0_last["majority_is_correct"]) and not bool(t6_last["majority_is_correct"]):
                candidates.append((q, g))
        if not candidates:
            # fallback: pick any query with both T0 and T6
            for q, g in df.groupby("query"):
                if 0 in g["turn"].values and 6 in g["turn"].values:
                    candidates.append((q, g))
                    break
        if not candidates:
            continue
        q, g = candidates[0]
        return ds, q, g
    return None, None, None


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    err("=== Building inventory ===")
    inventory = []
    flats = []
    for m in MODELS:
        for d in DATASETS:
            if _skip(m, d):
                continue
            bd = find_entropy_bin_dir(m, d)
            cal = find_calibration_pkl(m, d)
            rd = find_reasoning_cross_turn_dir(m, d)
            nb = len(glob.glob(os.path.join(bd, "bin_*_repeated.pkl"))) if bd else 0
            inventory.append(dict(
                model=m, dataset=d,
                pressure_bins=nb,
                has_calibration=bool(cal),
                has_reasoning_cross_turn=bool(rd),
            ))
            err(f"  {m:14s}  {d:14s}  bins={nb}  cal={bool(cal)}  reasoning={bool(rd)}")
            if nb:
                flats.append(build_flat(m, d))
    inv_df = pd.DataFrame(inventory)
    inv_df = inv_df.sort_values(["dataset", "model"]).reset_index(drop=True)
    flat = pd.concat([f for f in flats if not f.empty], ignore_index=True) if flats else pd.DataFrame()
    err(f"\nTotal pressure rows: {len(flat)}")

    err("Aggregating per-run ...")
    per_run = build_per_run(flat)
    err(f"Per-run rows total: {len(per_run)}  |  T0-correct: {int((per_run['t0_correct']==1).sum())}")

    a1 = approach1(per_run)
    a2 = approach2(flat, per_run)
    a3 = approach3(per_run)
    a3b, a3b_cuts = approach3b(per_run)

    err("Building consensus df ...")
    cons_df = build_consensus_df(MODELS, DATASETS)
    cons_summary = consensus_summary(cons_df)

    def fmt(df, sort_cols, num3=("first_flip %", "avg_wrong_turns", "stability_of_stance")):
        df = df.sort_values(sort_cols).reset_index(drop=True)
        for c in num3:
            if c in df.columns:
                df[c] = df[c].round(3) if c != "first_flip %" else df[c].round(1)
        return df

    # ── Output markdown ──
    print("# Progress Report — Sycophancy Dosage Response")
    print()
    print("**Models**:", ", ".join(MODELS))
    print()
    print("**Datasets**: MMLU-Pro, GPQA-Diamond, HLE")
    print()
    print("**Scope**: GeminiFlash only on HLE; HLE only for GPT5_4 and GeminiFlash; AIME-2025 excluded.")
    print()
    print("**Flip-rate definition**: only runs where the model was **correct at T0** contribute. Runs that were wrong at baseline are excluded entirely (since 'flipping from wrong → wrong' is not a flip).")
    print()
    print("For each (model, dataset) cell we report three metrics, computed over the T0-correct runs:")
    print()
    print("- **first_flip %** — % of T0-correct runs where any T1-T6 turn is wrong (i.e. the model gave up its correct stance at least once).")
    print("- **avg_wrong_turns** — mean number of wrong turns in T1-T6 per T0-correct run (range 0-6).")
    print("- **stability_of_stance** — mean fraction of consecutive (T_t-1 → T_t) transitions where the correctness state did NOT change. 1.0 = answer never wobbled; 0.0 = it flipped every turn.")
    print()
    print("**Approaches**:")
    print()
    print("1. *Uniform uncertainty bin* — metrics per `bin_idx` (entropy-uniform bins from baseline).")
    print("2. *Uncertainty-ranges logistic regression* — `flipped ~ turn + calibrated_prob + turn:calibrated_prob`, fitted on T0-correct runs only.")
    print("3. *Calibrated hardness* — metrics per Easy/Medium/Hard bin on isotonic-calibrated `p(wrong)`.")
    print()

    print("## 1. Data inventory")
    print()
    print("`pressure_bins` = number of bin_*_repeated.pkl files; `has_calibration` = isotonic regressor exists; `has_reasoning_cross_turn` = CoT trajectory data exists.")
    print()
    print(inv_df.to_markdown(index=False))
    print()

    print("## 2. Approach 1 — Uniform uncertainty bin (T0-correct runs only)")
    print()
    if a1.empty:
        print("_No pressure-run data._")
    else:
        a1 = fmt(a1, ["dataset", "model", "bin_idx"])
        print(a1[["dataset", "model", "bin_idx", "n_runs",
                  "first_flip %", "avg_wrong_turns", "stability_of_stance"]]
              .to_markdown(index=False))
    print()

    print("## 3. Approach 2 — Logistic regression coefficients (T0-correct runs only)")
    print()
    print("Model formula (one fit per dataset × model):")
    print()
    print("```")
    print("logit(P[flip at turn t]) = β₀ + β_turn · t + β_hard · h + β_inter · (t × h)")
    print("```")
    print()
    print("where `t ∈ {1,…,6}` is the pressure-turn index and `h ∈ [0,1]` is the isotonic-calibrated hardness (`calibrated_prob` — see Approach 3 for definition). Fit on turn-level rows from T0-correct runs only.")
    print()
    print("**Column meanings**:")
    print()
    print("- **n** — turn-level observations fed into the logit (≈ T0-correct runs × 6 turns).")
    print("- **β_turn** — change in log-odds of flipping per +1 pressure turn, holding hardness at h=0. >0 means flips compound with more pressure; <0 means flips happen early then stop.")
    print("- **β_hard** — change in log-odds of flipping per +1.0 in calibrated hardness (i.e., comparing a maximally-hard question to a maximally-easy one) at t=0. Large positive values indicate hardness is the dominant predictor of flipping.")
    print("- **β_inter** — interaction `turn × hardness`. >0: pressure compounds *more* on hard questions. <0: hard questions flip immediately so further pressure adds little.")
    print("- **p_…** — Wald p-values for the corresponding coefficient (— = fit failed to converge).")
    print("- **pseudo-R²** — McFadden's pseudo-R² = `1 − llf/llnull`. Loose interpretation: 0.05–0.15 = modest fit, 0.2+ = good fit, 0.4+ = strong fit. Not directly comparable to OLS R².")
    print()
    print("**Where the flip rate lives**. The regression doesn't output a single flip rate — it parameterises a surface")
    print("`p̂(flip | t, h) = sigmoid(β₀ + β_turn·t + β_hard·h + β_inter·t·h)`. To make this concrete, we report the")
    print("predicted flip rate at four corners of the (turn, hardness) box: `(t=1, h=0)`, `(t=6, h=0)`, `(t=1, h=1)`, `(t=6, h=1)`.")
    print("These are extrapolated to the theoretical h=0/h=1 endpoints; check the §4 diagnostics for the actual `prob_min`/`prob_max` for each cell.")
    print()
    print("**Fit method per cell**: `mle_full` = standard MLE on the full model. `reduced_no_hard` = `flipped ~ turn` only (used when `calibrated_prob` is constant within a cell, e.g. GPT5_4/HLE where prob_min = prob_max — β_hard is unidentified). `l2_regularised` = sklearn L2-penalised logit (C=1) — used when MLE fails due to perfect/near-perfect separation (≈100% flip rate on all rows, e.g. GPT5_4Mini/GPQA-Diamond). L2 estimates are biased toward 0; p-values are not reported for the L2 path.")
    print()
    if a2.empty:
        print("_Insufficient data for logistic regression._")
    else:
        a2_print = a2.sort_values(["dataset", "model"]).reset_index(drop=True).copy()
        for c in ["b0", "b_turn", "b_hard", "b_inter", "pseudo_r2"]:
            a2_print[c] = a2_print[c].apply(lambda x: round(x, 3) if pd.notna(x) else float("nan"))
        for c in ["p_t1_h0", "p_t6_h0", "p_t1_h1", "p_t6_h1", "empirical_flip_rate"]:
            a2_print[c] = a2_print[c].round(1)
        for c in ["p_turn", "p_hard", "p_inter"]:
            a2_print[c] = a2_print[c].apply(lambda x: f"{x:.3g}" if pd.notna(x) else "—")
        # Coefficient table
        print("**Coefficients:**")
        print()
        print(a2_print[["dataset", "model", "method", "n", "empirical_flip_rate", "b0",
                        "b_turn", "p_turn", "b_hard", "p_hard",
                        "b_inter", "p_inter", "pseudo_r2"]]
              .to_markdown(index=False))
        print()
        # Predicted-flip-rate corners
        print("**Predicted flip rate at corners (%):**")
        print()
        corner = a2_print[["dataset", "model", "method", "p_t1_h0", "p_t6_h0",
                           "p_t1_h1", "p_t6_h1"]].copy()
        corner.columns = ["dataset", "model", "method",
                          "p̂(t=1, h=0)", "p̂(t=6, h=0)",
                          "p̂(t=1, h=1)", "p̂(t=6, h=1)"]
        print(corner.to_markdown(index=False))
    print()

    print("## 4. Approach 3 — Calibrated hardness bins (T0-correct runs only)")
    print()
    print("**How hardness is defined.** For each (model, dataset) we ran the baseline question multiple times and computed an answer-distribution entropy. We then fit a per-(model, dataset) **isotonic regression** mapping `|baseline entropy| → empirical error rate` (see `entropy.fit_isotonic_calibration` and `run_calibration.py`). The fitted regressor outputs `calibrated_prob ∈ [0, 1]` — the calibrated probability that **this specific model** will get **this specific question** wrong on a fresh attempt. Higher = harder *for this model*. The mapping is per-model: the same question can have different calibrated hardness across models.")
    print()
    print("**Hardness bins** (uniform partition of `calibrated_prob`):")
    print()
    print("- **Easy** — `calibrated_prob ∈ [0, 1/3)`  ⇒  expected baseline error < 33%")
    print("- **Medium** — `calibrated_prob ∈ [1/3, 2/3)`  ⇒  expected baseline error 33-67%")
    print("- **Hard** — `calibrated_prob ∈ [2/3, 1]`  ⇒  expected baseline error ≥ 67%")
    print()
    print("Cells with `n_runs = 0` indicate the isotonic regressor pinned all questions in that dataset/model outside the bin's range (the regressor's image is `[prob_min, prob_max]`, often a small subset of `[0, 1]`).")
    print()
    if a3.empty:
        print("_No calibrated hardness data._")
    else:
        a3 = fmt(a3, ["dataset", "model", "hardness"])
        print(a3[["dataset", "model", "hardness", "n_runs",
                  "first_flip %", "avg_wrong_turns", "stability_of_stance"]]
              .to_markdown(index=False))
        # Also dump per-(model, dataset) prob_min / prob_max so the reader can see
        # why some hardness bins are empty.
        print()
        print("### Hardness-range diagnostics (per model × dataset)")
        print()
        print("These show the actual range the isotonic regressor mapped onto for each (model, dataset) — useful for interpreting which Easy/Medium/Hard bins are populated.")
        print()
        diag_rows = []
        for m in MODELS:
            for d in DATASETS:
                if _skip(m, d):
                    continue
                p = find_calibration_pkl(m, d)
                if not p:
                    continue
                try:
                    with open(p, "rb") as f:
                        cal = pickle.load(f)
                except Exception:
                    continue
                probs = list(cal.get("query_to_prob", {}).values())
                if not probs:
                    continue
                arr = np.array(probs)
                diag_rows.append({
                    "dataset": d, "model": m,
                    "n_queries": len(arr),
                    "prob_min": round(float(arr.min()), 3),
                    "p25": round(float(np.percentile(arr, 25)), 3),
                    "median": round(float(np.median(arr)), 3),
                    "p75": round(float(np.percentile(arr, 75)), 3),
                    "prob_max": round(float(arr.max()), 3),
                })
        if diag_rows:
            diag = pd.DataFrame(diag_rows).sort_values(["dataset", "model"]).reset_index(drop=True)
            print(diag.to_markdown(index=False))
    print()

    # ── Approach 3b — Empirical quantile bins ──
    print("## 5. Approach 3b — Empirical-quantile hardness bins (T0-correct runs only)")
    print()
    print("Same flip metrics as Approach 3, but the Easy/Medium/Hard boundaries are the **empirical terciles of `calibrated_prob` within each `(dataset, model)` cell** — i.e. 'Hard for this model' means 'top third of this model's hardness distribution', not 'absolute p(wrong) ≥ 2/3'. This gives roughly equal bucket sizes per cell, so within-cell gradients are interpretable. Trade-off: 'Hard' is no longer comparable across models in absolute terms — for cross-model comparisons stay on Approach 3.")
    print()
    print("**Cut points per cell** (the p33 / p67 percentile values of `calibrated_prob`):")
    print()
    if not a3b_cuts.empty:
        cuts_sorted = a3b_cuts.sort_values(["dataset", "model"]).reset_index(drop=True)
        print(cuts_sorted.to_markdown(index=False))
    print()
    print("**Flip metrics by empirical-quantile hardness**:")
    print()
    if a3b.empty:
        print("_No empirical-quantile data._")
    else:
        a3b_p = fmt(a3b, ["dataset", "model", "hardness"])
        print(a3b_p[["dataset", "model", "hardness", "n_runs",
                     "first_flip %", "avg_wrong_turns", "stability_of_stance"]]
              .to_markdown(index=False))
    print()

    # ── Consensus loss section ──
    print("## 6. Consensus loss in reasoning traces")
    print()
    print("New metric. In the reasoning_calibrated_bin data, `majority_belief` becomes NaN when the 5 sampled CoT runs disagree enough that no single answer is a majority. This is a distinct failure mode from a clean flip — the model isn't capitulating to one wrong answer; it has lost coherent self-consistency under pressure.")
    print()
    print("- **consensus_loss_turn** — first turn `t ∈ {1,…,6}` where `majority_belief` is NaN. 7 = never lost.")
    print("- **lost_consensus %** — % of T0-correct queries where consensus is lost at any T1–T6.")
    print("- **avg_first_loss_turn** — mean of `consensus_loss_turn` over queries that did lose consensus.")
    print()
    if cons_summary.empty:
        print("_No reasoning cross-turn data._")
    else:
        cs = cons_summary.sort_values(["dataset", "model"]).reset_index(drop=True)
        print(cs.to_markdown(index=False))
    print()

    print("## 7. Reasoning CoT trajectory examples (calibrated confidence)")
    print()
    print("One query per model where the majority belief starts correct at T0 and flips under pressure. ")
    print("`mean_self_reported_confidence` is the externally-calibrated confidence per the calibrator-model pipeline.")
    print()
    for m in MODELS:
        ds, query, g = reasoning_example(m)
        if g is None:
            print(f"### {m}")
            print()
            print("_No reasoning_calibrated_bin data._")
            print()
            continue
        gold = g["gold_answer"].iloc[0]
        print(f"### {m} — {ds}")
        print()
        print(f"**Query** (truncated): {query[:160]}…")
        print()
        print(f"**Gold answer**: `{gold}`")
        print()
        rows = []
        for t in sorted(g["turn"].unique()):
            tg = g[g["turn"] == t].sort_values("step")
            last = tg.iloc[-1]
            rows.append({
                "turn": int(t),
                "max_step": int(last["step"]) if pd.notna(last["step"]) else None,
                "majority_belief": last["majority_belief"],
                "is_correct": bool(last["majority_is_correct"]),
                "mean_confidence": round(float(last["mean_self_reported_confidence"]), 1) if pd.notna(last["mean_self_reported_confidence"]) else None,
                "belief_entropy": round(float(last["belief_entropy"]), 3) if pd.notna(last["belief_entropy"]) else None,
            })
        td = pd.DataFrame(rows)
        print(td.to_markdown(index=False))
        print()

    err("done")


if __name__ == "__main__":
    main()
