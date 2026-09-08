"""
build_analysis_dfs.py
---------------------
Two flat-dataframe builders for RQ2 and RQ4.

    build_flat_df()        → one row per (question × run × pressure turn)
                             used for logistic regression (RQ2)

    build_reasoning_df()   → one row per (question × CoT turn × reasoning step)
                             used for belief-trajectory analysis (RQ4)

    get_common_queries()   → returns the intersection of query sets across all
                             models for a given dataset; use to restrict
                             cross-model comparisons to a consistent question set
"""

from __future__ import annotations

import os
import pickle
from glob import glob
from typing import Optional

import numpy as np
import pandas as pd

from sycophancy.entropy import compute_entropy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPERIMENT_OUT     = "experiment_out"
REASONING_TRACE    = "reasoning_trace_output"

N_CAL_BINS = 5   # uniform [0,1] bins for calibrated_bin_idx

DATASET_SUBDIRS = {"gpqa_diamond", "aime_2025"}


def _load_entropy_map(model_dir: str) -> dict[str, dict]:
    """Returns {query: {entropy, uncertainty, category}} from baseline metadata in model_dir."""
    path = os.path.join(model_dir, "base_experiment_metadata.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            meta = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        return {}
    out = {}
    for item in meta:
        if "entropy" not in item:
            item["entropy"] = compute_entropy(item.get("answers_generated", []))
        out[item["query"]] = {
            "entropy":     item["entropy"],
            "uncertainty": item.get("uncertainty", None),
            "category":    item.get("category", None),
        }
    return out


def _load_calibration_map(model_dir: str) -> dict[str, float]:
    """Returns {query: calibrated_prob} from calibration.pkl sidecar in model_dir."""
    path = os.path.join(model_dir, "calibration.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            cal = pickle.load(f)
        return cal.get("query_to_prob", {})
    except (EOFError, pickle.UnpicklingError):
        return {}


def _calibrated_bin_idx(p: Optional[float], n_bins: int = N_CAL_BINS) -> Optional[int]:
    """Maps calibrated_prob ∈ [0,1] to a uniform bin index 0..n_bins-1."""
    if p is None:
        return None
    return min(int(p * n_bins), n_bins - 1)


def _iter_entropy_bin_dirs(model_dir: str):
    """
    Yields (dataset_name, entropy_bin_dir, entropy_map, cal_map) for a model directory.
    Covers:
      - <model_dir>/entropy_bin/                    → mmlu_pro
      - <model_dir>/gpqa_diamond/entropy_bin/       → gpqa_diamond
      - <model_dir>/aime_2025/entropy_bin/          → aime_2025
    """
    root_bin = os.path.join(model_dir, "entropy_bin")
    if os.path.isdir(root_bin):
        yield (
            "mmlu_pro", root_bin,
            _load_entropy_map(model_dir),
            _load_calibration_map(model_dir),
        )

    for subdir in DATASET_SUBDIRS:
        sub_bin = os.path.join(model_dir, subdir, "entropy_bin")
        if os.path.isdir(sub_bin):
            sub_dir_path = os.path.join(model_dir, subdir)
            yield (
                subdir, sub_bin,
                _load_entropy_map(sub_dir_path),
                _load_calibration_map(sub_dir_path),
            )


# ---------------------------------------------------------------------------
# Common-query helper
# ---------------------------------------------------------------------------

def get_common_queries(
    dataset: str,
    experiment_dir: str = EXPERIMENT_OUT,
    models: Optional[list[str]] = None,
) -> set[str]:
    """
    Returns the intersection of baseline query sets across all models for
    a given dataset.  Use this to restrict cross-model comparisons to the
    same question pool when models were run on different subsets.

    Example — GPQA Diamond: Claude models have 79 questions, GPT models
    have 198.  get_common_queries("gpqa_diamond") returns the 79 questions
    that all five models share.

    Parameters
    ----------
    dataset : "mmlu_pro" | "gpqa_diamond" | "aime_2025"
    """
    available = sorted(
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
        and d not in DATASET_SUBDIRS
    )
    if models:
        available = [m for m in available if m in models]

    common: Optional[set[str]] = None
    for model in available:
        if dataset == "mmlu_pro":
            meta_path = os.path.join(experiment_dir, model, "base_experiment_metadata.pkl")
        else:
            meta_path = os.path.join(experiment_dir, model, dataset, "base_experiment_metadata.pkl")

        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        qs = {item["query"] for item in meta}
        common = qs if common is None else common & qs

    return common or set()


# ---------------------------------------------------------------------------
# RQ2 — flat sycophancy dataframe
# ---------------------------------------------------------------------------

def build_flat_df(
    experiment_dir: str = EXPERIMENT_OUT,
    models: Optional[list[str]] = None,
    restrict_queries: Optional[dict[str, set[str]]] = None,
) -> pd.DataFrame:
    """
    Walks all entropy_bin directories across models and dataset subdirs, and
    expands raw_runs into one row per (question × run × pressure turn).

    Parameters
    ----------
    restrict_queries : optional dict mapping dataset name to a set of query
        strings to keep.  Rows for queries not in the set are dropped.
        Use get_common_queries() to build the filter, e.g.:
            restrict_queries={"gpqa_diamond": get_common_queries("gpqa_diamond")}

    Directory layout handled:
      experiment_dir/<MODEL>/entropy_bin/              → mmlu_pro
      experiment_dir/<MODEL>/gpqa_diamond/entropy_bin/ → gpqa_diamond
      experiment_dir/<MODEL>/aime_2025/entropy_bin/    → aime_2025

    Columns
    -------
    model                : str
    dataset              : str   ("mmlu_pro" | "gpqa_diamond" | "aime_2025")
    query                : str
    gold_answer          : str
    category             : str | None
    entropy              : float  (Shannon entropy, negative; 0 = maximally confident)
    uncertainty          : float  (1 - majority_correct_rate)
    calibrated_prob      : float | None  (isotonic-calibrated hardness ∈ [0,1])
    bin_idx              : int    (original entropy bin from experiment filename)
    calibrated_bin_idx   : int | None  (uniform bin on [0,1] hardness scale, 0=easy)
    run_idx              : int
    turn                 : int    (0 = baseline response, 1-6 = pressure turns T1-T6)
    pressure_level       : int    (same as turn; 0 means no pressure applied yet)
    flipped              : int    (1 if model answered wrong at this turn)
    turn_category        : str | None  (e.g. "personal_disagreement")
    first_wrong_turn     : int    (turn of first flip for this run; n_turns+1 = never)
    certain              : int    (1 if entropy == 0.0)
    """
    rows = []

    available = sorted(
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
        and d not in DATASET_SUBDIRS
    )
    if models:
        available = [m for m in available if m in models]

    for model in available:
        model_dir = os.path.join(experiment_dir, model)

        for dataset, bin_dir, entropy_map, cal_map in _iter_entropy_bin_dirs(model_dir):
            allowed = restrict_queries.get(dataset) if restrict_queries else None

            for pkl_path in sorted(glob(os.path.join(bin_dir, "bin_*_repeated.pkl"))):
                bin_idx = int(os.path.basename(pkl_path).split("_")[1])
                with open(pkl_path, "rb") as f:
                    questions = pickle.load(f)

                for q in questions:
                    query = q["query"]
                    if allowed is not None and query not in allowed:
                        continue
                    em = entropy_map.get(query, {})
                    cp = cal_map.get(query)

                    for run_idx, run in enumerate(q.get("raw_runs", [])):
                        is_wrong  = run.get("is_wrong", [])
                        turn_cats = run.get("turn_categories", [None] * len(is_wrong))
                        fwt       = run.get("first_wrong_turn", len(is_wrong))

                        for turn, (wrong, cat) in enumerate(zip(is_wrong, turn_cats)):
                            rows.append({
                                "model":               model,
                                "dataset":             dataset,
                                "query":               query,
                                "gold_answer":         q.get("gold_answer"),
                                "category":            em.get("category"),
                                "entropy":             em.get("entropy"),
                                "uncertainty":         em.get("uncertainty"),
                                "calibrated_prob":     cp,
                                "bin_idx":             bin_idx,
                                "calibrated_bin_idx":  _calibrated_bin_idx(cp),
                                "run_idx":             run_idx,
                                "turn":                turn,
                                "pressure_level":      turn,
                                "flipped":             int(wrong),
                                "turn_category":       cat,
                                "first_wrong_turn":    fwt,
                            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["certain"] = (df["entropy"] == 0.0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Backfill — build_flat_df-shaped rows from reasoning_calibrated_bin, for
# cells where entropy_bin (short-answer sycophancy) coverage is narrower
# than baseline/reasoning_calibrated_bin coverage.
# ---------------------------------------------------------------------------

def backfill_flat_from_reasoning(
    model: str,
    dataset: str,
    gap_queries: set,
    experiment_dir: str = EXPERIMENT_OUT,
) -> pd.DataFrame:
    """
    Fills in build_flat_df-shaped rows for questions that have full
    reasoning_calibrated_bin coverage but no entropy_bin (short-answer
    sycophancy) coverage — e.g. ClaudeHaiku/ClaudeSonnet on GPQA-Diamond,
    extended to the full 198-question set via reasoning-only generation
    rather than a full scripts/run_sycophancy.py re-run.

    Uses each reasoning sample's own final answer as the flip signal
    (one row per question x run x turn, same statistical unit as
    entropy_bin's per-run "flipped" — both have 5 samples/question), so the
    backfilled rows pool directly with the rest of the dataframe rather than
    changing the effective sample size for these cells.

    flipped = 1 if final_answer != gold_answer; an empty/failed generation
    (final_answer is None) produces NO row at all for that (query, run,
    turn) rather than flipped=0/1 or flipped=NaN — "empty turn construes as
    None" means the observation doesn't exist, so it can't silently inflate
    a denominator in any downstream len()/count() that isn't dropna'd on
    "flipped" specifically.
    """
    if not gap_queries:
        return pd.DataFrame()

    base_dir = os.path.join(experiment_dir, model, dataset)
    with open(os.path.join(base_dir, "base_experiment_metadata.pkl"), "rb") as f:
        meta_by_query = {m["query"]: m for m in pickle.load(f)}

    query_to_prob = {}
    cal_path = os.path.join(base_dir, "calibration.pkl")
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            query_to_prob = pickle.load(f).get("query_to_prob", {})

    rows = []
    rc_dir = os.path.join(base_dir, "reasoning_calibrated_bin")
    for pkl_path in sorted(glob(os.path.join(rc_dir, "bin_*_reasoning.pkl"))):
        with open(pkl_path, "rb") as f:
            trajs = pickle.load(f)
        for traj in trajs:
            if traj.query not in gap_queries:
                continue
            m_item    = meta_by_query.get(traj.query, {})
            entropy   = m_item.get("entropy")
            cp        = query_to_prob.get(traj.query)
            n_samples = len(traj.raw_traces[0]) if traj.raw_traces else 0

            for run_idx in range(n_samples):
                per_turn = []
                for turn, turn_traces in enumerate(traj.raw_traces):
                    if run_idx >= len(turn_traces):
                        continue
                    fa = turn_traces[run_idx].final_answer
                    flipped = np.nan if fa is None else int(fa != traj.gold_answer)
                    per_turn.append((turn, flipped))

                never = len(per_turn)
                fwt = next((t for t, fl in per_turn if t >= 1 and fl == 1), never)

                for turn, flipped in per_turn:
                    if isinstance(flipped, float) and np.isnan(flipped):
                        continue  # empty/failed generation — no observation, not a row
                    rows.append({
                        "model":               model,
                        "dataset":             dataset,
                        "query":               traj.query,
                        "gold_answer":         traj.gold_answer,
                        "category":            m_item.get("category"),
                        "entropy":             entropy,
                        "uncertainty":         m_item.get("uncertainty"),
                        "calibrated_prob":     cp,
                        "bin_idx":             -1,
                        "calibrated_bin_idx":  _calibrated_bin_idx(cp) if cp is not None else None,
                        "run_idx":             run_idx,
                        "turn":                turn,
                        "pressure_level":      turn,
                        "flipped":             flipped,
                        "turn_category":       None,
                        "first_wrong_turn":    fwt,
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["certain"] = (out["entropy"] == 0.0).astype(int)
    return out


def build_flat_df_backfilled(
    experiment_dir: str = EXPERIMENT_OUT,
    models: Optional[list[str]] = None,
    restrict_queries: Optional[dict[str, set[str]]] = None,
) -> pd.DataFrame:
    """
    build_flat_df(), plus backfill_flat_from_reasoning() applied to every
    (model, dataset) cell whose restrict_queries target set isn't fully
    covered by entropy_bin — the one shared entry point so run_rq_analysis_v2.py,
    run_combined_datasets_analysis.py, and run_separate_regressions.py all
    see the same backfilled coverage rather than three separate copies of
    this logic drifting apart.
    """
    df = build_flat_df(experiment_dir=experiment_dir, models=models,
                        restrict_queries=restrict_queries)
    if not restrict_queries:
        return df

    model_list = models or sorted(
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
        and d not in DATASET_SUBDIRS
        and os.path.exists(os.path.join(experiment_dir, d, "base_experiment_metadata.pkl"))
    )
    for dataset, target_queries in restrict_queries.items():
        for model in model_list:
            base_dir = os.path.join(experiment_dir, model, dataset)
            if not os.path.exists(os.path.join(base_dir, "base_experiment_metadata.pkl")):
                continue  # this model was never run on this dataset subdir at all
            existing = set(df.loc[(df["model"]==model) & (df["dataset"]==dataset), "query"])
            gap = target_queries - existing
            if not gap:
                continue
            backfill = backfill_flat_from_reasoning(model, dataset, gap, experiment_dir)
            if not backfill.empty:
                print(f"  Backfilled {model} {dataset}: {len(gap)} questions "
                      f"({backfill['query'].nunique()} recovered) from reasoning_calibrated_bin")
                df = pd.concat([df, backfill], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# RQ4 — reasoning trajectory dataframe
# ---------------------------------------------------------------------------

def build_reasoning_df(
    reasoning_dir: str = REASONING_TRACE,
    models: Optional[list[str]] = None,
    trace_types: tuple[str, ...] = ("reasoning_calibrated_bin", "reasoning_bin"),
) -> pd.DataFrame:
    """
    Walks reasoning_dir/<MODEL>/<trace_type>/bin_*_cross_turn.pkl and
    collects one row per (question × CoT turn × reasoning step).

    Columns
    -------
    model                       : str
    trace_type                  : str  ("reasoning_calibrated_bin" or "reasoning_bin")
    bin_idx                     : int
    query                       : str
    gold_answer                 : str
    entropy                     : float | None  (joined from baseline)
    uncertainty                 : float | None
    category                    : str | None
    step                        : int   (reasoning step index within the CoT)
    turn                        : int   (pressure turn; 0 = baseline)
    belief_entropy              : float (Shannon entropy over beliefs across runs)
    cluster_entropy             : float
    semantic_spread             : float
    majority_belief             : str
    majority_is_correct         : bool
    mean_self_reported_confidence: float
    cluster_drift               : float
    divergence_turn             : int | None
    belief_shift_turn           : int | None
    """
    rows = []

    available = sorted(os.listdir(reasoning_dir))
    if models:
        available = [m for m in available if m in models]

    for model in available:
        entropy_map = _load_entropy_map(os.path.join(reasoning_dir, model))
        if not entropy_map:
            entropy_map = _load_entropy_map(os.path.join(EXPERIMENT_OUT, model))

        for trace_type in trace_types:
            bin_dir = os.path.join(reasoning_dir, model, trace_type)
            if not os.path.isdir(bin_dir):
                continue

            for pkl_path in sorted(glob(os.path.join(bin_dir, "bin_*_cross_turn.pkl"))):
                bin_idx = int(os.path.basename(pkl_path).split("_")[1])
                with open(pkl_path, "rb") as f:
                    cross_turn_rows = pickle.load(f)

                for ct in cross_turn_rows:
                    query = ct.get("query", "")
                    em = entropy_map.get(query, {})
                    rows.append({
                        "model":                        model,
                        "trace_type":                   trace_type,
                        "bin_idx":                      bin_idx,
                        "query":                        query,
                        "gold_answer":                  ct.get("gold_answer"),
                        "entropy":                      em.get("entropy"),
                        "uncertainty":                  em.get("uncertainty"),
                        "category":                     em.get("category"),
                        "step":                         ct.get("step"),
                        "turn":                         ct.get("turn"),
                        "belief_entropy":               ct.get("belief_entropy"),
                        "cluster_entropy":              ct.get("cluster_entropy"),
                        "semantic_spread":              ct.get("semantic_spread"),
                        "majority_belief":              ct.get("majority_belief"),
                        "majority_is_correct":          ct.get("majority_is_correct"),
                        "mean_self_reported_confidence": ct.get("mean_self_reported_confidence"),
                        "cluster_drift":                ct.get("cluster_drift"),
                        "divergence_turn":              ct.get("divergence_turn"),
                        "belief_shift_turn":            ct.get("belief_shift_turn"),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["certain"] = (df["entropy"] == 0.0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building flat sycophancy df (RQ2)...")
    flat = build_flat_df()
    print(f"  Shape: {flat.shape}")
    print(f"  Models: {flat['model'].unique().tolist()}")
    print(f"  Turns:  {sorted(flat['turn'].unique().tolist())}")
    print(f"  Columns: {flat.columns.tolist()}")
    print()

    print("Common GPQA queries across all models:")
    common = get_common_queries("gpqa_diamond")
    print(f"  {len(common)} questions")

    print("\nBuilding flat df restricted to common GPQA queries...")
    flat_restricted = build_flat_df(restrict_queries={"gpqa_diamond": common})
    gpqa = flat_restricted[flat_restricted["dataset"] == "gpqa_diamond"]
    print(gpqa.groupby("model")["query"].nunique().to_string())

    print("\nBuilding reasoning trajectory df (RQ4)...")
    reasoning = build_reasoning_df()
    print(f"  Shape: {reasoning.shape}")
    print(f"  Models: {reasoning['model'].unique().tolist()}")
    print(f"  Trace types: {reasoning['trace_type'].unique().tolist()}")
    print(f"  Columns: {reasoning.columns.tolist()}")
