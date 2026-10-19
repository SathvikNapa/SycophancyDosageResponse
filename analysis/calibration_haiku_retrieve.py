"""
calibration_haiku_retrieve.py — Pulls results from the 5 calibration
batches, parses BELIEF/CONFIDENCE with the exact regex calibrator.py uses
for GPT-5.4 Mini, and compares Claude-Haiku-as-calibrator against the
original GPT-5.4-Mini-based values already used throughout the paper:
  1. final-step confidence (what RQ4's Conf_T{t}/Gap_t actually consumes)
  2. all-step calibration reliability (ECE/Brier), matching app:calibration

Usage: python analysis/calibration_haiku_retrieve.py
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

load_dotenv()

_BELIEF_RE = re.compile(r"BELIEF\s*:\s*([A-J])", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*:\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def parse_calibration(text):
    bm = _BELIEF_RE.search(text or "")
    cm = _CONF_RE.search(text or "")
    belief = bm.group(1).upper() if bm else None
    conf = float(cm.group(1)) if cm else None
    return belief, conf


def ece(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    total = len(probs)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        err += mask.sum() / total * abs(probs[mask].mean() - outcomes[mask].mean())
    return err


def main():
    client = Anthropic()
    ids = json.load(open(os.path.join(pp.OUT, "calibration_haiku_batch_ids.json")))["batch_ids"]

    all_ended = True
    for bid in ids:
        b = client.messages.batches.retrieve(bid)
        print(f"{bid}: {b.processing_status} {b.request_counts}")
        if b.processing_status != "ended":
            all_ended = False
    if not all_ended:
        print("not all batches finished, exiting")
        return

    beliefs, confs, errors = {}, {}, []
    for bid in ids:
        for result in client.messages.batches.results(bid):
            trace_id, step = (int(x) for x in result.custom_id[2:].split("_"))
            if result.result.type == "succeeded":
                text = next((b.text for b in result.result.message.content if b.type == "text"), "")
                belief, conf = parse_calibration(text)
                beliefs[(trace_id, step)] = belief
                confs[(trace_id, step)] = conf
            else:
                errors.append((trace_id, step, result.result.type))

    print(f"\nparsed: {len(confs):,}  failed: {len(errors):,}")
    n_noconf = sum(1 for v in confs.values() if v is None)
    print(f"responses with no parseable CONFIDENCE: {n_noconf:,} of {len(confs):,}")

    df = pd.read_pickle(os.path.join(pp.OUT, "calibration_haiku_headline.pkl"))

    # final-step comparison (what RQ4 actually uses)
    final_conf, final_belief = [], []
    for _, row in df.iterrows():
        last_step = len(row["step_texts"]) - 1
        final_conf.append(confs.get((row["trace_id"], last_step)))
        final_belief.append(beliefs.get((row["trace_id"], last_step)))
    df["haiku_final_conf"] = final_conf
    df["haiku_final_belief"] = final_belief

    out_path = os.path.join(pp.OUT, "calibration_haiku_results.pkl")
    df.to_pickle(out_path)
    print(f"wrote {out_path}")

    valid = df.dropna(subset=["haiku_final_conf", "orig_self_conf"])
    print(f"\nrows with both Haiku final-step conf and original GPT-5.4-Mini conf: {len(valid):,}")
    print("\nper (model, dataset): mean orig (GPT-5.4 Mini) conf vs mean Haiku conf, mean correct")
    g = valid.groupby(["model", "dataset"]).agg(
        n=("haiku_final_conf", "size"),
        orig_conf=("orig_self_conf", "mean"),
        haiku_conf=("haiku_final_conf", "mean"),
        correct_pct=("correct", lambda s: 100 * s.mean()),
    )
    print(g.to_string(float_format=lambda x: f"{x:.1f}"))
    g.to_csv(os.path.join(pp.OUT, "calibration_haiku_summary.csv"))

    print("\nSpearman(orig_conf, haiku_conf) per (model, dataset):")
    for (model, dataset), sub in valid.groupby(["model", "dataset"]):
        rho, p = stats.spearmanr(sub["orig_self_conf"], sub["haiku_final_conf"], nan_policy="omit")
        print(f"  {model:15s} {dataset:15s} n={len(sub):5,}  rho={rho:+.3f}  p={p:.3g}")

    # all-step ECE/Brier, matching app:calibration's own methodology
    step_rows = []
    for _, row in df.iterrows():
        for ell in range(len(row["step_texts"])):
            c = confs.get((row["trace_id"], ell))
            b = beliefs.get((row["trace_id"], ell))
            if c is None or b is None:
                continue
            step_rows.append({"model": row["model"], "dataset": row["dataset"],
                               "conf": c / 100.0, "hit": int(b == row["gold"])})
    sdf = pd.DataFrame(step_rows)
    sdf.to_pickle(os.path.join(pp.OUT, "calibration_haiku_steps.pkl"))
    print(f"\nstep-level rows for ECE/Brier: {len(sdf):,}")
    brier = ((sdf["conf"] - sdf["hit"]) ** 2).mean()
    e = ece(sdf["conf"].values, sdf["hit"].values)
    print(f"Claude-Haiku-as-calibrator: Brier={brier:.4f}  ECE={e:.4f}  (pooled, all headline steps)")
    print("compare against GPT-5.4 Mini's reported ECE in [0.26, 0.46], Brier in [0.28, 0.44] (app:calibration)")


if __name__ == "__main__":
    main()
