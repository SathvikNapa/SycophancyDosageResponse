"""
calibration_haiku_merge_redo.py — Pulls results from the redo batches
(calibration_haiku_redo_batch_ids.json), parses with the same BELIEF/
CONFIDENCE regex, and replaces the originally-truncated entries in the
step-level confidence/belief maps -- producing one clean, fully-covered
dataset from the original 57% (never touched) + the redone 43%.

Usage: python analysis/calibration_haiku_merge_redo.py
"""
import json
import os
import re
import sys

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

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


def main():
    client = Anthropic()

    # 1. original batches -> full belief/conf maps (as in calibration_haiku_retrieve.py)
    orig_ids = json.load(open(os.path.join(pp.OUT, "calibration_haiku_batch_ids.json")))["batch_ids"]
    beliefs, confs = {}, {}
    for bid in orig_ids:
        for result in client.messages.batches.results(bid):
            trace_id, step = (int(x) for x in result.custom_id[2:].split("_"))
            if result.result.type == "succeeded":
                text = next((b.text for b in result.result.message.content if b.type == "text"), "")
                belief, conf = parse_calibration(text)
                beliefs[(trace_id, step)] = belief
                confs[(trace_id, step)] = conf

    n_before = sum(1 for v in confs.values() if v is not None)
    print(f"original run: {n_before:,} of {len(confs):,} parsed")

    # 2. redo batches -> OVERWRITE the previously-missing entries
    redo_ids = json.load(open(os.path.join(pp.OUT, "calibration_haiku_redo_batch_ids.json")))["batch_ids"]
    n_redo_parsed, n_redo_total = 0, 0
    for bid in redo_ids:
        for result in client.messages.batches.results(bid):
            trace_id, step = (int(x) for x in result.custom_id[2:].split("_"))
            n_redo_total += 1
            if result.result.type == "succeeded":
                text = next((b.text for b in result.result.message.content if b.type == "text"), "")
                belief, conf = parse_calibration(text)
                beliefs[(trace_id, step)] = belief
                confs[(trace_id, step)] = conf
                if conf is not None:
                    n_redo_parsed += 1

    print(f"redo run: {n_redo_parsed:,} of {n_redo_total:,} newly parsed")
    n_after = sum(1 for v in confs.values() if v is not None)
    print(f"combined: {n_after:,} of {len(confs):,} parsed ({100*n_after/len(confs):.1f}%)")

    # 3. rebuild final-step comparison + step-level rows with the merged maps
    df = pd.read_pickle(os.path.join(pp.OUT, "calibration_haiku_headline.pkl"))
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
    print(f"step-level rows: {len(sdf):,} (was 62,692 before the redo)")

    brier = ((sdf["conf"] - sdf["hit"]) ** 2).mean()
    import numpy as np
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
    e = ece(sdf["conf"].values, sdf["hit"].values)
    print(f"Claude-Haiku-as-calibrator (merged): Brier={brier:.4f}  ECE={e:.4f}")
    print("compare against GPT-5.4 Mini's reported ECE in [0.26, 0.46], Brier in [0.28, 0.44] (app:calibration)")


if __name__ == "__main__":
    main()
