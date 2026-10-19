"""
judge_confidence_retrieve.py — Pulls results for a completed judge batch,
merges the independent judge's confidence back onto the trace mapping
(self-reported confidence + actual correctness), and reports:
  1. whether the judge's own P(correct) is calibrated against real
     correctness (a sanity check the judge signal is meaningful at all)
  2. per model/dataset, how the judge's confidence compares to the
     self-reported confidence -- the actual robustness-check numbers for
     app:calibration / RQ4.

Usage: python analysis/judge_confidence_retrieve.py --scope headline
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["headline", "full"], default="headline")
    args = ap.parse_args()

    client = Anthropic()
    info = json.load(open(os.path.join(pp.OUT, f"judge_batch_id_{args.scope}.json")))
    batch = client.messages.batches.retrieve(info["batch_id"])
    print(f"status={batch.processing_status} counts={batch.request_counts}")
    if batch.processing_status != "ended":
        print("not finished yet, exiting")
        return

    scores, errors = {}, []
    for result in client.messages.batches.results(info["batch_id"]):
        trace_id = int(result.custom_id.split("_", 1)[1])
        if result.result.type == "succeeded":
            text = next((b.text for b in result.result.message.content if b.type == "text"), None)
            try:
                conf = json.loads(text)["confidence"]
                scores[trace_id] = conf
            except Exception as e:
                errors.append((trace_id, f"parse: {e}"))
        else:
            errors.append((trace_id, result.result.type))

    print(f"succeeded+parsed: {len(scores):,}  failed: {len(errors):,}")
    if errors:
        print("sample failures:", errors[:10])

    df = pd.read_pickle(os.path.join(pp.OUT, f"judge_confidence_{args.scope}.pkl"))
    df["judge_conf"] = df["trace_id"].map(scores)
    out_path = os.path.join(pp.OUT, f"judge_confidence_{args.scope}_results.pkl")
    df.to_pickle(out_path)
    print(f"wrote {out_path}")

    valid = df.dropna(subset=["judge_conf", "correct"])
    print(f"\nrows with both judge score and correctness label: {len(valid):,}")

    # 1. is the judge itself calibrated? bin judge_conf into deciles, check
    #    observed correctness rate rises with judge_conf (Brier score too)
    brier = ((valid["judge_conf"] / 100 - valid["correct"].astype(float)) ** 2).mean()
    rho_judge, p_judge = stats.spearmanr(valid["judge_conf"], valid["correct"], nan_policy="omit")
    print(f"judge calibration: Brier={brier:.4f}  Spearman(judge_conf, correct)={rho_judge:+.3f} (p={p_judge:.3g})")

    # 2. per model/dataset: self-reported conf vs judge conf vs actual correctness
    print("\nper (model, dataset): mean self-reported conf, mean judge conf, mean correct")
    g = valid.groupby(["model", "dataset"]).agg(
        n=("judge_conf", "size"),
        self_conf=("self_conf", lambda s: 100 * s.mean()),
        judge_conf=("judge_conf", "mean"),
        correct_pct=("correct", lambda s: 100 * s.mean()),
    )
    print(g.to_string(float_format=lambda x: f"{x:.1f}"))

    # 3. correlation between self-reported and judge confidence, per cell
    print("\nSpearman(self_conf, judge_conf) per (model, dataset):")
    for (model, dataset), sub in valid.groupby(["model", "dataset"]):
        if sub["self_conf"].notna().sum() < 2:
            continue
        rho, p = stats.spearmanr(sub["self_conf"], sub["judge_conf"], nan_policy="omit")
        print(f"  {model:15s} {dataset:15s} n={len(sub):5,}  rho={rho:+.3f}  p={p:.3g}")

    g.to_csv(os.path.join(pp.OUT, f"judge_confidence_{args.scope}_summary.csv"))


if __name__ == "__main__":
    main()
