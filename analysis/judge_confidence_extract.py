"""
judge_confidence_extract.py — Pulls every (model, dataset, query, run, turn)
CoT trace from the canonical R-run pipeline, redacts the self-reported
CONFIDENCE:N% clause (keeping the stated belief/answer letter), and writes
one row per trace to analysis/paper_out/judge_confidence_input.pkl for an
independent Claude Sonnet judge to score.

Why redact only the confidence clause and not the answer letter: the check
this supports is "does an independent reader of the same reasoning, told
what the model concluded, agree that conclusion is well-supported" --
not "can text alone, with no stated conclusion, predict correctness."
Stripping the number but keeping the letter is what makes this a genuine
test of the self-reported CONFIDENCE figure specifically, rather than a
different and weaker claim about hidden conclusions.

paper_pipeline.load_cell() does not carry raw_text (it only keeps the
already-parsed answer/belief/conf fields), so this script re-reads the
same reasoning_calibrated_bin/*.pkl files directly instead of going
through load_cell.

Usage: python analysis/judge_confidence_extract.py
"""
import glob
import os
import pickle
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

_CONF_CLAUSE_RE = re.compile(
    r"[^\S\r\n]*\|?[^\S\r\n]*CONFIDENCE\s*:\s*\d+(?:\.\d+)?\s*(?:-\s*\d+(?:\.\d+)?)?\s*%",
    re.IGNORECASE,
)


def redact(raw_text):
    return _CONF_CLAUSE_RE.sub("", raw_text)


def options_map(model, dataset):
    d = pp.cell_dir(model, dataset)
    path = os.path.join(d, "base_experiment_metadata.pkl")
    meta = pickle.load(open(path, "rb"))
    return {m["query"]: m["options"] for m in meta}


def extract_cell(model, dataset, pool):
    paths = sorted(glob.glob(os.path.join(pp.cell_dir(model, dataset), "reasoning_calibrated_bin", "bin_*_reasoning.pkl")))
    if not paths:
        return []
    opts_map = options_map(model, dataset)
    rows = []
    for p in paths:
        trajs = pickle.load(open(p, "rb"))
        for traj in trajs:
            if traj.query not in pool:
                continue
            opts = opts_map.get(traj.query)
            for t, turn in enumerate(traj.raw_traces[:pp.N_TURNS]):
                for r, tr in enumerate(turn[:pp.R]):
                    if not tr.raw_text:
                        continue
                    conf = tr.steps[-1].confidence if tr.steps else None
                    rows.append({
                        "model": model, "dataset": dataset, "query": traj.query,
                        "options": opts, "gold": traj.gold_answer, "run": r, "turn": t,
                        "self_conf": conf / 100.0 if conf is not None else None,
                        "final_answer": tr.final_answer,
                        "correct": (tr.final_answer == traj.gold_answer) if tr.final_answer else None,
                        "redacted_text": redact(tr.raw_text),
                    })
    return rows


def main():
    all_rows = []
    for dataset in pp.DATASETS:
        pool = pp.canonical_pool(dataset)
        for model in pp.MODELS:
            rows = extract_cell(model, dataset, pool)
            print(f"{model:15s} {dataset:15s} {len(rows):6,} traces", flush=True)
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    print(f"\nTotal traces: {len(df):,}")
    out_path = os.path.join(pp.OUT, "judge_confidence_input.pkl")
    df.to_pickle(out_path)
    print(f"wrote {out_path}")

    # sanity check: redaction actually removed something in most rows,
    # and no row still contains a bare CONFIDENCE: N% clause
    leaked = df["redacted_text"].str.contains(r"CONFIDENCE\s*:\s*\d+(?:\.\d+)?\s*(?:-\s*\d+(?:\.\d+)?)?\s*%", case=False, regex=True, na=False)
    print(f"rows with a CONFIDENCE:N% clause surviving redaction: {int(leaked.sum())} (should be 0)")


if __name__ == "__main__":
    main()
