"""
calibration_haiku_batch.py — Submits the step-by-step calibration replication
to the Anthropic Message Batches API using Claude Haiku 4.5 as the calibrator,
reusing the EXACT prompts src/sycophancy/calibrator.py uses for GPT-5.4 Mini
(CALIBRATOR_SYSTEM_MSG, CALIBRATOR_STEP_PROMPT), so the only thing that
differs from the original run is the calibrator model.

One request per (trace, step) -- 110,566 requests across the 10,878 headline
traces (mean 10.16 steps/trace) -- split across multiple batches since the
Batches API caps each batch at 100,000 requests.

Usage:
    python analysis/calibration_haiku_batch.py --dry-run
    python analysis/calibration_haiku_batch.py
"""
import argparse
import json
import os
import sys

import pandas as pd
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp
from sycophancy.config import CALIBRATOR_STEP_PROMPT, CALIBRATOR_SYSTEM_MSG

load_dotenv()

MAX_PER_BATCH = 25_000  # 100k requests hit the 256MB-per-batch size cap; stay well under it


def build_requests(df):
    requests = []
    for _, row in df.iterrows():
        options_text = "\n".join(row["options"])
        steps = row["step_texts"]
        for ell in range(len(steps)):
            steps_so_far = steps[: ell + 1]
            steps_text = "\n\n".join(f"Step {i + 1}:\n{t}" for i, t in enumerate(steps_so_far))
            prompt = CALIBRATOR_STEP_PROMPT.format(
                question=row["query"], options=options_text,
                step_num=ell + 1, steps_so_far=steps_text,
            )
            custom_id = f"c_{row['trace_id']}_{ell}"
            requests.append(Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model="claude-haiku-4-5",
                    max_tokens=64,
                    system=CALIBRATOR_SYSTEM_MSG["content"],
                    messages=[{"role": "user", "content": prompt}],
                ),
            ))
    return requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_pickle(os.path.join(pp.OUT, "calibration_haiku_headline.pkl"))
    print(f"{len(df):,} traces, {df['step_texts'].apply(len).sum():,} total step-calls")

    requests = build_requests(df)
    print(f"built {len(requests):,} requests")

    chunks = [requests[i:i + MAX_PER_BATCH] for i in range(0, len(requests), MAX_PER_BATCH)]
    print(f"splitting into {len(chunks)} batch(es)")

    if args.dry_run:
        print("dry run: not submitting")
        return

    client = Anthropic()
    batch_ids = []
    for i, chunk in enumerate(chunks):
        batch = client.messages.batches.create(requests=chunk)
        print(f"batch {i}: id={batch.id} status={batch.processing_status} n={len(chunk):,}")
        batch_ids.append(batch.id)

    id_path = os.path.join(pp.OUT, "calibration_haiku_batch_ids.json")
    json.dump({"batch_ids": batch_ids, "n_requests": len(requests)}, open(id_path, "w"))
    print(f"wrote {id_path}")


if __name__ == "__main__":
    main()
