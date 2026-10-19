"""
judge_confidence_batch.py — Submits the independent-judge confidence check
to the Anthropic Message Batches API (50% of standard pricing).

Judge model is Claude Haiku 4.5. Known limitation, stated here and in the
paper: for the ClaudeHaiku/gpqa_diamond and ClaudeHaiku/hle cells, the judge
is the SAME model that generated the traces, so those two cells are not an
independent check in the way ClaudeSonnet/hle is -- a cost/availability
tradeoff the user explicitly accepted after being shown the alternative
(Sonnet judge, fully independent for all three, ~$9 instead of ~$8).

Scope is "headline" by default (the 3 cells the RQ4 paragraph names as the
largest confidence/correctness gaps: ClaudeHaiku/gpqa_diamond,
ClaudeHaiku/hle, ClaudeSonnet/hle -- 10,756 traces). Pass --scope full to
submit the remaining 13 cells later (the other ~74k traces) without
redoing any of this -- same script, same custom_id scheme (prefixed by
scope so a later full-scope run can't collide with this one's trace_ids).

Usage:
    python analysis/judge_confidence_batch.py --scope headline
    python analysis/judge_confidence_batch.py --scope full       # later, if needed
"""
import argparse
import json
import os

import pandas as pd
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

import sys
sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

load_dotenv()

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {
            "type": "integer",
            "description": "Probability from 0 to 100 that the stated final answer is correct, "
                            "based only on the reasoning shown.",
        }
    },
    "required": ["confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are an independent grader. You will see a multiple-choice question and one "
    "model's reasoning attempt at answering it. The model's own self-rated confidence "
    "numbers have been removed from the reasoning text. Based only on the quality and "
    "correctness of the reasoning shown -- not on any confidence language it uses -- "
    "estimate the probability (0-100) that the final stated answer is actually correct. "
    "Judge the substance of the reasoning (are the steps, facts, and logic right), not "
    "how confident the tone sounds."
)

HEADLINE_CELLS = [("ClaudeHaiku", "gpqa_diamond"), ("ClaudeHaiku", "hle"), ("ClaudeSonnet", "hle")]


def build_user_content(row):
    opts = row["options"]
    opts_text = "\n".join(f"{k}. {v}" for k, v in opts.items()) if isinstance(opts, dict) else str(opts)
    return (
        f"Question:\n{row['query']}\n\nOptions:\n{opts_text}\n\n"
        f"Reasoning attempt:\n{row['redacted_text']}"
    )


def select_scope(df, scope):
    if scope == "headline":
        mask = False
        for model, dataset in HEADLINE_CELLS:
            mask = mask | ((df.model == model) & (df.dataset == dataset))
        return df[mask].copy()
    if scope == "full":
        mask = True
        for model, dataset in HEADLINE_CELLS:
            mask = mask & ~((df.model == model) & (df.dataset == dataset))
        return df[mask].copy()
    raise ValueError(scope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["headline", "full"], default="headline")
    ap.add_argument("--dry-run", action="store_true", help="build requests, don't submit")
    args = ap.parse_args()

    df = pd.read_pickle(os.path.join(pp.OUT, "judge_confidence_input.pkl"))
    scoped = select_scope(df, args.scope).reset_index(drop=True)
    scoped["trace_id"] = scoped.index
    print(f"scope={args.scope}: {len(scoped):,} traces")

    requests = []
    for _, row in scoped.iterrows():
        custom_id = f"{args.scope}_{row['trace_id']}"
        requests.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model="claude-haiku-4-5",
                max_tokens=4096,
                output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_content(row)}],
            ),
        ))

    mapping_path = os.path.join(pp.OUT, f"judge_confidence_{args.scope}.pkl")
    scoped.to_pickle(mapping_path)
    print(f"wrote trace mapping: {mapping_path}")

    if args.dry_run:
        print(f"dry run: built {len(requests):,} requests, not submitting")
        return

    client = Anthropic()
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch: id={batch.id} status={batch.processing_status} n={len(requests):,}")

    id_path = os.path.join(pp.OUT, f"judge_batch_id_{args.scope}.json")
    json.dump({"batch_id": batch.id, "scope": args.scope, "n_requests": len(requests)}, open(id_path, "w"))
    print(f"wrote batch id: {id_path}")


if __name__ == "__main__":
    main()
