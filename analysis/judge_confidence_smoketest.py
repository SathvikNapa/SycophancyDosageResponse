"""
judge_confidence_smoketest.py — Sends a handful of real (non-batch) calls
to validate the judge prompt/schema and, critically, to measure ACTUAL
token usage before submitting the real batch. The earlier cost estimate
assumed ~10 output tokens per call with no thinking; if adaptive thinking
runs by default and is billed as output tokens, the real cost could be
materially higher, so this must be checked before spending on the batch.

Usage: python analysis/judge_confidence_smoketest.py
"""
import json
import os

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

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


def build_user_content(row):
    opts = row["options"]
    opts_text = "\n".join(f"{k}. {v}" for k, v in opts.items()) if isinstance(opts, dict) else str(opts)
    return (
        f"Question:\n{row['query']}\n\nOptions:\n{opts_text}\n\n"
        f"Reasoning attempt:\n{row['redacted_text']}"
    )


def judge_one(row):
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_content(row)}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage


def main():
    df = pd.read_pickle("analysis/paper_out/judge_confidence_input.pkl")
    headline = df[((df.model == "ClaudeHaiku") & (df.dataset == "gpqa_diamond")) |
                  ((df.model == "ClaudeHaiku") & (df.dataset == "hle")) |
                  ((df.model == "ClaudeSonnet") & (df.dataset == "hle"))]
    sample = headline.groupby(["model", "dataset"], group_keys=False).apply(
        lambda g: g.sample(min(4, len(g)), random_state=0)
    )

    in_rate, out_rate = 1.00, 5.00  # standard (non-batch) Haiku 4.5 rates for this smoke test
    total_cost = 0.0
    n_ok, n_fail = 0, 0
    for _, row in sample.iterrows():
        try:
            result, usage = judge_one(row)
            conf = result["confidence"]
            valid = isinstance(conf, int) and 0 <= conf <= 100
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
            continue
        if not valid:
            print(f"FAIL: confidence out of range/type: {conf!r}", flush=True)
            n_fail += 1
            continue
        n_ok += 1
        cost = usage.input_tokens * in_rate / 1e6 + usage.output_tokens * out_rate / 1e6
        total_cost += cost
        print(f"OK  self_conf={row['self_conf']}  correct={row['correct']}  "
              f"judge={conf}  in_tok={usage.input_tokens}  "
              f"out_tok={usage.output_tokens}  cost=${cost:.5f}", flush=True)

    print(f"\nformat gate: {n_ok}/{len(sample)} valid responses ({n_fail} failures)")
    if n_ok:
        print(f"avg cost/call (standard, non-batch rate): ${total_cost/n_ok:.5f}")
        print(f"=> batch rate (50% off): ${total_cost/n_ok/2:.5f} per call")
        print(f"=> projected headline batch (10,756 traces): ${total_cost/n_ok/2*10756:.2f}")
    print("GATE PASSED" if n_fail == 0 else "GATE FAILED")


if __name__ == "__main__":
    main()
