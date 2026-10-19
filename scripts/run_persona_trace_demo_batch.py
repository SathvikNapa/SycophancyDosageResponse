"""
run_persona_trace_demo_batch.py — Batch-inference version of
run_persona_trace_demo.py. Each example is a genuinely sequential 7-turn
conversation (T0 baseline + 6 pressure doses), so this can't be one batch
submission: round t's request needs the model's own real response from
round t-1. Instead, this runs 7 sequential rounds, each one batching ALL
examples' turn-t request together (Anthropic Batches for Claude Sonnet,
OpenAI Batches for GPT-5.4 Mini), polling each round to completion before
building the next. Still gets the 50% batch discount; just needs 7 rounds
instead of 1.

Produces the same output shape as run_persona_trace_demo.py
(experiment_out/persona_trace_demo_results_<persona>.pkl), so existing
downstream code (the paper's example-trace table) can read it unchanged.

Usage: python scripts/run_persona_trace_demo_batch.py --persona novice
"""
import argparse
import os
import pickle
import sys
import time

import anthropic
import openai
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from run_persona_trace_demo import (
    DOSE_TEXT_BY_CATEGORY, MODELS, PERSONA_SENTENCES, select_examples, results_path,
)
from sycophancy.config import MODELS as MODEL_API_IDS, SYSTEM_MSG
from sycophancy.generator import extract_letter
from sycophancy.sycophancy import build_dose_user_prompt

load_dotenv()

N = 10  # N_PER_STRATUM, matching the original script's default
POLL_S = 20


def persona_system_content(persona_key: str) -> str:
    return f"{SYSTEM_MSG['content']}\n\n{PERSONA_SENTENCES[persona_key]}"


def build_examples(persona_key: str):
    """One entry per (model, stratum, example); each carries its own growing
    `messages` list and per-turn record, mirroring run_sycophancy_conversation
    but driven externally, round by round."""
    examples = []
    for model in MODELS:
        strata = select_examples(model, N)
        for stratum, runs in strata.items():
            for run in runs:
                dose_categories = run["turn_categories"][1:]
                dose_statements = [DOSE_TEXT_BY_CATEGORY[c] for c in dose_categories]
                examples.append({
                    "model": model, "stratum": stratum, "run": run,
                    "dose_categories": dose_categories, "dose_statements": dose_statements,
                    "messages": [{"role": "system", "content": persona_system_content(persona_key)},
                                 {"role": "user", "content": run["prompt"]}],
                    "raw_turns": [], "parsed_turns": [],
                })
    return examples


def anthropic_model_id(model_key):
    return MODEL_API_IDS[model_key].removeprefix("anthropic/")


def openai_model_id(model_key):
    return MODEL_API_IDS[model_key].removeprefix("openai/")


def run_round(examples, turn_idx, a_client, o_client):
    """turn_idx=0 is T0 (no dose appended yet); turn_idx=1..6 appends dose
    turn_idx-1 first. Submits one Anthropic batch (Claude Sonnet examples)
    and one OpenAI batch (GPT-5.4 Mini examples), polls both, applies results."""
    active = [e for e in examples if turn_idx == 0 or len(e["raw_turns"]) == turn_idx]
    for e in active:
        if turn_idx > 0:
            ds = e["dose_statements"][turn_idx - 1]
            dose_content = build_dose_user_prompt(ds, e["run"]["wrong_answer_text"])
            e["messages"].append({"role": "user", "content": dose_content})

    a_examples = [e for e in active if e["model"] == "ClaudeSonnet"]
    o_examples = [e for e in active if e["model"] == "GPT5_4Mini"]
    print(f"  round {turn_idx}: {len(a_examples)} Claude Sonnet, {len(o_examples)} GPT-5.4 Mini", flush=True)

    a_batch_id = None
    if a_examples:
        reqs = [
            anthropic.types.messages.batch_create_params.Request(
                custom_id=str(i),
                params=anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                    model=anthropic_model_id(e["model"]), max_tokens=1024, messages=e["messages"][1:],
                    system=e["messages"][0]["content"],
                ),
            )
            for i, e in enumerate(a_examples)
        ]
        batch = a_client.messages.batches.create(requests=reqs)
        a_batch_id = batch.id

    o_batch_id = None
    if o_examples:
        import io
        lines = []
        for i, e in enumerate(o_examples):
            lines.append({
                "custom_id": str(i),
                "method": "POST", "url": "/v1/chat/completions",
                "body": {"model": openai_model_id(e["model"]), "messages": e["messages"],
                         "max_completion_tokens": 1024},
            })
        import json
        buf = io.BytesIO(("\n".join(json.dumps(l) for l in lines)).encode())
        f = o_client.files.create(file=("round.jsonl", buf), purpose="batch")
        batch = o_client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
        o_batch_id = batch.id

    # poll both to completion
    a_results, o_results = {}, {}
    while True:
        a_done = a_batch_id is None
        o_done = o_batch_id is None
        if a_batch_id:
            b = a_client.messages.batches.retrieve(a_batch_id)
            a_done = b.processing_status == "ended"
        if o_batch_id:
            b = o_client.batches.retrieve(o_batch_id)
            o_done = b.status in ("completed", "failed", "expired", "cancelled")
        if a_done and o_done:
            break
        time.sleep(POLL_S)

    if a_batch_id:
        for r in a_client.messages.batches.results(a_batch_id):
            if r.result.type == "succeeded":
                text = next((b.text for b in r.result.message.content if b.type == "text"), "")
                a_results[r.custom_id] = text
            else:
                a_results[r.custom_id] = ""
    if o_batch_id:
        b = o_client.batches.retrieve(o_batch_id)
        content = o_client.files.content(b.output_file_id).read().decode()
        import json
        for line in content.splitlines():
            row = json.loads(line)
            body = row.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            o_results[row["custom_id"]] = text or ""

    for i, e in enumerate(a_examples):
        raw = a_results.get(str(i), "")
        e["messages"].append({"role": "assistant", "content": raw})
        e["raw_turns"].append(raw)
        e["parsed_turns"].append(extract_letter(raw))
    for i, e in enumerate(o_examples):
        raw = o_results.get(str(i), "")
        e["messages"].append({"role": "assistant", "content": raw})
        e["raw_turns"].append(raw)
        e["parsed_turns"].append(extract_letter(raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True, choices=list(PERSONA_SENTENCES.keys()))
    args = ap.parse_args()

    examples = build_examples(args.persona)
    print(f"{len(examples)} total examples across {len(MODELS)} models", flush=True)

    a_client = anthropic.Anthropic()
    o_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for turn_idx in range(7):  # T0..T6
        run_round(examples, turn_idx, a_client, o_client)

    out = {}
    for e in examples:
        wrong_letter = (e["run"]["wrong_answer_letter"] or "").strip().upper()
        first_wrong = next((i for i, p in enumerate(e["parsed_turns"]) if p == wrong_letter), None)
        out.setdefault(e["model"], {}).setdefault(e["stratum"], []).append({
            "model": e["model"], "query": e["run"]["query"], "persona": args.persona,
            "baseline_raw_turns": e["run"]["raw_turns"], "baseline_first_wrong_turn": e["run"]["first_wrong_turn"],
            "adversarial_raw_turns": e["raw_turns"], "adversarial_parsed_turns": e["parsed_turns"],
            "adversarial_first_wrong_turn": first_wrong,
        })

    out_path = results_path(args.persona)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
