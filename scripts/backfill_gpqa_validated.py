"""
backfill_gpqa_validated.py — Targeted, low-risk backfill of ClaudeSonnet's
GPQA-Diamond reasoning_calibrated_bin data.

Context
-------
A large fraction of GPQA-Diamond's reasoning_calibrated_bin traces are
completely empty (raw_text == "") — the original collection run hit a
sustained failure window (confirmed: credit-balance exhaustion on sibling
pipelines) and nothing ever retried. Continuing a broken dialogue safely
requires knowing which wrong answer was being pushed in the dose turns —
but that choice is made once per item via a random draw that is NEVER
stored in ReasoningTrace/UncertaintyTrajectory, only implied by the
conversation itself. RNG-replay reconstruction was tested exhaustively
(3,000 candidate seeds against the real canonical item order from
entropy.py's own binning code) and tops out at ~53% match against
ground truth — not recoverable that way.

What IS recoverable: the pushed answer is sometimes echoed back by the
model in its own reply ("knowing the answer is D", "the answer should be
A"). Where 2+ independent traces for the same item agree on the same
letter with zero contradictions, we trust it. Only 30 of GPQA's 198
items clear that bar (29 in bin0, 1 in bin1) — by design, this script
does NOT fill every gap, only the ones with real corroborating evidence.
Everything else is left untouched.

Policy per validated item
--------------------------
For each affected sample (any blank turn):
  - Turns before the first blank are kept as-is (their raw_text + the
    validated wrong-answer text are used to reconstruct the exact prior
    message history).
  - If T0 itself is blank, it's simply regenerated fresh — no wrong-answer
    dependency there at all.
  - Turns from the first blank onward are regenerated via fresh API calls
    (now benefiting from generator.py's process-wide rate limiter +
    retry-on-transport-error, the same fix that drove the persona
    experiments to 0 errors this session), then calibrated exactly as
    run_reasoning_calibrated.py does.

Output
------
Backs up bin_0/1_reasoning.pkl (+ summary/cross_turn) before writing,
then writes patched versions with ONLY the validated items' blank
samples filled in. Every other item/sample is byte-identical to before.

Usage
-----
    python scripts/backfill_gpqa_validated.py --dry_run   # show scope only
    python scripts/backfill_gpqa_validated.py             # actually run
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import pickle
import re
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))  # for `from run_reasoning_calibrated import ...`

import numpy as np
from dotenv import load_dotenv

import sycophancy.reasoning_uncertainty as ru
sys.modules["reasoning_uncertainty"] = ru  # pickle compatibility shim (bare module name)

from sycophancy.calibrator import calibrate_trace
from sycophancy.config import DEFAULT_CALIBRATOR_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_N_CLUSTERS, DEFAULT_TIMEOUT_S
from sycophancy.generator import ResponseGenerator
from sycophancy.reasoning_uncertainty import (
    ReasoningTrace,
    UncertaintyTrajectory,
    _load_encoder,
    build_uncertainty_trajectory,
    parse_reasoning_steps_freeform,
    summarise_cross_turn_comparison,
    summarise_trajectory,
)
from sycophancy.sycophancy import option_letter

from run_reasoning_calibrated import build_free_dose_prompt, build_free_prompt, DEFAULT_DOSE_STATEMENTS
from sycophancy.config import COT_FREE_SYSTEM_MSG

MODEL = "ClaudeSonnet"
DATASET_DIR = "experiment_out/ClaudeSonnet/gpqa_diamond"
BIN_DIR = os.path.join(DATASET_DIR, "reasoning_calibrated_bin")
META_PATH = os.path.join(DATASET_DIR, "base_experiment_metadata.pkl")

PATTERNS = [
    re.compile(r"knowing the answer is\s*\**\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"\bthe (?:correct )?answer (?:is|should be)\s*\**\s*([A-J])\b", re.IGNORECASE),
    re.compile(r"feedback that the answer is\s*\**\s*([A-J])\b", re.IGNORECASE),
]


def extract_echoed_letter(raw_text: str) -> Optional[str]:
    cut = len(raw_text)
    for marker in ("**Step", "CURRENT BELIEF", "Step 1"):
        idx = raw_text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    head = raw_text[:cut]
    for pat in PATTERNS:
        m = pat.search(head)
        if m:
            return m.group(1).upper()
    return None


def find_validated_items(trajs: List[UncertaintyTrajectory]) -> Dict[str, str]:
    """query -> confirmed wrong-answer letter, for items with 2+ agreeing,
    zero-contradicting echoed-letter traces."""
    validated: Dict[str, str] = {}
    for traj in trajs:
        if not traj.raw_traces:
            continue
        letters_seen: collections.Counter = collections.Counter()
        n_turns = len(traj.raw_traces)
        for t in range(1, n_turns):
            for tr in traj.raw_traces[t]:
                if tr.raw_text:
                    letter = extract_echoed_letter(tr.raw_text)
                    if letter:
                        letters_seen[letter] += 1
        if len(letters_seen) == 1 and sum(letters_seen.values()) >= 2:
            validated[traj.query] = next(iter(letters_seen))
    return validated


def wrong_answer_text_for(item: dict, letter: str) -> str:
    for opt in item.get("options", []):
        if option_letter(opt) == letter:
            return opt
    raise ValueError(f"letter {letter!r} not found among options for query {item.get('query','')[:60]!r}")


async def regenerate_sample(
    sem: asyncio.Semaphore,
    rg: ResponseGenerator,
    item: dict,
    wrong_text: str,
    existing_traces: List[ReasoningTrace],
    calibrator_model: str,
    timeout_s: float,
    seed: int,
) -> List[ReasoningTrace]:
    question = item.get("query", "")
    options = item.get("options", [])
    gold_answer = (item.get("actual_answer") or "").strip().upper() or None

    good = [bool(t.raw_text) for t in existing_traces]
    first_bad = good.index(False) if False in good else len(good)
    n_turns = len(existing_traces)

    new_traces = list(existing_traces)
    messages: List[dict] = [
        COT_FREE_SYSTEM_MSG,
        {"role": "user", "content": build_free_prompt(question, options)},
    ]

    # Reconstruct the known-good prefix of the conversation verbatim.
    for t in range(0, first_bad):
        if t >= 1:
            ds = DEFAULT_DOSE_STATEMENTS[t - 1]
            messages.append({"role": "user", "content": build_free_dose_prompt(ds, wrong_text)})
        messages.append({"role": "assistant", "content": existing_traces[t].raw_text})

    # Regenerate from the first blank turn onward.
    for t in range(first_bad, n_turns):
        if t >= 1:
            ds = DEFAULT_DOSE_STATEMENTS[t - 1]
            messages.append({"role": "user", "content": build_free_dose_prompt(ds, wrong_text)})

        turn_seed = seed + t * 100
        raw: Optional[str] = None
        async with sem:
            try:
                raw = await rg.acomplete(messages=messages, model=MODEL, timeout_s=timeout_s, seed=turn_seed)
            except Exception:
                raw = None
        raw = raw or ""

        trace = parse_reasoning_steps_freeform(raw, gold_answer=gold_answer)
        trace = await calibrate_trace(
            sem=sem, rg=rg, question=question, options=options, trace=trace,
            calibrator_model=calibrator_model, timeout_s=timeout_s, base_seed=turn_seed,
        )
        new_traces[t] = trace
        messages.append({"role": "assistant", "content": raw})

    return new_traces


async def process_bin(
    bin_idx: int,
    meta_by_query: dict,
    calibrator_model: str,
    timeout_s: float,
    concurrency: int,
    base_seed: int,
    dry_run: bool,
    encoder,
) -> None:
    path = os.path.join(BIN_DIR, f"bin_{bin_idx}_reasoning.pkl")
    with open(path, "rb") as f:
        trajs: List[UncertaintyTrajectory] = pickle.load(f)

    validated = find_validated_items(trajs)
    traj_by_query = {t.query: t for t in trajs}

    tasks = []
    task_meta = []  # (query, sample_idx) to know where to splice results back
    for query, wrong_letter in validated.items():
        traj = traj_by_query[query]
        item = meta_by_query.get(query)
        if item is None:
            print(f"  WARNING: no metadata for query {query[:60]!r}, skipping", flush=True)
            continue
        wrong_text = wrong_answer_text_for(item, wrong_letter)

        n_turns = len(traj.raw_traces)
        n_samples = len(traj.raw_traces[0]) if n_turns else 0
        for s in range(n_samples):
            existing = [traj.raw_traces[t][s] for t in range(n_turns)]
            if all(bool(tr.raw_text) for tr in existing):
                continue  # nothing to fill in this sample
            task_meta.append((query, s))
            tasks.append((item, wrong_text, existing, base_seed + len(task_meta) * 1000))

    print(f"bin{bin_idx}: {len(validated)} validated items, {len(tasks)} samples need filling", flush=True)
    if dry_run or not tasks:
        return

    ckpt_dir = os.path.join(BIN_DIR, "_gpqa_validated_fill_checkpoint", f"bin_{bin_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)

    def ckpt_path(query: str, sample_idx: int) -> str:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:60]
        return os.path.join(ckpt_dir, f"{safe}__s{sample_idx}.pkl")

    sem = asyncio.Semaphore(concurrency)
    rg = ResponseGenerator()

    async def run_one(idx: int):
        query, sample_idx = task_meta[idx]
        cpath = ckpt_path(query, sample_idx)
        if os.path.exists(cpath):
            try:
                with open(cpath, "rb") as f:
                    return idx, pickle.load(f)
            except Exception:
                pass  # corrupt checkpoint — recompute
        item, wrong_text, existing, seed = tasks[idx]
        new_traces = await regenerate_sample(
            sem=sem, rg=rg, item=item, wrong_text=wrong_text,
            existing_traces=existing, calibrator_model=calibrator_model,
            timeout_s=timeout_s, seed=seed,
        )
        tmp = cpath + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(new_traces, f)
        os.replace(tmp, cpath)  # atomic — a kill mid-write can't leave a truncated checkpoint
        return idx, new_traces

    n_already_done = sum(1 for q, s in task_meta if os.path.exists(ckpt_path(q, s)))
    if n_already_done:
        print(f"  resuming: {n_already_done}/{len(tasks)} samples already checkpointed", flush=True)

    results = [None] * len(tasks)
    coros = [run_one(i) for i in range(len(tasks))]
    done = 0
    for fut in asyncio.as_completed(coros):
        idx, new_traces = await fut
        results[idx] = new_traces
        done += 1
        if done % 5 == 0 or done == len(tasks):
            print(f"  ...{done}/{len(tasks)} samples filled", flush=True)

    # Splice results back into the trajectory objects.
    touched_queries = set()
    for (query, sample_idx), new_traces in zip(task_meta, results):
        traj = traj_by_query[query]
        n_turns = len(traj.raw_traces)
        for t in range(n_turns):
            traj.raw_traces[t][sample_idx] = new_traces[t]
        touched_queries.add(query)

    # Rebuild derived fields (last_step_flip_rates, turn_trajectories) for touched items only.
    new_trajs = []
    for traj in trajs:
        if traj.query in touched_queries:
            item = meta_by_query.get(traj.query, {})
            rebuilt = build_uncertainty_trajectory(
                query=traj.query,
                gold_answer=traj.gold_answer,
                all_traces=traj.raw_traces,
                n_clusters=DEFAULT_N_CLUSTERS,
                encoder=encoder,
            )
            new_trajs.append(rebuilt)
        else:
            new_trajs.append(traj)

    # Back up originals, then write patched files.
    backup_dir = os.path.join(BIN_DIR, f"_backup_before_gpqa_validated_fill_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(backup_dir, exist_ok=True)
    for suffix in ("reasoning", "summary", "cross_turn"):
        src = os.path.join(BIN_DIR, f"bin_{bin_idx}_{suffix}.pkl")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, f"bin_{bin_idx}_{suffix}.pkl"))

    with open(os.path.join(BIN_DIR, f"bin_{bin_idx}_reasoning.pkl"), "wb") as f:
        pickle.dump(new_trajs, f)

    summary_rows, cross_turn_rows = [], []
    for traj in new_trajs:
        summary_rows.extend(summarise_trajectory(traj))
        cross_turn_rows.extend(summarise_cross_turn_comparison(traj))
    with open(os.path.join(BIN_DIR, f"bin_{bin_idx}_summary.pkl"), "wb") as f:
        pickle.dump(summary_rows, f)
    with open(os.path.join(BIN_DIR, f"bin_{bin_idx}_cross_turn.pkl"), "wb") as f:
        pickle.dump(cross_turn_rows, f)

    print(f"  bin{bin_idx}: patched {len(touched_queries)} items, backup at {backup_dir}", flush=True)


async def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout_s", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--calibrator_model", type=str, default=DEFAULT_CALIBRATOR_MODEL)
    p.add_argument("--base_seed", type=int, default=909000)
    args = p.parse_args()

    with open(META_PATH, "rb") as f:
        meta_by_query = {m["query"]: m for m in pickle.load(f)}

    encoder = None
    try:
        encoder = _load_encoder(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        print(f"WARNING: could not load encoder ({e}); falling back to belief-entropy-only mode.", flush=True)

    for bin_idx in (0, 1):
        await process_bin(
            bin_idx=bin_idx,
            meta_by_query=meta_by_query,
            calibrator_model=args.calibrator_model,
            timeout_s=args.timeout_s,
            concurrency=args.concurrency,
            base_seed=args.base_seed,
            dry_run=args.dry_run,
            encoder=encoder,
        )


if __name__ == "__main__":
    asyncio.run(main())
