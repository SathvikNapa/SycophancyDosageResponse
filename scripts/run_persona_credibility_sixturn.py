"""
run_persona_credibility_sixturn.py
------------------------------------
Six-turn confirmatory run for Appendix "Source-Credibility Framing"
(paper/active_version/sections/appendix.tex, \\label{app:persona}).

Companion to run_persona_credibility.py, which uses the paper's SINGLE-TURN
robustness-check design (app:single_turn). This script instead uses the
paper's PRIMARY design: a full sequential six-turn escalating dialogue
(T0 baseline, then T1..T6 each applying the next pressure category in the
fixed canonical order from tables/pressure_levels.tex -- not shuffled, to
avoid a category-order confound in the write-up). Same unbiased question
sampling (confident + mid-uncertainty strata, T0-correct, pooled MMLU-Pro +
GPQA-Diamond) and same four personas as run_persona_credibility.py.

Call cost per (question, persona, rep): T0 is free (reused from the
existing baseline pkl, never regenerated under a persona's system message);
T1..T6 are 6 fresh sequential calls advancing one growing conversation --
same per-unit call count as the single-turn design's 6 independent
one-shot calls, but each call carries more context on average (the
conversation accumulates turn by turn), so this run costs roughly ~2x per
call despite half the repetitions (R=2 vs R=4).

Wrong-answer choice is seeded by query only (not persona, not rep) -- every
persona and every repetition of a given question is pressured with the
identical planted wrong answer across all six turns, matching the paper's
own multi-turn protocol (a single fixed a^\\dagger per dialogue) and the
single-turn script's convention of holding the stimulus constant across
personas.

Usage:
    uv run python scripts/run_persona_credibility_sixturn.py
    uv run python scripts/run_persona_credibility_sixturn.py --model ClaudeSonnet
    uv run python scripts/run_persona_credibility_sixturn.py --n 40 --reps 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import random
from collections import defaultdict

from tqdm.auto import tqdm

from sycophancy.config import DEFAULT_MAX_CONCURRENT, DEFAULT_SEED, SYSTEM_MSG
from sycophancy.generator import ResponseGenerator
from sycophancy.sycophancy import pick_wrong_option_from_item, run_sycophancy_conversation
from sycophancy.sycophancy_dosage import DOSES

# Reuse the exact question-selection logic from the single-turn confirmatory
# script, so both designs draw from the identical unbiased sample.
from run_persona_credibility import (
    DATASETS,
    MID_CAP_OVERRIDE,
    N_PER_STRATUM,
    PERSONAS,
    build_system_msg,
    first_correct_answer,
    load_baseline_meta,
    select_confident,
    select_mid,
)

EXPERIMENT_OUT = "experiment_out"
OUT_SUBDIR = "persona_credibility_sixturn"
MODELS = ["ClaudeSonnet", "GPT5_4Mini"]
N_REPS = 2

# Fixed canonical order (matches tables/pressure_levels.tex), not shuffled.
DOSE_STATEMENTS = [template for _, template in DOSES]
DOSE_CATEGORIES = [slug for slug, _ in DOSES]


async def run_one_dialogue(
    gen: ResponseGenerator,
    sem: asyncio.Semaphore,
    model_key: str,
    dataset: str,
    stratum: str,
    persona_key: str,
    rep_idx: int,
    item: dict,
    t0_answer: str,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    wrong_text, wrong_letter = pick_wrong_option_from_item(item, rng)

    # `sem` is passed straight through to run_sycophancy_conversation, which
    # acquires/releases it per individual call (T0, T1, ..., T6) rather than
    # for the whole dialogue -- matching run_persona_trace_demo.py's working
    # pattern. Wrapping the whole dialogue in our own `async with sem` would
    # instead reserve one concurrency slot for a full 7-call sequential
    # dialogue's entire duration, collapsing effective throughput.
    res = await run_sycophancy_conversation(
        sem=sem,
        rg=gen,
        q_prompt=item["prompt"],
        query=item["query"],
        answer=item["actual_answer"],
        model=model_key,
        wrong_answer_text=wrong_text,
        wrong_answer_letter=wrong_letter,
        dose_statements=DOSE_STATEMENTS,
        dose_categories=DOSE_CATEGORIES,
        timeout_s=60.0,
        seed=seed,
        n_confidence_polls=0,
        max_retries=1,
        system_msg=build_system_msg(persona_key),
    )

    return {
        "model": model_key,
        "dataset": dataset,
        "stratum": stratum,
        "persona": persona_key,
        "rep": rep_idx,
        "query": item["query"],
        "category": item.get("category"),
        "gold_answer": item["actual_answer"],
        "t0_answer": t0_answer,
        "baseline_entropy": item.get("entropy"),
        "wrong_answer_text": wrong_text,
        "wrong_answer_letter": wrong_letter,
        "raw_turns": res["raw_turns"],
        "parsed_turns": res["parsed_turns"],
        "is_wrong": res["is_wrong"],
        "is_correct": res["is_correct"],
        "turn_categories": res["turn_categories"],
        "first_wrong_turn": res["first_wrong_turn"],
        "first_flip_category": res["first_flip_category"],
        "errors": res["errors"],
    }


async def run_model(
    model_key: str, n_per_stratum: int, n_reps: int, seed: int, concurrency: int, out_dir: str, force: bool
) -> None:
    out_path = os.path.join(out_dir, model_key, OUT_SUBDIR, "results.pkl")
    if os.path.exists(out_path) and not force:
        print(f"  [skip] {out_path} already exists (--force to overwrite)")
        return

    gen = ResponseGenerator()
    sem = asyncio.Semaphore(concurrency)

    mmlu_meta = load_baseline_meta(model_key, "mmlu_pro", out_dir)
    confident = select_confident(mmlu_meta, n_per_stratum)
    mid_cap = MID_CAP_OVERRIDE.get(model_key, n_per_stratum)
    mid = select_mid(model_key, out_dir, mmlu_meta, mid_cap)
    print(f"{model_key}: confident (MMLU-Pro, entropy=0) -> {len(confident)}/{n_per_stratum}, "
          f"mid (pooled, entropy<0) -> {len(mid)}/{mid_cap if mid_cap is not None else 'uncapped'}")

    tasks = []
    for stratum, items in (("confident", confident), ("mid", mid)):
        for item, t0, dataset in items:
            stimulus_seed = seed + abs(hash(item["query"])) % 100_000
            for persona_key in PERSONAS:
                for rep_idx in range(n_reps):
                    tasks.append(run_one_dialogue(
                        gen, sem, model_key, dataset, stratum, persona_key, rep_idx,
                        item, t0, stimulus_seed + rep_idx,
                    ))

    n_questions = len(confident) + len(mid)
    print(f"  Dispatching {len(tasks)} six-turn dialogues "
          f"({n_questions} questions x 4 personas x {n_reps} reps, 6 sequential calls each "
          f"= {len(tasks) * 6} total API calls) ...")
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=model_key):
        results.append(await coro)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    n_valid = sum(1 for r in results if all(t is not None for t in r["parsed_turns"]))
    n_error = sum(1 for r in results if r["errors"])
    print(f"  Saved {len(results)} dialogues -> {out_path}")
    print(f"  Fully-parsed dialogues: {n_valid}/{len(results)}  With errors: {n_error}/{len(results)}")

    by_key: dict[tuple, list] = defaultdict(list)
    for r in results:
        flipped = r["first_wrong_turn"] is not None
        by_key[(r["stratum"], r["persona"])].append(flipped)
    print("  Per (stratum, persona) ever-flipped rate:")
    for key in sorted(by_key):
        flips = by_key[key]
        print(f"    {key}: {sum(flips)}/{len(flips)} = {sum(flips) / len(flips) * 100:.1f}%")


async def backfill_model(model_key: str, seed: int, concurrency: int, out_dir: str) -> None:
    """Fully re-runs any dialogue that had a call failure partway through,
    rather than patching individual turns -- a mid-dialogue failure leaves a
    blank assistant reply spliced into the transcript, which would corrupt
    every turn after it if only the failed call were retried in place.
    Identifies affected rows by non-empty `errors` (an actual API failure),
    not by a None in parsed_turns (which can legitimately happen for an
    unparseable-but-successful response -- that's real data, not an error)."""
    out_path = os.path.join(out_dir, model_key, OUT_SUBDIR, "results.pkl")
    with open(out_path, "rb") as f:
        results: list[dict] = pickle.load(f)

    error_idxs = [i for i, r in enumerate(results) if r["errors"]]
    if not error_idxs:
        print(f"{model_key}: no errored dialogues to backfill.")
        return
    print(f"{model_key}: backfilling {len(error_idxs)}/{len(results)} errored dialogues "
          f"({len(error_idxs) * 6} fresh API calls) ...")

    mmlu_meta = load_baseline_meta(model_key, "mmlu_pro", out_dir)
    gpqa_meta = load_baseline_meta(model_key, "gpqa_diamond", out_dir)
    items_by_query = {m["query"]: m for m in mmlu_meta + gpqa_meta}

    gen = ResponseGenerator()
    sem = asyncio.Semaphore(concurrency)

    async def redo(idx: int) -> tuple[int, dict]:
        old = results[idx]
        item = items_by_query[old["query"]]
        stimulus_seed = seed + abs(hash(old["query"])) % 100_000
        new = await run_one_dialogue(
            gen, sem, model_key, old["dataset"], old["stratum"], old["persona"], old["rep"],
            item, old["t0_answer"], stimulus_seed + old["rep"],
        )
        return idx, new

    tasks = [redo(i) for i in error_idxs]
    n_fixed = 0
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{model_key}/backfill"):
        idx, new = await coro
        results[idx] = new
        if not new["errors"]:
            n_fixed += 1

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    n_error_now = sum(1 for r in results if r["errors"])
    print(f"  Backfilled {n_fixed}/{len(error_idxs)}; {n_error_now}/{len(results)} dialogues still erroring.")


async def main() -> None:
    p = argparse.ArgumentParser(description="Six-turn source-credibility confirmatory run")
    p.add_argument("--model", type=str, default="", help="Model key. Omit to run both.")
    p.add_argument("--n", type=int, default=N_PER_STRATUM, help="Questions per stratum, per model.")
    p.add_argument("--reps", type=int, default=N_REPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENT)
    p.add_argument("--out_dir", type=str, default=EXPERIMENT_OUT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--backfill", action="store_true",
                    help="Fully re-run errored dialogues in an existing results.pkl instead of a fresh run.")
    args = p.parse_args()

    models = [args.model] if args.model else MODELS
    for model_key in models:
        if args.backfill:
            await backfill_model(model_key, args.seed, args.concurrency, args.out_dir)
        else:
            await run_model(model_key, args.n, args.reps, args.seed, args.concurrency, args.out_dir, args.force)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
