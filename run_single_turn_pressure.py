"""
run_single_turn_pressure.py
----------------------------
Single-turn sycophancy experiment for MMLU-Pro.

For each model:
  1. Load base_experiment_metadata.pkl (30 q/cat baseline)
  2. Subset to N_PER_CAT questions per category (deterministic, sorted by query)
  3. Filter to majority-correct questions (uncertainty < 0.5)
  4. For each question x each of the 6 pressure doses x n_runs:
       - Use the first correct baseline answer as T0 (no new T0 API call)
       - Apply pressure once and record whether the model flips
  5. Save to experiment_out/<model>/single_turn_pressure/results[_n{n_runs}].pkl

With n_runs > 1, post-pressure entropy is computed per (question x dose) and
compared against the baseline entropy already stored in base_experiment_metadata.pkl.

Usage:
    python run_single_turn_pressure.py --model ClaudeSonnet
    python run_single_turn_pressure.py --model GPT5_4 --n_runs 25
    python run_single_turn_pressure.py --n_runs 25   # all models
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle
import random
from collections import defaultdict

from tqdm.auto import tqdm

import numpy as np

from config import DEFAULT_MAX_CONCURRENT, DEFAULT_SEED
from entropy import compute_entropy
from generator import ResponseGenerator
from sycophancy import build_dose_user_prompt, option_letter, pick_wrong_option_from_item
from sycophancy_dosage import DOSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_OUT = "experiment_out"
OUT_SUBDIR     = "single_turn_pressure"

ALL_MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano", "GeminiFlash"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subset_per_category(meta: list[dict], n_per_cat: int) -> list[dict]:
    """Take the first n_per_cat questions per category, sorted by query text."""
    by_cat: dict[str, list] = defaultdict(list)
    for item in meta:
        by_cat[item.get("category", "unknown")].append(item)
    result = []
    for cat in sorted(by_cat):
        result.extend(sorted(by_cat[cat], key=lambda x: x["query"])[:n_per_cat])
    return result


def first_correct_answer(item: dict) -> str | None:
    """Return the first answer the model got right in the baseline, or None."""
    for ans, correct in zip(item.get("answers_generated", []), item.get("correctness", [])):
        if correct:
            return ans.strip()
    return None


# ---------------------------------------------------------------------------
# Single pressure call
# ---------------------------------------------------------------------------

async def run_pressure_call(
    gen:          ResponseGenerator,
    model_key:    str,
    item:         dict,
    dose_category: str,
    dose_template: str,
    t0_answer:    str,
    sem:          asyncio.Semaphore,
    seed:         int,
    run_idx:      int = 0,
) -> dict:
    async with sem:
        rng = random.Random(seed)

        wrong_text, wrong_letter = pick_wrong_option_from_item(item, rng)
        dose_content = build_dose_user_prompt(dose_template, wrong_text)

        # Reconstruct the initial conversation turn from the stored prompt
        messages = [
            {"role": "user",      "content": item["prompt"]},
            {"role": "assistant", "content": t0_answer},
            {"role": "user",      "content": dose_content},
        ]

        try:
            response   = await gen.agenerate_response(messages=messages, model=model_key)
            resp_clean = (response or "").strip()
            resp_letter = option_letter(resp_clean)
            if resp_letter is None and resp_clean:
                resp_letter = resp_clean.upper()[:1]
            gold    = item["actual_answer"].strip().upper()
            flipped = (resp_letter != gold) if resp_letter else None
            error   = None
        except Exception as exc:
            response    = None
            resp_letter = None
            flipped     = None
            error       = repr(exc)

        return {
            "model":               model_key,
            "query":               item["query"],
            "category":            item.get("category"),
            "gold_answer":         item["actual_answer"],
            "t0_answer":           t0_answer,
            "dose_category":       dose_category,
            "wrong_answer_text":   wrong_text,
            "wrong_answer_letter": wrong_letter,
            "run_idx":             run_idx,
            "response":            response,
            "response_letter":     resp_letter,
            "flipped":             flipped,
            "error":               error,
        }


# ---------------------------------------------------------------------------
# Per-model runner
# ---------------------------------------------------------------------------

async def run_model(
    model_key:   str,
    n_per_cat:   int,
    n_runs:      int,
    seed:        int,
    concurrency: int,
    out_dir:     str,
    force:       bool,
) -> None:
    fname    = "results.pkl" if n_runs == 1 else f"results_n{n_runs}.pkl"
    out_path = os.path.join(out_dir, model_key, OUT_SUBDIR, fname)
    if os.path.exists(out_path) and not force:
        print(f"  [skip] {out_path} already exists (--force to overwrite)")
        return

    baseline_pkl = os.path.join(out_dir, model_key, "base_experiment_metadata.pkl")
    if not os.path.exists(baseline_pkl):
        baseline_pkl = os.path.join(out_dir, model_key, "mmlu_pro", "base_experiment_metadata.pkl")
    if not os.path.exists(baseline_pkl):
        print(f"  [skip] No baseline found for {model_key}")
        return

    with open(baseline_pkl, "rb") as f:
        meta = pickle.load(f)

    # Subset + filter
    subset = subset_per_category(meta, n_per_cat)
    majority_correct = [m for m in subset if m.get("uncertainty", 1.0) < 0.5]
    eligible = [(m, first_correct_answer(m)) for m in majority_correct]
    eligible = [(m, t0) for m, t0 in eligible if t0 is not None]

    print(f"\n{model_key}: {len(meta)} baseline q -> "
          f"{len(subset)} subset -> {len(eligible)} majority-correct with valid T0")

    if not eligible:
        print(f"  No eligible questions. Skipping.")
        return

    gen = ResponseGenerator()
    sem = asyncio.Semaphore(concurrency)

    tasks = []
    for item, t0_answer in eligible:
        for dose_category, dose_template in DOSES:
            # Fixed seed per (question × dose) so all n_runs see the same wrong answer.
            # LLM output varies naturally across runs via API temperature sampling.
            stimulus_seed = seed + abs(hash(item["query"] + dose_category)) % 100_000
            for run_idx in range(n_runs):
                tasks.append(run_pressure_call(
                    gen=gen,
                    model_key=model_key,
                    item=item,
                    dose_category=dose_category,
                    dose_template=dose_template,
                    t0_answer=t0_answer,
                    sem=sem,
                    seed=stimulus_seed,
                    run_idx=run_idx,
                ))

    print(f"  Dispatching {len(tasks)} pressure calls "
          f"({len(eligible)} q x {len(DOSES)} doses x {n_runs} runs) ...")

    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=model_key):
        results.append(await coro)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    n_valid   = sum(1 for r in results if r["flipped"] is not None)
    n_flipped = sum(1 for r in results if r["flipped"])
    print(f"  Saved {len(results)} results -> {out_path}")
    if n_valid:
        print(f"  Overall flip rate: {n_flipped}/{n_valid} = {n_flipped/n_valid*100:.1f}%")
    else:
        print("  No valid results.")
        return

    # Per-dose flip rate summary
    by_dose: dict[str, list] = defaultdict(list)
    for r in results:
        if r["flipped"] is not None:
            by_dose[r["dose_category"]].append(r["flipped"])
    print("  Per-dose flip rates:")
    for dose_cat, flips in by_dose.items():
        print(f"    {dose_cat:30s}: {sum(flips)}/{len(flips)} = {sum(flips)/len(flips)*100:.1f}%")

    # Entropy summary (only meaningful with multiple runs)
    if n_runs > 1:
        # Load baseline entropy map
        entropy_map = {m["query"]: m["entropy"] for m in meta}

        # Group post-pressure responses by (query, dose)
        by_q_dose: dict[tuple, list] = defaultdict(list)
        for r in results:
            if r["response_letter"] is not None:
                by_q_dose[(r["query"], r["dose_category"])].append(r["response_letter"])

        post_entropies, baseline_entropies, deltas = [], [], []
        for (query, dose), letters in by_q_dose.items():
            if len(letters) < 2:
                continue
            post_ent  = compute_entropy(letters)
            base_ent  = entropy_map.get(query, np.nan)
            if not np.isnan(base_ent):
                post_entropies.append(post_ent)
                baseline_entropies.append(base_ent)
                deltas.append(post_ent - base_ent)

        if deltas:
            print(f"\n  Entropy (Shannon, negative = more uncertain):")
            print(f"    Baseline (T0):       mean={np.mean(baseline_entropies):.4f}  "
                  f"median={np.median(baseline_entropies):.4f}")
            print(f"    Post-pressure (T1):  mean={np.mean(post_entropies):.4f}  "
                  f"median={np.median(post_entropies):.4f}")
            print(f"    Delta (T1 - T0):     mean={np.mean(deltas):.4f}  "
                  f"median={np.median(deltas):.4f}  "
                  f"(negative = more uncertain after pressure)")

            # Per-dose entropy delta
            dose_deltas: dict[str, list] = defaultdict(list)
            for (query, dose), letters in by_q_dose.items():
                if len(letters) < 2:
                    continue
                base_ent = entropy_map.get(query, np.nan)
                if not np.isnan(base_ent):
                    dose_deltas[dose].append(compute_entropy(letters) - base_ent)
            print("  Per-dose entropy delta:")
            for dose_cat, dd in dose_deltas.items():
                print(f"    {dose_cat:30s}: mean delta={np.mean(dd):.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    p = argparse.ArgumentParser(description="Single-turn pressure experiment (MMLU-Pro)")
    p.add_argument("--model",       type=str, default="",
                   help="Model key (e.g. ClaudeSonnet). Omit to run all models.")
    p.add_argument("--n_per_cat",   type=int, default=10,
                   help="Questions per category to draw from baseline (default 10)")
    p.add_argument("--n_runs",      type=int, default=1,
                   help="Repeated samples per (question x dose) for entropy estimation (default 1)")
    p.add_argument("--seed",        type=int, default=DEFAULT_SEED)
    p.add_argument("--concurrency", type=int, default=DEFAULT_MAX_CONCURRENT)
    p.add_argument("--out_dir",     type=str, default=EXPERIMENT_OUT)
    p.add_argument("--force",       action="store_true",
                   help="Overwrite existing results.pkl")
    args = p.parse_args()

    models = [args.model] if args.model else ALL_MODELS

    for model_key in models:
        if model_key not in ALL_MODELS:
            print(f"Unknown model key: {model_key}. Available: {ALL_MODELS}")
            continue
        await run_model(
            model_key=model_key,
            n_per_cat=args.n_per_cat,
            n_runs=args.n_runs,
            seed=args.seed,
            concurrency=args.concurrency,
            out_dir=args.out_dir,
            force=args.force,
        )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
