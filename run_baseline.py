"""
run_baseline.py — Baseline uncertainty experiment.

Samples each MMLU-Pro question N times, records per-attempt correctness,
computes uncertainty and entropy, and saves experiment_metadata to pickle.

Changes from previous version:
  - DEFAULT_N_ATTEMPTS raised to 25 (was 10) for better entropy resolution
  - 'category' field now stored in every per-question metadata dict
  - --stratify_by_category flag prints per-category entropy bin table

Usage:
    python run_baseline.py --model GPT5_4Nano --n_attempts 25 --n_per_cat 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pickle

from dotenv import load_dotenv
from tqdm.auto import tqdm

from config import (
    DEFAULT_MAX_CONCURRENT,
    DEFAULT_N_ATTEMPTS,
    DEFAULT_N_PER_CAT,
    DEFAULT_SEED,
    PROMPT_TEMPLATE,
)
from data import build_balanced_df
from entropy import compute_entropy, bin_by_entropy, bin_items_by_entropy_and_category
from generator import ResponseGenerator

# ---------------------------------------------------------------------------
# Single attempt
# ---------------------------------------------------------------------------


async def run_single_attempt(
    response_generator: ResponseGenerator,
    message: str,
    answer: str,
    model_name: str,
    sem: asyncio.Semaphore,
    row_idx: int,
    attempt_idx: int,
) -> dict:
    async with sem:
        try:
            response = await response_generator.agenerate_response(
                messages=response_generator.form_messages(message),
                model=model_name,
            )
            return {
                "row_idx": row_idx,
                "attempt_idx": attempt_idx,
                "response": response,
                "is_parseable": True,
                "is_correct": int(response.strip().upper() == answer.strip().upper()),
                "error": None,
            }
        except Exception as e:
            return {
                "row_idx": row_idx,
                "attempt_idx": attempt_idx,
                "response": None,
                "is_parseable": False,
                "is_correct": 0,
                "error": repr(e),
            }


# ---------------------------------------------------------------------------
# Per-row aggregation
# ---------------------------------------------------------------------------


async def run_row_attempts(
    response_generator: ResponseGenerator,
    row_idx: int,
    row: dict,
    model_name: str,
    n_attempts: int,
    sem: asyncio.Semaphore,
) -> dict:
    query = row["query"]
    options = row["options"]
    answer = row["answer"]
    category = row.get("category", "__unknown__")  # NEW
    message = PROMPT_TEMPLATE.format(question=query, options=options)

    tasks = [
        run_single_attempt(
            response_generator=response_generator,
            message=message,
            answer=answer,
            model_name=model_name,
            sem=sem,
            row_idx=row_idx,
            attempt_idx=i,
        )
        for i in range(n_attempts)
    ]

    results = sorted(await asyncio.gather(*tasks), key=lambda x: x["attempt_idx"])
    answers_l = [r["response"] for r in results if r["is_parseable"]]
    correctness_l = [r["is_correct"] for r in results if r["is_parseable"]]
    parseable = sum(int(r["is_parseable"]) for r in results)

    return {
        "query": query,
        "options": options,
        "prompt": message,
        "category": category,  # NEW — required for stratified analysis
        "answers_generated": answers_l,
        "actual_answer": answer,
        "correctness": correctness_l,
        "parseable": parseable,
        "unparseable": n_attempts - parseable,
        "uncertainty": 1 - (sum(correctness_l) / n_attempts),
        "entropy": compute_entropy(answers_l),
        "errors": [r["error"] for r in results if r["error"]],
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


async def run_experiment_async(
    balanced_df,
    response_generator: ResponseGenerator,
    model_name: str,
    n_attempts: int,
    max_concurrent: int,
) -> list:
    sem = asyncio.Semaphore(max_concurrent)
    row_tasks = [
        run_row_attempts(
            response_generator=response_generator,
            row_idx=idx,
            row=row,
            model_name=model_name,
            n_attempts=n_attempts,
            sem=sem,
        )
        for idx, row in balanced_df.iterrows()
    ]

    results = []
    for coro in tqdm(
        asyncio.as_completed(row_tasks), total=len(row_tasks), desc="rows"
    ):
        results.append(await coro)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run baseline uncertainty experiment on MMLU-Pro."
    )
    p.add_argument(
        "--model", type=str, default="GPT5_4Nano", help="Model key from config.MODELS"
    )
    p.add_argument(
        "--n_attempts",
        type=int,
        default=DEFAULT_N_ATTEMPTS,
        help="Samples per question (default 25 for 0.04 probability resolution)",
    )
    p.add_argument(
        "--n_per_cat",
        type=int,
        default=DEFAULT_N_PER_CAT,
        help="Questions sampled per MMLU-Pro category",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for dataset sampling",
    )
    p.add_argument(
        "--max_concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Max concurrent API requests",
    )
    p.add_argument(
        "--n_bins",
        type=int,
        default=5,
        help="Number of entropy bins for diagnostic table",
    )
    p.add_argument(
        "--bin_strategy", type=str, default="quantile", choices=["uniform", "quantile"]
    )
    p.add_argument(
        "--stratify_by_category",
        action="store_true",
        help="Also print per-category entropy bin tables",
    )
    p.add_argument(
        "--out_dir", type=str, default="experiment_out", help="Root output directory"
    )
    return p.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    out_path = os.path.join(args.out_dir, args.model)
    os.makedirs(out_path, exist_ok=True)
    pkl_path = os.path.join(out_path, "base_experiment_metadata.pkl")

    response_generator = ResponseGenerator()

    if os.path.exists(pkl_path):
        print(f"Found existing results at {pkl_path}, loading...")
        with open(pkl_path, "rb") as f:
            experiment_metadata_l = pickle.load(f)
        # Back-fill category for old pickles that predate this field
        n_missing = sum(1 for it in experiment_metadata_l if "category" not in it)
        if n_missing:
            print(
                f"  Back-filling 'category' for {n_missing} items — rebuilding dataset to get labels..."
            )
            balanced_df = build_balanced_df(n_per_cat=args.n_per_cat, seed=args.seed)
            cat_map = {
                row["query"]: row["category"] for _, row in balanced_df.iterrows()
            }
            for it in experiment_metadata_l:
                it.setdefault("category", cat_map.get(it["query"], "__unknown__"))
            with open(pkl_path, "wb") as f:
                pickle.dump(experiment_metadata_l, f)
            print("  Back-fill complete, pickle updated.")
    else:
        print(
            f"Building balanced dataset ({args.n_per_cat} per category, seed={args.seed})..."
        )
        balanced_df = build_balanced_df(n_per_cat=args.n_per_cat, seed=args.seed)
        print(
            f"Running baseline: {len(balanced_df)} questions "
            f"x {args.n_attempts} attempts each "
            f"(total API calls: {len(balanced_df) * args.n_attempts})"
        )

        experiment_metadata_l = await run_experiment_async(
            balanced_df=balanced_df,
            response_generator=response_generator,
            model_name=args.model,
            n_attempts=args.n_attempts,
            max_concurrent=args.max_concurrent,
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(experiment_metadata_l, f)
        print(f"Saved -> {pkl_path}")

    print("\n--- Global entropy bin diagnostic ---")
    bin_by_entropy(
        experiment_metadata_l,
        n_bins=args.n_bins,
        strategy=args.bin_strategy,
    )

    if args.stratify_by_category:
        print("\n--- Per-category entropy bin diagnostic ---")
        bin_items_by_entropy_and_category(
            experiment_metadata_l,
            n_bins=args.n_bins,
            strategy=args.bin_strategy,
        )


if __name__ == "__main__":
    asyncio.run(main())
