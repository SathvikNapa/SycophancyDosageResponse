from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from config import DEFAULT_N_BINS, DEFAULT_BIN_STRATEGY


def compute_entropy(answers: List[str]) -> float:
    """
    Returns sum(p * log(p)) over the answer distribution — negative entropy.
    Value of 0.0 means maximally uncertain (uniform); more negative means more confident.
    """
    if not answers:
        return 0.0
    counts = Counter(answers)
    total = len(answers)
    probs = np.array([c / total for c in counts.values()])
    return float((probs * np.log(probs)).sum())


def patch_entropy(experiment_metadata_l: List[dict]) -> None:
    """
    Adds 'entropy' key in-place to any item missing it.
    Safe to call on pickles saved before entropy was added to run_row_attempts.
    """
    for item in experiment_metadata_l:
        if "entropy" not in item:
            item["entropy"] = compute_entropy(item["answers_generated"])


def bin_items_by_entropy(
    experiment_metadata_l: List[dict],
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = DEFAULT_BIN_STRATEGY,
) -> Tuple[dict, KBinsDiscretizer]:
    """
    Fits a KBinsDiscretizer on per-item entropy values, assigns each item to a bin,
    prints a summary table, and returns (bins_dict, fitted_binner).

    bins_dict: {bin_idx (int): [list of item dicts]}
    """
    patch_entropy(experiment_metadata_l)

    entropies = np.array(
        [item["entropy"] for item in experiment_metadata_l]
    ).reshape(-1, 1)

    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    binner.fit(entropies)
    bin_indices = binner.transform(entropies).flatten().astype(int)

    bins: dict = {i: [] for i in range(n_bins)}
    for item, bin_idx in zip(experiment_metadata_l, bin_indices):
        bins[bin_idx].append(item)

    _print_bin_summary(bins, binner, n_bins)
    return bins, binner


def _print_bin_summary(
    bins: dict,
    binner: KBinsDiscretizer,
    n_bins: int,
) -> None:
    edges = binner.bin_edges_[0]
    print(f"\n{'Bin':<6} {'Entropy Range':<22} {'Items':<8} {'Accuracy'}")
    print("-" * 50)
    for i in range(n_bins):
        items = bins[i]
        if items:
            acc = np.mean([
                int(
                    Counter(it["answers_generated"]).most_common(1)[0][0]
                    == it["actual_answer"]
                )
                for it in items
            ])
        else:
            acc = 0.0
        print(f"{i:<6} {edges[i]:.3f} to {edges[i+1]:.3f}    {len(items):<8} {acc:.3f}")


def bin_by_entropy(
    experiment_metadata_l: List[dict],
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = DEFAULT_BIN_STRATEGY,
) -> None:
    """
    Standalone diagnostic function: prints the accuracy-per-entropy-bin table.
    Does not return bins — use bin_items_by_entropy for the full pipeline.
    """
    patch_entropy(experiment_metadata_l)

    records = []
    for item in experiment_metadata_l:
        answers = item["answers_generated"]
        label   = item["actual_answer"]
        entropy = item["entropy"]
        majority = Counter(answers).most_common(1)[0][0] if answers else None
        correct  = int(majority == label) if majority is not None else 0
        records.append({"entropy": entropy, "correct": correct})

    entropies  = np.array([r["entropy"] for r in records]).reshape(-1, 1)
    binner     = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    binner.fit(entropies)
    bin_indices = binner.transform(entropies).flatten().astype(int)

    bin_results: dict = {i: [] for i in range(n_bins)}
    for idx, rec in enumerate(records):
        bin_results[bin_indices[idx]].append(rec["correct"])

    edges = binner.bin_edges_[0]
    print(f"\n{'Entropy Bin Range':<25} | {'Samples':<10} | {'Accuracy'}")
    print("-" * 52)
    for i in range(n_bins):
        scores = bin_results[i]
        acc    = np.mean(scores) if scores else 0.0
        rng    = f"{edges[i]:.3f} to {edges[i+1]:.3f}"
        print(f"{rng:<25} | {len(scores):<10} | {acc:.3f}")
