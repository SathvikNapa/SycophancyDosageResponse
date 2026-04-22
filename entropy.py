from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from config import DEFAULT_N_BINS, DEFAULT_BIN_STRATEGY

# ---------------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------------


def compute_entropy(answers: List[str]) -> float:
    """
    Shannon entropy H = -sum(p * log(p)) over the answer distribution.
    Returns a NEGATIVE value (matching original convention):
      0.0  = all samples agree  → maximally confident
      more negative = more spread → more uncertain
    Empty list returns 0.0 (treated as confident).
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
    Safe to call on pickles saved before entropy was added.
    """
    for item in experiment_metadata_l:
        if "entropy" not in item:
            item["entropy"] = compute_entropy(item["answers_generated"])


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------


def _safe_quantile_bins(
    entropies: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, int]:
    """
    Compute quantile bin edges robustly.

    KBinsDiscretizer silently merges bins when many samples share the same
    entropy value (common for confident models where entropy == 0.0 for a
    large fraction of questions). This produces fewer actual bins than
    requested without warning.

    Fix: compute percentile edges → deduplicate with np.unique → use
    np.searchsorted for assignment. The resulting actual_n_bins is always
    <= n_bins and is correct.

    Returns (bin_ids array, actual_n_bins).
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(entropies, quantiles))
    actual_n_bins = len(edges) - 1

    if actual_n_bins < n_bins:
        print(
            f"  WARNING: requested {n_bins} bins but only {actual_n_bins} distinct "
            f"quantile edges exist in this data (common when many samples share the "
            f"same entropy value). Proceeding with {actual_n_bins} bins."
        )

    # searchsorted assigns each value to the bin whose RIGHT edge it falls
    # under. Clamp so the maximum-entropy item lands in the last valid bin.
    bin_ids = np.searchsorted(edges[1:], entropies, side="right")
    bin_ids = np.clip(bin_ids, 0, actual_n_bins - 1)
    return bin_ids, actual_n_bins, edges


def _uniform_bins(
    entropies: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    bin_ids = binner.fit_transform(entropies.reshape(-1, 1)).astype(int).ravel()
    edges = binner.bin_edges_[0]
    return bin_ids, len(edges) - 1, edges


# ---------------------------------------------------------------------------
# Primary binning function
# ---------------------------------------------------------------------------


def bin_items_by_entropy(
    experiment_metadata_l: List[dict],
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = DEFAULT_BIN_STRATEGY,
) -> Tuple[Dict[int, List[dict]], np.ndarray]:
    """
    Bins items by their entropy value.

    Returns:
        bins  : dict mapping bin_id -> list of items
        edges : array of bin boundary values (length actual_n_bins + 1)

    Previously returned (bins, KBinsDiscretizer); now returns (bins, edges)
    so callers have the raw boundaries without needing the sklearn object.
    """
    for item in experiment_metadata_l:
        if "entropy" not in item:
            item["entropy"] = compute_entropy(item["answers_generated"])

    entropies = np.array([item["entropy"] for item in experiment_metadata_l])

    if strategy == "quantile":
        bin_ids, actual_n_bins, edges = _safe_quantile_bins(entropies, n_bins)
    else:
        bin_ids, actual_n_bins, edges = _uniform_bins(entropies, n_bins)

    bins: Dict[int, List[dict]] = {i: [] for i in range(actual_n_bins)}
    for item, bid in zip(experiment_metadata_l, bin_ids):
        bins[int(bid)].append(item)

    _print_bin_summary(bins, edges, actual_n_bins)
    return bins, edges


# ---------------------------------------------------------------------------
# Category-stratified binning (Change 3)
# ---------------------------------------------------------------------------


def bin_items_by_entropy_and_category(
    experiment_metadata_l: List[dict],
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = DEFAULT_BIN_STRATEGY,
) -> Dict[str, Dict[int, List[dict]]]:
    """
    Bins items by entropy WITHIN each MMLU-Pro category.

    Bin edges are fit independently per category so the entropy ranges are
    comparable within a subject domain. This means bin 0 for 'math' and
    bin 0 for 'biology' both represent the most uncertain quartile of that
    subject — not a global entropy threshold.

    Returns:
        {category: {bin_id: [items]}}

    Each item must have a 'category' key (added by run_baseline.py).
    Items missing 'category' are grouped under '__unknown__'.
    """
    patch_entropy(experiment_metadata_l)

    by_cat: Dict[str, List[dict]] = defaultdict(list)
    for item in experiment_metadata_l:
        cat = item.get("category", "__unknown__")
        by_cat[cat].append(item)

    result: Dict[str, Dict[int, List[dict]]] = {}
    for cat, items in sorted(by_cat.items()):
        if len(items) < n_bins:
            print(f"  Category '{cat}': only {len(items)} items, skipping bin split.")
            result[cat] = {0: items}
            continue
        print(f"\n  === Category: {cat} ({len(items)} items) ===")
        bins, _ = bin_items_by_entropy(items, n_bins=n_bins, strategy=strategy)
        result[cat] = bins

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _print_bin_summary(
    bins: Dict[int, List[dict]],
    edges: np.ndarray,
    actual_n_bins: int,
) -> None:
    print(f"\n{'Bin':<6} {'Entropy Range':<22} {'Items':<8} {'Accuracy'}")
    print("-" * 50)
    for i in range(actual_n_bins):
        items = bins[i]
        if items:
            acc = np.mean(
                [
                    int(
                        Counter(it["answers_generated"]).most_common(1)[0][0]
                        == it["actual_answer"]
                    )
                    for it in items
                ]
            )
        else:
            acc = 0.0
        print(f"{i:<6} {edges[i]:.3f} to {edges[i+1]:.3f}    {len(items):<8} {acc:.3f}")


def bin_by_entropy(
    experiment_metadata_l: List[dict],
    n_bins: int = DEFAULT_N_BINS,
    strategy: str = DEFAULT_BIN_STRATEGY,
) -> None:
    """
    Standalone diagnostic: prints accuracy-per-entropy-bin table.
    Does not return bins — use bin_items_by_entropy for the full pipeline.
    """
    patch_entropy(experiment_metadata_l)

    records = []
    for item in experiment_metadata_l:
        answers = item["answers_generated"]
        label = item["actual_answer"]
        entropy = item["entropy"]
        majority = Counter(answers).most_common(1)[0][0] if answers else None
        correct = int(majority == label) if majority is not None else 0
        records.append({"entropy": entropy, "correct": correct})

    entropies = np.array([r["entropy"] for r in records])

    if strategy == "quantile":
        bin_ids, actual_n_bins, edges = _safe_quantile_bins(entropies, n_bins)
    else:
        bin_ids, actual_n_bins, edges = _uniform_bins(entropies, n_bins)

    bin_results: Dict[int, List[int]] = {i: [] for i in range(actual_n_bins)}
    for idx, rec in enumerate(records):
        bin_results[bin_ids[idx]].append(rec["correct"])

    for i in range(actual_n_bins):
        scores = bin_results[i]
        acc = np.mean(scores) if scores else 0.0
        rng = f"{edges[i]:.3f} to {edges[i+1]:.3f}"
        print(f"{rng:<25} | {len(scores):<10} | {acc:.3f}")
