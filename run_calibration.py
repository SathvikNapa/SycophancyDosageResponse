"""
run_calibration.py — Batch isotonic calibration for all baseline pkls.

Finds every base_experiment_metadata.pkl under experiment_out/, fits an
isotonic regression mapping |entropy| → uncertainty (expected error rate),
and saves a calibration.pkl sidecar in the same directory.

The sidecar is intentionally separate from the baseline pkl so baseline data
is never mutated.  All downstream scripts that need calibrated_prob load it
from calibration.pkl via entropy.load_calibration().

calibration.pkl structure:
  {
    "regressor":     IsotonicRegression,   # fitted sklearn object
    "query_to_prob": Dict[str, float],     # query string → calibrated_prob
    "model":         str,
    "dataset":       str,
    "n_items":       int,
    "prob_min":      float,
    "prob_max":      float,
  }

Usage:
    python run_calibration.py
    python run_calibration.py --out_dir experiment_out
    python run_calibration.py --force   # overwrite existing calibration.pkl files
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from entropy import compute_entropy, fit_isotonic_calibration


def calibrate_one(pkl_path: str, out_dir: str, force: bool = False) -> dict | None:
    """
    Fits isotonic calibration for one baseline pkl and saves calibration.pkl
    in the same directory.  Returns the calibration dict, or None if skipped.
    """
    pkl_path = str(pkl_path)
    cal_path = os.path.join(os.path.dirname(pkl_path), "calibration.pkl")

    if os.path.exists(cal_path) and not force:
        print(f"  [skip] calibration.pkl already exists (use --force to overwrite)")
        return None

    with open(pkl_path, "rb") as f:
        items = pickle.load(f)

    for item in items:
        if "entropy" not in item:
            item["entropy"] = compute_entropy(item.get("answers_generated", []))

    regressor = fit_isotonic_calibration(items)
    X = np.array([-item["entropy"] for item in items])
    probs = regressor.predict(X)
    query_to_prob = {item["query"]: float(p) for item, p in zip(items, probs)}

    # Infer model/dataset from path structure relative to out_dir:
    #   <out_dir>/<model>/base_experiment_metadata.pkl          → mmlu_pro (legacy)
    #   <out_dir>/<model>/<dataset>/base_experiment_metadata.pkl
    rel_parts = Path(pkl_path).relative_to(out_dir).parts[:-1]  # drop filename
    if len(rel_parts) == 1:
        model, dataset = rel_parts[0], "mmlu_pro"
    elif len(rel_parts) == 2:
        model, dataset = rel_parts[0], rel_parts[1]
    else:
        model, dataset = "unknown", "unknown"

    calibration = {
        "regressor":     regressor,
        "query_to_prob": query_to_prob,
        "model":         model,
        "dataset":       dataset,
        "n_items":       len(items),
        "prob_min":      float(probs.min()),
        "prob_max":      float(probs.max()),
    }

    with open(cal_path, "wb") as f:
        pickle.dump(calibration, f)

    print(f"  Saved -> {cal_path}")
    print(f"    model={model}, dataset={dataset}, n={len(items)}, "
          f"prob_range=[{probs.min():.4f}, {probs.max():.4f}]")
    return calibration


def main() -> None:
    p = argparse.ArgumentParser(description="Batch calibration for all baseline pkls.")
    p.add_argument("--out_dir", type=str, default="experiment_out",
                   help="Root experiment output directory to search for baseline pkls")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing calibration.pkl files")
    args = p.parse_args()

    pkls = sorted(Path(args.out_dir).rglob("base_experiment_metadata.pkl"))
    if not pkls:
        print(f"No base_experiment_metadata.pkl files found under {args.out_dir}")
        return

    print(f"Found {len(pkls)} baseline pkl(s) to calibrate.\n")
    for pkl_path in pkls:
        print(f"Processing: {pkl_path}")
        calibrate_one(pkl_path, args.out_dir, force=args.force)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
