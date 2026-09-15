"""
wrong_answer_plausibility.py
------------------------------
Reviewer-requested check (does flip rate scale with the plausibility of
the planted wrong answer?), using data already collected -- no new API
calls. limitations.tex already names this as feasible from the existing
K=25 baseline samples:

    "testing this, available from the existing K=25 baseline samples
    without further API calls, would separate question-level
    uncertainty (RQ1) from planted-answer plausibility"

This does NOT replace the full "targeted adversary" experiment the
reviewer also asked for (that needs new pressure runs with a
deliberately-chosen a-dagger, real API spend). It answers a narrower,
already-answerable question: across the runs we already have, where
a-dagger was drawn uniformly, did the ones that happened to draw a
more-plausible a-dagger flip more often?

For each (question, run), a-dagger's baseline plausibility is
p_hat_q(a-dagger) = fraction of that question's K=25 baseline samples
that independently produced a-dagger, i.e. how "tempting" that wrong
answer already was to the model before any pressure existed.

Usage:
    uv run python analysis/wrong_answer_plausibility.py
"""

from __future__ import annotations

import glob
import pickle
import re

import numpy as np
import pandas as pd
from scipy import stats

EXPERIMENT_OUT = "experiment_out"
MODELS = ["ClaudeHaiku", "ClaudeSonnet", "GPT5_4", "GPT5_4Mini", "GPT5_4Nano"]

_LETTER_RE = re.compile(r"^\s*([A-J])\b")


def _baseline_prob_map(model: str) -> dict[str, dict[str, float]]:
    """query -> {letter: p_hat} from the K=25 baseline samples."""
    with open(f"{EXPERIMENT_OUT}/{model}/base_experiment_metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    out = {}
    for item in meta:
        letters = []
        for ans in item.get("answers_generated", []):
            m = _LETTER_RE.match(str(ans))
            if m:
                letters.append(m.group(1))
        n = len(letters)
        if n == 0:
            continue
        counts: dict[str, int] = {}
        for letter in letters:
            counts[letter] = counts.get(letter, 0) + 1
        out[item["query"]] = {letter: c / n for letter, c in counts.items()}
    return out


def build_rows(model: str) -> list[dict]:
    prob_map = _baseline_prob_map(model)
    rows = []
    for pkl in sorted(glob.glob(f"{EXPERIMENT_OUT}/{model}/entropy_bin/bin_*_repeated.pkl")):
        with open(pkl, "rb") as f:
            items = pickle.load(f)
        for item in items:
            q = item["query"]
            q_probs = prob_map.get(q, {})
            for run in item.get("raw_runs", []):
                a_dagger = run.get("wrong_answer_letter")
                if a_dagger is None:
                    continue
                p_hat = q_probs.get(a_dagger, 0.0)
                ever_flipped = int(run.get("first_wrong_turn") is not None)
                rows.append({
                    "model": model, "query": q,
                    "a_dagger_p_hat": p_hat, "ever_flipped": ever_flipped,
                })
    return rows


def main() -> None:
    all_rows = []
    for model in MODELS:
        all_rows.extend(build_rows(model))
    df = pd.DataFrame(all_rows)
    print(f"Total (question, run) pairs: {len(df):,}\n")

    print("=" * 72)
    print("Does flip rate scale with the planted wrong answer's baseline")
    print("plausibility p_hat_q(a-dagger)? (Spearman rho, per model)")
    print("=" * 72)
    for model in MODELS:
        sub = df[df["model"] == model]
        if sub["a_dagger_p_hat"].nunique() < 2:
            print(f"  {model:15s} insufficient variation in a_dagger_p_hat, skipped")
            continue
        rho, p = stats.spearmanr(sub["a_dagger_p_hat"], sub["ever_flipped"])
        print(f"  {model:15s} n={len(sub):6,}  rho={rho:+.3f}  p={p:.4g}")

    print()
    print("Pooled across all 5 models:")
    rho, p = stats.spearmanr(df["a_dagger_p_hat"], df["ever_flipped"])
    print(f"  n={len(df):,}  rho={rho:+.3f}  p={p:.4g}")

    # Logistic regression, pooling models with model dummies, for an
    # effect-size read (not just direction/significance).
    import statsmodels.formula.api as smf
    model_res = smf.logit("ever_flipped ~ a_dagger_p_hat + C(model)", data=df).fit(disp=0)
    print()
    print("Pooled logistic regression (ever_flipped ~ a_dagger_p_hat + model dummies):")
    print(f"  beta(a_dagger_p_hat) = {model_res.params['a_dagger_p_hat']:+.4f}"
          f"  (SE={model_res.bse['a_dagger_p_hat']:.4f},"
          f"  p={model_res.pvalues['a_dagger_p_hat']:.4g})")


if __name__ == "__main__":
    main()
