"""
rq4_divergence_plot.py — Joint correctness/confidence divergence plot
(NAACL plan Experiment 4.5), the three headline cells, T0-T6.

Three panels (Claude Haiku/GPQA-Diamond, Claude Haiku/HLE, Claude Sonnet/
HLE), each showing: majority correctness (ground truth), GPT-5.4-Mini-
based confidence (the original calibrator used throughout the paper),
and Claude-Haiku-based confidence (a second, independent calibrator run
with the same prompts; the initial run's max_tokens=64 truncated ~43% of
responses before reaching the BELIEF/CONFIDENCE line, fixed by redoing
exactly those with a stronger no-preamble system prompt and
max_tokens=300 -- see calibration_haiku_redo.py -- bringing coverage to
97.6% step-level / 99.5% final-step).

Usage: python analysis/rq4_divergence_plot.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import paper_pipeline as pp

HEADLINE_CELLS = [("ClaudeHaiku", "gpqa_diamond", "Claude Haiku / GPQA-Diamond"),
                   ("ClaudeHaiku", "hle", "Claude Haiku / HLE"),
                   ("ClaudeSonnet", "hle", "Claude Sonnet / HLE")]

COLOR_CORRECT = "#111827"     # near-black: ground truth
COLOR_ORIG_CONF = "#2563EB"   # blue: original GPT-5.4-Mini confidence
COLOR_HAIKU_CONF = "#EA580C"  # orange: Claude Haiku confidence (partial)


def main():
    df = pd.read_pickle(os.path.join(pp.OUT, "calibration_haiku_results.pkl"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    turns = np.arange(0, pp.N_TURNS)

    for ax, (model, dataset, title) in zip(axes, HEADLINE_CELLS):
        sub = df[(df.model == model) & (df.dataset == dataset)]
        g = sub.groupby("turn").agg(
            correct_pct=("correct", lambda s: 100 * s.mean()),
            orig_conf=("orig_self_conf", "mean"),
            haiku_conf=("haiku_final_conf", "mean"),
            n_haiku=("haiku_final_conf", "count"),
            n=("correct", "size"),
        ).reindex(turns)

        ax.plot(turns, g["correct_pct"], "-o", color=COLOR_CORRECT, label="Majority correct (%)", markersize=4)
        ax.plot(turns, g["orig_conf"], "-s", color=COLOR_ORIG_CONF,
                 label="Confidence (GPT-5.4 Mini)", markersize=4)
        ax.plot(turns, g["haiku_conf"], "-^", color=COLOR_HAIKU_CONF,
                 label="Confidence (Claude Haiku)", markersize=4, alpha=0.85)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Pressure turn $T$")
        ax.set_ylim(-2, 102)
        ax.set_xticks(turns)

    axes[0].set_ylabel("Percent")
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("RQ4: correctness vs. confidence trajectories, two independent calibrators", fontsize=12)
    fig.text(0.5, -0.02,
              "Claude Haiku confidence from an independent step-by-step calibration run (final-step coverage "
              "99.5%; see app:calibration_haiku), using the same prompts as the original GPT-5.4 Mini calibrator.",
              ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()

    path = "paper/active_version/sections/fig/rq4_divergence.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
