"""
An empty/failed generation must produce NO row at all (not flipped=0/1 and
not flipped=NaN-in-row) — see backfill_flat_from_reasoning's docstring.
This is the invariant the GPQA-Diamond coverage backfill relies on so an
empty response can't silently inflate a flip-rate denominator.
"""

import pickle
from types import SimpleNamespace

from sycophancy.build_analysis_dfs import backfill_flat_from_reasoning


def _traj(query, gold_answer, final_answers_by_turn):
    """final_answers_by_turn: list of lists, one inner list per turn, one
    entry per run (None = empty/failed generation for that run's turn)."""
    raw_traces = [
        [SimpleNamespace(final_answer=fa) for fa in turn_answers]
        for turn_answers in final_answers_by_turn
    ]
    return SimpleNamespace(query=query, gold_answer=gold_answer, raw_traces=raw_traces)


def _write_experiment_dir(tmp_path, model, dataset, trajs):
    base = tmp_path / model / dataset
    (base / "reasoning_calibrated_bin").mkdir(parents=True)

    metadata = [{"query": t.query, "entropy": 0.0, "category": "test"} for t in trajs]
    with open(base / "base_experiment_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    with open(base / "reasoning_calibrated_bin" / "bin_0_reasoning.pkl", "wb") as f:
        pickle.dump(trajs, f)

    return str(tmp_path)


def test_empty_generation_produces_no_row(tmp_path):
    # 2 runs; turn 0 both answer "A" (correct); turn 1 run 0 flips to "B",
    # run 1's turn-1 generation failed (None).
    traj = _traj(
        query="q1",
        gold_answer="A",
        final_answers_by_turn=[["A", "A"], ["B", None]],
    )
    experiment_dir = _write_experiment_dir(tmp_path, "TestModel", "gpqa_diamond", [traj])

    df = backfill_flat_from_reasoning(
        model="TestModel",
        dataset="gpqa_diamond",
        gap_queries={"q1"},
        experiment_dir=experiment_dir,
    )

    # 2 runs x 2 turns = 4 observations if nothing were missing; one failed
    # generation means exactly 3 rows, never 4 and never a flipped=NaN row.
    assert len(df) == 3
    assert not df["flipped"].isna().any()

    run0 = df[df["run_idx"] == 0].sort_values("turn")
    assert list(run0["flipped"]) == [0, 1]  # turn0 correct, turn1 flipped

    run1 = df[df["run_idx"] == 1]
    assert list(run1["turn"]) == [0]  # turn 1 dropped entirely, not zeroed/NaNed


def test_no_gap_queries_returns_empty_df(tmp_path):
    traj = _traj(query="q1", gold_answer="A", final_answers_by_turn=[["A"]])
    experiment_dir = _write_experiment_dir(tmp_path, "TestModel", "gpqa_diamond", [traj])

    df = backfill_flat_from_reasoning(
        model="TestModel",
        dataset="gpqa_diamond",
        gap_queries=set(),
        experiment_dir=experiment_dir,
    )
    assert df.empty
