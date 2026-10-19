"""
repair_gpqa_backfill.py — Repair pass after backfill_gpqa_validated.py.

The first backfill run (concurrency 24, 120s timeout) silently recorded failed
API calls as empty: ~100 turns came back with no text and ~half of the
calibrator steps in backfilled turns have confidence=None. A single-sample
probe at low load had zero errors, so the failures were load-dependent
(timeouts / transient gateway errors with no retry).

Phase 1 (regen): move aside the checkpoints of samples that still have an
  empty turn, then rerun backfill_gpqa_validated.main() — it regenerates each
  such sample from its first empty turn, with the known-good prefix as context.
Phase 2 (recal): for every step with confidence=None in a trace that has text,
  re-query the calibrator for that step only (incremental context, as in
  calibrate_trace), then rebuild derived fields and rewrite the bin files
  (backing up first).

Both phases wrap ResponseGenerator.acomplete with retries, and treat an empty
completion as a failure.

Usage:
    python scripts/repair_gpqa_backfill.py [--phase all|regen|recal] [--concurrency 8] [--timeout_s 300]
"""

import argparse
import asyncio
import glob
import os
import pickle
import random
import re
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import backfill_gpqa_validated as bf  # also installs the reasoning_uncertainty pickle shim
from dotenv import load_dotenv

from sycophancy.calibrator import _parse_calibration
from sycophancy.config import CALIBRATOR_STEP_PROMPT, CALIBRATOR_SYSTEM_MSG
from sycophancy.generator import ResponseGenerator
from sycophancy.reasoning_uncertainty import ReasoningStep, ReasoningTrace

CKPT_ROOT = os.path.join(bf.BIN_DIR, "_gpqa_validated_fill_checkpoint")
RECAL_CKPT_ROOT = os.path.join(bf.BIN_DIR, "_gpqa_recalibration_checkpoint")
MAX_ATTEMPTS = 5


def install_retries() -> None:
    orig = ResponseGenerator.acomplete

    async def acomplete_with_retry(self, messages, model, timeout_s=None, seed=1234):
        last_err = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                out = await orig(self, messages=messages, model=model, timeout_s=timeout_s, seed=seed)
                if out:
                    return out
                last_err = RuntimeError("empty completion")
            except Exception as e:
                last_err = e
            await asyncio.sleep(min(60, 2 ** attempt * 3) + random.random() * 3)
        print(f"    [retry] giving up on {model} after {MAX_ATTEMPTS} attempts: "
              f"{type(last_err).__name__}: {str(last_err)[:200]}", flush=True)
        raise last_err

    ResponseGenerator.acomplete = acomplete_with_retry


# ---------------------------------------------------------------------------
# Phase 1 — regenerate empty turns
# ---------------------------------------------------------------------------

async def phase_regen(concurrency: int, timeout_s: float) -> None:
    stale_dir = os.path.join(bf.BIN_DIR, f"_stale_fill_checkpoints_{datetime.now():%Y%m%d_%H%M%S}")
    n_moved = 0
    for bin_dir in sorted(glob.glob(os.path.join(CKPT_ROOT, "bin_*"))):
        for path in glob.glob(os.path.join(bin_dir, "*.pkl")):
            with open(path, "rb") as f:
                traces = pickle.load(f)
            if any(not tr.raw_text for tr in traces):
                dest = os.path.join(stale_dir, os.path.basename(bin_dir))
                os.makedirs(dest, exist_ok=True)
                shutil.move(path, os.path.join(dest, os.path.basename(path)))
                n_moved += 1
    print(f"[regen] moved {n_moved} incomplete checkpoints aside -> {stale_dir}", flush=True)

    sys.argv = [bf.__file__, "--concurrency", str(concurrency), "--timeout_s", str(timeout_s)]
    await bf.main()


# ---------------------------------------------------------------------------
# Phase 2 — recalibrate steps with confidence=None
# ---------------------------------------------------------------------------

async def recalibrate_trace_missing(sem, rg, question, options, trace, calibrator_model, timeout_s, seed):
    missing = [i for i, s in enumerate(trace.steps) if s.confidence is None]
    if not missing:
        return trace, 0

    async def one(ell):
        steps_text = "\n\n".join(f"Step {i + 1}:\n{s.step_text}" for i, s in enumerate(trace.steps[: ell + 1]))
        prompt = CALIBRATOR_STEP_PROMPT.format(
            question=question, options="\n".join(options), step_num=ell + 1, steps_so_far=steps_text,
        )
        async with sem:
            try:
                raw = await rg.acomplete(
                    messages=[CALIBRATOR_SYSTEM_MSG, {"role": "user", "content": prompt}],
                    model=calibrator_model, timeout_s=timeout_s, seed=seed + ell,
                )
            except Exception:
                return ell, (None, None)
        return ell, _parse_calibration(raw)

    results = dict(await asyncio.gather(*[one(ell) for ell in missing]))
    new_steps = []
    for i, s in enumerate(trace.steps):
        belief, conf = results.get(i, (None, None))
        new_steps.append(ReasoningStep(
            step_index=s.step_index,
            step_text=s.step_text,
            current_belief=belief if belief is not None else s.current_belief,
            confidence=conf if conf is not None else s.confidence,
        ))
    last_belief = next((s.current_belief for s in reversed(new_steps) if s.current_belief), None)
    last_step_flip = (
        trace.final_answer is not None and last_belief is not None and trace.final_answer != last_belief
    )
    new_trace = ReasoningTrace(
        raw_text=trace.raw_text, steps=new_steps, final_answer=trace.final_answer,
        last_step_belief=last_belief, last_step_flip=last_step_flip,
        gold_answer=trace.gold_answer, final_confidence=trace.final_confidence,
    )
    n_fixed = sum(1 for i in missing if new_steps[i].confidence is not None)
    return new_trace, n_fixed


async def phase_recal(concurrency: int, timeout_s: float, calibrator_model: str, encoder) -> None:
    with open(bf.META_PATH, "rb") as f:
        meta_by_query = {m["query"]: m for m in pickle.load(f)}
    sem = asyncio.Semaphore(concurrency)
    rg = ResponseGenerator()

    for bin_idx in (0, 1):
        path = os.path.join(bf.BIN_DIR, f"bin_{bin_idx}_reasoning.pkl")
        with open(path, "rb") as f:
            trajs = pickle.load(f)
        ckpt_dir = os.path.join(RECAL_CKPT_ROOT, f"bin_{bin_idx}")
        os.makedirs(ckpt_dir, exist_ok=True)

        jobs = []  # (traj, turn, sample)
        for traj in trajs:
            for t, turn in enumerate(traj.raw_traces):
                for s, tr in enumerate(turn):
                    if tr.raw_text and any(st.confidence is None for st in tr.steps):
                        jobs.append((traj, t, s))
        n_steps = sum(
            sum(st.confidence is None for st in traj.raw_traces[t][s].steps) for traj, t, s in jobs
        )
        print(f"[recal] bin{bin_idx}: {len(jobs)} traces, {n_steps} steps missing confidence", flush=True)
        if not jobs:
            continue

        def cpath(query, t, s):
            safe = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:60]
            return os.path.join(ckpt_dir, f"{safe}__t{t}__s{s}.pkl")

        async def run(job):
            traj, t, s = job
            cp = cpath(traj.query, t, s)
            if os.path.exists(cp):
                with open(cp, "rb") as f:
                    return job, pickle.load(f), 0
            item = meta_by_query.get(traj.query, {})
            new_tr, n_fixed = await recalibrate_trace_missing(
                sem, rg, item.get("query", traj.query), item.get("options", []),
                traj.raw_traces[t][s], calibrator_model, timeout_s, seed=515000 + t * 100 + s * 10,
            )
            with open(cp + ".tmp", "wb") as f:
                pickle.dump(new_tr, f)
            os.replace(cp + ".tmp", cp)
            return job, new_tr, n_fixed

        done = fixed = 0
        touched = set()
        for fut in asyncio.as_completed([run(j) for j in jobs]):
            (traj, t, s), new_tr, n_fixed = await fut
            traj.raw_traces[t][s] = new_tr
            touched.add(traj.query)
            done += 1
            fixed += n_fixed
            if done % 20 == 0 or done == len(jobs):
                print(f"  ...{done}/{len(jobs)} traces recalibrated", flush=True)

        remaining = sum(
            st.confidence is None
            for traj in trajs for turn in traj.raw_traces for tr in turn if tr.raw_text for st in tr.steps
        )
        print(f"[recal] bin{bin_idx}: steps still missing confidence after recal: {remaining}", flush=True)
        write_bin(bin_idx, trajs, touched, encoder, tag="recal")


def write_bin(bin_idx, trajs, touched_queries, encoder, tag):
    new_trajs = []
    for traj in trajs:
        if traj.query in touched_queries:
            new_trajs.append(bf.build_uncertainty_trajectory(
                query=traj.query, gold_answer=traj.gold_answer, all_traces=traj.raw_traces,
                n_clusters=bf.DEFAULT_N_CLUSTERS, encoder=encoder,
            ))
        else:
            new_trajs.append(traj)

    backup_dir = os.path.join(bf.BIN_DIR, f"_backup_before_gpqa_{tag}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(backup_dir, exist_ok=True)
    for suffix in ("reasoning", "summary", "cross_turn"):
        src = os.path.join(bf.BIN_DIR, f"bin_{bin_idx}_{suffix}.pkl")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(backup_dir, f"bin_{bin_idx}_{suffix}.pkl"))

    with open(os.path.join(bf.BIN_DIR, f"bin_{bin_idx}_reasoning.pkl"), "wb") as f:
        pickle.dump(new_trajs, f)
    summary_rows, cross_turn_rows = [], []
    for traj in new_trajs:
        summary_rows.extend(bf.summarise_trajectory(traj))
        cross_turn_rows.extend(bf.summarise_cross_turn_comparison(traj))
    with open(os.path.join(bf.BIN_DIR, f"bin_{bin_idx}_summary.pkl"), "wb") as f:
        pickle.dump(summary_rows, f)
    with open(os.path.join(bf.BIN_DIR, f"bin_{bin_idx}_cross_turn.pkl"), "wb") as f:
        pickle.dump(cross_turn_rows, f)
    print(f"  bin{bin_idx}: wrote {tag} results for {len(touched_queries)} items, backup at {backup_dir}", flush=True)


async def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["all", "regen", "recal"], default="all")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout_s", type=float, default=300.0)
    p.add_argument("--calibrator_model", type=str, default=bf.DEFAULT_CALIBRATOR_MODEL)
    args = p.parse_args()

    install_retries()

    if args.phase in ("all", "regen"):
        await phase_regen(args.concurrency, args.timeout_s)
    if args.phase in ("all", "recal"):
        encoder = None
        try:
            encoder = bf._load_encoder(bf.DEFAULT_EMBEDDING_MODEL)
        except Exception as e:
            print(f"WARNING: could not load encoder ({e}); belief-entropy-only rebuild.", flush=True)
        await phase_recal(args.concurrency, args.timeout_s, args.calibrator_model, encoder)


if __name__ == "__main__":
    asyncio.run(main())
