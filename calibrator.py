"""
calibrator.py — External per-step belief and confidence calibration.

After a reasoning model produces a free-form chain-of-thought (no inline
CURRENT BELIEF / CONFIDENCE markers), a separate calibrator model reads
the partial reasoning trace up to each step ℓ and assigns:
  - belief   : which answer option does the reasoning lean toward?
  - confidence: how certain does the reasoning appear? (0-100)

The calibration is incremental: when assessing step ℓ the calibrator
sees steps 1..ℓ only, simulating a reader who evaluates the emerging
argument as it unfolds rather than retrospectively.

The returned ReasoningTrace is fully compatible with
build_uncertainty_trajectory() — the calibrator-assigned .current_belief
and .confidence fields slot into the same dataclass fields that the
structured CoT pipeline populates via inline markers.
"""

from __future__ import annotations

import asyncio
import re
from typing import List, Optional, Tuple

from config import (
    CALIBRATOR_STEP_PROMPT,
    CALIBRATOR_SYSTEM_MSG,
    DEFAULT_CALIBRATOR_MODEL,
)
from generator import ResponseGenerator
from reasoning_uncertainty import ReasoningStep, ReasoningTrace

_BELIEF_RE = re.compile(r"BELIEF\s*:\s*([A-J])", re.IGNORECASE)
_CONF_RE   = re.compile(r"CONFIDENCE\s*:\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_calibration(raw: str) -> Tuple[Optional[str], Optional[float]]:
    belief_m = _BELIEF_RE.search(raw or "")
    conf_m   = _CONF_RE.search(raw or "")
    belief   = belief_m.group(1).upper() if belief_m else None
    conf     = float(conf_m.group(1))    if conf_m   else None
    return belief, conf


# ---------------------------------------------------------------------------
# Single-step calibration call
# ---------------------------------------------------------------------------

async def calibrate_step(
    sem:              asyncio.Semaphore,
    rg:               ResponseGenerator,
    question:         str,
    options:          List[str],
    steps_so_far:     List[str],
    step_num:         int,
    calibrator_model: str,
    timeout_s:        Optional[float],
    seed:             int,
) -> Tuple[Optional[str], Optional[float]]:
    """
    Queries the calibrator to assess belief and confidence at step `step_num`.
    `steps_so_far` contains the text of steps 1..step_num (inclusive).
    Returns (belief_letter, confidence_0_to_100).
    """
    options_text = "\n".join(options)
    steps_text   = "\n\n".join(
        f"Step {i + 1}:\n{text}" for i, text in enumerate(steps_so_far)
    )
    prompt = CALIBRATOR_STEP_PROMPT.format(
        question=question,
        options=options_text,
        step_num=step_num,
        steps_so_far=steps_text,
    )
    messages = [
        CALIBRATOR_SYSTEM_MSG,
        {"role": "user", "content": prompt},
    ]
    async with sem:
        try:
            raw = await rg.acomplete(
                messages=messages,
                model=calibrator_model,
                timeout_s=timeout_s,
                seed=seed,
            )
        except Exception:
            return None, None
    return _parse_calibration(raw)


# ---------------------------------------------------------------------------
# Full-trace calibration
# ---------------------------------------------------------------------------

async def calibrate_trace(
    sem:              asyncio.Semaphore,
    rg:               ResponseGenerator,
    question:         str,
    options:          List[str],
    trace:            ReasoningTrace,
    calibrator_model: str,
    timeout_s:        Optional[float],
    base_seed:        int,
) -> ReasoningTrace:
    """
    Returns a new ReasoningTrace with each step's .current_belief and
    .confidence populated by the calibrator model.

    All step calibrations for one trace are launched concurrently.
    Step ℓ receives steps 1..ℓ as context (incremental calibration).
    """
    if not trace.steps:
        return trace

    tasks = [
        calibrate_step(
            sem=sem,
            rg=rg,
            question=question,
            options=options,
            steps_so_far=[s.step_text for s in trace.steps[: ell + 1]],
            step_num=ell + 1,
            calibrator_model=calibrator_model,
            timeout_s=timeout_s,
            seed=base_seed + ell,
        )
        for ell in range(len(trace.steps))
    ]
    calibrations: List[Tuple[Optional[str], Optional[float]]] = (
        await asyncio.gather(*tasks)
    )

    new_steps = [
        ReasoningStep(
            step_index=step.step_index,
            step_text=step.step_text,
            current_belief=belief if belief is not None else step.current_belief,
            confidence=conf   if conf   is not None else step.confidence,
        )
        for step, (belief, conf) in zip(trace.steps, calibrations)
    ]

    last_belief    = next(
        (s.current_belief for s in reversed(new_steps) if s.current_belief), None
    )
    last_step_flip = (
        trace.final_answer is not None
        and last_belief    is not None
        and trace.final_answer != last_belief
    )

    return ReasoningTrace(
        raw_text=trace.raw_text,
        steps=new_steps,
        final_answer=trace.final_answer,
        last_step_belief=last_belief,
        last_step_flip=last_step_flip,
        gold_answer=trace.gold_answer,
        final_confidence=trace.final_confidence,
    )
