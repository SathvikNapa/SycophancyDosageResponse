"""
reasoning_uncertainty.py — Step-wise uncertainty estimation during CoT reasoning.

Implements the professor's suggestion: prompt the model to decompose reasoning
into explicit steps, apply semantic clustering across N runs to measure
uncertainty at each point in the reasoning chain, and track how that uncertainty
changes under social pressure across turns.

Three advances over the previous version
─────────────────────────────────────────
1. Global clustering (was: positional per-step KMeans)
   A SINGLE KMeans is fit on ALL step embeddings from ALL turns and ALL runs.
   Cluster IDs are therefore consistent — step (T0, run=2, step=3) and step
   (T2, run=0, step=1) share a cluster if they contain similar reasoning.
   This enables tracking cluster membership trajectories rather than just
   per-position entropy snapshots.

2. Cross-turn trajectory comparison
   compare_trajectories_across_turns() answers the core question: does pressure
   shift the model's reasoning mid-chain before the final answer flips, or does
   the flip only happen at the last step? Per step position it computes:
     - cluster_entropy change from T0 to each subsequent turn
     - cluster_drift: fraction of runs whose cluster assignment changed from T0
     - divergence_turn: first turn where entropy exceeds baseline by a threshold
     - belief_shift_turn: first turn where the majority belief at this step changes

3. Free-form CoT parsing
   parse_reasoning_steps_freeform() does not require CURRENT BELIEF markers.
   It segments responses on natural boundaries and heuristically extracts
   letter signals, making it suitable for unconstrained model reasoning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import DEFAULT_N_CLUSTERS, DEFAULT_EMBEDDING_MODEL
from entropy import compute_entropy

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    _HAVE_ST = True
except ImportError:
    _HAVE_ST = False

_CURRENT_BELIEF_RE = re.compile(r"CURRENT\s+BELIEF\s*:\s*([A-J])", re.IGNORECASE)
_FINAL_ANSWER_RE   = re.compile(r"FINAL\s+ANSWER\s*:\s*([A-J])",   re.IGNORECASE)
_LETTER_SIGNAL_RE  = re.compile(r"\b([A-J])\b",                     re.IGNORECASE)

_FREEFORM_STEP_RE = re.compile(
    r"(?:^|\n)(?:\d+[\.\)]\s|Step\s+\d+[:\.]?\s|\*\*Step)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    step_index:     int
    step_text:      str
    current_belief: Optional[str]


@dataclass
class ReasoningTrace:
    raw_text:         str
    steps:            List[ReasoningStep]
    final_answer:     Optional[str]
    last_step_belief: Optional[str]
    last_step_flip:   bool
    gold_answer:      Optional[str] = None


@dataclass
class StepUncertainty:
    step_index:          int
    n_runs:              int
    belief_entropy:      float
    cluster_entropy:     float
    semantic_spread:     float
    belief_counts:       Dict[str, int]
    cluster_labels:      List[int]        # global cluster id per run (-1 = missing)
    majority_belief:     Optional[str]
    majority_is_correct: Optional[bool]


@dataclass
class CrossTurnStepComparison:
    """Cross-turn analysis for one step position."""
    step_index:               int
    belief_entropy_by_turn:   List[float]
    cluster_entropy_by_turn:  List[float]
    semantic_spread_by_turn:  List[float]
    majority_belief_by_turn:  List[Optional[str]]
    majority_correct_by_turn: List[Optional[bool]]
    cluster_drift_by_turn:    List[float]   # fraction of runs whose cluster changed from T0
    divergence_turn:          Optional[int] # first turn where cluster_H exceeds T0 + threshold
    belief_shift_turn:        Optional[int] # first turn where majority_belief differs from T0


@dataclass
class UncertaintyTrajectory:
    query:                str
    gold_answer:          Optional[str]
    turn_trajectories:    List[List[StepUncertainty]]         # [turn][step]
    last_step_flip_rates: List[float]
    cross_turn:           Optional[List[CrossTurnStepComparison]] = None
    raw_traces:           List[List[ReasoningTrace]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing — structured
# ---------------------------------------------------------------------------

def parse_reasoning_steps(
    raw_text:    str,
    gold_answer: Optional[str] = None,
) -> ReasoningTrace:
    """
    Parses a structured CoT response that uses CURRENT BELIEF / FINAL ANSWER markers.
    Lenient — captures whatever signals are present even if format is partial.
    """
    fa_match     = _FINAL_ANSWER_RE.search(raw_text)
    final_answer = fa_match.group(1).upper() if fa_match else None
    text_body    = _FINAL_ANSWER_RE.sub("", raw_text).strip()

    belief_matches = list(_CURRENT_BELIEF_RE.finditer(text_body))

    if not belief_matches:
        return ReasoningTrace(
            raw_text=raw_text,
            steps=[ReasoningStep(0, text_body.strip(), final_answer)],
            final_answer=final_answer,
            last_step_belief=final_answer,
            last_step_flip=False,
            gold_answer=gold_answer,
        )

    steps: List[ReasoningStep] = []
    prev_end = 0
    for i, m in enumerate(belief_matches):
        step_text = text_body[prev_end:m.start()].strip()
        belief    = m.group(1).upper()
        steps.append(ReasoningStep(i, step_text, belief))
        prev_end  = m.end()

    last_belief    = steps[-1].current_belief if steps else final_answer
    last_step_flip = (
        final_answer is not None
        and last_belief is not None
        and final_answer != last_belief
    )

    return ReasoningTrace(
        raw_text=raw_text,
        steps=steps,
        final_answer=final_answer,
        last_step_belief=last_belief,
        last_step_flip=last_step_flip,
        gold_answer=gold_answer,
    )


# ---------------------------------------------------------------------------
# Parsing — free-form
# ---------------------------------------------------------------------------

def _extract_letter_signal(text: str) -> Optional[str]:
    answer_signal = re.search(
        r"(?:answer|therefore|conclude|so|thus|=|is)\s*[:\s]*([A-J])\b",
        text, re.IGNORECASE,
    )
    if answer_signal:
        return answer_signal.group(1).upper()
    matches = _LETTER_SIGNAL_RE.findall(text)
    return matches[-1].upper() if matches else None


def parse_reasoning_steps_freeform(
    raw_text:    str,
    gold_answer: Optional[str] = None,
) -> ReasoningTrace:
    """
    Parses a free-form CoT response without requiring CURRENT BELIEF markers.

    Segments on numbered items / step headers / blank lines and heuristically
    extracts a letter signal from each segment. More faithful to unconstrained
    reasoning; beliefs are noisier so analysis relies more on semantic clustering.
    """
    fa_match     = _FINAL_ANSWER_RE.search(raw_text)
    final_answer = fa_match.group(1).upper() if fa_match else None

    if final_answer is None:
        for line in reversed(raw_text.strip().split("\n")[-3:]):
            sig = _extract_letter_signal(line)
            if sig:
                final_answer = sig
                break

    text_body = _FINAL_ANSWER_RE.sub("", raw_text).strip()

    split_positions = [m.start() for m in _FREEFORM_STEP_RE.finditer(text_body)]
    if len(split_positions) < 2:
        segments = [s.strip() for s in re.split(r"\n\s*\n", text_body) if s.strip()]
    else:
        segments = []
        for i, pos in enumerate(split_positions):
            end = split_positions[i + 1] if i + 1 < len(split_positions) else len(text_body)
            segments.append(text_body[pos:end].strip())

    if not segments:
        segments = [text_body.strip()]

    steps = [ReasoningStep(i, seg, _extract_letter_signal(seg)) for i, seg in enumerate(segments)]
    last_belief    = next((s.current_belief for s in reversed(steps) if s.current_belief), None)
    last_step_flip = (
        final_answer is not None
        and last_belief is not None
        and final_answer != last_belief
    )

    return ReasoningTrace(
        raw_text=raw_text,
        steps=steps,
        final_answer=final_answer,
        last_step_belief=last_belief,
        last_step_flip=last_step_flip,
        gold_answer=gold_answer,
    )


# ---------------------------------------------------------------------------
# Last-step flip detection
# ---------------------------------------------------------------------------

def detect_last_step_flip(trace: ReasoningTrace) -> bool:
    """
    True if the model reasoned to the correct belief at its last step
    but committed a different (wrong) final answer.
    """
    if trace.gold_answer is None:
        return False
    return (
        trace.last_step_belief == trace.gold_answer
        and trace.final_answer != trace.gold_answer
    )


# ---------------------------------------------------------------------------
# Global cluster model
# ---------------------------------------------------------------------------

def _load_encoder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> "SentenceTransformer":
    if not _HAVE_ST:
        raise ImportError("pip install sentence-transformers")
    return SentenceTransformer(model_name)


def fit_global_cluster_model(
    all_traces: List[List[ReasoningTrace]],
    encoder:    "SentenceTransformer",
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> Tuple["KMeans", np.ndarray, List[Tuple[int, int, int]]]:
    """
    Embeds ALL step texts from ALL turns and ALL runs into a shared space
    and fits ONE KMeans model so cluster IDs are consistent everywhere.

    Returns:
        km             : fitted KMeans
        embeddings     : (N_total_steps, d) float array
        step_index_map : list of (turn_idx, run_idx, step_idx) — one per row
    """
    texts: List[str] = []
    step_index_map: List[Tuple[int, int, int]] = []

    for t_idx, turn_traces in enumerate(all_traces):
        for r_idx, trace in enumerate(turn_traces):
            for s_idx, step in enumerate(trace.steps):
                texts.append(step.step_text or "")
                step_index_map.append((t_idx, r_idx, s_idx))

    if not texts:
        raise ValueError("No step texts found across all traces.")

    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    k  = min(n_clusters, len(texts))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(embeddings)
    return km, embeddings, step_index_map


def _assign_global_clusters(
    km:                 "KMeans",
    embeddings:         np.ndarray,
    step_index_map:     List[Tuple[int, int, int]],
    n_turns:            int,
    n_runs_per_turn:    List[int],
    max_steps_per_turn: List[int],
) -> List[List[List[int]]]:
    """
    Returns cluster_grid[turn][run][step] = global cluster label (int, -1 = missing).
    """
    labels = km.predict(embeddings)
    grid: List[List[List[int]]] = [
        [[-1] * max_steps_per_turn[t] for _ in range(n_runs_per_turn[t])]
        for t in range(n_turns)
    ]
    for row_idx, (t, r, s) in enumerate(step_index_map):
        if s < len(grid[t][r]):
            grid[t][r][s] = int(labels[row_idx])
    return grid


# ---------------------------------------------------------------------------
# Per-step uncertainty from global clusters
# ---------------------------------------------------------------------------

def _step_uncertainty_from_global(
    step_index:     int,
    turn_idx:       int,
    cluster_grid:   List[List[List[int]]],
    embeddings:     np.ndarray,
    step_index_map: List[Tuple[int, int, int]],
    traces:         List[ReasoningTrace],
    gold_answer:    Optional[str],
    n_clusters:     int,
) -> StepUncertainty:
    labels  = []
    beliefs = []
    emb_rows: List[np.ndarray] = []

    for r_idx, trace in enumerate(traces):
        lbl = cluster_grid[turn_idx][r_idx][step_index] if step_index < len(cluster_grid[turn_idx][r_idx]) else -1
        labels.append(lbl)
        bel = trace.steps[step_index].current_belief if step_index < len(trace.steps) else None
        beliefs.append(bel)

    for row_idx, (t, r, s) in enumerate(step_index_map):
        if t == turn_idx and s == step_index:
            emb_rows.append(embeddings[row_idx])

    valid_labels   = [l for l in labels if l >= 0]
    if valid_labels:
        counts     = np.bincount(valid_labels, minlength=n_clusters)
        probs      = counts / counts.sum()
        cl_entropy = float(-(probs * np.log(probs + 1e-9)).sum())
    else:
        cl_entropy = 0.0

    if len(emb_rows) >= 2:
        E      = np.stack(emb_rows)
        cos    = E @ E.T
        upper  = np.triu_indices(len(emb_rows), k=1)
        spread = float(1.0 - cos[upper].mean())
    else:
        spread = 0.0

    valid_beliefs = [b for b in beliefs if b is not None]
    belief_counts = {b: valid_beliefs.count(b) for b in set(valid_beliefs)}
    majority      = max(belief_counts, key=belief_counts.get) if belief_counts else None

    return StepUncertainty(
        step_index=step_index,
        n_runs=len(traces),
        belief_entropy=compute_entropy(valid_beliefs),
        cluster_entropy=cl_entropy,
        semantic_spread=spread,
        belief_counts=belief_counts,
        cluster_labels=labels,
        majority_belief=majority,
        majority_is_correct=(majority == gold_answer) if majority and gold_answer else None,
    )


# ---------------------------------------------------------------------------
# Trajectory builder
# ---------------------------------------------------------------------------

def build_uncertainty_trajectory(
    query:                str,
    gold_answer:          Optional[str],
    all_traces:           List[List[ReasoningTrace]],
    n_clusters:           int   = DEFAULT_N_CLUSTERS,
    encoder:              Optional["SentenceTransformer"] = None,
    divergence_threshold: float = 0.1,
) -> UncertaintyTrajectory:
    """
    Builds the full uncertainty trajectory for one question.
    all_traces[turn_idx][run_idx] = ReasoningTrace.
    Uses global clustering so cluster IDs are consistent across turns.
    """
    if encoder is None and _HAVE_ST:
        encoder = _load_encoder()

    n_turns             = len(all_traces)
    n_runs_per_turn     = [len(t) for t in all_traces]
    max_steps_per_turn  = [max((len(t.steps) for t in traces), default=0) for traces in all_traces]

    # Global clustering
    if encoder is not None and any(max_steps_per_turn):
        km, embeddings, step_index_map = fit_global_cluster_model(
            all_traces=all_traces, encoder=encoder, n_clusters=n_clusters,
        )
        cluster_grid = _assign_global_clusters(
            km=km, embeddings=embeddings, step_index_map=step_index_map,
            n_turns=n_turns, n_runs_per_turn=n_runs_per_turn,
            max_steps_per_turn=max_steps_per_turn,
        )
        use_global = True
    else:
        embeddings = cluster_grid = step_index_map = None
        use_global = False

    # Per-turn, per-step uncertainty
    turn_trajectories:    List[List[StepUncertainty]] = []
    last_step_flip_rates: List[float]                 = []

    for t_idx, turn_traces in enumerate(all_traces):
        step_uncerts: List[StepUncertainty] = []
        for s_pos in range(max_steps_per_turn[t_idx]):
            if use_global:
                su = _step_uncertainty_from_global(
                    step_index=s_pos, turn_idx=t_idx,
                    cluster_grid=cluster_grid, embeddings=embeddings,
                    step_index_map=step_index_map, traces=turn_traces,
                    gold_answer=gold_answer, n_clusters=n_clusters,
                )
            else:
                beliefs = [
                    t.steps[s_pos].current_belief if s_pos < len(t.steps) else None
                    for t in turn_traces
                ]
                valid   = [b for b in beliefs if b is not None]
                bc      = {b: valid.count(b) for b in set(valid)}
                maj     = max(bc, key=bc.get) if bc else None
                su = StepUncertainty(
                    step_index=s_pos, n_runs=len(turn_traces),
                    belief_entropy=compute_entropy(valid),
                    cluster_entropy=0.0, semantic_spread=0.0,
                    belief_counts=bc, cluster_labels=[-1] * len(turn_traces),
                    majority_belief=maj,
                    majority_is_correct=(maj == gold_answer) if maj and gold_answer else None,
                )
            step_uncerts.append(su)

        turn_trajectories.append(step_uncerts)
        last_step_flip_rates.append(
            float(np.mean([detect_last_step_flip(t) for t in turn_traces]))
        )

    # Cross-turn comparison
    cross_turn = compare_trajectories_across_turns(
        turn_trajectories=turn_trajectories,
        cluster_grid=cluster_grid,
        divergence_threshold=divergence_threshold,
    )

    return UncertaintyTrajectory(
        query=query,
        gold_answer=gold_answer,
        turn_trajectories=turn_trajectories,
        last_step_flip_rates=last_step_flip_rates,
        cross_turn=cross_turn,
        raw_traces=all_traces,
    )


# ---------------------------------------------------------------------------
# Cross-turn comparison
# ---------------------------------------------------------------------------

def compare_trajectories_across_turns(
    turn_trajectories:   List[List[StepUncertainty]],
    cluster_grid:        Optional[List[List[List[int]]]],
    divergence_threshold: float = 0.1,
) -> List[CrossTurnStepComparison]:
    """
    For each step position in T0, compares how reasoning content and beliefs
    change across T1..TK.

    cluster_drift : fraction of runs whose global cluster assignment changed
                    from T0 — measures whether pressure causes genuinely different
                    reasoning content at this step, not just different final answers.

    divergence_turn   : first turn where cluster_entropy > T0_entropy + threshold
    belief_shift_turn : first turn where majority_belief at this step differs from T0
    """
    if not turn_trajectories:
        return []

    n_turns        = len(turn_trajectories)
    baseline_steps = turn_trajectories[0]
    comparisons: List[CrossTurnStepComparison] = []

    for s_pos, baseline_su in enumerate(baseline_steps):
        bel_H:   List[float]         = []
        cl_H:    List[float]         = []
        spread:  List[float]         = []
        maj_bel: List[Optional[str]] = []
        maj_ok:  List[Optional[bool]]= []
        drift:   List[float]         = []

        baseline_labels = baseline_su.cluster_labels

        for t_idx in range(n_turns):
            su = turn_trajectories[t_idx][s_pos] if s_pos < len(turn_trajectories[t_idx]) else None

            bel_H.append(su.belief_entropy    if su else float("nan"))
            cl_H.append(su.cluster_entropy    if su else float("nan"))
            spread.append(su.semantic_spread  if su else float("nan"))
            maj_bel.append(su.majority_belief if su else None)
            maj_ok.append(su.majority_is_correct if su else None)

            if t_idx == 0 or su is None or cluster_grid is None:
                drift.append(0.0)
            else:
                curr = su.cluster_labels
                pairs = [(b, c) for b, c in zip(baseline_labels, curr) if b >= 0 and c >= 0]
                drift.append(sum(1 for b, c in pairs if b != c) / len(pairs) if pairs else 0.0)

        baseline_ce = cl_H[0]
        divergence_turn = next(
            (t for t in range(1, n_turns)
             if not np.isnan(cl_H[t]) and cl_H[t] > baseline_ce + divergence_threshold),
            None,
        )
        baseline_belief = maj_bel[0]
        belief_shift_turn = next(
            (t for t in range(1, n_turns)
             if maj_bel[t] is not None and maj_bel[t] != baseline_belief),
            None,
        )

        comparisons.append(CrossTurnStepComparison(
            step_index=s_pos,
            belief_entropy_by_turn=bel_H,
            cluster_entropy_by_turn=cl_H,
            semantic_spread_by_turn=spread,
            majority_belief_by_turn=maj_bel,
            majority_correct_by_turn=maj_ok,
            cluster_drift_by_turn=drift,
            divergence_turn=divergence_turn,
            belief_shift_turn=belief_shift_turn,
        ))

    return comparisons


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarise_trajectory(traj: UncertaintyTrajectory) -> List[dict]:
    """One row per (turn, step)."""
    rows = []
    for t_idx, turn_steps in enumerate(traj.turn_trajectories):
        lsf = traj.last_step_flip_rates[t_idx] if t_idx < len(traj.last_step_flip_rates) else None
        for su in turn_steps:
            rows.append({
                "query":               traj.query[:80],
                "gold_answer":         traj.gold_answer,
                "turn":                t_idx,
                "step":                su.step_index,
                "belief_entropy":      su.belief_entropy,
                "cluster_entropy":     su.cluster_entropy,
                "semantic_spread":     su.semantic_spread,
                "majority_belief":     su.majority_belief,
                "majority_is_correct": su.majority_is_correct,
                "last_step_flip_rate": lsf,
            })
    return rows


def summarise_cross_turn_comparison(traj: UncertaintyTrajectory) -> List[dict]:
    """One row per (step, turn) — use for plotting entropy/drift trajectories."""
    if not traj.cross_turn:
        return []
    rows = []
    for comp in traj.cross_turn:
        for t_idx in range(len(comp.belief_entropy_by_turn)):
            rows.append({
                "query":               traj.query[:80],
                "gold_answer":         traj.gold_answer,
                "step":                comp.step_index,
                "turn":                t_idx,
                "belief_entropy":      comp.belief_entropy_by_turn[t_idx],
                "cluster_entropy":     comp.cluster_entropy_by_turn[t_idx],
                "semantic_spread":     comp.semantic_spread_by_turn[t_idx],
                "majority_belief":     comp.majority_belief_by_turn[t_idx],
                "majority_is_correct": comp.majority_correct_by_turn[t_idx],
                "cluster_drift":       comp.cluster_drift_by_turn[t_idx],
                "divergence_turn":     comp.divergence_turn,
                "belief_shift_turn":   comp.belief_shift_turn,
            })
    return rows


def print_trajectory_summary(traj: UncertaintyTrajectory) -> None:
    """Human-readable summary for quick inspection during a run."""
    print(f"\n{'─'*60}")
    print(f"Query: {traj.query[:70]}...")
    print(f"Gold:  {traj.gold_answer}")
    for t_idx, (steps, lsf) in enumerate(
        zip(traj.turn_trajectories, traj.last_step_flip_rates)
    ):
        label = "baseline" if t_idx == 0 else f"pressure T{t_idx}"
        print(f"\n  Turn {t_idx} ({label})  lsf_rate={lsf:.2f}")
        for su in steps:
            marker = " ✓" if su.majority_is_correct else (" ✗" if su.majority_is_correct is False else "")
            print(
                f"    step {su.step_index}: "
                f"bel_H={su.belief_entropy:.2f}  "
                f"cl_H={su.cluster_entropy:.2f}  "
                f"spread={su.semantic_spread:.2f}  "
                f"majority={su.majority_belief}{marker}"
            )
    if traj.cross_turn:
        print(f"\n  Cross-turn divergence:")
        for comp in traj.cross_turn:
            drift_str = "  ".join(
                f"T{i}:{d:.2f}" for i, d in enumerate(comp.cluster_drift_by_turn)
            )
            print(
                f"    step {comp.step_index}: "
                f"div_turn={comp.divergence_turn}  "
                f"bel_shift={comp.belief_shift_turn}  "
                f"drift=[{drift_str}]"
            )
