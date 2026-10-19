"""
fill_cot_batch.py — Fill missing Claude CoT turns (reasoning_calibrated_bin)
with real responses, for the canonical design (first R=3 runs, the question
pool shared by every model; see analysis/paper_pipeline.py).

Many Claude CoT turns were stored empty after failed API calls (credit
exhaustion, timeouts, rate limits). This script regenerates each affected
dialogue from its first missing turn, replaying the known-good prefix
verbatim, with the same system message, question prompt and dose template as
scripts/run_reasoning_calibrated.py. Three phases:

  gen   Anthropic Message Batches API, one round per turn (dialogues are
        sequential): each round submits the next pending turn of every
        unfinished dialogue. Failed/empty replies are retried in later rounds
        (MAX_ATTEMPTS per turn) and otherwise recorded as failed.
  cal   OpenAI Batch API (gpt-5.4-mini, the paper's confidence estimator):
        per-step belief/confidence for every new turn, using the same
        incremental prompt as sycophancy.calibrator.calibrate_step.
  merge Writes the new traces into bin_*_reasoning.pkl, rebuilding the
        trajectory, summary and cross_turn files (originals backed up first),
        and records every refilled turn in fill_log.json.

Planted wrong answer per question — the dose prompt must name the same
wrong option the dialogue's existing turns were given:
  seed        run_reasoning_calibrated's deterministic per-bin draw
              (reconstruct_cot_wrong_letters; checked against echoes)
  echo        the single non-gold letter the model's replies echo back
  fresh       no pressure turn exists in runs 0..R-1: a new deterministic draw
  fresh_reset pressure turns exist but their letter cannot be recovered:
              keep T0, draw a new letter, regenerate T1..T6 for runs 0..R-1

State is saved after every round/batch; rerunning resumes.

Usage:
  python scripts/fill_cot_batch.py all    [--limit N] [--state_dir DIR]   # gen -> cal -> merge
  python scripts/fill_cot_batch.py gen|cal|merge|status [--state_dir DIR]
  --limit N  : test mode, N dialogues per cell; never merges
"""

import argparse
import asyncio
import collections
import glob
import io
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from datetime import datetime

import anthropic
import httpx
import openai
from dotenv import dotenv_values, load_dotenv

load_dotenv(".env")  # before importing sycophancy.config, which reads the gateway key at import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import backfill_gpqa_validated as bf  # extract_echoed_letter, wrong_answer_text_for, pickle shim
from run_reasoning_calibrated import DEFAULT_DOSE_STATEMENTS, build_free_dose_prompt, build_free_prompt
from sycophancy.build_analysis_dfs import reconstruct_cot_wrong_letters
from sycophancy.calibrator import _parse_calibration
from sycophancy.config import (CALIBRATOR_STEP_PROMPT, CALIBRATOR_SYSTEM_MSG, COT_FREE_SYSTEM_MSG,
                               DEFAULT_EMBEDDING_MODEL, DEFAULT_N_CLUSTERS, WSE_COMPAT_ENDPOINT, WSE_MODEL_NAMES)
from sycophancy.generator import _wse_headers
from sycophancy.reasoning_uncertainty import (ReasoningStep, ReasoningTrace, _load_encoder, build_uncertainty_trajectory,
                                              parse_reasoning_steps_freeform, summarise_cross_turn_comparison,
                                              summarise_trajectory)
from sycophancy.sycophancy import option_letter, pick_wrong_option_from_item

import paper_pipeline as pp  # canonical_pool, R, N_TURNS

EXP = "experiment_out"
DEFAULT_STATE_DIR = os.path.join(EXP, "_cot_fill")
CELLS = [("ClaudeSonnet", "mmlu_pro"), ("ClaudeSonnet", "gpqa_diamond"),
         ("ClaudeHaiku", "hle"), ("ClaudeSonnet", "hle")]
API_MODELS = {"ClaudeHaiku": "claude-haiku-4-5", "ClaudeSonnet": "claude-sonnet-4-6"}
CALIBRATOR = "gpt-5.4-mini"
MAX_TOKENS = 16000  # originals reach ~13k tokens; 8192 truncated long Sonnet replies
MAX_ATTEMPTS = 3
POLL_S = 60
CAL_CHUNK = 20000  # requests per OpenAI batch file (keeps files well under the size limit)
FATAL = {"authentication_error", "permission_error", "billing_error"}
BATCH_PRICE = {"claude-haiku-4-5": (0.50, 2.50), "claude-sonnet-4-6": (1.50, 7.50)}  # $/MTok, batch
FINAL_RE = re.compile(r"FINAL ANSWER\s*[:\-]?\s*\**\s*\(?([A-J])\b", re.IGNORECASE)
R, N_TURNS = pp.R, pp.N_TURNS

STATE_PATH = None


def save(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH + ".tmp", "wb") as f:
        pickle.dump(state, f)
    os.replace(STATE_PATH + ".tmp", STATE_PATH)


def load():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "rb") as f:
            return pickle.load(f)
    return None


def keys():
    env = dotenv_values(".env")
    return env["ANTHROPIC_API_KEY"], env["OPENAI_API_KEY"]


# ---------------------------------------------------------------------------
# Build: which dialogues, which planted letter, which prefix
# ---------------------------------------------------------------------------

def is_missing(tr):
    return not tr.raw_text or tr.final_answer is None


def decide_letter(model, dataset, traj, item, seed_letters):
    gold = traj.gold_answer
    echoes = collections.Counter(
        e for turn in traj.raw_traces[1:] for tr in turn if tr.raw_text
        for e in [bf.extract_echoed_letter(tr.raw_text)] if e and e != gold)
    seed = seed_letters.get(traj.query) if dataset != "gpqa_diamond" else None
    if seed and (not echoes or echoes.most_common(1)[0][0] == seed):
        return seed, "seed"
    if len(echoes) == 1:
        return next(iter(echoes)), "echo"
    rng = random.Random(f"{model}|{dataset}|{traj.query}")
    _, fresh = pick_wrong_option_from_item(item, rng)
    has_pressure = any(tr.raw_text for turn in traj.raw_traces[1:] for tr in turn[:R])
    return fresh, ("fresh_reset" if has_pressure else "fresh")


def build(limit):
    dialogues = {}
    for model, dataset in CELLS:
        pool = pp.canonical_pool(dataset)
        d = pp.cell_dir(model, dataset)
        with open(os.path.join(d, "base_experiment_metadata.pkl"), "rb") as f:
            meta = {m["query"]: m for m in pickle.load(f)}
        seed_letters = reconstruct_cot_wrong_letters(model, dataset) if dataset != "gpqa_diamond" else {}
        n_cell = 0
        for path in sorted(glob.glob(os.path.join(d, "reasoning_calibrated_bin", "bin_*_reasoning.pkl"))):
            bin_idx = int(os.path.basename(path).split("_")[1])
            with open(path, "rb") as f:
                trajs = pickle.load(f)
            for traj in trajs:
                if traj.query not in pool:
                    continue
                firsts = {r: next((t for t in range(N_TURNS) if is_missing(traj.raw_traces[t][r])), None)
                          for r in range(R)}
                if all(v is None for v in firsts.values()):
                    continue
                item = meta[traj.query]
                letter, src = decide_letter(model, dataset, traj, item, seed_letters)
                wrong_text = bf.wrong_answer_text_for(item, letter)
                for r in range(R):
                    start = firsts[r]
                    if src == "fresh_reset":
                        start = 0 if is_missing(traj.raw_traces[0][r]) else 1
                    if start is None or (limit and n_cell >= limit):
                        continue
                    history = [{"role": "user", "content": build_free_prompt(item["query"], item["options"])}]
                    for t in range(start):
                        if t >= 1:
                            history.append({"role": "user", "content": build_free_dose_prompt(DEFAULT_DOSE_STATEMENTS[t - 1], wrong_text)})
                        history.append({"role": "assistant", "content": traj.raw_traces[t][r].raw_text})
                    dialogues[(model, dataset, bin_idx, traj.query, r)] = {
                        "history": history, "next": start, "start": start, "letter": letter, "letter_src": src,
                        "wrong_text": wrong_text, "gold": traj.gold_answer, "question": item["query"],
                        "options": item["options"], "new": {}, "attempts": 0, "failed": None,
                    }
                    n_cell += 1
        print(f"  {model}/{dataset}: {n_cell} dialogues to fill", flush=True)
    return {"dialogues": dialogues, "round": 0, "gen_batches": [], "cal_batches": [], "cal": {},
            "limit": limit, "created": str(datetime.now())}


# ---------------------------------------------------------------------------
# Generation (Anthropic batches, one round per turn)
# ---------------------------------------------------------------------------

def accept(text, dlg):
    """A reply counts only with an explicit FINAL ANSWER naming one of the question's options."""
    if not text:
        return None
    m = FINAL_RE.findall(text)
    valid = {option_letter(o) for o in dlg["options"]}
    if not m or m[-1].upper() not in valid:
        return None
    trace = parse_reasoning_steps_freeform(text, gold_answer=dlg["gold"])
    return trace if trace.final_answer == m[-1].upper() else None


def next_messages(dlg):
    msgs = list(dlg["history"])
    t = dlg["next"]
    if t >= 1:
        msgs.append({"role": "user", "content": build_free_dose_prompt(DEFAULT_DOSE_STATEMENTS[t - 1], dlg["wrong_text"])})
    return msgs


def gen_round(client, state, budget, cells):
    pend = [k for k, d in state["dialogues"].items()
            if d["failed"] is None and d["next"] < N_TURNS and k[:2] in cells]
    if not pend:
        return False
    spent = state.get("spent", 0.0)
    if state.get("last_round_cost") and spent + state["last_round_cost"] > budget:
        print(f"BUDGET STOP: spent ${spent:.2f}; next round (~${state['last_round_cost']:.2f}) would exceed ${budget:.2f}. "
              f"State saved — rerun with a higher --budget to continue.", flush=True)
        return False
    ids, reqs = {}, []
    for i, k in enumerate(pend):
        cid = f"g{state['round']}-{i}"
        ids[cid] = k
        reqs.append({"custom_id": cid, "params": {
            "model": API_MODELS[k[0]], "max_tokens": MAX_TOKENS,
            "system": COT_FREE_SYSTEM_MSG["content"], "messages": next_messages(state["dialogues"][k])}})
    try:
        batch = client.messages.batches.create(requests=reqs)
    except anthropic.APIStatusError as e:
        if e.status_code in (400, 402, 403) and ("credit" in str(e).lower() or "billing" in str(e).lower()):
            print(f"ANTHROPIC CREDIT STOP: {str(e)[:200]} — remaining turns go to the gateway", flush=True)
            return False
        raise
    state["gen_batches"].append((state["round"], batch.id, len(reqs)))
    save(state)
    print(f"[gen round {state['round']}] {batch.id}: {len(reqs)} requests", flush=True)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"  ...processing {b.request_counts.processing}, succeeded {b.request_counts.succeeded}, "
              f"errored {b.request_counts.errored}", flush=True)
        time.sleep(POLL_S)
    got, fatal, cost = {}, None, 0.0
    for res in client.messages.batches.results(batch.id):
        k = ids[res.custom_id]
        if res.result.type == "succeeded":
            msg = res.result.message
            pi, po = BATCH_PRICE[API_MODELS[k[0]]]
            cost += msg.usage.input_tokens / 1e6 * pi + msg.usage.output_tokens / 1e6 * po
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
            got[k] = (text, msg.stop_reason)
        else:
            err = getattr(getattr(res.result, "error", None), "error", None)
            etype = getattr(err, "type", res.result.type)
            fatal = etype if etype in FATAL else fatal
            got[k] = (None, etype)
    ok = retry = fail = 0
    for k in pend:
        d = state["dialogues"][k]
        text, info = got.get(k, (None, "missing"))
        trace = accept(text, d)
        if trace is not None:
            t = d["next"]
            if t >= 1:
                d["history"].append({"role": "user", "content": build_free_dose_prompt(DEFAULT_DOSE_STATEMENTS[t - 1], d["wrong_text"])})
            d["history"].append({"role": "assistant", "content": text})
            d["new"][t] = {"raw": text, "stop_reason": info}
            d["next"] += 1
            d["attempts"] = 0
            ok += 1
        else:
            d["attempts"] += 1
            if info == "invalid_request" or d["attempts"] >= MAX_ATTEMPTS:
                d["failed"] = f"T{d['next']}: {info if text is None else f'no valid FINAL ANSWER ({info})'}"
                fail += 1
            else:
                retry += 1
    state["round"] += 1
    state["spent"] = state.get("spent", 0.0) + cost
    state["last_round_cost"] = cost
    save(state)
    print(f"[gen round {state['round'] - 1}] {ok} turns filled, {retry} to retry, {fail} gave up | "
          f"round ${cost:.2f}, total ${state['spent']:.2f}", flush=True)
    if fatal:
        print(f"ANTHROPIC STOP: {fatal} in batch results — remaining turns go to the gateway", flush=True)
        return False
    return True


# ---------------------------------------------------------------------------
# WSE gateway (free, RPM-limited): generation and last-step calibration
# ---------------------------------------------------------------------------

GW_TIMEOUT_S = 1200        # long Sonnet replies took >120 s, the timeout that created these gaps
GW_TRANSIENT = {408, 409, 429, 500, 502, 503, 504}


class GatewayFatal(RuntimeError):
    pass


async def gw_call(client, model_key, messages, max_tokens, sem):
    """One gateway completion, retrying transient errors (incl. RPM 429s) until it succeeds.
    Returns (text, finish_reason) or (None, error) for a non-retryable client error."""
    canon = WSE_MODEL_NAMES[model_key]
    tok_param = "max_completion_tokens" if canon.startswith("openai/") else "max_tokens"
    payload = {"model": canon, "messages": messages, tok_param: max_tokens}
    delay = 5.0
    while True:
        async with sem:
            try:
                r = await client.post(WSE_COMPAT_ENDPOINT, headers=_wse_headers(), json=payload)
            except httpx.TransportError as e:
                r, err = None, f"{type(e).__name__}"
        if r is not None:
            if r.status_code == 200:
                ch = r.json()["choices"][0]
                return (ch["message"]["content"] or "").strip(), ch.get("finish_reason")
            body = r.text[:300].lower()
            if r.status_code in (401, 402, 403) or "credit" in body or "billing" in body:
                raise GatewayFatal(f"{r.status_code}: {r.text[:200]}")
            if r.status_code not in GW_TRANSIENT:
                return None, f"http {r.status_code}"
            ra = r.headers.get("retry-after")
            wait = float(ra) if ra and ra.replace(".", "").isdigit() else delay
        else:
            wait = delay
        await asyncio.sleep(wait + random.random() * 3)
        delay = min(delay * 2, 120.0)


async def gw_generate(state, concurrency):
    for d in state["dialogues"].values():   # fresh attempt budget on the new backend, except
        if d["failed"] is None or "no valid FINAL ANSWER" not in d["failed"]:  # replies truncated at 16k
            d["failed"], d["attempts"] = None, 0
    pend = [k for k, d in state["dialogues"].items() if d["failed"] is None and d["next"] < N_TURNS]
    print(f"[gateway gen] {len(pend)} dialogues, {sum(N_TURNS - state['dialogues'][k]['next'] for k in pend)} turns", flush=True)
    sem = asyncio.Semaphore(concurrency)
    done = {"turns": 0, "last_save": time.time()}

    async def run_dialogue(client, k):
        d = state["dialogues"][k]
        while d["failed"] is None and d["next"] < N_TURNS:
            msgs = [COT_FREE_SYSTEM_MSG] + next_messages(d)
            text, info = await gw_call(client, k[0], msgs, MAX_TOKENS, sem)
            trace = accept(text, d)
            if trace is not None:
                t = d["next"]
                if t >= 1:
                    d["history"].append({"role": "user", "content": build_free_dose_prompt(DEFAULT_DOSE_STATEMENTS[t - 1], d["wrong_text"])})
                d["history"].append({"role": "assistant", "content": text})
                d["new"][t] = {"raw": text, "stop_reason": info, "backend": "wse_gateway"}
                d["next"] += 1
                d["attempts"] = 0
                done["turns"] += 1
            else:
                d["attempts"] += 1
                if d["attempts"] >= MAX_ATTEMPTS:
                    d["failed"] = f"T{d['next']}: {info if text is None else f'no valid FINAL ANSWER ({info})'}"
            if time.time() - done["last_save"] > 60:
                save(state)
                done["last_save"] = time.time()
                left = sum(N_TURNS - state["dialogues"][x]["next"] for x in pend if state["dialogues"][x]["failed"] is None)
                print(f"  ...gateway gen: {done['turns']} turns filled, {left} left", flush=True)

    async with httpx.AsyncClient(timeout=GW_TIMEOUT_S) as client:
        await asyncio.gather(*[run_dialogue(client, k) for k in pend])
    save(state)
    print(f"[gateway gen] done: {done['turns']} turns filled", flush=True)


def last_step_requests(state):
    """One calibrator request per new turn, for its final step (the only step RQ4 reads).
    The prompt is identical to calibrate_step's at the last step (all steps as context)."""
    reqs = {}
    for i, (k, d) in enumerate(state["dialogues"].items()):
        for t, nt in d["new"].items():
            steps = parse_reasoning_steps_freeform(nt["raw"], gold_answer=d["gold"]).steps
            if not steps:
                continue
            ell = len(steps) - 1
            cid = f"{i}-{t}-{ell}"
            if cid in state["cal"]:
                continue
            steps_text = "\n\n".join(f"Step {j + 1}:\n{s.step_text}" for j, s in enumerate(steps))
            prompt = CALIBRATOR_STEP_PROMPT.format(question=d["question"], options="\n".join(d["options"]),
                                                   step_num=ell + 1, steps_so_far=steps_text)
            reqs[cid] = [CALIBRATOR_SYSTEM_MSG, {"role": "user", "content": prompt}]
    return reqs


async def gw_calibrate(state, concurrency, max_rounds=3):
    sem = asyncio.Semaphore(concurrency)
    for rnd in range(max_rounds):
        reqs = last_step_requests(state)
        if not reqs:
            break
        print(f"[gateway cal round {rnd}] {len(reqs)} last-step requests", flush=True)
        n = {"done": 0, "last_save": time.time()}

        async def one(client, cid, msgs):
            text, _ = await gw_call(client, "GPT5_4Mini", msgs, 256, sem)
            belief, conf = _parse_calibration(text or "")
            if conf is not None:
                state["cal"][cid] = (belief, conf)
            n["done"] += 1
            if time.time() - n["last_save"] > 60:
                save(state)
                n["last_save"] = time.time()
                print(f"  ...gateway cal: {n['done']}/{len(reqs)}", flush=True)

        async with httpx.AsyncClient(timeout=300) as client:
            await asyncio.gather(*[one(client, cid, m) for cid, m in reqs.items()])
        save(state)
        print(f"[gateway cal round {rnd}] calibrated steps total {len(state['cal'])}", flush=True)


def last_step_requests_oa(state):
    """Like last_step_requests, but as OpenAI Batch API JSONL request dicts
    (one per new turn's final step -- the only step RQ4 reads)."""
    reqs = {}
    for i, (k, d) in enumerate(state["dialogues"].items()):
        for t, nt in d["new"].items():
            steps = parse_reasoning_steps_freeform(nt["raw"], gold_answer=d["gold"]).steps
            if not steps:
                continue
            ell = len(steps) - 1
            cid = f"{i}-{t}-{ell}"
            if cid in state["cal"]:
                continue
            steps_text = "\n\n".join(f"Step {j + 1}:\n{s.step_text}" for j, s in enumerate(steps))
            prompt = CALIBRATOR_STEP_PROMPT.format(question=d["question"], options="\n".join(d["options"]),
                                                   step_num=ell + 1, steps_so_far=steps_text)
            reqs[cid] = {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {
                "model": CALIBRATOR, "messages": [CALIBRATOR_SYSTEM_MSG, {"role": "user", "content": prompt}]}}
    return reqs


# ---------------------------------------------------------------------------
# Calibration (OpenAI batches)
# ---------------------------------------------------------------------------

def cal_requests(state):
    reqs = {}
    for i, (k, d) in enumerate(state["dialogues"].items()):
        for t, nt in d["new"].items():
            steps = parse_reasoning_steps_freeform(nt["raw"], gold_answer=d["gold"]).steps
            for ell in range(len(steps)):
                cid = f"{i}-{t}-{ell}"
                if cid in state["cal"]:
                    continue
                steps_text = "\n\n".join(f"Step {j + 1}:\n{s.step_text}" for j, s in enumerate(steps[: ell + 1]))
                prompt = CALIBRATOR_STEP_PROMPT.format(question=d["question"], options="\n".join(d["options"]),
                                                       step_num=ell + 1, steps_so_far=steps_text)
                reqs[cid] = {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {
                    "model": CALIBRATOR, "messages": [CALIBRATOR_SYSTEM_MSG, {"role": "user", "content": prompt}]}}
    return reqs


def calibrate(oclient, state, max_rounds=2, request_fn=cal_requests):
    for rnd in range(max_rounds):
        reqs = request_fn(state)
        if not reqs:
            break
        items = list(reqs.values())
        print(f"[cal round {rnd}] {len(items)} step requests", flush=True)
        batch_ids = []
        for c in range(0, len(items), CAL_CHUNK):
            buf = io.BytesIO("\n".join(json.dumps(x) for x in items[c:c + CAL_CHUNK]).encode())
            f = oclient.files.create(file=("cal.jsonl", buf), purpose="batch")
            b = oclient.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
            batch_ids.append(b.id)
            state["cal_batches"].append(b.id)
            save(state)
            print(f"  submitted {b.id} ({len(items[c:c + CAL_CHUNK])} requests)", flush=True)
        for bid in batch_ids:
            while True:
                b = oclient.batches.retrieve(bid)
                if b.status in ("completed", "failed", "expired", "cancelled"):
                    break
                print(f"  ...{bid} {b.status}: {b.request_counts.completed}/{b.request_counts.total}", flush=True)
                time.sleep(POLL_S)
            if b.output_file_id:
                for line in oclient.files.content(b.output_file_id).text.splitlines():
                    row = json.loads(line)
                    resp = row.get("response") or {}
                    if resp.get("status_code") == 200:
                        text = resp["body"]["choices"][0]["message"]["content"] or ""
                        belief, conf = _parse_calibration(text)
                        if conf is not None:
                            state["cal"][row["custom_id"]] = (belief, conf)
            save(state)
            print(f"  {bid} {b.status}; calibrated steps so far {len(state['cal'])}", flush=True)


def calibrated_trace(state, i, d, t):
    base = parse_reasoning_steps_freeform(d["new"][t]["raw"], gold_answer=d["gold"])
    steps = []
    for ell, s in enumerate(base.steps):
        belief, conf = state["cal"].get(f"{i}-{t}-{ell}", (None, None))
        steps.append(ReasoningStep(step_index=s.step_index, step_text=s.step_text,
                                   current_belief=belief if belief is not None else s.current_belief,
                                   confidence=conf if conf is not None else s.confidence))
    last = next((s.current_belief for s in reversed(steps) if s.current_belief), None)
    return ReasoningTrace(raw_text=base.raw_text, steps=steps, final_answer=base.final_answer,
                          last_step_belief=last,
                          last_step_flip=base.final_answer is not None and last is not None and base.final_answer != last,
                          gold_answer=base.gold_answer, final_confidence=base.final_confidence)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(state):
    if state["limit"]:
        raise SystemExit("test state (--limit) — refusing to merge into the real data")
    encoder = None
    try:
        encoder = _load_encoder(DEFAULT_EMBEDDING_MODEL)
    except Exception as e:
        print(f"WARNING: could not load encoder ({e}); falling back to belief-entropy-only mode.", flush=True)
    by_file = collections.defaultdict(list)
    for i, (k, d) in enumerate(state["dialogues"].items()):
        if d["new"]:
            by_file[k[:3]].append((i, k, d))
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    log = []
    for (model, dataset, bin_idx), items in sorted(by_file.items()):
        bdir = os.path.join(pp.cell_dir(model, dataset), "reasoning_calibrated_bin")
        backup = os.path.join(bdir, f"_backup_before_cot_fill_{stamp}")
        os.makedirs(backup, exist_ok=True)
        for suffix in ("reasoning", "summary", "cross_turn"):
            src = os.path.join(bdir, f"bin_{bin_idx}_{suffix}.pkl")
            if os.path.exists(src):
                shutil.copy2(src, backup)
        with open(os.path.join(bdir, f"bin_{bin_idx}_reasoning.pkl"), "rb") as f:
            trajs = pickle.load(f)
        by_q = {t.query: t for t in trajs}
        touched = set()
        for i, k, d in items:
            traj = by_q[k[3]]
            r = k[4]
            for t in d["new"]:
                traj.raw_traces[t][r] = calibrated_trace(state, i, d, t)
            touched.add(traj.query)
            log.append({"model": model, "dataset": dataset, "query": k[3][:120], "run": r,
                        "turns": sorted(d["new"]), "letter": d["letter"], "letter_src": d["letter_src"],
                        "failed": d["failed"]})
        new_trajs = [build_uncertainty_trajectory(query=t.query, gold_answer=t.gold_answer, all_traces=t.raw_traces,
                                                  n_clusters=DEFAULT_N_CLUSTERS, encoder=encoder)
                     if t.query in touched else t for t in trajs]
        with open(os.path.join(bdir, f"bin_{bin_idx}_reasoning.pkl"), "wb") as f:
            pickle.dump(new_trajs, f)
        summ, cross = [], []
        for t in new_trajs:
            summ.extend(summarise_trajectory(t))
            cross.extend(summarise_cross_turn_comparison(t))
        with open(os.path.join(bdir, f"bin_{bin_idx}_summary.pkl"), "wb") as f:
            pickle.dump(summ, f)
        with open(os.path.join(bdir, f"bin_{bin_idx}_cross_turn.pkl"), "wb") as f:
            pickle.dump(cross, f)
        print(f"  {model}/{dataset} bin_{bin_idx}: {len(items)} dialogues merged; backup {backup}", flush=True)
    with open(os.path.join(os.path.dirname(STATE_PATH), f"fill_log_{stamp}.json"), "w") as f:
        json.dump(log, f, indent=1)


def status(state):
    ds = state["dialogues"]
    by = collections.Counter((k[0], k[1], "done" if d["next"] >= N_TURNS else "failed" if d["failed"] else "pending")
                             for k, d in ds.items())
    src = collections.Counter((k[0], k[1], d["letter_src"]) for k, d in ds.items())
    print(f"gen round {state['round']}; Anthropic spend ${state.get('spent', 0):.2f}; "
          f"calibrated steps {len(state['cal'])}; limit {state['limit']}")
    for c in sorted({k[:2] for k in by}):
        print(f"  {c[0]}/{c[1]}: " + ", ".join(f"{s} {by[c + (s,)]}" for s in ("done", "pending", "failed")) +
              " | letters " + ", ".join(f"{s} {n}" for (m, dd, s), n in sorted(src.items()) if (m, dd) == c))


def main():
    global STATE_PATH
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "gen", "gw_gen", "gw_cal", "cal", "oa_cal", "merge", "status", "build"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--state_dir", default=DEFAULT_STATE_DIR)
    p.add_argument("--budget", type=float, default=100.0, help="stop generation before Anthropic spend exceeds this ($)")
    p.add_argument("--anthropic_cells", nargs="*", default=["ClaudeSonnet/mmlu_pro", "ClaudeSonnet/gpqa_diamond"],
                   help="cells generated on the Anthropic Batches API; everything else (and any remainder) uses the gateway")
    p.add_argument("--gw_concurrency", type=int, default=6)
    args = p.parse_args()
    STATE_PATH = os.path.join(args.state_dir, "state.pkl")
    state = load()
    if state is None:
        print("building fill plan...", flush=True)
        state = build(args.limit)
        save(state)
    if args.cmd in ("status", "build"):
        return status(state)
    akey, okey = keys()
    cells = {tuple(c.split("/")) for c in args.anthropic_cells}
    if args.cmd in ("all", "gen"):
        client = anthropic.Anthropic(api_key=akey)
        while gen_round(client, state, args.budget, cells):
            pass
        all_cells = {k[:2] for k in state["dialogues"]}
        while gen_round(client, state, args.budget, all_cells):  # leftover credit -> remaining cells
            pass
    if args.cmd in ("all", "gw_gen"):
        asyncio.run(gw_generate(state, args.gw_concurrency))
    if args.cmd in ("all", "gw_cal"):
        asyncio.run(gw_calibrate(state, args.gw_concurrency))
    if args.cmd == "cal":  # OpenAI Batch API, all steps (not used by `all`)
        calibrate(openai.OpenAI(api_key=okey), state)
    if args.cmd == "oa_cal":  # OpenAI Batch API, last step only (the gw_cal replacement)
        calibrate(openai.OpenAI(api_key=okey), state, request_fn=last_step_requests_oa)
    if args.cmd == "merge" or (args.cmd == "all" and not state["limit"]):
        merge(state)
    status(state)


if __name__ == "__main__":
    main()
