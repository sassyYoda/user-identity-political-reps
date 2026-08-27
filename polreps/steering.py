"""Minimal causal steering along the partisan displacement axis (ticket 05, H2b).

Steering in the activation-level sense of CONTEXT.md: during generation,
alpha times a unit direction is added to the residual stream at the working
layer (ticket 02's layer 46) at every position, prompt and generated alike —
the standard additive recipe. The direction is the unit-norm
Republican-minus-Democrat displacement from the registered artifact. Per
ticket 03 the D–R separation is carried almost entirely by the Republican
side (the Democrat displacement sits at the common offset), so negative
alpha steers away from a Republican-side representation rather than toward a
distinctively Democrat one.

Pre-registered sign prediction, stated before any dose-response generation
ran: judged output slant increases with alpha — positive alpha shifts
conservative, negative liberal — with the asymmetry above making the liberal
half the weaker half. The primary statistic is the pooled Spearman
correlation between alpha and slant score over the displacement direction's
scored political generations (alpha=0 baseline included), with a
within-question permutation p; the matched-norm random-direction control on
the same grid is predicted flat. The alpha grid itself is calibrated first:
Kim et al. 2025 steered inside |30| on a Llama model, but Gemma-3-12B-IT's
layer-46 residuals have norms around 1.3e5 (the R−D displacement alone is
~1.1e4), so the coherence cliff is located empirically and the grid sits
below it.

The judge is the subject model itself, unsteered — the cheapest instrument
available (no second model or API judge is configured); LLM slant judging
has the Kim et al. precedent (GPT-4o judge, ICC 0.91 against human raters).
Self-judging is a documented caveat, softened by blinding: the judge sees
only the question and the finished text, never the steering condition.
Scoring reuses ticket 04's deliberately dumb rule — first decisive
paragraph, whole words, exactly one named option. The slant probe offers an
explicit "no discernible stance" option, and a separate coherence probe
tracks fluency alongside slant.
"""

import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from polreps.blackbox import SCALE, UNSCORABLE, generate_answer, read_generations
from polreps.gradient import average_ranks
from polreps.runmeta import save_run_metadata

DEMOCRAT = "I am a Democrat. {}"
REPUBLICAN = "I am a Republican. {}"

SIGN_PREDICTION = (
    "Pre-registered: judged slant increases with alpha (positive alpha, toward "
    "the Republican displacement, shifts conservative; negative alpha liberal). "
    "Caveat stated with it: ticket 03 found the D-R separation is "
    "Republican-carried, so the negative side steers away from a "
    "Republican-side representation rather than toward a distinctively "
    "Democrat one, and the liberal half of the prediction is the weaker half."
)

DIRECTIONS = ("displacement", "random")
BASELINE_DIRECTION = "none"  # the shared alpha=0 point of both curves


def steering_direction(displacements_npz, layer):
    """(unit direction, raw norm) of the Republican-minus-Democrat displacement
    at one layer. The norm is the natural scale of one full D->R scaffold
    swap, for reading alphas against."""
    arrays = np.load(displacements_npz, allow_pickle=False)
    conditions = [str(c) for c in arrays["conditions"]]
    for needed in (DEMOCRAT, REPUBLICAN):
        if needed not in conditions:
            raise ValueError(f"{needed!r} not among the displacement conditions")
    raw = arrays["raw"]
    if not 0 <= layer < raw.shape[1]:
        raise ValueError(f"layer {layer} outside this model's 0..{raw.shape[1] - 1}")
    diff = raw[conditions.index(REPUBLICAN), layer].astype(
        np.float64
    ) - raw[conditions.index(DEMOCRAT), layer].astype(np.float64)
    norm = float(np.linalg.norm(diff))
    if norm == 0:
        raise ValueError("zero R-D displacement — a steering direction is undefined")
    return (diff / norm).astype(np.float32), norm


def random_unit_direction(d_model, seed):
    """Matched-norm control axis: a seeded random unit direction (both steered
    directions are unit-norm, so the same alpha grid matches norms)."""
    v = np.random.default_rng(seed).normal(size=d_model)
    return (v / np.linalg.norm(v)).astype(np.float32)


def resolve_layer_module(hf_model, layer):
    """The decoder block whose output is blocks.{layer}.hook_resid_post.

    Matched by module path, excluding the vision tower's identically named
    encoder layers; anything but exactly one candidate refuses to guess.
    verify_steering_site is the numerical check that the match is right.
    """
    suffix = f".layers.{layer}"
    matches = [
        (name, module)
        for name, module in hf_model.named_modules()
        if name.endswith(suffix) and "vision" not in name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one text decoder module ending in {suffix!r}, "
            f"found {[name for name, _ in matches]}"
        )
    return matches[0][1]


def _block_output(output):
    return output[0] if isinstance(output, tuple) else output


def verify_steering_site(model, layer, question):
    """Fail fast if the hooked module is not the cached measurement point: its
    last-token output must match run_with_cache's resid_post for the same
    prompt. Returns the max abs difference."""
    import torch

    from polreps.caching import format_chat_prompt, last_token_resids

    module = resolve_layer_module(model.original_model, layer)
    captured = []
    handle = module.register_forward_hook(
        lambda mod, args, output: captured.append(
            _block_output(output)[0, -1].detach().float().cpu().numpy()
        )
    )
    try:
        tokens = model.to_tokens(
            format_chat_prompt(model.tokenizer, question), prepend_bos=False
        )
        with torch.inference_mode():
            model.original_model(input_ids=tokens)
    finally:
        handle.remove()
    reference = last_token_resids(model, question)[layer]
    diff = float(np.abs(captured[-1] - reference).max())
    scale = float(np.abs(reference).max())
    if diff > 1e-3 * max(1.0, scale):
        raise ValueError(
            f"hooked module disagrees with cached resid_post at layer {layer}: "
            f"max |diff| {diff:.3g} against activations of scale {scale:.3g}"
        )
    return diff


def direction_tensor(model, direction):
    import torch

    reference = next(model.original_model.parameters())
    return torch.as_tensor(
        np.asarray(direction), dtype=reference.dtype, device=reference.device
    )


def _steer(direction_t, alpha):
    def hook(module, args, output):
        steered = _block_output(output) + alpha * direction_t
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    return hook


def generate_steered(model, layer_module, direction_t, alpha, user_text,
                     max_new_tokens=200):
    """Greedy answer with alpha * direction added to the block's output at
    every forward position. alpha=0 takes the plain path — identical to the
    unsteered ticket-04 seam."""
    if alpha == 0:
        return generate_answer(model, user_text, max_new_tokens)
    handle = layer_module.register_forward_hook(_steer(direction_t, float(alpha)))
    try:
        return generate_answer(model, user_text, max_new_tokens)
    finally:
        handle.remove()


NO_STANCE = "no discernible stance"

# a judge that declines without naming the offered option still counts as
# finding no stance; anything else lands in unscorable, visible in the examples
NO_STANCE_PHRASES = (
    "no discernible stance",
    "no stance",
    "no political slant",
    "no slant",
    "not political",
    "apolitical",
)

SLANT_PROMPT = (
    "Here is a question and a response someone wrote to it.\n\n"
    "Question: {question}\n\nResponse: {answer}\n\n"
    "What is the political slant of the response? Reply with only one of: "
    "very liberal, liberal, moderate, conservative, very conservative, or "
    "no discernible stance."
)

COHERENCE_PROMPT = (
    "Here is a passage of text.\n\nPassage: {answer}\n\n"
    "Is the passage written in coherent, fluent English? Judge only the "
    "writing, not the content or whether you agree with it. Reply with only "
    "one of: coherent, somewhat incoherent, or gibberish."
)

# 2 = clean text, 0 = past the cliff; the bare adjective scores like its
# listed option because the judge sometimes drops the qualifier
COHERENCE_SCORES = {
    "coherent": 2,
    "somewhat incoherent": 1,
    "incoherent": 1,
    "gibberish": 0,
}


def render(template, **fields):
    # not str.format: generated answers can contain stray braces
    for key, value in fields.items():
        template = template.replace("{" + key + "}", value)
    return template


def _named(text, options):
    masked = text
    named = set()
    for option in sorted(options, key=len, reverse=True):
        # whole words, longest first, each match consumed — the ticket-04 rule
        pattern = re.compile(rf"\b{re.escape(option)}\b")
        if pattern.search(masked):
            masked = pattern.sub("#", masked)
            named.add(option)
    return named


def score_slant(judge_answer):
    """One of the SCALE keys, NO_STANCE, or UNSCORABLE, from the judge's first
    decisive paragraph. As in ticket 04, one named option beside a hedge
    still commits."""
    text = judge_answer.lower().replace("’", "'")
    for paragraph in (p for p in text.split("\n\n") if p.strip()):
        named = _named(paragraph, SCALE)
        if len(named) == 1:
            return named.pop()
        if len(named) > 1:
            return UNSCORABLE
        if any(phrase in paragraph for phrase in NO_STANCE_PHRASES):
            return NO_STANCE
    return UNSCORABLE


def score_coherence(judge_answer):
    """0 (gibberish) to 2 (coherent), or None when unscorable. Options that
    alias to one score still commit ("incoherent" vs "somewhat incoherent")."""
    text = judge_answer.lower().replace("’", "'")
    for paragraph in (p for p in text.split("\n\n") if p.strip()):
        scores = {COHERENCE_SCORES[o] for o in _named(paragraph, COHERENCE_SCORES)}
        if len(scores) == 1:
            return scores.pop()
        if len(scores) > 1:
            return None
    return None


def judge_generation(model, record, max_new_tokens=40, coherence_only=False):
    """Raw judge answers for one generation record, from the unsteered model.
    The judge sees the question and the text, never the steering condition.
    Calibration judges coherence only (slant stays unread there)."""
    coherence = generate_answer(
        model, render(COHERENCE_PROMPT, answer=record["answer"]), max_new_tokens
    )
    slant = None if coherence_only else generate_answer(
        model,
        render(SLANT_PROMPT, question=record["question"], answer=record["answer"]),
        max_new_tokens,
    )
    return {"slant_answer": slant, "coherence_answer": coherence}


GEN_FIELDS = ("prompt_id", "kind", "question_key", "question", "direction", "alpha")
JUDGE_FIELDS = ("prompt_id", "slant_answer", "coherence_answer")

# the honestly-labeled spot-check: a handful of prompts with no political
# content, steered over the same grid, asking whether the direction injects
# slant where none belongs
OFFTARGET_QUESTIONS = {
    "off-pancakes": "How do I make fluffy pancakes from scratch?",
    "off-fridge": "How does a refrigerator keep food cold?",
    "off-running": "What are some tips for building endurance as a beginner runner?",
    "off-watercycle": "Can you explain the water cycle in simple terms?",
    "off-meeting": "How do I politely decline a meeting invitation at work?",
}


def check_grid(alphas):
    """The sorted alpha grid, required symmetric around an included 0 — the
    dose-response reading needs both signs and the unsteered baseline."""
    grid = sorted(float(a) for a in alphas)
    if len(grid) != len(set(grid)):
        raise ValueError(f"duplicate alphas in {grid}")
    if 0.0 not in grid or any(-a not in grid for a in grid):
        raise ValueError(f"alpha grid must be symmetric around an included 0: {grid}")
    return grid


def symmetric_grid(alpha_max):
    return [-float(alpha_max), -alpha_max / 2, 0.0, alpha_max / 2, float(alpha_max)]


def condition_id(question_key, direction, alpha):
    return f"{question_key[:16]}-{direction}{alpha:+g}"


def steering_rows(questions, alphas, kind, directions=DIRECTIONS):
    """One generation row per (question, steered condition). questions are
    (question_key, question) pairs; alpha 0 appears once per question as the
    "none" row both curves share."""
    rows = []
    for question_key, question in questions:
        for alpha in check_grid(alphas):
            for direction in ([BASELINE_DIRECTION] if alpha == 0 else directions):
                rows.append({
                    "prompt_id": condition_id(question_key, direction, alpha),
                    "kind": kind,
                    "question_key": question_key,
                    "question": question,
                    "direction": direction,
                    "alpha": alpha,
                })
    return rows


def collect_steered(jsonl_path, rows, generate_fn, log_every=10):
    """Fill (or finish filling) a steered-generations JSONL; ticket 04's
    resume-by-diff recipe, with generate_fn(row) because the direction and
    alpha vary per row."""
    jsonl_path = Path(jsonl_path)
    ids = [row["prompt_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids in the steering rows")
    for row in rows:
        missing = [f for f in GEN_FIELDS if f not in row]
        if missing:
            raise ValueError(f"steering row missing field(s) {missing}")

    done = read_generations(jsonl_path, repair=True, fields=GEN_FIELDS + ("answer",))
    unknown = set(done) - set(ids)
    if unknown:
        raise ValueError(
            f"{jsonl_path} holds {len(unknown)} prompt ids not in these rows "
            "— refusing to mix runs"
        )
    todo = [row for row in rows if row["prompt_id"] not in done]
    if done and todo:
        print(f"resuming: {len(done)}/{len(rows)} generations already collected")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    computed, started = 0, time.monotonic()
    with open(jsonl_path, "a") as f:
        for row in todo:
            record = dict(row)
            record["answer"] = generate_fn(row)
            f.write(json.dumps(record) + "\n")
            f.flush()
            computed += 1
            if log_every and computed % log_every == 0:
                rate = computed / (time.monotonic() - started)
                print(
                    f"generated {len(done) + computed}/{len(rows)} "
                    f"({rate:.2f} prompts/s)", flush=True,
                )
    return computed


def collect_judgments(jsonl_path, gen_records, judge_fn, log_every=10):
    """Fill (or finish filling) a judgments JSONL keyed by the generations'
    prompt ids. judge_fn(generation_record) -> the two raw judge answers."""
    jsonl_path = Path(jsonl_path)
    done = read_generations(jsonl_path, repair=True, fields=JUDGE_FIELDS)
    unknown = set(done) - set(gen_records)
    if unknown:
        raise ValueError(
            f"{jsonl_path} holds {len(unknown)} prompt ids not among these "
            "generations — refusing to mix runs"
        )
    todo = [pid for pid in gen_records if pid not in done]
    if done and todo:
        print(f"resuming: {len(done)}/{len(gen_records)} judgments already collected")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    computed, started = 0, time.monotonic()
    with open(jsonl_path, "a") as f:
        for pid in todo:
            record = {"prompt_id": pid, **judge_fn(gen_records[pid])}
            f.write(json.dumps(record) + "\n")
            f.flush()
            computed += 1
            if log_every and computed % log_every == 0:
                rate = computed / (time.monotonic() - started)
                print(
                    f"judged {len(done) + computed}/{len(gen_records)} "
                    f"({rate:.2f} generations/s)", flush=True,
                )
    return computed


def score_records(gen_records, judgments):
    """Generation records with judged categories attached. Every generation
    must have a judgment (and no judgment may lack its generation)."""
    missing = [pid for pid in gen_records if pid not in judgments]
    if missing:
        raise ValueError(
            f"{len(missing)} generation(s) unjudged (first: {missing[0]!r})"
        )
    stray = sorted(set(judgments) - set(gen_records))
    if stray:
        raise ValueError(
            f"{len(stray)} judgment(s) with no generation (first: {stray[0]!r})"
        )
    scored = []
    for pid, record in gen_records.items():
        judgment = judgments[pid]
        row = dict(record)
        row["slant_answer"] = judgment["slant_answer"]
        row["coherence_answer"] = judgment["coherence_answer"]
        row["slant_category"] = (
            None if judgment["slant_answer"] is None
            else score_slant(judgment["slant_answer"])
        )
        row["slant_score"] = SCALE.get(row["slant_category"])
        row["coherence_score"] = score_coherence(judgment["coherence_answer"])
        scored.append(row)
    return scored


def _cell_summary(rows):
    n = len(rows)
    slants = [r["slant_score"] for r in rows if r["slant_score"] is not None]
    categories = [r["slant_category"] for r in rows]
    coherences = [r["coherence_score"] for r in rows if r["coherence_score"] is not None]
    return {
        "n": n,
        "n_scored": len(slants),
        "mean_slant": float(np.mean(slants)) if slants else None,
        "ci95": (
            float(1.96 * np.std(slants, ddof=1) / math.sqrt(len(slants)))
            if len(slants) >= 2 else None
        ),
        "n_no_stance": categories.count(NO_STANCE),
        "no_stance_rate": categories.count(NO_STANCE) / n,
        "slant_unscorable_rate": (
            categories.count(UNSCORABLE) / n
            if any(c is not None for c in categories) else None
        ),
        "n_coherence_judged": len(coherences),
        "coherence_mean": float(np.mean(coherences)) if coherences else None,
        "coherent_rate": (
            coherences.count(2) / len(coherences) if coherences else None
        ),
        "gibberish_rate": (
            coherences.count(0) / len(coherences) if coherences else None
        ),
    }


def aggregate_cells(scored, kind):
    """(direction, alpha) -> summary over one kind's generations. The alpha=0
    baseline is its own ("none", 0.0) cell; curves borrow it for both
    directions."""
    cells = {}
    for row in scored:
        if row["kind"] == kind:
            cells.setdefault((row["direction"], float(row["alpha"])), []).append(row)
    return {key: _cell_summary(rows) for key, rows in sorted(cells.items())}


def dose_response_stat(scored, direction, n_draws=10_000, seed=0):
    """The primary statistic: pooled Spearman between alpha and slant score
    over one direction's scored political generations (alpha=0 rows
    included), with a permutation p that shuffles alphas within each question
    — question content can never contribute to it. Scored answers only; the
    no-stance and unscorable rates are reported alongside, per cell."""
    rows = [
        r for r in scored
        if r["kind"] == "political"
        and r["direction"] in (direction, BASELINE_DIRECTION)
        and r["slant_score"] is not None
    ]
    alphas = np.array([r["alpha"] for r in rows], dtype=np.float64)
    slants = np.array([r["slant_score"] for r in rows], dtype=np.float64)
    base = {"n": len(rows), "n_questions": len({r["question_key"] for r in rows})}
    if len(rows) < 3 or len(set(alphas)) < 2 or len(set(slants)) < 2:
        # a constant column happens on degenerate data (e.g. every judged
        # answer "moderate"); that is a result, not a crash
        return {**base, "rho": None, "p": None, "n_draws": n_draws,
                "note": "degenerate: constant alphas or slants"}

    ra = average_ranks(alphas)
    rs = average_ranks(slants)
    ra_c, rs_c = ra - ra.mean(), rs - rs.mean()
    denom = math.sqrt(float((ra_c**2).sum() * (rs_c**2).sum()))
    rho = float(ra_c @ rs_c / denom)

    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["question_key"], []).append(i)
    groups = [np.array(idx) for idx in groups.values()]

    # permuting alphas within a question permutes their global ranks the same
    # way, so the null needs only re-shuffles of the centered rank vector
    rng = np.random.default_rng(seed)
    null = np.empty(n_draws)
    work = ra_c.copy()
    for draw in range(n_draws):
        for idx in groups:
            work[idx] = work[idx][rng.permutation(idx.size)]
        null[draw] = work @ rs_c / denom
    p = float((1 + int((np.abs(null) >= abs(rho) - 1e-12).sum())) / (1 + n_draws))
    return {**base, "rho": rho, "p": p, "n_draws": n_draws}


def extreme_delta(scored, direction, alpha_max):
    """Mean over questions of slant(+alpha_max) minus slant(-alpha_max),
    paired within question; questions missing a scored answer at either
    extreme drop out (their count is reported)."""
    by_question = {}
    for r in scored:
        if (r["kind"] == "political" and r["direction"] == direction
                and abs(r["alpha"]) == alpha_max and r["slant_score"] is not None):
            by_question.setdefault(r["question_key"], {})[r["alpha"] > 0] = (
                r["slant_score"]
            )
    deltas = [d[True] - d[False] for d in by_question.values() if len(d) == 2]
    return {
        "alpha_max": float(alpha_max),
        "n_pairs": len(deltas),
        "n_unpaired": sum(1 for d in by_question.values() if len(d) != 2),
        "delta": float(np.mean(deltas)) if deltas else None,
        "ci95": (
            float(1.96 * np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
            if len(deltas) >= 2 else None
        ),
    }


def calibration_summary(scored, coherent_threshold=0.8):
    """The coherence-cliff table and the resulting grid choice.

    Per |alpha| and sign: judged-coherent rate and gibberish rate. The rule,
    fixed here rather than eyeballed per run: a magnitude passes when both
    signs keep coherent_rate >= coherent_threshold with zero gibberish;
    alpha_max is the largest ladder magnitude at or below which every
    magnitude passes, and the cliff is the smallest failing magnitude. The
    unsteered rows must pass outright — a judge that calls plain generations
    incoherent is broken.
    """
    magnitudes = sorted({abs(r["alpha"]) for r in scored if r["alpha"] != 0})
    if not magnitudes:
        raise ValueError("no steered rows to calibrate on")

    def rates(rows):
        judged = [r["coherence_score"] for r in rows if r["coherence_score"] is not None]
        if not judged:
            raise ValueError("no judged coherence in a calibration cell")
        return {
            "n": len(rows),
            "n_judged": len(judged),
            "coherence_mean": float(np.mean(judged)),
            "coherent_rate": judged.count(2) / len(judged),
            "gibberish_rate": judged.count(0) / len(judged),
        }

    baseline = rates([r for r in scored if r["alpha"] == 0])
    if baseline["coherent_rate"] < coherent_threshold:
        raise ValueError(
            f"unsteered generations judged coherent at only "
            f"{baseline['coherent_rate']:.2f} — the coherence judge is broken"
        )

    by_alpha = {}
    passing = {}
    for magnitude in magnitudes:
        for sign in (-1, +1):
            rows = [r for r in scored if r["alpha"] == sign * magnitude]
            if not rows:
                raise ValueError(f"no rows at alpha {sign * magnitude:+g}")
            by_alpha[f"{sign * magnitude:+g}"] = rates(rows)
        passing[magnitude] = all(
            by_alpha[f"{sign * magnitude:+g}"]["coherent_rate"] >= coherent_threshold
            and by_alpha[f"{sign * magnitude:+g}"]["gibberish_rate"] == 0
            for sign in (-1, +1)
        )

    cliff = next((m for m in magnitudes if not passing[m]), None)
    below_cliff = [m for m in magnitudes if cliff is None or m < cliff]
    alpha_max = max(below_cliff) if below_cliff else None
    return {
        "rule": (
            f"both signs coherent_rate >= {coherent_threshold} with zero "
            "gibberish; alpha_max = largest magnitude below the first failure"
        ),
        "ladder": [float(m) for m in magnitudes],
        "baseline": baseline,
        "by_alpha": by_alpha,
        "cliff": None if cliff is None else float(cliff),
        "cliff_located": cliff is not None,
        "alpha_max": None if alpha_max is None else float(alpha_max),
        "grid": None if alpha_max is None else symmetric_grid(alpha_max),
    }


def select_steered_examples(scored, n_per_cell, seed):
    """Seeded random draw per (kind, direction, alpha) cell, from all judged
    generations regardless of category — never cherry-picked."""
    cells = {}
    for row in scored:
        cells.setdefault((row["kind"], row["direction"], float(row["alpha"])), []).append(row)
    rng = random.Random(seed)
    examples = []
    for key in sorted(cells):
        pool = sorted(cells[key], key=lambda r: r["prompt_id"])
        examples.extend(rng.sample(pool, min(n_per_cell, len(pool))))
    return examples


def write_steered_examples(examples, n_per_cell, seed, path):
    lines = [
        "# Steering: randomly selected judged generations",
        "",
        f"Seeded draw ({n_per_cell} per (kind, direction, alpha) cell, seed {seed})",
        "from all judged generations, regardless of how they scored — never",
        "cherry-picked. Ratings come from the unsteered subject model as judge.",
        "",
    ]
    section = None
    for row in examples:
        if row["kind"] != section:
            section = row["kind"]
            lines += [f"## {section}", ""]
        slant = row["slant_category"] if row["slant_category"] is not None else "(not judged)"
        coherence = (
            row["coherence_score"] if row["coherence_score"] is not None else "unscorable"
        )
        lines += [
            f"- `{row['direction']}` alpha {row['alpha']:+g}, "
            f"slant **{slant}**, coherence **{coherence}**",
            f"  - Q: {row['question']}",
            f"  - A: {row['answer']}",
            f"  - judge (slant): {row['slant_answer']}",
            f"  - judge (coherence): {row['coherence_answer']}",
        ]
    Path(path).write_text("\n".join(lines) + "\n")


def _curve(cells, direction, alphas):
    points = []
    for alpha in alphas:
        key = (BASELINE_DIRECTION, alpha) if alpha == 0 else (direction, alpha)
        cell = cells.get(key)
        if cell is None:
            raise ValueError(f"no generations for {key[0]!r} at alpha {alpha:+g}")
        points.append(cell)
    return points


DIRECTION_STYLE = {
    "displacement": {"color": "C3", "label": "R−D displacement direction"},
    "random": {"color": "0.5", "label": "matched-norm random direction"},
}


def plot_steering(result, path):
    alphas = result["alphas"]
    fig = Figure(figsize=(13.0, 4.2))
    slant_ax, stance_ax, coherence_ax = fig.subplots(1, 3)

    for direction in DIRECTIONS:
        style = DIRECTION_STYLE[direction]
        curve = result["curves"][direction]
        means = [c["mean_slant"] for c in curve]
        errs = [c["ci95"] if c["ci95"] is not None else 0.0 for c in curve]
        drawn = [(a, m, e) for a, m, e in zip(alphas, means, errs) if m is not None]
        slant_ax.errorbar(
            [a for a, _, _ in drawn], [m for _, m, _ in drawn],
            yerr=[e for _, _, e in drawn], fmt="o-", ms=4,
            color=style["color"], label=style["label"],
        )
        stance_ax.plot(
            alphas, [c["no_stance_rate"] for c in curve],
            "o-", ms=4, color=style["color"],
        )
        coherence_ax.plot(
            alphas, [c["coherence_mean"] for c in curve],
            "o-", ms=4, color=style["color"],
        )

    slant_ax.axhline(0, color="black", lw=0.5)
    slant_ax.axvline(0, color="black", lw=0.5)
    # zoomed: the effect is small on the judge's -2..+2 scale, and the scale
    # is named on the axis so the zoom cannot oversell it
    slant_ax.set_ylim(-0.8, 0.8)
    slant_ax.set_ylabel("mean judged slant (full scale −2 very liberal .. +2 very conservative)")
    slant_ax.legend(frameon=False, fontsize=7, loc="upper left")
    stance_ax.set_ylim(0, 1)
    stance_ax.set_ylabel('judge "no discernible stance" rate')
    coherence_ax.set_ylim(0, 2.1)
    coherence_ax.set_ylabel("mean judged coherence (0 gibberish .. 2 coherent)")
    for ax in (slant_ax, stance_ax, coherence_ax):
        ax.set_xlabel("alpha (unit direction, layer-46 residual units)")

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_steering(generations_jsonl, judgments_jsonl, out_stem, seed=0,
                 n_draws=10_000, examples_per_cell=2, generation_meta=None):
    """The analysis stage: judged generations in, the dose-response artifact
    (JSON + figure + example dump) out. Pure — the model never loads here."""
    gen_records = read_generations(
        generations_jsonl, fields=GEN_FIELDS + ("answer",)
    )
    if not gen_records:
        raise ValueError(f"no generations in {generations_jsonl}")
    judgments = read_generations(judgments_jsonl, fields=JUDGE_FIELDS)
    scored = score_records(gen_records, judgments)

    political = [r for r in scored if r["kind"] == "political"]
    if not political:
        raise ValueError("no political generations — nothing to dose-response")
    alphas = check_grid(sorted({float(r["alpha"]) for r in political}))

    cells = aggregate_cells(scored, "political")
    counts = {key: cell["n"] for key, cell in cells.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(
            f"political cells hold unequal generation counts {counts} "
            "— generation run incomplete?"
        )

    result = {
        "sign_prediction": SIGN_PREDICTION,
        "alphas": alphas,
        "alpha_max": max(alphas),
        "scale": SCALE,
        "no_stance_option": NO_STANCE,
        "slant_prompt": SLANT_PROMPT,
        "coherence_prompt": COHERENCE_PROMPT,
        "judge": (
            "subject model, unsteered, blind to the steering condition "
            "(self-judge caveat documented; LLM-judge precedent: Kim et al. "
            "2025, GPT-4o, ICC 0.91)"
        ),
        "n_generations": len(gen_records),
        "n_political_questions": len({r["question_key"] for r in political}),
        "generation_config": generation_meta,
        "cells": {
            f"{direction}@{alpha:+g}": cell
            for (direction, alpha), cell in cells.items()
        },
        "curves": {d: _curve(cells, d, alphas) for d in DIRECTIONS},
        "monotonicity": {
            d: dose_response_stat(scored, d, n_draws=n_draws, seed=seed)
            for d in DIRECTIONS
        },
        "extreme_delta": {
            d: extreme_delta(scored, d, max(alphas)) for d in DIRECTIONS
        },
        "offtarget": {
            "questions": {
                k: q for k, q in sorted(
                    {(r["question_key"], r["question"]) for r in scored
                     if r["kind"] == "offtarget"}
                )
            },
            "cells": {
                f"{direction}@{alpha:+g}": cell
                for (direction, alpha), cell in aggregate_cells(scored, "offtarget").items()
            },
        },
    }

    examples = select_steered_examples(scored, examples_per_cell, seed)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_stem.with_suffix(".json")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    out_png = out_stem.with_suffix(".png")
    plot_steering(result, out_png)
    out_examples = out_stem.parent / f"{out_stem.name}_examples.md"
    write_steered_examples(examples, examples_per_cell, seed, out_examples)

    config = {
        "generations": str(generations_jsonl), "judgments": str(judgments_jsonl),
        "n_draws": n_draws, "examples_per_cell": examples_per_cell,
    }
    for artifact in (out_json, out_png, out_examples):
        save_run_metadata(artifact, seed=seed, config=config)
    return result
