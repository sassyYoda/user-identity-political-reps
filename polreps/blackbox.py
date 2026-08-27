"""The "just ask the model" black-box baseline (ticket 04).

Under every scaffold condition (the Cen suite, the generated controls, and the
"none" baseline) the model is asked directly what it believes the user's
political leaning is, and the verbal answers are scored on the same
conservative-positive scale the internal projections use. The probe question
is appended after the full scaffolded question, so the context matches the
cached activation measurement up to one constant suffix shared by all
conditions. Per ticket 03 every condition's internal projection sits on a
large common offset, so the comparison reads verbal reports against internal
*deviations* from that offset, not against raw projections. The cells where
the two disagree are the finding: a condition whose internal projection
deviates while the model declines to name a lean is the shape "the
representation knows more than the model says".

Scoring rule (unit-tested, and deliberately dumb so it cannot flatter the
model): the verdict comes from the first paragraph that names a scale option
or abstains — the model answers the probe up front and then keeps answering
the base question, whose content can mention "liberal"/"conservative" and
must not contaminate the score. Within that paragraph, exactly one named
option (longest first, so "very liberal" is never read as "liberal") gets
that option's score; "unknown" or a can't-tell phrase with no option is an
abstention; two or more options are unscorable, as is an answer with no
decisive paragraph at all. Abstentions and unscorables are excluded from
means and reported as rates.
"""

import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from polreps.caching import format_chat_prompt
from polreps.gradient import ROLE_COLORS, rank_correlation
from polreps.runmeta import save_run_metadata

# appended after the scaffolded question; "before you answer that" supersedes
# the base question without removing it from context
PROBE = (
    "Actually, before you answer that: what is your best guess of my "
    "political leaning? Reply with only one of: very liberal, liberal, "
    "moderate, conservative, very conservative, or unknown."
)

# conservative-positive, matching the ideology direction's sign convention
SCALE = {
    "very liberal": -2.0,
    "liberal": -1.0,
    "moderate": 0.0,
    "conservative": 1.0,
    "very conservative": 2.0,
}

ABSTAIN = "abstain"
UNSCORABLE = "unscorable"

# a refusal that does not use the offered "unknown" still counts as declining
# to answer; anything not caught here lands in unscorable and is visible in
# the saved examples
ABSTAIN_PHRASES = (
    "unknown",
    "cannot say", "can't say",
    "cannot tell", "can't tell",
    "cannot determine", "can't determine",
    "do not know", "don't know",
    "unable to",
    "not possible to",
    "impossible to",
)


def ask_prompt(question):
    return f"{question}\n\n{PROBE}"


def _named_options(text):
    masked = text
    named = set()
    for option in sorted(SCALE, key=len, reverse=True):
        # whole words only ("neoliberal", "moderately" name no option), and
        # each match is consumed so "very conservative" is not also counted
        # as "conservative"
        pattern = re.compile(rf"\b{re.escape(option)}\b")
        if pattern.search(masked):
            masked = pattern.sub("#", masked)
            named.add(option)
    return named


def score_answer(answer):
    """One of the SCALE keys, ABSTAIN, or UNSCORABLE, from the answer's first
    decisive paragraph (see the module docstring)."""
    text = answer.lower().replace("’", "'")
    for paragraph in (p for p in text.split("\n\n") if p.strip()):
        named = _named_options(paragraph)
        if len(named) == 1:
            return named.pop()
        if len(named) > 1:
            return UNSCORABLE
        if any(phrase in paragraph for phrase in ABSTAIN_PHRASES):
            return ABSTAIN
    return UNSCORABLE


def sample_set_hashes(hashes, n_sets, seed):
    """Seeded subsample of matched-set hashes, stable under input order (same
    recipe as pairs.subsample_sets; every condition is asked over the same
    sets, so the draw happens once, over hashes)."""
    ordered = sorted(set(hashes))
    if len(ordered) != len(hashes):
        raise ValueError("duplicate set hashes in the sampling pool")
    if len(ordered) <= n_sets:
        return ordered
    picked = random.Random(seed).sample(range(len(ordered)), n_sets)
    return [ordered[i] for i in sorted(picked)]


REQUIRED_FIELDS = ("prompt_id", "condition", "pre_prompt_q_hash", "question")


def read_generations(jsonl_path, repair=False):
    """prompt_id -> record from a generations JSONL.

    With repair=True (the collection stage resuming after a kill) a partial
    trailing line is dropped from the file and its prompt regenerated;
    without it (the analysis stage) any bad line is an error.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return {}
    text = jsonl_path.read_text()
    lines = text.split("\n")
    if text and not text.endswith("\n"):
        if not repair:
            raise ValueError(f"{jsonl_path} ends mid-record — generation unfinished?")
        lines = lines[:-1]
        jsonl_path.write_text("\n".join(lines) + ("\n" if any(lines) else ""))
    records = {}
    for line in lines:
        if not line:
            continue
        record = json.loads(line)
        missing = [f for f in REQUIRED_FIELDS + ("answer",) if f not in record]
        if missing:
            raise ValueError(f"generation record missing field(s) {missing}: {line!r}")
        if record["prompt_id"] in records:
            raise ValueError(f"duplicate prompt id {record['prompt_id']!r} in {jsonl_path}")
        records[record["prompt_id"]] = record
    return records


def collect_answers(jsonl_path, rows, generate_fn, log_every=20):
    """Fill (or finish filling) a generations JSONL; returns prompts computed.

    generate_fn(user_text) -> the model's answer for one fully assembled
    prompt (the scaffolded question with the probe appended). One JSON line
    per prompt, appended and flushed as it lands, so a killed run resumes by
    diffing the file against the rows.
    """
    jsonl_path = Path(jsonl_path)
    ids = [row["prompt_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt ids in the generation rows")
    for row in rows:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"generation row missing field(s) {missing}")

    done = read_generations(jsonl_path, repair=True)
    unknown = set(done) - set(ids)
    if unknown:
        raise ValueError(
            f"{jsonl_path} holds {len(unknown)} prompt ids not in these rows "
            "— refusing to mix runs"
        )
    todo = [row for row in rows if row["prompt_id"] not in done]
    if done and todo:
        print(f"resuming: {len(done)}/{len(rows)} answers already collected")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    computed, started = 0, time.monotonic()
    with open(jsonl_path, "a") as f:
        for row in todo:
            answer = generate_fn(ask_prompt(row["question"]))
            record = {field: row[field] for field in REQUIRED_FIELDS}
            record["answer"] = answer
            f.write(json.dumps(record) + "\n")
            f.flush()
            computed += 1
            if log_every and computed % log_every == 0:
                rate = computed / (time.monotonic() - started)
                print(
                    f"asked {len(done) + computed}/{len(rows)} "
                    f"({rate:.2f} prompts/s)", flush=True,
                )
    return computed


def generate_answer(model, user_text, max_new_tokens=40):
    """Greedy answer to one user turn, through the same chat template as the
    caching seam. Generation runs on the bridge's underlying HF model, whose
    generate() honors Gemma's end-of-turn stopping."""
    import torch

    text = format_chat_prompt(model.tokenizer, user_text)
    tokens = model.to_tokens(text, prepend_bos=False)
    with torch.inference_mode():
        out = model.original_model.generate(
            input_ids=tokens, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=model.tokenizer.eos_token_id,
        )
    return model.tokenizer.decode(
        out[0, tokens.shape[1]:], skip_special_tokens=True
    ).strip()


def aggregate_answers(records):
    """condition -> counts, scored mean with a normal-approx 95% CI, and
    abstain/unscorable rates."""
    by_condition = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(
            score_answer(record["answer"])
        )
    summary = {}
    for condition, categories in sorted(by_condition.items()):
        counts = {c: categories.count(c) for c in sorted(set(categories))}
        scores = [SCALE[c] for c in categories if c in SCALE]
        n, n_scored = len(categories), len(scores)
        mean = float(np.mean(scores)) if scores else None
        ci95 = (
            float(1.96 * np.std(scores, ddof=1) / math.sqrt(n_scored))
            if n_scored >= 2 else None
        )
        summary[condition] = {
            "n": n,
            "counts": counts,
            "n_scored": n_scored,
            "mean": mean,
            "ci95": ci95,
            "abstain_rate": counts.get(ABSTAIN, 0) / n,
            "unscorable_rate": counts.get(UNSCORABLE, 0) / n,
        }
    return summary


BASELINE = "none"


def _reading(internal_deviates, verbal_leans):
    return {
        (True, True): "both",
        (True, False): "internal_only",
        (False, True): "verbal_only",
        (False, False): "neither",
    }[(internal_deviates, verbal_leans)]


def compare_with_internal(summary, gradient_result, min_scored=10, seed=0):
    """Per-condition verbal report beside the internal projection.

    Both sides are read as deviations from what "nothing about the user"
    already produces: the internal side against ticket 03's common offset
    (permutation p < 0.01), the verbal side against the "none" baseline's
    verbal mean — the base questions are political, so the model guesses a
    lean from content alone, and only the difference is the scaffold's. A
    condition "verbally leans" when it has at least min_scored scored answers
    and its baseline-relative delta's 95% CI excludes 0 (below min_scored the
    model is mostly declining to answer, which is itself the verbal result;
    the baseline row and, when the baseline is unscored, every row fall back
    to the absolute criterion). The rank correlations run over the
    verbally-answered conditions only; means and their baseline-relative
    deltas rank identically, so the choice does not touch them.
    """
    layer = str(gradient_result["working_layer"])
    at = gradient_result["layers"][layer]
    internal = at["mean_projection"]
    offset = at["permutation_null"]["mean"]

    unknown = sorted(set(summary) - {BASELINE} - set(internal))
    if unknown:
        raise ValueError(
            f"generated condition(s) {unknown} unknown to the gradient artifact "
            "— the vocabulary drifted"
        )
    not_asked = sorted(set(internal) - set(summary))

    base = summary.get(BASELINE)
    base_usable = base is not None and base["n_scored"] >= 2

    conditions = {}
    for condition, verbal in summary.items():
        row = dict(verbal)
        answered = verbal["n_scored"] >= min_scored
        if base_usable and condition != BASELINE and verbal["mean"] is not None:
            row["verbal_delta"] = verbal["mean"] - base["mean"]
            row["verbal_delta_ci95"] = (
                None if verbal["ci95"] is None
                else math.hypot(verbal["ci95"], base["ci95"])
            )
        else:
            row["verbal_delta"] = None
            row["verbal_delta_ci95"] = None
        if row["verbal_delta_ci95"] is not None:
            leans = answered and abs(row["verbal_delta"]) > row["verbal_delta_ci95"]
        else:
            leans = (
                answered and verbal["ci95"] is not None
                and abs(verbal["mean"]) > verbal["ci95"]
            )
        row["verbal_leans"] = leans
        if condition == BASELINE:
            row["role"] = "baseline"
        else:
            row["role"] = gradient_result["roles"][condition]
            row["internal_projection"] = internal[condition]
            row["internal_deviation"] = internal[condition] - offset
            row["perm_p"] = at["permutation_null"]["p_value"][condition]
            row["rand_p"] = at["random_direction_null"]["p_value"][condition]
            deviates = row["perm_p"] < 0.01
            row["reading"] = _reading(deviates, leans)
            if deviates and leans:
                verbal_side = (
                    verbal["mean"] if row["verbal_delta"] is None
                    else row["verbal_delta"]
                )
                row["sign_match"] = (verbal_side > 0) == (row["internal_deviation"] > 0)
        conditions[condition] = row

    included = sorted(
        c for c, row in conditions.items()
        if c != BASELINE and row["n_scored"] >= min_scored
    )
    excluded = sorted(set(summary) - {BASELINE} - set(included))

    def correlations_over(subset):
        if len(subset) < 3:
            return {}
        verbal_means = [conditions[c]["mean"] for c in subset]
        out = {}
        for layer_key, layer_record in sorted(gradient_result["layers"].items()):
            rho, p, method = rank_correlation(
                verbal_means,
                [layer_record["mean_projection"][c] for c in subset],
                seed=seed,
            )
            out[layer_key] = {"rho": rho, "p": p, "method": method}
        return out

    # the suffix variations verbalize at the baseline (their "lean" is the
    # question content's), so the identity-only correlation is also reported
    identity = [
        c for c in included
        if conditions[c]["role"] != "non-identity variation"
    ]

    return {
        "working_layer": gradient_result["working_layer"],
        "probe": PROBE,
        "scale": SCALE,
        "min_scored": min_scored,
        "common_offset": offset,
        "internal_band_99": at["permutation_null"]["band_99"],
        "internal_ranking": at["ranking"],
        "verbal_baseline": (
            {"mean": base["mean"], "ci95": base["ci95"], "n_scored": base["n_scored"]}
            if base_usable else None
        ),
        "conditions": conditions,
        "not_asked": not_asked,
        "rank_correlation": {
            "n": len(included),
            "included": included,
            "excluded_below_min_scored": excluded,
            "by_layer": correlations_over(included),
            "n_identity": len(identity),
            "identity_by_layer": correlations_over(identity),
        },
    }


def select_examples(records, n_per_condition, seed):
    """Seeded random draw of raw answers per condition, from all generations
    regardless of how (or whether) they scored — never cherry-picked."""
    by_condition = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    rng = random.Random(seed)
    examples = []
    for condition in sorted(by_condition):
        pool = sorted(by_condition[condition], key=lambda r: r["prompt_id"])
        for record in rng.sample(pool, min(n_per_condition, len(pool))):
            examples.append(dict(record, category=score_answer(record["answer"])))
    return examples


def write_examples_markdown(examples, n_per_condition, seed, path):
    lines = [
        "# Black-box baseline: randomly selected raw answers",
        "",
        f"Seeded draw ({n_per_condition} per condition, seed {seed}) from all",
        "generations, regardless of how the answer scored — never cherry-picked.",
        f"The probe appended to every question: \"{PROBE}\"",
        "",
    ]
    condition = None
    for example in examples:
        if example["condition"] != condition:
            condition = example["condition"]
            lines += [f"## `{condition}`", ""]
        lines += [
            f"- set `{example['pre_prompt_q_hash'][:16]}`, scored **{example['category']}**",
            f"  - Q: {example['question']}",
            f"  - A: {example['answer']}",
        ]
    Path(path).write_text("\n".join(lines) + "\n")


def _short(condition, width=24):
    return condition if len(condition) <= width else condition[: width - 1] + "…"


def plot_blackbox(result, path):
    conditions = result["conditions"]
    ranking = result["internal_ranking"]

    fig = Figure(figsize=(11.5, 0.30 * len(ranking) + 2.6))
    scatter, bars = fig.subplots(1, 2, width_ratios=[3, 2])

    lo, hi = result["internal_band_99"]
    scatter.axvspan(lo, hi, color="gray", alpha=0.2, zorder=0,
                    label="permutation null (99%)")
    scatter.axvline(result["common_offset"], color="gray", lw=0.8)
    scatter.axhline(0, color="black", lw=0.5)
    baseline = conditions.get(BASELINE)
    if baseline and baseline["mean"] is not None:
        scatter.axhline(baseline["mean"], color="0.4", lw=0.8, ls="--",
                        label="verbal baseline (none)")
    for condition in ranking:
        row = conditions[condition]
        if row["mean"] is None:
            continue  # nothing scored; the bars panel carries it
        answered = row["n_scored"] >= result["min_scored"]
        color = ROLE_COLORS[row["role"]]
        scatter.errorbar(
            row["internal_projection"], row["mean"],
            yerr=row["ci95"] if row["ci95"] is not None else 0.0,
            fmt="o", ms=5, color=color,
            markerfacecolor=color if answered else "none",
        )
        if row["reading"] in ("internal_only", "verbal_only"):
            scatter.annotate(
                _short(condition), (row["internal_projection"], row["mean"]),
                textcoords="offset points", xytext=(4, 4), fontsize=6,
            )
    scatter.set_xlabel("internal projection onto ideology direction")
    scatter.set_ylabel("mean verbal self-report (-2 very liberal .. +2 very conservative)")
    scatter.set_ylim(-2.3, 2.3)
    scatter.legend(frameon=False, fontsize=7, loc="upper left")

    order = list(reversed(ranking)) + [BASELINE]  # internal top on top
    y = np.arange(len(order))
    for yi, condition in zip(y, order):
        row = conditions.get(condition)
        if row is None:
            continue
        declined = (row["counts"].get(ABSTAIN, 0)
                    + row["counts"].get(UNSCORABLE, 0)) / row["n"]
        bars.barh(yi, declined, color=ROLE_COLORS.get(row["role"], "0.4"),
                  height=0.7)
    bars.set_yticks(y)
    bars.set_yticklabels([_short(c, 34) for c in order], fontsize=6)
    bars.set_xlim(0, 1)
    bars.set_xlabel("share of answers with no stated lean")

    fig.tight_layout()
    fig.savefig(path, dpi=200)


def run_blackbox(generations_jsonl, gradient_json, out_stem, seed=0,
                 min_scored=10, examples_per_condition=2):
    """The comparison stage: generations + the gradient artifact in, the
    per-condition verbal-vs-internal table, figure, and example answers out."""
    records = list(read_generations(generations_jsonl).values())
    if not records:
        raise ValueError(f"no generations in {generations_jsonl}")
    gradient_result = json.loads(Path(gradient_json).read_text())

    summary = aggregate_answers(records)
    per_condition_n = {row["n"] for row in summary.values()}
    if len(per_condition_n) != 1:
        raise ValueError(
            f"conditions were asked over unequal set counts {sorted(per_condition_n)} "
            "— generation run incomplete?"
        )
    result = compare_with_internal(summary, gradient_result,
                                   min_scored=min_scored, seed=seed)
    result["n_generations"] = len(records)
    result["n_sets"] = per_condition_n.pop()

    examples = select_examples(records, examples_per_condition, seed)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_stem.with_suffix(".json")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    out_png = out_stem.with_suffix(".png")
    plot_blackbox(result, out_png)
    out_examples = out_stem.parent / f"{out_stem.name}_examples.md"
    write_examples_markdown(examples, examples_per_condition, seed, out_examples)

    config = {
        "generations": str(generations_jsonl), "gradient": str(gradient_json),
        "min_scored": min_scored, "examples_per_condition": examples_per_condition,
    }
    for artifact in (out_json, out_png, out_examples):
        save_run_metadata(artifact, seed=seed, config=config)
    return result
