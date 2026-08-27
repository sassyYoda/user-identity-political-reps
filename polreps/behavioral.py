"""The internal-to-behavioral link (ticket 06, the stretch item).

Does the size of a scaffold's activation displacement predict how much it
moves the model's actual answers? Per condition, the model answers the *base*
questions (no leaning probe — ticket 04's generations answered the probe, not
the question) under the scaffold and under "none", and the behavioral
displacement is Cen et al.'s output-embedding measure: cosine distance
between the two answers' MiniLM embeddings, paired within matched set,
averaged over sets. The internal side is read from the registered gradient
artifact at the working layer.

Pre-registered, fixed here before any generation ran:

- Conditions: the identity scaffolds, partisan paraphrases, and the three
  controls (16 non-baseline conditions) plus "none", over ticket 04's seeded
  60-set subsample. The ten non-identity suffix variations are excluded for
  the generation budget; they were also the unstable cut in tickets 03-05,
  and the exclusion is a stated limitation, not a robustness choice.
- Generation: greedy, 100 new tokens, the ticket-04 harness and chat
  template. Embedding: sentence-transformers/all-MiniLM-L6-v2 at the pinned
  revision below, normalized, so cosine distance is 1 minus the dot product
  (its 256-wordpiece window covers the answers untruncated).
- Primary statistic: Spearman between the conditions' layer-46 mean
  displacement norms (gradient.json, already registered) and their mean
  output-embedding distances, permutation p from rank_correlation.
  Sign prediction: positive — conditions that displace the residual more
  move the answers more. Secondary, same test: |deviation of the mean
  projection from the common offset| (the ideology-specific component)
  against the same behavioral distances.
- Diagnostics for the boring alternatives, reported alongside: answer word
  count deltas vs "none" (a scaffold that merely changes verbosity moves the
  embedding), and the scaffold-echo rate (an answer that repeats "as a
  Democrat..." moves the embedding without any opinion shift).

Ticket 05 found tiny behavioral effects from much larger interventions, so a
flat result here is a live possibility and, per the spec, an acceptable one.
"""

import json
import math
import random
import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from polreps.blackbox import read_generations
from polreps.gradient import ROLE_COLORS, rank_correlation
from polreps.runmeta import save_run_metadata

EMBEDDER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# pinned 2026-08-27, same discipline as the subject model's revision
EMBEDDER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

BASELINE = "none"

# scaffold words an answer can echo; everything else in the templates is
# first-person plumbing
_STOPWORDS = {"i", "m", "am", "a", "an", "the", "was", "do", "not", "as", "or", "in", "of"}


def echo_terms(condition):
    words = re.findall(r"[a-z]+(?:-[a-z]+)*", condition.lower())
    return sorted(set(words) - _STOPWORDS)


def echoes_scaffold(condition, answer):
    text = answer.lower()
    return any(
        re.search(rf"\b{re.escape(term)}\b", text) for term in echo_terms(condition)
    )


def _word_count(text):
    return len(re.findall(r"\S+", text))


def load_embedder(device="cpu"):
    """embed_fn over the pinned MiniLM; normalized rows, so downstream cosine
    distance is 1 - dot. CPU is deliberate: the model is tiny and the GPU may
    be busy generating."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER_NAME, revision=EMBEDDER_REVISION, device=device)
    return lambda texts: np.asarray(
        model.encode(texts, batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False),
        dtype=np.float64,
    )


def paired_output_displacement(records, embed_fn, baseline=BASELINE):
    """Per-condition mean cosine distance between each answer's embedding and
    the same matched set's baseline answer, plus the length/echo diagnostics.

    Complete sets are required — the measure is paired within set by
    construction, and a missing cell would silently unpair the average.
    """
    answer_of = {}
    for record in records:
        key = (record["pre_prompt_q_hash"], record["condition"])
        if key in answer_of:
            raise ValueError(f"duplicate (set, condition) cell {key!r}")
        if not record["answer"].strip():
            raise ValueError(f"empty answer for {record['prompt_id']!r}")
        answer_of[key] = record["answer"]

    sets = sorted({s for s, _ in answer_of})
    conditions = sorted({c for _, c in answer_of} - {baseline})
    if not conditions:
        raise ValueError(f"only {baseline!r} rows present; nothing to compare")
    absent = [
        (s, c) for c in [baseline] + conditions for s in sets if (s, c) not in answer_of
    ]
    if absent:
        raise ValueError(
            f"{len(absent)} (set, condition) cell(s) missing (first: {absent[0]!r}) "
            "— the pairing needs complete sets"
        )

    keys = [(s, c) for s in sets for c in [baseline] + conditions]
    embeddings = embed_fn([answer_of[k] for k in keys])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_of = dict(zip(keys, embeddings))

    n_sets = len(sets)
    per_set, mean, ci95, wc_delta, echo = {}, {}, {}, {}, {}
    for condition in conditions:
        distances = np.array([
            1.0 - float(embedding_of[(s, condition)] @ embedding_of[(s, baseline)])
            for s in sets
        ])
        per_set[condition] = {s: float(d) for s, d in zip(sets, distances)}
        mean[condition] = float(distances.mean())
        ci95[condition] = float(1.96 * distances.std(ddof=1) / math.sqrt(n_sets))
        wc_delta[condition] = float(np.mean([
            abs(_word_count(answer_of[(s, condition)]) - _word_count(answer_of[(s, baseline)]))
            for s in sets
        ]))
        echo[condition] = float(np.mean([
            echoes_scaffold(condition, answer_of[(s, condition)]) for s in sets
        ]))

    # scale reference: how far apart the baseline answers to *different*
    # questions sit — the ceiling the paired distances live under
    base = np.stack([embedding_of[(s, baseline)] for s in sets])
    off_diagonal = 1.0 - (base @ base.T)[np.triu_indices(n_sets, k=1)]

    return {
        "baseline": baseline,
        "n_sets": n_sets,
        "conditions": conditions,
        "mean_distance": mean,
        "ci95": ci95,
        "per_set": per_set,
        "wordcount_delta": wc_delta,
        "echo_rate": echo,
        "between_question_reference": {
            "mean": float(off_diagonal.mean()),
            "n_pairs": int(off_diagonal.size),
        },
    }


def _correlation(x, y, seed):
    if np.std(x) == 0 or np.std(y) == 0:
        return {"rho": None, "p": None, "note": "one side is constant"}
    rho, p, method = rank_correlation(x, y, seed=seed)
    return {"rho": rho, "p": p, "method": method}


def link_with_internal(behavioral, gradient_result, seed=0):
    """The correlation table: behavioral displacement against the internal
    displacement norm (primary) and the |ideology-axis deviation| (secondary)
    at every layer the gradient artifact carries, plus the length and echo
    diagnostics."""
    conditions = behavioral["conditions"]
    unknown = sorted(set(conditions) - set(gradient_result["conditions"]))
    if unknown:
        raise ValueError(
            f"generated condition(s) {unknown} unknown to the gradient artifact "
            "— the vocabulary drifted"
        )
    distances = [behavioral["mean_distance"][c] for c in conditions]

    working = str(gradient_result["working_layer"])
    by_layer = {}
    for layer_key, at in sorted(gradient_result["layers"].items(), key=lambda kv: int(kv[0])):
        offset = at["permutation_null"]["mean"]
        by_layer[layer_key] = {
            "displacement_norm": _correlation(
                [at["displacement_norm"][c] for c in conditions], distances, seed
            ),
            "abs_deviation": _correlation(
                [abs(at["mean_projection"][c] - offset) for c in conditions],
                distances, seed,
            ),
        }

    at = gradient_result["layers"][working]
    offset = at["permutation_null"]["mean"]
    rows = {}
    for condition in conditions:
        rows[condition] = {
            "role": gradient_result["roles"][condition],
            "mean_distance": behavioral["mean_distance"][condition],
            "ci95": behavioral["ci95"][condition],
            "wordcount_delta": behavioral["wordcount_delta"][condition],
            "echo_rate": behavioral["echo_rate"][condition],
            "internal_norm": at["displacement_norm"][condition],
            "internal_deviation": at["mean_projection"][condition] - offset,
        }

    return {
        "working_layer": gradient_result["working_layer"],
        "baseline": behavioral["baseline"],
        "n_sets": behavioral["n_sets"],
        "n_conditions": len(conditions),
        "between_question_reference": behavioral["between_question_reference"],
        "conditions": rows,
        "rank_correlation": {
            "primary": dict(
                by_layer[working]["displacement_norm"],
                layer=int(working), internal="displacement_norm",
            ),
            "secondary": dict(
                by_layer[working]["abs_deviation"],
                layer=int(working), internal="abs_deviation_from_offset",
            ),
            "by_layer": by_layer,
            "diagnostics": {
                "wordcount_delta": _correlation(
                    [behavioral["wordcount_delta"][c] for c in conditions],
                    distances, seed,
                ),
                "echo_rate": _correlation(
                    [behavioral["echo_rate"][c] for c in conditions],
                    distances, seed,
                ),
            },
        },
    }


def _short(condition, width=28):
    return condition if len(condition) <= width else condition[: width - 1] + "…"


def plot_behavioral_link(result, path):
    rows = result["conditions"]
    order = sorted(rows)
    correlation = result["rank_correlation"]

    fig = Figure(figsize=(11.0, 4.6))
    axes = fig.subplots(1, 2, sharey=True)
    panels = (
        ("internal_norm", "primary", "mean displacement norm (layer %d)"),
        ("internal_deviation", "secondary",
         "projection deviation from the common offset (layer %d)"),
    )
    for ax, (field, stat_key, xlabel) in zip(axes, panels):
        for condition in order:
            row = rows[condition]
            x = row[field] if field == "internal_norm" else abs(row[field])
            ax.errorbar(
                x, row["mean_distance"], yerr=row["ci95"], fmt="o", ms=5,
                color=ROLE_COLORS[row["role"]],
            )
            ax.annotate(_short(condition), (x, row["mean_distance"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6)
        stat = correlation[stat_key]
        ax.set_title(
            f"{stat_key}: Spearman {stat['rho']:+.3f} (p {stat['p']:.3g}, "
            f"n={result['n_conditions']})", fontsize=9,
        )
        ax.set_xlabel(xlabel % result["working_layer"]
                      if field == "internal_norm"
                      else "|" + xlabel % result["working_layer"] + "|")
    axes[0].set_ylabel("mean output-embedding cosine distance from the none answer")

    from matplotlib.lines import Line2D
    roles = sorted({row["role"] for row in rows.values()})
    axes[1].legend(
        handles=[Line2D([], [], marker="o", ls="", color=ROLE_COLORS[r], label=r)
                 for r in roles],
        frameon=False, fontsize=7, loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def select_examples(records, n_per_condition, seed):
    """Seeded random draw of raw base-question answers per condition — never
    cherry-picked (same discipline as ticket 04, without its scorer)."""
    by_condition = {}
    for record in records:
        by_condition.setdefault(record["condition"], []).append(record)
    rng = random.Random(seed)
    examples = []
    for condition in sorted(by_condition):
        pool = sorted(by_condition[condition], key=lambda r: r["prompt_id"])
        examples.extend(rng.sample(pool, min(n_per_condition, len(pool))))
    return examples


def write_examples_markdown(examples, per_set, n_per_condition, seed, path):
    lines = [
        "# Internal-to-behavioral link: randomly selected base-question answers",
        "",
        f"Seeded draw ({n_per_condition} per condition, seed {seed}) from all",
        "generations — never cherry-picked. Each non-baseline answer carries its",
        "output-embedding cosine distance from the same set's none answer.",
        "",
    ]
    condition = None
    for example in examples:
        if example["condition"] != condition:
            condition = example["condition"]
            lines += [f"## `{condition}`", ""]
        distance = per_set.get(condition, {}).get(example["pre_prompt_q_hash"])
        tag = "" if distance is None else f", distance {distance:.3f}"
        lines += [
            f"- set `{example['pre_prompt_q_hash'][:16]}`{tag}",
            f"  - Q: {example['question']}",
            f"  - A: {example['answer']}",
        ]
    Path(path).write_text("\n".join(lines) + "\n")


def run_behavioral_link(generations_jsonl, gradient_json, out_stem, embed_fn,
                        embedder_config, seed=0, examples_per_condition=2):
    """The analysis stage: base-question generations + the gradient artifact
    in; the correlation table, figure, and example answers out."""
    records = list(read_generations(generations_jsonl).values())
    if not records:
        raise ValueError(f"no generations in {generations_jsonl}")
    gradient_result = json.loads(Path(gradient_json).read_text())

    behavioral = paired_output_displacement(records, embed_fn)
    result = link_with_internal(behavioral, gradient_result, seed=seed)
    result["embedder"] = embedder_config
    result["n_generations"] = len(records)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_stem.with_suffix(".json")
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    out_png = out_stem.with_suffix(".png")
    plot_behavioral_link(result, out_png)
    out_examples = out_stem.parent / f"{out_stem.name}_examples.md"
    examples = select_examples(records, examples_per_condition, seed)
    write_examples_markdown(examples, behavioral["per_set"],
                            examples_per_condition, seed, out_examples)

    config = {
        "generations": str(generations_jsonl), "gradient": str(gradient_json),
        "embedder": embedder_config,
        "examples_per_condition": examples_per_condition,
    }
    for artifact in (out_json, out_png, out_examples):
        save_run_metadata(artifact, seed=seed, config=config)
    return result
