"""Matched-set reconstruction from the Cen et al. 2025 prompt bank.

The release's flattened question bank has one row per final question: base_q
(the question *template*, placeholders unfilled), prompt_type (the variation,
"none" for the identity variation "{}"), and question (the final text). The
matched-set unit is the pre-prompt question — placeholders filled, variation
not yet applied — which the bank does not store directly; we recover it by
inverting the variation around its "{}" slot and key sets on its sha256, the
release's own pre_prompt_q_hash recipe. Keying on base_q instead would mix
different placeholder contents (candidates, issues) inside one set: 105
templates vs 573 questions in the real bank, and a matched set must hold
content constant with only the scaffold varying. base_q_template_hash is kept
as a column so cross-validation can still group at the template level.
"""

import hashlib
import random
from collections import Counter

REQUIRED_FIELDS = ("question", "type", "category", "base_q", "prompt_type")

# Cen et al. ship 21 prompt variations plus the identity variation "{}";
# anything else means the upstream dataset moved under us
EXPECTED_N_CONDITIONS = 22


def hash_string(s):
    # the release's hash recipe (README "Hash Generation")
    return hashlib.sha256(s.encode()).hexdigest()


def expected_conditions(prompt_variants):
    """Variation strings from the release's prompt_variations.json, renamed
    the way bank rows name them: the identity variation "{}" is "none"."""
    conditions = {"none" if v == "{}" else v for v in prompt_variants}
    if len(conditions) != EXPECTED_N_CONDITIONS:
        raise ValueError(
            f"expected {EXPECTED_N_CONDITIONS} prompt variations (21 + none), "
            f"got {len(conditions)}: {sorted(conditions)}"
        )
    return conditions


def recover_pre_prompt_q(question, prompt_type):
    if prompt_type == "none":
        return question
    prefix, slot, suffix = prompt_type.partition("{}")
    if not slot or not question.startswith(prefix) or not question.endswith(suffix):
        raise ValueError(
            f"question does not embed its variation template: "
            f"variation {prompt_type!r}, question {question!r}"
        )
    return question[len(prefix) : len(question) - len(suffix)]


def build_matched_sets(bank_rows, conditions):
    """Group bank rows into matched sets: one per pre-prompt question, holding
    the final question text under every condition, anchored on "none".

    Returns (sets, report). Sets are sorted by pre_prompt_q_hash so downstream
    order never depends on bank row order. Excluded material is counted in the
    report; schema drift (missing fields, vocabulary mismatch in either
    direction, uninvertible questions, duplicate rows, no anchored sets at
    all) raises instead.
    """
    for row in bank_rows:
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"bank row missing field(s) {missing}: {row!r}")

    kept = [r for r in bank_rows if r["type"] != "baseline"]
    dropped = Counter((r["type"], r["category"]) for r in bank_rows if r["type"] == "baseline")

    observed = {r["prompt_type"] for r in kept}
    if observed != set(conditions):
        raise ValueError(
            f"prompt-type vocabulary drifted: unexpected {sorted(observed - set(conditions))}, "
            f"absent {sorted(set(conditions) - observed)}; "
            f"observed vocabulary is {sorted(observed)}"
        )

    by_question = {}
    for row in kept:
        pre_prompt_q = recover_pre_prompt_q(row["question"], row["prompt_type"])
        q_hash = hash_string(pre_prompt_q)
        s = by_question.setdefault(
            q_hash,
            {
                "pre_prompt_q": pre_prompt_q,
                "pre_prompt_q_hash": q_hash,
                "base_q_template": row["base_q"],
                "base_q_template_hash": hash_string(row["base_q"]),
                "type": row["type"],
                "category": row["category"],
                "subcategory": row.get("subcategory") or "",
                "questions": {},
            },
        )
        if row["prompt_type"] in s["questions"]:
            raise ValueError(
                f"duplicate condition {row['prompt_type']!r} for question {pre_prompt_q!r}"
            )
        if s["base_q_template"] != row["base_q"]:
            raise ValueError(
                f"question {pre_prompt_q!r} claims two templates: "
                f"{s['base_q_template']!r} and {row['base_q']!r}"
            )
        s["questions"][row["prompt_type"]] = row["question"]

    # a bank with no "none" rows at all never reaches here: "none" is always
    # among the expected conditions, so the vocabulary check above fires first
    anchored = sorted(
        (s for s in by_question.values() if "none" in s["questions"]),
        key=lambda s: s["pre_prompt_q_hash"],
    )

    report = {
        "rows_in": len(bank_rows),
        "rows_dropped": dict(dropped),
        "sets_missing_none": len(by_question) - len(anchored),
        "incomplete_sets": sum(
            1 for s in anchored if set(s["questions"]) != set(conditions)
        ),
        "n_sets": len(anchored),
        "condition_counts": Counter(
            c for s in anchored for c in s["questions"]
        ),
    }
    return anchored, report


def subsample_sets(matched_sets, max_sets, seed):
    """Seeded subsample of whole sets, stable under input order (sets are
    re-sorted by hash before drawing, and the draw is order-preserving)."""
    ordered = sorted(matched_sets, key=lambda s: s["pre_prompt_q_hash"])
    if len(ordered) <= max_sets:
        return ordered
    picked = random.Random(seed).sample(range(len(ordered)), max_sets)
    return [ordered[i] for i in sorted(picked)]


def condition_order(condition):
    # "none" first, then alphabetical — the display and table order everywhere
    return (condition != "none", condition)


def prompt_table_rows(matched_sets):
    """Flatten sets to prompt-table rows, "none" first within each set. Row
    order and prompt_id must be reproducible across rebuilds: the activation
    cache (polreps.actcache) is keyed by prompt_id."""
    rows = []
    for s in matched_sets:
        for condition in sorted(s["questions"], key=condition_order):
            rows.append(
                {
                    "prompt_id": f"{s['pre_prompt_q_hash'][:16]}-{hash_string(condition)[:16]}",
                    "condition": condition,
                    "question": s["questions"][condition],
                    "pre_prompt_q_hash": s["pre_prompt_q_hash"],
                    "base_q_template_hash": s["base_q_template_hash"],
                    "type": s["type"],
                    "category": s["category"],
                    "subcategory": s["subcategory"],
                }
            )
    return rows
