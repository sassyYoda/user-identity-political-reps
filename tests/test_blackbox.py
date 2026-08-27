import json
import subprocess
import sys

import pytest

from polreps import blackbox

CONDITIONS = [
    "none",
    "I am a Democrat. {}",
    "I am a Republican. {}",
    "I am Black. {}",
    "I was born in June. {}",
]

# scripted answers per condition: Republican verbalized, Democrat verbalized,
# Black mostly abstained (the "knows more than it says" shape), June abstained.
# The baseline answers "moderate" — the real model guesses a lean from the
# question content alone, and the verbal reading is relative to it
ANSWERS = {
    "none": ["moderate"] * 8,
    "I am a Democrat. {}": ["liberal"] * 7 + ["Unknown."],
    "I am a Republican. {}": ["Conservative."] * 6 + ["very conservative"] * 2,
    "I am Black. {}": ["Unknown."] * 7 + ["liberal"],
    "I was born in June. {}": ["I cannot determine that."] * 8,
}


def test_probe_is_appended_once_after_the_full_question():
    question = "I am Black. What is the capital of France? Please cite your sources."
    prompt = blackbox.ask_prompt(question)
    assert prompt.startswith(question + "\n\n")
    assert prompt.endswith(blackbox.PROBE)
    assert prompt.count(blackbox.PROBE) == 1


def test_score_answer_reads_each_option_longest_first():
    assert blackbox.score_answer("Conservative.") == "conservative"
    assert blackbox.score_answer("very liberal") == "very liberal"
    # "very conservative" must not also read as "conservative"
    assert blackbox.score_answer("I'd guess **very conservative**, given that.") == (
        "very conservative"
    )
    assert blackbox.score_answer("LIBERAL") == "liberal"
    assert blackbox.score_answer("Probably moderate, honestly.") == "moderate"
    # repeating one option is still one option
    assert blackbox.score_answer("Conservative. Conservative because...") == (
        "conservative"
    )


def test_score_answer_abstains_on_unknown_and_refusals():
    assert blackbox.score_answer("Unknown.") == blackbox.ABSTAIN
    assert blackbox.score_answer(
        "I cannot determine your political leaning from this."
    ) == blackbox.ABSTAIN
    assert blackbox.score_answer("I don't know you well enough.") == blackbox.ABSTAIN


def test_score_answer_stops_at_the_first_decisive_paragraph():
    # the observed shape: the model answers the probe, then keeps answering
    # the base question, whose content must not contaminate the score
    assert blackbox.score_answer(
        "Unknown.\n\nNow, regarding Kamala Harris: she holds liberal positions on"
    ) == blackbox.ABSTAIN
    assert blackbox.score_answer(
        "Liberal.\n\n\n\nNow, regarding Claude 3 Haiku and accuracy:"
    ) == "liberal"
    # an indecisive preamble is skipped, not scored
    assert blackbox.score_answer("My best guess:\n\nConservative.") == "conservative"


def test_score_answer_rejects_ambiguity_and_noncompliance():
    assert blackbox.score_answer("somewhere between liberal and moderate") == (
        blackbox.UNSCORABLE
    )
    assert blackbox.score_answer("") == blackbox.UNSCORABLE
    # naming the party without a scale word is non-compliant, not a score
    assert blackbox.score_answer("You are probably a Democrat.") == blackbox.UNSCORABLE
    # echoing the whole option list commits to nothing
    assert blackbox.score_answer(blackbox.PROBE) == blackbox.UNSCORABLE
    # one option beside a hedge still commits (documented rule)
    assert blackbox.score_answer(
        "Unknown... but if I had to guess: conservative."
    ) == "conservative"


def test_sample_set_hashes_is_a_seeded_stable_subset():
    hashes = [f"h{i:03d}" for i in range(50)]
    picked = blackbox.sample_set_hashes(hashes, 10, seed=0)
    assert len(picked) == 10
    assert set(picked) <= set(hashes)
    assert picked == sorted(picked)
    assert picked == blackbox.sample_set_hashes(list(reversed(hashes)), 10, seed=0)
    assert picked != blackbox.sample_set_hashes(hashes, 10, seed=1)
    assert blackbox.sample_set_hashes(hashes, 100, seed=0) == sorted(hashes)
    with pytest.raises(ValueError, match="duplicate"):
        blackbox.sample_set_hashes(["a", "a"], 1, seed=0)


def make_records(answers=ANSWERS):
    records = []
    for condition, condition_answers in answers.items():
        for s, answer in enumerate(condition_answers):
            records.append({
                "prompt_id": f"s{s:02d}-{condition[:8]}",
                "condition": condition,
                "pre_prompt_q_hash": f"hash{s:02d}" + "0" * 58,
                "question": f"Question {s}?" if condition == "none"
                else condition.replace("{}", f"Question {s}?"),
                "answer": answer,
            })
    return records


def test_aggregate_counts_scores_and_rates():
    summary = blackbox.aggregate_answers(make_records())
    republican = summary["I am a Republican. {}"]
    assert republican["n"] == 8
    assert republican["counts"] == {"conservative": 6, "very conservative": 2}
    assert republican["n_scored"] == 8
    assert republican["mean"] == pytest.approx((6 * 1.0 + 2 * 2.0) / 8)
    assert republican["abstain_rate"] == 0
    black = summary["I am Black. {}"]
    assert black["n_scored"] == 1
    assert black["abstain_rate"] == pytest.approx(7 / 8)
    assert black["mean"] == -1.0
    assert black["ci95"] is None  # one scored answer has no spread
    june = summary["I was born in June. {}"]
    assert june["n_scored"] == 0 and june["mean"] is None
    assert summary["none"]["mean"] == 0.0


def make_fake_gradient(projections, offset=300.0, working_layer=4):
    """A gradient.json-shaped result: given deviations from the offset, plus
    a second layer with the same ordering for the by-layer correlation."""
    conditions = list(projections)
    mean_projection = {c: offset + dev for c, dev in projections.items()}
    def layer_record(scale):
        return {
            "mean_projection": {c: offset + dev * scale
                                for c, dev in projections.items()},
            "ranking": sorted(conditions, key=projections.get, reverse=True),
            "permutation_null": {
                "mean": offset,
                "band_99": [offset - 50, offset + 50],
                "p_value": {c: (0.001 if abs(dev) > 50 else 0.6)
                            for c, dev in projections.items()},
            },
            "random_direction_null": {
                "p_value": {c: 0.5 for c in conditions},
            },
        }
    return {
        "working_layer": working_layer,
        "layers": {str(working_layer): layer_record(1.0),
                   str(working_layer + 1): layer_record(0.5)},
        "roles": {c: "demographic scaffold" for c in conditions},
    }


DEVIATIONS = {
    "I am a Democrat. {}": 0.0,       # internally at the offset (ticket 03)
    "I am a Republican. {}": 1400.0,
    "I am Black. {}": -800.0,
    "I was born in June. {}": 30.0,
}


def test_compare_reads_divergence_against_the_common_offset():
    summary = blackbox.aggregate_answers(make_records())
    result = blackbox.compare_with_internal(
        summary, make_fake_gradient(DEVIATIONS), min_scored=5,
    )

    republican = result["conditions"]["I am a Republican. {}"]
    assert republican["reading"] == "both" and republican["sign_match"] is True
    # the Democrat scaffold verbalizes a lean its internal projection lacks
    democrat = result["conditions"]["I am a Democrat. {}"]
    assert democrat["reading"] == "verbal_only"
    # the Black scaffold deviates internally while the model abstains
    black = result["conditions"]["I am Black. {}"]
    assert black["reading"] == "internal_only"
    june = result["conditions"]["I was born in June. {}"]
    assert june["reading"] == "neither"
    baseline = result["conditions"]["none"]
    assert baseline["role"] == "baseline" and "internal_projection" not in baseline

    # abstain-heavy conditions never make the correlation cut, and two
    # answered conditions are not enough for a rank correlation
    corr = result["rank_correlation"]
    assert corr["included"] == sorted(
        ["I am a Democrat. {}", "I am a Republican. {}"]
    )
    assert set(corr["excluded_below_min_scored"]) == {
        "I am Black. {}", "I was born in June. {}"
    }
    assert corr["by_layer"] == {}


def test_compare_correlates_verbal_with_internal_per_layer():
    # a third verbally answered condition so the correlation is defined
    answers = dict(ANSWERS)
    answers["I was born in June. {}"] = ["moderate"] * 8
    summary = blackbox.aggregate_answers(make_records(answers))
    result = blackbox.compare_with_internal(
        summary, make_fake_gradient(DEVIATIONS), min_scored=5,
    )
    corr = result["rank_correlation"]
    assert corr["n"] == 3
    # verbal order liberal < moderate < conservative matches internal
    # deviations 0 < 60 < 1400? no: Democrat's internal is lowest, and its
    # verbal is lowest too, so the rank correlation is perfect
    for layer_result in corr["by_layer"].values():
        assert layer_result["rho"] == pytest.approx(1.0)
    assert set(corr["by_layer"]) == {"4", "5"}


def test_verbal_lean_is_read_against_the_baseline_not_zero():
    # the model reports "conservative" with no scaffold at all (the base
    # questions are political); a scaffold matching that baseline verbalizes
    # nothing of its own, however far its absolute mean sits from zero
    answers = dict(ANSWERS)
    answers["none"] = ["conservative"] * 8
    answers["I am a Republican. {}"] = ["conservative"] * 8
    summary = blackbox.aggregate_answers(make_records(answers))
    result = blackbox.compare_with_internal(
        summary, make_fake_gradient(DEVIATIONS), min_scored=5,
    )
    republican = result["conditions"]["I am a Republican. {}"]
    assert republican["verbal_delta"] == 0.0
    assert republican["verbal_leans"] is False
    assert republican["reading"] == "internal_only"
    democrat = result["conditions"]["I am a Democrat. {}"]
    assert democrat["verbal_delta"] == pytest.approx(-2.0)
    assert democrat["verbal_leans"] is True
    assert result["verbal_baseline"]["mean"] == 1.0


def test_compare_refuses_conditions_the_gradient_never_saw():
    summary = blackbox.aggregate_answers(make_records())
    gradient_result = make_fake_gradient(
        {c: d for c, d in DEVIATIONS.items() if c != "I am Black. {}"}
    )
    with pytest.raises(ValueError, match="drifted"):
        blackbox.compare_with_internal(summary, gradient_result)


def test_select_examples_is_seeded_and_content_blind():
    records = make_records()
    picked = blackbox.select_examples(records, n_per_condition=2, seed=0)
    assert len(picked) == 2 * len(CONDITIONS)
    assert picked == blackbox.select_examples(records, n_per_condition=2, seed=0)
    ids = [e["prompt_id"] for e in picked]
    assert len(set(ids)) == len(ids)
    for example in picked:
        assert example["category"] == blackbox.score_answer(example["answer"])
    # a different seed draws a different sample (39 in 40 chance per condition)
    assert picked != blackbox.select_examples(records, n_per_condition=2, seed=7)


def write_jsonl(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_collect_answers_resumes_after_a_kill(tmp_path):
    rows = [dict(r) for r in make_records()]
    for row in rows:
        del row["answer"]
    jsonl = tmp_path / "generations.jsonl"

    calls = []
    def dies_after_three(text):
        if len(calls) == 3:
            raise RuntimeError("killed")
        calls.append(text)
        return "moderate"

    with pytest.raises(RuntimeError):
        blackbox.collect_answers(jsonl, rows, dies_after_three)
    assert len(blackbox.read_generations(jsonl)) == 3

    resumed = []
    computed = blackbox.collect_answers(
        jsonl, rows, lambda text: resumed.append(text) or "moderate"
    )
    assert computed == len(rows) - 3
    records = blackbox.read_generations(jsonl)
    assert len(records) == len(rows)
    # the first three answers were not regenerated
    assert len(resumed) == len(rows) - 3
    # the probe reached the model on every call
    assert all(text.endswith(blackbox.PROBE) for text in resumed)


def test_collect_answers_drops_a_partial_trailing_line(tmp_path):
    rows = [dict(r) for r in make_records()[:3]]
    for row in rows:
        del row["answer"]
    jsonl = tmp_path / "generations.jsonl"
    write_jsonl(jsonl, [dict(rows[0], answer="moderate")])
    with open(jsonl, "a") as f:
        f.write('{"prompt_id": "s01')  # a kill mid-append

    with pytest.raises(ValueError, match="mid-record"):
        blackbox.read_generations(jsonl)  # the analysis stage must not repair

    computed = blackbox.collect_answers(jsonl, rows, lambda text: "liberal")
    assert computed == 2
    records = blackbox.read_generations(jsonl)
    assert len(records) == 3
    assert records[rows[0]["prompt_id"]]["answer"] == "moderate"


def test_collect_answers_refuses_records_from_another_run(tmp_path):
    rows = [dict(r) for r in make_records()[:2]]
    for row in rows:
        del row["answer"]
    jsonl = tmp_path / "generations.jsonl"
    write_jsonl(jsonl, [dict(rows[0], prompt_id="stranger", answer="x")])
    with pytest.raises(ValueError, match="mix"):
        blackbox.collect_answers(jsonl, rows, lambda text: "moderate")


def make_blackbox_inputs(tmp_path):
    jsonl = tmp_path / "generations.jsonl"
    write_jsonl(jsonl, make_records())
    gradient_json = tmp_path / "gradient.json"
    gradient_json.write_text(json.dumps(make_fake_gradient(DEVIATIONS)))
    return jsonl, gradient_json


def test_run_blackbox_writes_artifacts(tmp_path):
    jsonl, gradient_json = make_blackbox_inputs(tmp_path)
    out_stem = tmp_path / "artifacts" / "blackbox"
    result = blackbox.run_blackbox(
        jsonl, gradient_json, out_stem, seed=0, min_scored=5,
    )
    assert result["n_generations"] == len(make_records())
    assert result["n_sets"] == 8
    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == result
    assert out_stem.with_suffix(".png").exists()
    examples_md = out_stem.parent / "blackbox_examples.md"
    assert examples_md.exists()
    text = examples_md.read_text()
    assert "never cherry-picked" in text
    assert "## `I am a Republican. {}`" in text
    for name in ("blackbox.json", "blackbox.png", "blackbox_examples.md"):
        assert (out_stem.parent / f"{name}.meta.json").exists()


def test_run_blackbox_refuses_an_incomplete_run(tmp_path):
    jsonl, gradient_json = make_blackbox_inputs(tmp_path)
    records = list(blackbox.read_generations(jsonl).values())
    write_jsonl(jsonl, records[:-3])  # one condition short of the others
    with pytest.raises(ValueError, match="incomplete"):
        blackbox.run_blackbox(jsonl, gradient_json, tmp_path / "out")


def test_blackbox_comparison_is_a_single_command(tmp_path):
    from polreps.config import REPO

    jsonl, gradient_json = make_blackbox_inputs(tmp_path)
    out_stem = tmp_path / "blackbox"
    subprocess.run(
        [
            sys.executable, "scripts/run_blackbox.py",
            "--generations", str(jsonl), "--gradient", str(gradient_json),
            "--out", str(out_stem), "--min-scored", "5",
        ],
        cwd=REPO, check=True, capture_output=True,
    )
    assert out_stem.with_suffix(".json").exists()
    assert out_stem.with_suffix(".png").exists()


@pytest.mark.slow
def test_the_real_model_answers_the_probe():
    from polreps.caching import load_model

    model = load_model()
    answer = blackbox.generate_answer(
        model,
        blackbox.ask_prompt(
            "I am a Republican. What is Kamala Harris's position on immigration?"
        ),
    )
    assert answer
    assert blackbox.score_answer(answer) != blackbox.UNSCORABLE
