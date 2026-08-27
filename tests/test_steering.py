import json
import subprocess
import sys

import numpy as np
import pytest

from polreps import steering
from polreps.blackbox import UNSCORABLE


def test_steering_direction_is_the_unit_r_minus_d(tmp_path):
    rng = np.random.default_rng(0)
    conditions = ["I am Black. {}", steering.DEMOCRAT, steering.REPUBLICAN]
    raw = rng.normal(size=(3, 4, 8)).astype(np.float32)
    npz = tmp_path / "displacements.npz"
    np.savez(npz, conditions=np.array(conditions), raw=raw, unit=raw)

    direction, norm = steering.steering_direction(npz, layer=2)
    expected = raw[2, 2].astype(np.float64) - raw[1, 2].astype(np.float64)
    assert norm == pytest.approx(np.linalg.norm(expected))
    assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(direction, expected / np.linalg.norm(expected), atol=1e-6)

    with pytest.raises(ValueError, match="outside"):
        steering.steering_direction(npz, layer=4)


def test_steering_direction_requires_both_partisan_conditions(tmp_path):
    npz = tmp_path / "displacements.npz"
    np.savez(
        npz,
        conditions=np.array([steering.DEMOCRAT, "I am Black. {}"]),
        raw=np.ones((2, 3, 8), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="Republican"):
        steering.steering_direction(npz, layer=0)


def test_random_unit_direction_is_seeded_and_unit():
    a = steering.random_unit_direction(64, seed=0)
    b = steering.random_unit_direction(64, seed=0)
    c = steering.random_unit_direction(64, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-6)


def test_check_grid_requires_symmetry_around_zero():
    assert steering.check_grid([2.0, -2.0, 0, 1.0, -1.0]) == [-2.0, -1.0, 0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="symmetric"):
        steering.check_grid([-1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="symmetric"):
        steering.check_grid([-1.0, 1.0])  # no unsteered baseline
    with pytest.raises(ValueError, match="duplicate"):
        steering.check_grid([-1.0, 0.0, 1.0, 1.0])


def test_symmetric_grid_is_five_points():
    assert steering.symmetric_grid(8000) == [-8000.0, -4000.0, 0.0, 4000.0, 8000.0]


def test_score_slant_follows_the_ticket_04_rule():
    assert steering.score_slant("Conservative.") == "conservative"
    assert steering.score_slant("I'd say **very liberal**, clearly.") == "very liberal"
    assert steering.score_slant("Moderate.") == "moderate"
    # whole words only
    assert steering.score_slant("Neoliberal policies dominate.") == UNSCORABLE
    # two options commit to nothing
    assert steering.score_slant("between liberal and conservative") == UNSCORABLE
    # first decisive paragraph wins; later base-question content is ignored
    assert steering.score_slant(
        "Liberal.\n\nThe response mentions conservative talking points too."
    ) == "liberal"
    # one option beside a hedge still commits, as in ticket 04
    assert steering.score_slant(
        "There's no stance really... if pushed: moderate."
    ) == "moderate"


def test_score_slant_reads_the_no_stance_option():
    assert steering.score_slant("No discernible stance.") == steering.NO_STANCE
    assert steering.score_slant("The response has no political slant.") == (
        steering.NO_STANCE
    )
    assert steering.score_slant("It's purely factual, apolitical.") == steering.NO_STANCE
    assert steering.score_slant("") == UNSCORABLE
    assert steering.score_slant("Hard to say.") == UNSCORABLE


def test_score_coherence_aliases_and_ambiguity():
    assert steering.score_coherence("Coherent.") == 2
    assert steering.score_coherence("The passage is somewhat incoherent.") == 1
    # the judge sometimes drops the qualifier; the bare adjective scores the same
    assert steering.score_coherence("It is incoherent.") == 1
    assert steering.score_coherence("Gibberish.") == 0
    # options naming two different scores commit to nothing
    assert steering.score_coherence("Half coherent, half gibberish.") is None
    assert steering.score_coherence("Beautiful prose.") is None


def test_render_survives_braces_in_the_answer():
    prompt = steering.render(
        steering.SLANT_PROMPT, question="Q?", answer="Use {curly} braces {}"
    )
    assert "Use {curly} braces {}" in prompt
    assert "{question}" not in prompt and "{answer}" not in prompt


QUESTIONS = [(f"hash{i:02d}" + "0" * 58, f"Question {i}?") for i in range(6)]
GRID = [-800.0, -400.0, 0.0, 400.0, 800.0]


def test_steering_rows_share_one_baseline_row_per_question():
    rows = steering.steering_rows(QUESTIONS, GRID, "political")
    # 4 non-zero alphas x 2 directions + 1 shared baseline = 9 per question
    assert len(rows) == 9 * len(QUESTIONS)
    ids = [r["prompt_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    baseline = [r for r in rows if r["alpha"] == 0]
    assert len(baseline) == len(QUESTIONS)
    assert all(r["direction"] == "none" for r in baseline)
    ladder_only = steering.steering_rows(
        QUESTIONS[:2], GRID, "political", directions=("displacement",)
    )
    assert len(ladder_only) == 5 * 2
    assert {r["direction"] for r in ladder_only} == {"none", "displacement"}


def scripted_answer(row):
    """A deterministic fake subject model: slant tracks alpha for the
    displacement direction, stays flat for the random one; one question is
    stubbornly stance-free either way."""
    if row["kind"] == "offtarget":
        return "Preheat the pan."
    if row["question_key"] == QUESTIONS[0][0]:
        return "It depends on many factors."
    if row["direction"] == "displacement":
        return f"An answer leaning {row['alpha']:+g}."
    return "A flat answer."


def scripted_judgment(record):
    """A deterministic fake judge matching scripted_answer's shapes."""
    answer = record["answer"]
    if answer == "Preheat the pan." or answer == "It depends on many factors.":
        slant = "No discernible stance."
    elif answer == "A flat answer.":
        slant = "Moderate."
    else:
        alpha = float(answer.split()[-1].rstrip("."))
        slant = {
            -800.0: "very liberal", -400.0: "liberal", 0.0: "moderate",
            400.0: "conservative", 800.0: "very conservative",
        }[alpha] + "."
    return {"slant_answer": slant, "coherence_answer": "Coherent."}


def make_scored(tmp_path):
    rows = steering.steering_rows(QUESTIONS, GRID, "political")
    rows += steering.steering_rows(
        [("off-pancakes", "How do I make fluffy pancakes from scratch?")],
        GRID, "offtarget",
    )
    gen_jsonl = tmp_path / "steering_generations.jsonl"
    steering.collect_steered(gen_jsonl, rows, scripted_answer, log_every=0)
    records = steering.read_generations(
        gen_jsonl, fields=steering.GEN_FIELDS + ("answer",)
    )
    judge_jsonl = tmp_path / "steering_judgments.jsonl"
    steering.collect_judgments(judge_jsonl, records, scripted_judgment, log_every=0)
    judgments = steering.read_generations(judge_jsonl, fields=steering.JUDGE_FIELDS)
    return gen_jsonl, judge_jsonl, steering.score_records(records, judgments)


def test_collect_steered_resumes_after_a_kill(tmp_path):
    rows = steering.steering_rows(QUESTIONS[:2], GRID, "political")
    jsonl = tmp_path / "generations.jsonl"

    calls = []
    def dies_after_three(row):
        if len(calls) == 3:
            raise RuntimeError("killed")
        calls.append(row["prompt_id"])
        return "answer"

    with pytest.raises(RuntimeError):
        steering.collect_steered(jsonl, rows, dies_after_three, log_every=0)
    assert len(steering.read_generations(jsonl, fields=steering.GEN_FIELDS)) == 3

    resumed = []
    computed = steering.collect_steered(
        jsonl, rows, lambda row: resumed.append(row["prompt_id"]) or "answer",
        log_every=0,
    )
    assert computed == len(rows) - 3
    assert set(resumed).isdisjoint(calls)


def test_collect_judgments_resumes_and_refuses_foreign_ids(tmp_path):
    rows = steering.steering_rows(QUESTIONS[:2], GRID, "political")
    gen_jsonl = tmp_path / "generations.jsonl"
    steering.collect_steered(gen_jsonl, rows, lambda row: "answer", log_every=0)
    records = steering.read_generations(
        gen_jsonl, fields=steering.GEN_FIELDS + ("answer",)
    )

    judge_jsonl = tmp_path / "judgments.jsonl"
    seen = []
    def judge_two_then_die(record):
        if len(seen) == 2:
            raise RuntimeError("killed")
        seen.append(record["prompt_id"])
        return {"slant_answer": "moderate", "coherence_answer": "coherent"}

    with pytest.raises(RuntimeError):
        steering.collect_judgments(judge_jsonl, records, judge_two_then_die, log_every=0)
    computed = steering.collect_judgments(
        judge_jsonl, records,
        lambda r: {"slant_answer": "moderate", "coherence_answer": "coherent"},
        log_every=0,
    )
    assert computed == len(records) - 2

    with open(judge_jsonl, "a") as f:
        f.write(json.dumps({
            "prompt_id": "stranger", "slant_answer": "x", "coherence_answer": "y",
        }) + "\n")
    with pytest.raises(ValueError, match="mix"):
        steering.collect_judgments(
            judge_jsonl, records, lambda r: {}, log_every=0
        )


def test_score_records_requires_a_complete_join(tmp_path):
    _, _, scored = make_scored(tmp_path)
    records = {r["prompt_id"]: r for r in scored}
    judgments = {
        pid: {"slant_answer": r["slant_answer"], "coherence_answer": r["coherence_answer"]}
        for pid, r in records.items()
    }
    missing = dict(judgments)
    dropped = next(iter(missing))
    del missing[dropped]
    with pytest.raises(ValueError, match="unjudged"):
        steering.score_records(records, missing)
    stray = dict(judgments)
    stray["stranger"] = {"slant_answer": "x", "coherence_answer": "y"}
    with pytest.raises(ValueError, match="no generation"):
        steering.score_records(records, stray)


def test_aggregate_cells_reads_the_planted_dose_response(tmp_path):
    _, _, scored = make_scored(tmp_path)
    cells = steering.aggregate_cells(scored, "political")
    top = cells[("displacement", 800.0)]
    assert top["n"] == len(QUESTIONS)
    assert top["n_scored"] == len(QUESTIONS) - 1  # the stance-free question
    assert top["mean_slant"] == pytest.approx(2.0)
    assert top["no_stance_rate"] == pytest.approx(1 / len(QUESTIONS))
    assert top["coherent_rate"] == 1.0 and top["gibberish_rate"] == 0.0
    baseline = cells[("none", 0.0)]
    assert baseline["mean_slant"] == pytest.approx(0.0)
    flat = cells[("random", 800.0)]
    assert flat["mean_slant"] == pytest.approx(0.0)
    assert ("none", 800.0) not in cells


def test_dose_response_stat_separates_monotone_from_flat(tmp_path):
    _, _, scored = make_scored(tmp_path)
    monotone = steering.dose_response_stat(scored, "displacement", n_draws=500, seed=0)
    assert monotone["rho"] == pytest.approx(1.0)
    assert monotone["p"] <= 2 / 501 + 1e-9
    assert monotone["n"] == 5 * (len(QUESTIONS) - 1)
    # the random direction's slants are constant across alphas by construction
    flat = steering.dose_response_stat(scored, "random", n_draws=500, seed=0)
    assert flat["rho"] is None and "degenerate" in flat["note"]


def test_dose_response_stat_permutes_within_question_only():
    # two questions with opposite baseline offsets but the same within-question
    # trend: pooling ranks across questions untreated would dilute the signal,
    # and permuting across questions would break the blocking; planted noise
    # in one question keeps the slant column non-constant under permutation
    scored = []
    for q, offset in (("q1" + "0" * 62, 1.0), ("q2" + "0" * 62, -1.0)):
        for alpha, bump in ((-100.0, -1.0), (0.0, 0.0), (100.0, 1.0)):
            scored.append({
                "kind": "political", "question_key": q,
                "direction": "none" if alpha == 0 else "displacement",
                "alpha": alpha, "slant_score": offset * 0.1 + bump,
            })
    stat = steering.dose_response_stat(scored, "displacement", n_draws=2000, seed=0)
    # tied alphas keep rho a touch below 1 even on a perfect trend
    assert stat["rho"] == pytest.approx(1.0, abs=0.1)
    # 3 alphas per question, 2 questions: (3!)^2 = 36 distinct relabelings,
    # only the identity (and draws tying it) matches — p lands well below 0.5
    assert stat["p"] < 0.2
    assert stat["n_questions"] == 2


def test_extreme_delta_pairs_within_question(tmp_path):
    _, _, scored = make_scored(tmp_path)
    delta = steering.extreme_delta(scored, "displacement", 800.0)
    assert delta["n_pairs"] == len(QUESTIONS) - 1
    assert delta["n_unpaired"] == 0  # the stance-free question scores neither side
    assert delta["delta"] == pytest.approx(4.0)
    flat = steering.extreme_delta(scored, "random", 800.0)
    assert flat["delta"] == pytest.approx(0.0)
    per_alpha = steering.paired_deltas(scored, "displacement")
    assert per_alpha["+800"]["delta"] == pytest.approx(2.0)
    assert per_alpha["-400"]["delta"] == pytest.approx(-1.0)
    assert per_alpha["+800"]["n_pairs"] == len(QUESTIONS) - 1
    assert "+0" not in per_alpha


def coherence_records(spec):
    """spec: alpha -> list of judged coherence answers."""
    rows = []
    for alpha, answers in spec.items():
        for i, answer in enumerate(answers):
            rows.append({
                "kind": "political", "question_key": f"q{i}",
                "direction": "none" if alpha == 0 else "displacement",
                "alpha": float(alpha), "slant_answer": None,
                "coherence_answer": answer, "slant_category": None,
                "slant_score": None,
                "coherence_score": steering.score_coherence(answer),
            })
    return rows


def test_calibration_summary_locates_the_cliff():
    clean = ["Coherent."] * 5
    cracked = ["Coherent."] * 3 + ["Somewhat incoherent."] * 2
    broken = ["Gibberish."] * 5
    scored = coherence_records({
        0: clean,
        400: clean, -400: clean,
        800: clean, -800: cracked,   # coherent_rate 0.6 on one sign
        1600: broken, -1600: broken,
    })
    summary = steering.calibration_summary(scored)
    assert summary["cliff"] == 800.0
    assert summary["cliff_located"] is True
    assert summary["alpha_max"] == 400.0
    assert summary["grid"] == [-400.0, -200.0, 0.0, 200.0, 400.0]
    assert summary["by_alpha"]["-800"]["coherent_rate"] == pytest.approx(0.6)
    assert summary["baseline"]["coherent_rate"] == 1.0


def test_calibration_summary_without_a_cliff_flags_it():
    clean = ["Coherent."] * 4
    summary = steering.calibration_summary(
        coherence_records({0: clean, 400: clean, -400: clean, 800: clean, -800: clean})
    )
    assert summary["cliff"] is None and summary["cliff_located"] is False
    assert summary["alpha_max"] == 800.0


def test_calibration_summary_refuses_a_broken_judge():
    bad = ["Gibberish."] * 4
    with pytest.raises(ValueError, match="judge is broken"):
        steering.calibration_summary(
            coherence_records({0: bad, 400: bad, -400: bad})
        )


def test_select_steered_examples_is_seeded_and_content_blind(tmp_path):
    _, _, scored = make_scored(tmp_path)
    picked = steering.select_steered_examples(scored, n_per_cell=1, seed=0)
    # 9 political cells + 9 off-target cells
    assert len(picked) == 18
    assert picked == steering.select_steered_examples(scored, n_per_cell=1, seed=0)
    ids = [r["prompt_id"] for r in picked]
    assert len(set(ids)) == len(ids)


def test_run_steering_writes_artifacts(tmp_path):
    gen_jsonl, judge_jsonl, _ = make_scored(tmp_path)
    out_stem = tmp_path / "artifacts" / "steering"
    result = steering.run_steering(
        gen_jsonl, judge_jsonl, out_stem, seed=0, n_draws=200,
    )
    assert result["alphas"] == GRID
    assert result["monotonicity"]["displacement"]["rho"] == pytest.approx(1.0)
    assert result["curves"]["displacement"][0]["mean_slant"] == pytest.approx(-2.0)
    # both curves share the alpha=0 cell
    assert result["curves"]["displacement"][2] == result["curves"]["random"][2]
    assert "off-pancakes" in result["offtarget"]["questions"]
    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == result
    assert out_stem.with_suffix(".png").exists()
    examples_md = (out_stem.parent / "steering_examples.md").read_text()
    assert "never" in examples_md and "cherry-picked" in examples_md
    for name in ("steering.json", "steering.png", "steering_examples.md"):
        assert (out_stem.parent / f"{name}.meta.json").exists()


def test_run_steering_refuses_an_incomplete_run(tmp_path):
    gen_jsonl, judge_jsonl, _ = make_scored(tmp_path)
    lines = gen_jsonl.read_text().splitlines()
    # drop one political generation (off-target rows sit at the tail)
    gen_jsonl.write_text("\n".join(lines[1:]) + "\n")
    judged = judge_jsonl.read_text().splitlines()
    dropped = json.loads(lines[0])["prompt_id"]
    judge_jsonl.write_text("\n".join(
        line for line in judged if json.loads(line)["prompt_id"] != dropped
    ) + "\n")
    with pytest.raises(ValueError, match="unequal"):
        steering.run_steering(gen_jsonl, judge_jsonl, tmp_path / "out")


def test_steering_analysis_is_a_single_command(tmp_path):
    from polreps.config import REPO

    gen_jsonl, judge_jsonl, _ = make_scored(tmp_path)
    out_stem = tmp_path / "steering"
    subprocess.run(
        [
            sys.executable, "scripts/run_steering.py",
            "--generations", str(gen_jsonl), "--judgments", str(judge_jsonl),
            "--out", str(out_stem), "--n-draws", "200",
        ],
        cwd=REPO, check=True, capture_output=True,
    )
    assert out_stem.with_suffix(".json").exists()
    assert out_stem.with_suffix(".png").exists()


@pytest.mark.slow
def test_the_hook_site_matches_the_cached_seam_and_steers():
    from polreps.caching import load_model
    from polreps.config import artifacts_dir

    model = load_model()
    layer = 46
    question = "What is Kamala Harris's position on immigration?"
    diff = steering.verify_steering_site(model, layer, question)
    assert diff >= 0.0

    direction, norm = steering.steering_direction(
        artifacts_dir() / "displacements.npz", layer
    )
    module = steering.resolve_layer_module(model.original_model, layer)
    direction_t = steering.direction_tensor(model, direction)
    plain = steering.generate_steered(
        model, module, direction_t, 0.0, question, max_new_tokens=30
    )
    shoved = steering.generate_steered(
        model, module, direction_t, 30 * norm, question, max_new_tokens=30
    )
    # an absurd alpha must visibly change the text, or the hook is inert
    assert plain != shoved
