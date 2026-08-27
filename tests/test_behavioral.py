import json
import math

import numpy as np
import pytest

from polreps import behavioral, blackbox

# planted embedding angles per condition; the none answer sits at angle 0, so
# the cosine distance from it is 1 - cos(angle), monotone over this range
ANGLES = {
    "I am a Republican. {}": 1.2,
    "I am Black. {}": 0.9,
    "I am a Democrat. {}": 0.6,
    "I was born in June. {}": 0.3,
}

# internal sides planted so the primary link is perfectly monotone with the
# angles and the secondary is perfectly anti-monotone — distinguishable
NORMS = {c: 1000.0 * angle for c, angle in ANGLES.items()}
DEVIATIONS = {c: 500.0 - 300.0 * angle for c, angle in ANGLES.items()}


def embed_stub(texts):
    angles = [float(t.split()[0]) for t in texts]
    return np.array([[math.cos(a), math.sin(a)] for a in angles])


def make_records(n_sets=4):
    records = []
    for s in range(n_sets):
        set_hash = f"hash{s:02d}" + "0" * 58
        rows = [("none", "0.0")] + [
            # padding words vary the word count by condition so the length
            # diagnostic has spread; the stub embeds the first token only
            (c, f"{angle} " + "pad " * int(angle * 5)) for c, angle in ANGLES.items()
        ]
        for condition, answer in rows:
            records.append({
                "prompt_id": f"s{s:02d}-{condition[:8]}",
                "condition": condition,
                "pre_prompt_q_hash": set_hash,
                "question": f"Question {s}?" if condition == "none"
                else condition.replace("{}", f"Question {s}?"),
                "answer": answer.strip(),
            })
    return records


def make_fake_gradient(norms=NORMS, deviations=DEVIATIONS, offset=300.0,
                       working_layer=4):
    conditions = sorted(norms)
    def layer_record(scale):
        return {
            "mean_projection": {c: offset + deviations[c] * scale for c in conditions},
            "displacement_norm": {c: norms[c] * scale for c in conditions},
            "permutation_null": {"mean": offset},
        }
    return {
        "working_layer": working_layer,
        "conditions": conditions,
        "roles": {c: "demographic scaffold" for c in conditions},
        "layers": {str(working_layer): layer_record(1.0),
                   str(working_layer + 1): layer_record(0.5)},
    }


def test_echo_terms_strip_the_first_person_plumbing():
    assert behavioral.echo_terms("I am left-handed. {}") == ["left-handed"]
    assert behavioral.echo_terms("I was born in June. {}") == ["born", "june"]
    assert behavioral.echo_terms("I'm a member of the Republican party. {}") == (
        ["member", "party", "republican"]
    )
    assert behavioral.echoes_scaffold("I am a Democrat. {}", "As a Democrat, you may…")
    # embedded words are not echoes
    assert not behavioral.echoes_scaffold(
        "I am a Democrat. {}", "Democratic norms suggest…"
    )
    assert not behavioral.echoes_scaffold("I am Black. {}", "a blackboard")


def test_paired_displacement_is_cosine_distance_from_the_none_answer():
    out = behavioral.paired_output_displacement(make_records(), embed_stub)
    assert out["n_sets"] == 4
    assert out["conditions"] == sorted(ANGLES)
    for condition, angle in ANGLES.items():
        assert out["mean_distance"][condition] == pytest.approx(1 - math.cos(angle))
        assert out["ci95"][condition] == pytest.approx(0.0)  # identical across sets
        assert out["wordcount_delta"][condition] == pytest.approx(int(angle * 5))
        assert out["echo_rate"][condition] == 0.0  # numeric answers echo nothing
    # every none answer is identical, so the between-question reference is 0
    assert out["between_question_reference"]["mean"] == pytest.approx(0.0)
    assert out["between_question_reference"]["n_pairs"] == 6


def test_paired_displacement_requires_complete_sets():
    records = make_records()
    with pytest.raises(ValueError, match="missing"):
        behavioral.paired_output_displacement(records[:-1], embed_stub)
    with pytest.raises(ValueError, match="duplicate"):
        behavioral.paired_output_displacement(records + [records[0]], embed_stub)
    empty = dict(records[0], answer="  ")
    with pytest.raises(ValueError, match="empty"):
        behavioral.paired_output_displacement(records[1:] + [empty], embed_stub)


def test_link_recovers_a_planted_monotone_link():
    out = behavioral.paired_output_displacement(make_records(), embed_stub)
    result = behavioral.link_with_internal(out, make_fake_gradient(), seed=0)

    primary = result["rank_correlation"]["primary"]
    assert primary["internal"] == "displacement_norm"
    assert primary["layer"] == 4
    assert primary["rho"] == pytest.approx(1.0)
    assert primary["method"] == "exact"
    secondary = result["rank_correlation"]["secondary"]
    assert secondary["rho"] == pytest.approx(-1.0)

    # both layers carry the same ranks by construction
    other = result["rank_correlation"]["by_layer"]["5"]
    assert other["displacement_norm"]["rho"] == pytest.approx(1.0)

    diagnostics = result["rank_correlation"]["diagnostics"]
    assert diagnostics["wordcount_delta"]["rho"] == pytest.approx(1.0)
    assert diagnostics["echo_rate"]["rho"] is None  # constant at zero

    republican = result["conditions"]["I am a Republican. {}"]
    assert republican["internal_norm"] == NORMS["I am a Republican. {}"]
    assert republican["internal_deviation"] == pytest.approx(
        DEVIATIONS["I am a Republican. {}"]
    )


def test_link_refuses_conditions_the_gradient_never_saw():
    out = behavioral.paired_output_displacement(make_records(), embed_stub)
    gradient = make_fake_gradient()
    gradient["conditions"] = [c for c in gradient["conditions"] if "June" not in c]
    with pytest.raises(ValueError, match="vocabulary drifted"):
        behavioral.link_with_internal(out, gradient)


def test_collect_answers_prompt_fn_leaves_the_question_bare(tmp_path):
    rows = [
        {k: r[k] for k in blackbox.REQUIRED_FIELDS}
        for r in make_records(n_sets=2)
    ]
    asked = []
    blackbox.collect_answers(
        tmp_path / "generations.jsonl", rows,
        lambda text: asked.append(text) or "0.5",
        prompt_fn=lambda question: question,
    )
    assert asked == [r["question"] for r in rows]
    assert not any(blackbox.PROBE in text for text in asked)


def test_run_behavioral_link_writes_artifacts(tmp_path):
    jsonl = tmp_path / "behavioral_generations.jsonl"
    with open(jsonl, "w") as f:
        for record in make_records():
            f.write(json.dumps(record) + "\n")
    gradient_json = tmp_path / "gradient.json"
    gradient_json.write_text(json.dumps(make_fake_gradient()))

    out_stem = tmp_path / "artifacts" / "behavioral_link"
    result = behavioral.run_behavioral_link(
        jsonl, gradient_json, out_stem,
        embed_fn=embed_stub, embedder_config={"name": "stub"}, seed=0,
    )
    assert result["n_generations"] == len(make_records())
    assert result["n_conditions"] == len(ANGLES)
    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == result
    assert out_stem.with_suffix(".png").exists()
    examples_md = out_stem.parent / "behavioral_link_examples.md"
    text = examples_md.read_text()
    assert "never cherry-picked" in text
    assert "## `I am a Republican. {}`" in text
    assert ", distance " in text
    for name in ("behavioral_link.json", "behavioral_link.png",
                 "behavioral_link_examples.md"):
        assert (out_stem.parent / f"{name}.meta.json").exists()


@pytest.mark.slow
def test_the_pinned_embedder_loads_and_normalizes():
    embed = behavioral.load_embedder()
    vectors = embed(["a short answer", "a rather different short answer"])
    assert vectors.shape == (2, 384)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(2), abs=1e-5)
    # batch composition changes padding and with it the numerics at ~1e-7;
    # the measured distances are of order 1e-1
    assert np.allclose(embed(["a short answer"])[0], vectors[0], atol=1e-5)
