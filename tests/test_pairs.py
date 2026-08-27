import pytest

from polreps import pairs

# a miniature prompt-variation suite in the release's format: "{}" is the
# identity variation the bank calls "none"
VARIANTS = ["{}", "I am a Democrat. {}", "{} Please cite your sources."]
CONDITIONS = {"none", "I am a Democrat. {}", "{} Please cite your sources."}


def bank_row(base_q, prompt_type, pre_prompt_q=None, **overrides):
    pre_prompt_q = pre_prompt_q or base_q
    question = (
        pre_prompt_q if prompt_type == "none" else prompt_type.replace("{}", pre_prompt_q)
    )
    row = {
        "question": question,
        "type": "endo",
        "category": "issues",
        "subcategory": None,
        "base_q": base_q,
        "prompt_type": prompt_type,
    }
    row.update(overrides)
    return row


def template_rows(base_q, pre_prompt_qs, conditions=CONDITIONS):
    return [
        bank_row(base_q, cond, pre_prompt_q=q) for q in pre_prompt_qs for cond in conditions
    ]


def test_sets_key_on_question_content_not_template():
    # one template, two placeholder instantiations: must yield two sets,
    # never one set mixing the two questions
    rows = template_rows(
        "What is {c}'s stance?",
        ["What is Harris's stance?", "What is Trump's stance?"],
    )
    sets, report = pairs.build_matched_sets(rows, CONDITIONS)

    assert len(sets) == 2
    assert {s["pre_prompt_q"] for s in sets} == {
        "What is Harris's stance?",
        "What is Trump's stance?",
    }
    for s in sets:
        assert set(s["questions"]) == CONDITIONS
        assert s["questions"]["none"] == s["pre_prompt_q"]
        assert s["questions"]["I am a Democrat. {}"] == f"I am a Democrat. {s['pre_prompt_q']}"
        assert s["base_q_template"] == "What is {c}'s stance?"


def test_set_without_none_baseline_is_excluded_and_counted():
    rows = template_rows("kept?", ["kept?"])
    rows += [
        bank_row("dropped?", cond, pre_prompt_q="dropped?")
        for cond in CONDITIONS - {"none"}
    ]
    sets, report = pairs.build_matched_sets(rows, CONDITIONS)

    assert [s["pre_prompt_q"] for s in sets] == ["kept?"]
    assert report["sets_missing_none"] == 1


def test_bank_with_no_none_rows_at_all_is_a_loud_failure():
    # caught by the vocabulary assertion: "none" is an expected condition
    rows = [bank_row("q?", "I am a Democrat. {}")]
    with pytest.raises(ValueError, match="none"):
        pairs.build_matched_sets(rows, CONDITIONS)


def test_baseline_type_rows_are_filtered_and_counted():
    rows = template_rows("q?", ["q?"])
    rows.append(
        bank_row("2+2?", "none", type="baseline", category="GSM8K_multiple_choice")
    )
    sets, report = pairs.build_matched_sets(rows, CONDITIONS)

    assert len(sets) == 1
    assert report["rows_dropped"] == {("baseline", "GSM8K_multiple_choice"): 1}


def test_unexpected_condition_fails_with_observed_vocabulary():
    rows = template_rows("q?", ["q?"])
    rows.append(bank_row("q?", "I am a Whig. {}"))
    with pytest.raises(ValueError) as err:
        pairs.build_matched_sets(rows, CONDITIONS)
    assert "I am a Whig. {}" in str(err.value)


def test_expected_condition_absent_from_bank_fails_with_observed_vocabulary():
    # upstream dropping a variation entirely is drift too, not just adding one
    rows = template_rows("q?", ["q?"], conditions=CONDITIONS - {"I am a Democrat. {}"})
    with pytest.raises(ValueError) as err:
        pairs.build_matched_sets(rows, CONDITIONS)
    assert "I am a Democrat. {}" in str(err.value)


def test_incomplete_sets_are_counted():
    # one set has every condition, the other is missing one row (but the
    # vocabulary as a whole is intact, so this is not drift — just a gap)
    rows = template_rows("full?", ["full?"])
    rows += [
        bank_row("partial?", cond)
        for cond in CONDITIONS - {"{} Please cite your sources."}
    ]
    sets, report = pairs.build_matched_sets(rows, CONDITIONS)

    assert len(sets) == 2
    assert report["incomplete_sets"] == 1


def test_missing_field_fails_loudly():
    rows = template_rows("q?", ["q?"])
    del rows[0]["prompt_type"]
    with pytest.raises(ValueError, match="prompt_type"):
        pairs.build_matched_sets(rows, CONDITIONS)


def test_question_not_matching_its_variation_fails():
    # variation says the question must start with the scaffold; upstream drift
    # in either field must not silently produce a wrong pre-prompt question
    rows = template_rows("q?", ["q?"])
    rows.append(
        {**bank_row("other?", "I am a Democrat. {}"), "question": "I am Green. other?"}
    )
    with pytest.raises(ValueError, match="variation"):
        pairs.build_matched_sets(rows, CONDITIONS)


def test_duplicate_condition_within_a_set_fails():
    rows = template_rows("q?", ["q?"])
    rows.append(bank_row("q?", "none"))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        pairs.build_matched_sets(rows, CONDITIONS)


def test_expected_conditions_renames_identity_variant_and_checks_count():
    # the release ships 22 variants where "{}" is what rows call "none"
    variants = ["{}"] + [f"variant {i}. {{}}" for i in range(21)]
    conditions = pairs.expected_conditions(variants)
    assert "none" in conditions
    assert "{}" not in conditions
    assert len(conditions) == 22

    with pytest.raises(ValueError, match="22"):
        pairs.expected_conditions(variants[:5])


def many_sets(n):
    return template_rows("q {i}?", [f"question {i}?" for i in range(n)])


def test_subsample_is_deterministic_and_keeps_whole_sets():
    sets, _ = pairs.build_matched_sets(many_sets(20), CONDITIONS)

    once = pairs.subsample_sets(sets, max_sets=8, seed=7)
    again = pairs.subsample_sets(list(reversed(sets)), max_sets=8, seed=7)
    other_seed = pairs.subsample_sets(sets, max_sets=8, seed=8)

    assert [s["pre_prompt_q_hash"] for s in once] == [s["pre_prompt_q_hash"] for s in again]
    assert [s["pre_prompt_q_hash"] for s in once] != [s["pre_prompt_q_hash"] for s in other_seed]
    assert all(set(s["questions"]) == CONDITIONS for s in once)


def test_subsample_below_target_returns_everything():
    sets, _ = pairs.build_matched_sets(many_sets(5), CONDITIONS)
    assert len(pairs.subsample_sets(sets, max_sets=8, seed=0)) == 5


def test_prompt_table_rows_have_unique_stable_ids():
    sets, _ = pairs.build_matched_sets(many_sets(4), CONDITIONS)
    rows = pairs.prompt_table_rows(sets)

    assert len(rows) == 4 * len(CONDITIONS)
    ids = [r["prompt_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    # rebuilding from the same bank must produce the same ids in the same order,
    # since the activation cache is keyed by them
    rebuilt, _ = pairs.build_matched_sets(many_sets(4), CONDITIONS)
    assert [r["prompt_id"] for r in pairs.prompt_table_rows(rebuilt)] == ids
    # every row carries what downstream stages join on
    assert all(r["condition"] in CONDITIONS for r in rows)
    assert all(r["question"] and r["pre_prompt_q_hash"] and r["base_q_template_hash"] for r in rows)
