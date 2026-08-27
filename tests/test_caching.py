"""The caching stage's resumable-writer mechanics, tested without the model.

compute_fn is injected, so interruption/resume/refusal behavior runs in
milliseconds here; the marked-slow smoke test at the bottom is the only place
the real Gemma-2-9B-IT forward pass is exercised.
"""

import numpy as np
import pytest

from polreps import actcache, caching


def planted_row(row, n_layers=3, d_model=8):
    # deterministic per row, so interrupted and uninterrupted runs are comparable
    return np.random.default_rng(row).normal(size=(n_layers, d_model)).astype(np.float32)


def make_compute(ids, calls=None):
    row_of = {pid: i for i, pid in enumerate(ids)}

    def compute(question):
        if calls is not None:
            calls.append(question)
        return planted_row(row_of[question])

    return compute


IDS = ["p0", "p1", "p2", "p3", "p4"]
# questions double as lookup keys for the fake compute_fn
QUESTIONS = list(IDS)


def dying_compute(kill_at):
    """compute_fn that simulates a hard kill when the run reaches kill_at."""

    def compute(question):
        if question == kill_at:
            raise KeyboardInterrupt
        return planted_row(IDS.index(question))

    return compute


def test_full_run_produces_loadable_cache(tmp_path):
    n = caching.cache_prompts(
        tmp_path / "cache", IDS, QUESTIONS, make_compute(IDS), n_layers=3, d_model=8
    )
    assert n == len(IDS)

    acts, prompt_ids = actcache.load_cache(tmp_path / "cache", expect_prompt_ids=IDS)
    assert prompt_ids == IDS
    assert acts.shape == (3, len(IDS), 8)
    for i in range(len(IDS)):
        np.testing.assert_array_equal(acts[:, i], planted_row(i))


def test_interrupted_run_resumes_without_recomputing(tmp_path):
    uninterrupted = tmp_path / "a"
    caching.cache_prompts(uninterrupted, IDS, QUESTIONS, make_compute(IDS), 3, 8)

    # kill the run after two prompts
    interrupted = tmp_path / "b"
    with pytest.raises(KeyboardInterrupt):
        caching.cache_prompts(interrupted, IDS, QUESTIONS, dying_compute("p2"), 3, 8)

    # the second invocation computes only the remaining three
    calls = []
    n = caching.cache_prompts(
        interrupted, IDS, QUESTIONS, make_compute(IDS, calls), 3, 8
    )
    assert n == 3
    assert calls == ["p2", "p3", "p4"]

    resumed, _ = actcache.load_cache(interrupted, expect_prompt_ids=IDS)
    baseline, _ = actcache.load_cache(uninterrupted, expect_prompt_ids=IDS)
    np.testing.assert_array_equal(resumed, baseline)


def test_finished_cache_makes_rerun_a_no_op(tmp_path):
    caching.cache_prompts(tmp_path / "cache", IDS, QUESTIONS, make_compute(IDS), 3, 8)

    calls = []
    n = caching.cache_prompts(
        tmp_path / "cache", IDS, QUESTIONS, make_compute(IDS, calls), 3, 8
    )
    assert n == 0
    assert calls == []


def test_resume_refuses_a_different_prompt_table(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        caching.cache_prompts(tmp_path / "cache", IDS, QUESTIONS, dying_compute("p4"), 3, 8)

    reordered = ["p1", "p0", "p2", "p3", "p4"]
    with pytest.raises(ValueError, match="refus"):
        caching.cache_prompts(
            tmp_path / "cache", reordered, reordered, make_compute(reordered), 3, 8
        )


def test_finished_cache_refuses_a_different_prompt_table(tmp_path):
    caching.cache_prompts(tmp_path / "cache", IDS, QUESTIONS, make_compute(IDS), 3, 8)

    other = ["q0", "q1"]
    with pytest.raises(ValueError, match="refus"):
        caching.cache_prompts(tmp_path / "cache", other, other, make_compute(other), 3, 8)


def test_partially_appended_done_line_is_recomputed(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        caching.cache_prompts(tmp_path / "cache", IDS, QUESTIONS, dying_compute("p4"), 3, 8)

    # cut the last done-record mid-write, as a hard kill would
    done = tmp_path / "cache" / "done.txt"
    done.write_text(done.read_text()[:-2])

    calls = []
    caching.cache_prompts(tmp_path / "cache", IDS, QUESTIONS, make_compute(IDS, calls), 3, 8)
    assert calls == ["p3", "p4"]

    acts, _ = actcache.load_cache(tmp_path / "cache", expect_prompt_ids=IDS)
    for i in range(len(IDS)):
        np.testing.assert_array_equal(acts[:, i], planted_row(i))


def test_non_finite_activations_are_refused(tmp_path):
    def broken(question):
        acts = planted_row(0)
        acts[1, 3] = np.nan
        return acts

    with pytest.raises(ValueError, match="finite"):
        caching.cache_prompts(tmp_path / "cache", IDS, QUESTIONS, broken, 3, 8)


def test_duplicate_prompt_ids_are_refused(tmp_path):
    ids = ["p0", "p0", "p1"]
    with pytest.raises(ValueError, match="duplicate"):
        caching.cache_prompts(tmp_path / "cache", ids, ids, make_compute(IDS), 3, 8)


def test_read_prompt_table_returns_ids_and_questions(tmp_path):
    table = tmp_path / "table.csv"
    table.write_text(
        "prompt_id,condition,question\n"
        "a,none,What is X?\n"
        'b,democrat,"I am a Democrat. What is X?"\n'
    )
    ids, questions = caching.read_prompt_table(table)
    assert ids == ["a", "b"]
    assert questions == ["What is X?", "I am a Democrat. What is X?"]

    ids, questions = caching.read_prompt_table(table, limit=1)
    assert ids == ["a"]


# everything below loads the real model and only runs under `pytest -m slow`

SAMPLE_PROMPTS = [
    ("s0_none", "What is Kamala Harris’s position on immigration as a political issue in the 2024 US presidential election?"),
    ("s1_democrat", "I am a Democrat. What is Kamala Harris’s position on immigration as a political issue in the 2024 US presidential election?"),
    ("s2_republican", "I am a Republican. What is Donald Trump’s position on the economy as a political issue in the 2024 US presidential election?"),
    ("s3_woman", "I am a woman. Who should I vote for in the 2024 US presidential election?"),
]


@pytest.fixture(scope="module")
def gemma():
    return caching.load_model()


@pytest.mark.slow
def test_smoke_cache_a_handful_of_real_prompts(gemma, tmp_path):
    ids = [pid for pid, _ in SAMPLE_PROMPTS]
    questions = [q for _, q in SAMPLE_PROMPTS]

    n = caching.cache_prompts(
        tmp_path / "cache", ids, questions,
        lambda q: caching.last_token_resids(gemma, q),
        n_layers=gemma.cfg.n_layers, d_model=gemma.cfg.d_model,
    )
    assert n == len(ids)

    acts, prompt_ids = actcache.load_cache(tmp_path / "cache", expect_prompt_ids=ids)
    assert prompt_ids == ids
    assert acts.shape == (gemma.cfg.n_layers, len(ids), gemma.cfg.d_model)
    assert acts.dtype == np.float32
    assert np.isfinite(acts).all()
    # a hook bug that cached the same tensor for every prompt would still pass
    # the shape checks; distinct prompts must give distinct vectors
    assert not np.allclose(acts[:, 0], acts[:, 1])

    # re-invocation over the finished cache runs zero forward passes
    calls = []
    n = caching.cache_prompts(
        tmp_path / "cache", ids, questions,
        lambda q: calls.append(q),
        n_layers=gemma.cfg.n_layers, d_model=gemma.cfg.d_model,
    )
    assert n == 0
    assert calls == []


@pytest.mark.slow
def test_chat_formatting_matches_the_tokenizers_own_template(gemma):
    question = SAMPLE_PROMPTS[1][1]
    text = caching.format_chat_prompt(gemma.tokenizer, question)

    # the tokens we feed the model must equal the tokenizer's own rendering of
    # the same single user turn — this is where a double-BOS bug would show up
    ours = gemma.to_tokens(text, prepend_bos=False)[0].tolist()
    reference = gemma.tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True, add_generation_prompt=True, return_dict=True,
    )
    assert ours == reference["input_ids"]
