"""Shared synthetic-activation fixtures.

make_planted_data is the reference input for the probe-sweep seam and, written
through actcache, the shared fixture that pins the on-disk cache format for the
caching stage: a real cache must load through the same actcache.load_cache call
these tests exercise.
"""

import csv

import numpy as np
import pytest

from polreps import actcache

CONDITIONS = ("none", "democrat", "republican", "woman")
SIGNAL_LAYER = 3


def write_prompt_table(path, planted, drop=(), group_column="base_q_template_hash"):
    rows = [
        {"prompt_id": pid, "condition": cond, group_column: group}
        for pid, cond, group in zip(
            planted["prompt_ids"], planted["labels"], planted["groups"]
        )
        if pid not in drop
    ]
    rows.reverse()  # cache order and table order differ; the join must be by id
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "condition", group_column])
        writer.writeheader()
        writer.writerows(rows)


def make_planted_data(
    n_groups=30,
    conditions=CONDITIONS,
    n_layers=6,
    d_model=16,
    signal_layer=SIGNAL_LAYER,
    signal_scale=3.0,
    seed=0,
):
    """One row per (base question, condition); pure noise at every layer except
    a linear class separation planted at signal_layer."""
    rng = np.random.default_rng(seed)
    class_means = rng.normal(size=(len(conditions), d_model))
    class_means *= signal_scale / np.linalg.norm(class_means, axis=1, keepdims=True)

    rows, labels, groups, prompt_ids = [], [], [], []
    for g in range(n_groups):
        for c, cond in enumerate(conditions):
            x = rng.normal(size=(n_layers, d_model))
            x[signal_layer] += class_means[c]
            rows.append(x)
            labels.append(cond)
            groups.append(f"q{g:03d}")
            prompt_ids.append(f"q{g:03d}_{cond}")

    acts = np.stack(rows, axis=1)  # (n_layers, n_rows, d_model)
    return acts, np.array(labels), np.array(groups), prompt_ids


@pytest.fixture
def planted(tmp_path):
    acts, labels, groups, prompt_ids = make_planted_data()
    cache_dir = tmp_path / "planted_cache"
    actcache.save_cache(cache_dir, acts, prompt_ids)
    return {
        "cache_dir": cache_dir,
        "labels": labels,
        "groups": groups,
        "prompt_ids": prompt_ids,
        "signal_layer": SIGNAL_LAYER,
    }
