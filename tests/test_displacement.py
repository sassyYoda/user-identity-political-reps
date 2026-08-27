import json
import subprocess
import sys

import numpy as np
import pytest

from polreps import displacement
from polreps.config import REPO
from tests.conftest import write_prompt_table

N_LAYERS, D_MODEL = 4, 12


def make_paired_data(n_sets=9, conditions=("democrat", "republican"), noise=0.0, seed=0):
    """One row per (set, condition) plus a "none" anchor per set. Each set gets
    its own large base-question vector; each condition adds a fixed per-layer
    offset. The offsets are what displacement_means must recover — the base
    vectors cancel only if the estimator actually pairs within sets."""
    rng = np.random.default_rng(seed)
    offsets = {c: rng.normal(size=(N_LAYERS, D_MODEL)) for c in conditions}
    rows, labels, set_ids = [], [], []
    for g in range(n_sets):
        base_q = 50.0 * rng.normal(size=(N_LAYERS, D_MODEL))
        for cond in ("none", *conditions):
            x = base_q + noise * rng.normal(size=(N_LAYERS, D_MODEL))
            if cond != "none":
                x = x + offsets[cond]
            rows.append(x)
            labels.append(cond)
            set_ids.append(f"set{g:03d}")
    acts = np.stack(rows, axis=1).astype(np.float32)
    return acts, np.array(labels), np.array(set_ids), offsets


def test_difference_in_means_recovers_planted_offsets():
    acts, labels, set_ids, offsets = make_paired_data()
    conditions, raw, n_sets = displacement.displacement_means(acts, labels, set_ids)

    assert conditions == ["democrat", "republican"]
    assert raw.shape == (2, N_LAYERS, D_MODEL)
    assert raw.dtype == np.float32
    # noise-free, so pairing must cancel the base-question vectors exactly (to
    # fp32 rounding against 50x-larger base terms)
    for c, cond in enumerate(conditions):
        np.testing.assert_allclose(raw[c], offsets[cond], atol=1e-3)
        assert n_sets[cond] == 9


def test_unpaired_content_does_not_survive_noise():
    # same check with per-row noise: the estimate converges on the offset, not
    # on anything question-specific
    acts, labels, set_ids, offsets = make_paired_data(n_sets=200, noise=0.5, seed=1)
    conditions, raw, _ = displacement.displacement_means(acts, labels, set_ids)
    for c, cond in enumerate(conditions):
        np.testing.assert_allclose(raw[c], offsets[cond], atol=0.35)


def test_unit_copies_are_unit_norm_and_parallel_to_raw():
    acts, labels, set_ids, _ = make_paired_data()
    _, raw, _ = displacement.displacement_means(acts, labels, set_ids)
    unit = displacement.unit_normalize(raw)

    norms = np.linalg.norm(unit, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    cos = (unit * raw).sum(axis=-1) / np.linalg.norm(raw, axis=-1)
    np.testing.assert_allclose(cos, 1.0, atol=1e-5)


def test_unit_normalize_refuses_zero_vectors():
    raw = np.zeros((1, N_LAYERS, D_MODEL), dtype=np.float32)
    with pytest.raises(ValueError, match="zero"):
        displacement.unit_normalize(raw)


def test_missing_baseline_row_raises():
    acts, labels, set_ids, _ = make_paired_data(n_sets=3)
    keep = ~((set_ids == "set001") & (labels == "none"))
    with pytest.raises(ValueError, match="none"):
        displacement.displacement_means(acts[:, keep], labels[keep], set_ids[keep])


def test_duplicate_condition_within_set_raises():
    acts, labels, set_ids, _ = make_paired_data(n_sets=3)
    labels = labels.copy()
    labels[np.argmax((set_ids == "set002") & (labels == "republican"))] = "democrat"
    with pytest.raises(ValueError, match="duplicate"):
        displacement.displacement_means(acts, labels, set_ids)


def test_incomplete_sets_average_over_the_sets_that_have_the_condition():
    # drop one set's "republican" row: its mean now spans 8 sets, democrat's
    # still 9, and both are reported so downstream knows the support
    acts, labels, set_ids, offsets = make_paired_data()
    keep = ~((set_ids == "set000") & (labels == "republican"))
    conditions, raw, n_sets = displacement.displacement_means(
        acts[:, keep], labels[keep], set_ids[keep]
    )

    assert n_sets == {"democrat": 9, "republican": 8}
    r = conditions.index("republican")
    np.testing.assert_allclose(raw[r], offsets["republican"], atol=1e-3)


def test_run_displacements_writes_arrays_manifest_and_metadata(tmp_path):
    from polreps import actcache

    acts, labels, set_ids, offsets = make_paired_data()
    prompt_ids = [f"p{i:03d}" for i in range(acts.shape[1])]
    cache_dir = tmp_path / "cache"
    actcache.save_cache(cache_dir, acts, prompt_ids)
    table = tmp_path / "prompt_table.csv"
    write_prompt_table(
        table,
        {"prompt_ids": prompt_ids, "labels": labels, "groups": set_ids},
        group_column="pre_prompt_q_hash",
    )
    out_stem = tmp_path / "artifacts" / "displacements"

    manifest = displacement.run_displacements(cache_dir, table, out_stem)

    arrays = np.load(out_stem.with_suffix(".npz"))
    assert list(arrays["conditions"]) == ["democrat", "republican"]
    assert arrays["raw"].shape == (2, N_LAYERS, D_MODEL)
    assert arrays["raw"].dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(arrays["unit"], axis=-1), 1.0, atol=1e-5
    )
    d = list(arrays["conditions"]).index("democrat")
    np.testing.assert_allclose(arrays["raw"][d], offsets["democrat"], atol=1e-3)

    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == manifest
    assert manifest["baseline"] == "none"
    assert manifest["n_sets"] == {"democrat": 9, "republican": 9}
    assert manifest["n_layers"] == N_LAYERS
    assert (tmp_path / "artifacts" / "displacements.npz.meta.json").exists()
    assert (tmp_path / "artifacts" / "displacements.json.meta.json").exists()


def test_displacements_is_a_single_command(tmp_path):
    from polreps import actcache

    acts, labels, set_ids, _ = make_paired_data(n_sets=3)
    prompt_ids = [f"p{i:03d}" for i in range(acts.shape[1])]
    cache_dir = tmp_path / "cache"
    actcache.save_cache(cache_dir, acts, prompt_ids)
    table = tmp_path / "prompt_table.csv"
    write_prompt_table(
        table,
        {"prompt_ids": prompt_ids, "labels": labels, "groups": set_ids},
        group_column="pre_prompt_q_hash",
    )
    out_stem = tmp_path / "displacements"

    subprocess.run(
        [
            sys.executable, "scripts/compute_displacements.py",
            "--cache", str(cache_dir),
            "--labels", str(table),
            "--out", str(out_stem),
        ],
        cwd=REPO, check=True, capture_output=True,
    )

    assert (tmp_path / "displacements.npz").exists()
    assert (tmp_path / "displacements.json").exists()
