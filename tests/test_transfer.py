import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from polreps import transfer

N_LAYERS, D_MODEL, SIGNAL_LAYER = 4, 16, 1


def make_transfer_data(n_groups=40, gap=3.0, noise=1.0, seed=0):
    """Two-class activations separated along a known axis at one layer only,
    two rows per group so the grouped CV has something to keep together."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=D_MODEL)
    axis /= np.linalg.norm(axis)

    labels, groups, rows = [], [], []
    for g in range(n_groups):
        for label in ("democrat", "republican"):
            x = noise * rng.normal(size=(N_LAYERS, D_MODEL))
            sign = 1.0 if label == "republican" else -1.0
            x[SIGNAL_LAYER] += sign * (gap / 2) * axis
            rows.append(x)
            labels.append(label)
            groups.append(f"g{g:03d}")
    acts = np.stack(rows, axis=1).astype(np.float32)
    return acts, np.array(labels), np.array(groups), axis


def test_projection_shapes_and_zero_direction_refused():
    acts, _, _, axis = make_transfer_data()
    directions = np.tile(axis, (N_LAYERS, 1))
    projections = transfer.project(acts, directions)
    assert projections.shape == acts.shape[:2]
    with pytest.raises(ValueError, match="zero direction"):
        transfer.project(acts, np.zeros((N_LAYERS, D_MODEL)))


def test_planted_axis_transfers_at_signal_layer_only():
    acts, labels, groups, axis = make_transfer_data(gap=6.0)
    directions = np.tile(axis, (N_LAYERS, 1))
    fold_accs = transfer.transfer_curve(acts, labels, groups, directions)

    assert fold_accs.shape == (N_LAYERS, 5)
    means = fold_accs.mean(axis=1)
    assert means[SIGNAL_LAYER] > 0.95
    # the same axis is meaningless at the noise layers
    noise_layers = [l for l in range(N_LAYERS) if l != SIGNAL_LAYER]
    assert np.all(means[noise_layers] < 0.7)


def test_sign_flip_cannot_change_the_score():
    acts, labels, groups, axis = make_transfer_data(gap=6.0)
    directions = np.tile(axis, (N_LAYERS, 1))
    flipped = transfer.transfer_curve(acts, labels, groups, -directions)
    straight = transfer.transfer_curve(acts, labels, groups, directions)
    np.testing.assert_allclose(flipped, straight)


def test_shuffled_labels_score_at_chance():
    acts, labels, groups, axis = make_transfer_data(gap=6.0, n_groups=100)
    directions = np.tile(axis, (N_LAYERS, 1))
    shuffled = np.random.default_rng(0).permutation(labels)
    fold_accs = transfer.transfer_curve(acts, shuffled, groups, directions)
    assert abs(fold_accs.mean() - 0.5) < 0.06


def test_threshold_accuracy_refuses_non_binary_labels():
    acts, labels, groups, axis = make_transfer_data()
    projections = transfer.project(acts, np.tile(axis, (N_LAYERS, 1)))
    labels = labels.copy()
    labels[0] = "green"
    with pytest.raises(ValueError, match="binary"):
        transfer.threshold_accuracy(projections[0], labels, groups)


def test_mismatched_direction_count_refused():
    acts, labels, groups, axis = make_transfer_data()
    with pytest.raises(ValueError, match="directions"):
        transfer.transfer_curve(acts, labels, groups, np.tile(axis, (N_LAYERS + 1, 1)))


def write_table(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_transfer_fixture(tmp_path, seed=0):
    """Both sides of the transfer test sharing one planted axis at
    SIGNAL_LAYER: a scaffold cache of (none, democrat, republican) matched
    sets and a content cache of D/R statements, plus their label tables."""
    from polreps import actcache, displacement, ideology

    rng = np.random.default_rng(seed)
    axis = rng.normal(size=D_MODEL)
    axis /= np.linalg.norm(axis)

    rows, table = [], []
    for g in range(30):
        base_q = 20.0 * rng.normal(size=(N_LAYERS, D_MODEL))
        for cond, sign in (("none", 0.0), ("democrat", -1.0), ("republican", 1.0)):
            x = base_q + 0.3 * rng.normal(size=(N_LAYERS, D_MODEL))
            x[SIGNAL_LAYER] += sign * 2.0 * axis
            rows.append(x)
            table.append(
                {
                    "prompt_id": f"s{g:03d}-{cond}",
                    "condition": cond,
                    "base_q_template_hash": f"t{g % 10:02d}",
                    "pre_prompt_q_hash": f"q{g:03d}",
                }
            )
    scaffold_cache = tmp_path / "scaffold_cache"
    actcache.save_cache(
        scaffold_cache, np.stack(rows, axis=1).astype(np.float32),
        [r["prompt_id"] for r in table],
    )
    scaffold_table = tmp_path / "prompt_table.csv"
    write_table(
        scaffold_table,
        ["prompt_id", "condition", "base_q_template_hash", "pre_prompt_q_hash"],
        table,
    )

    rows, table = [], []
    for i in range(120):
        party = "D" if i % 2 == 0 else "R"
        sign = -1.0 if party == "D" else 1.0
        x = rng.normal(size=(N_LAYERS, D_MODEL))
        x[SIGNAL_LAYER] += sign * 2.5 * axis
        rows.append(x)
        table.append(
            {
                "prompt_id": f"crec-{i:04d}",
                "party": party,
                "nominate_dim1": f"{sign * rng.uniform(0.1, 0.8):.3f}",
                "icpsr": f"{10000 + i % 40}",
                "mentions_party": str(i % 3 == 0 and 1 or 0),
            }
        )
    content_cache = tmp_path / "content_cache"
    actcache.save_cache(
        content_cache, np.stack(rows, axis=1).astype(np.float32),
        [r["prompt_id"] for r in table],
    )
    content_table = tmp_path / "content_corpus.csv"
    write_table(
        content_table,
        ["prompt_id", "party", "nominate_dim1", "icpsr", "mentions_party"],
        table,
    )

    displacement.run_displacements(
        scaffold_cache, scaffold_table, tmp_path / "displacements"
    )
    ideology.run_ideology_directions(
        content_cache, content_table, tmp_path / "ideology_direction"
    )
    return {
        "scaffold_cache": scaffold_cache,
        "scaffold_table": scaffold_table,
        "content_cache": content_cache,
        "content_table": content_table,
        "displacements": tmp_path / "displacements.npz",
        "ideology": tmp_path / "ideology_direction.npz",
    }


def test_run_transfer_test_end_to_end(tmp_path):
    paths = make_transfer_fixture(tmp_path)
    out_stem = tmp_path / "artifacts" / "transfer_test"

    curve = transfer.run_transfer_test(
        paths["scaffold_cache"], paths["scaffold_table"],
        paths["content_cache"], paths["content_table"],
        paths["displacements"], paths["ideology"], out_stem,
        dem="democrat", rep="republican", null_draws=2000,
    )

    # the planted shared axis must carry both ways at the signal layer only
    s2c = curve["scaffold_to_content"]["mean_accuracy"]
    c2s = curve["content_to_scaffold_diff"]["mean_accuracy"]
    assert s2c[SIGNAL_LAYER] > 0.95 and c2s[SIGNAL_LAYER] > 0.95
    assert curve["content_to_scaffold_clean_diff"]["mean_accuracy"][SIGNAL_LAYER] > 0.95
    assert curve["working_layer"]["layer"] == SIGNAL_LAYER
    assert "content_to_scaffold_diff" in curve["working_layer"]["informative_variants"]
    assert curve["scaffold_to_content_no_party_token"]["n_rows"] == 80
    assert curve["alignment"]["cosine_diff"][SIGNAL_LAYER] > 0.9
    assert curve["working_layer"]["alignment_p"] < 0.01
    assert max(curve["scaffold_to_content"]["shuffled_mean_accuracy"]) < 0.65

    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == curve
    assert out_stem.with_suffix(".png").exists()
    assert (tmp_path / "artifacts" / "transfer_test.json.meta.json").exists()


def test_run_transfer_test_refuses_unknown_condition(tmp_path):
    paths = make_transfer_fixture(tmp_path)
    with pytest.raises(ValueError, match="not in displacements"):
        transfer.run_transfer_test(
            paths["scaffold_cache"], paths["scaffold_table"],
            paths["content_cache"], paths["content_table"],
            paths["displacements"], paths["ideology"], tmp_path / "out",
            dem="I am a Democrat. {}", rep="republican", null_draws=100,
        )


def test_transfer_test_is_a_single_command(tmp_path):
    from polreps.config import REPO

    paths = make_transfer_fixture(tmp_path)
    out_stem = tmp_path / "transfer_test"
    subprocess.run(
        [
            sys.executable, "scripts/run_transfer_test.py",
            "--scaffold-cache", str(paths["scaffold_cache"]),
            "--scaffold-table", str(paths["scaffold_table"]),
            "--content-cache", str(paths["content_cache"]),
            "--content-table", str(paths["content_table"]),
            "--displacements", str(paths["displacements"]),
            "--ideology", str(paths["ideology"]),
            "--out", str(out_stem),
            "--dem", "democrat", "--rep", "republican",
            "--null-draws", "1000",
        ],
        cwd=REPO, check=True, capture_output=True,
    )
    assert out_stem.with_suffix(".json").exists()
    assert out_stem.with_suffix(".png").exists()


def test_paired_scorer_cancels_question_content():
    # per-set base vectors 20x the offset: the unpaired threshold scorer
    # drowns, the paired scorer must not
    rng = np.random.default_rng(3)
    axis = rng.normal(size=D_MODEL)
    axis /= np.linalg.norm(axis)
    rows, labels, sets = [], [], []
    for g in range(50):
        base_q = 20.0 * rng.normal(size=(N_LAYERS, D_MODEL))
        for label, sign in (("democrat", -1.0), ("republican", 1.0)):
            x = base_q + 0.3 * rng.normal(size=(N_LAYERS, D_MODEL))
            x[SIGNAL_LAYER] += sign * axis
            rows.append(x)
            labels.append(label)
            sets.append(f"q{g:03d}")
    acts = np.stack(rows, axis=1).astype(np.float32)
    directions = np.tile(axis, (N_LAYERS, 1))

    accs, n_pairs = transfer.paired_transfer_curve(
        acts, np.array(labels), np.array(sets), directions,
        pos="republican", neg="democrat",
    )
    assert n_pairs == 50
    assert accs[SIGNAL_LAYER] == 1.0

    # anti-aligned direction lands below chance — nothing rescues the sign
    flipped, _ = transfer.paired_transfer_curve(
        acts, np.array(labels), np.array(sets), -directions,
        pos="republican", neg="democrat",
    )
    assert flipped[SIGNAL_LAYER] == 0.0

    shuffled = transfer.shuffled_pair_labels(
        np.array(labels), np.array(sets), "republican", "democrat", seed=0
    )
    shuf_accs, _ = transfer.paired_transfer_curve(
        acts, shuffled, np.array(sets), directions,
        pos="republican", neg="democrat",
    )
    assert 0.3 < shuf_accs[SIGNAL_LAYER] < 0.7


def test_paired_scorer_refuses_duplicates_and_empty_pairs():
    projections = np.zeros((N_LAYERS, 4))
    with pytest.raises(ValueError, match="duplicate"):
        transfer.paired_sign_accuracy(
            projections, ["a", "a", "b", "b"], ["s1", "s1", "s1", "s2"], "a", "b"
        )
    with pytest.raises(ValueError, match="no matched set"):
        transfer.paired_sign_accuracy(
            projections, ["a", "a", "b", "b"], ["s1", "s2", "s3", "s4"], "a", "b"
        )


def test_random_cosine_null_is_tight_in_high_dimension():
    null = transfer.random_cosine_null(d_model=3840, n_draws=20000, seed=0)
    assert abs(null.mean()) < 0.001
    # sd of a random cosine is ~1/sqrt(d)
    assert abs(null.std() - 1 / np.sqrt(3840)) < 0.002
    again = transfer.random_cosine_null(d_model=3840, n_draws=20000, seed=0)
    np.testing.assert_array_equal(null, again)
