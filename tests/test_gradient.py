import csv
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from polreps import displacement, gradient, pairs

D_MODEL = 32
LAYER_A, LAYER_B = 1, 3
N_LAYERS = 4

# planted per-condition projections onto the ideology axis, in the graded
# order ADR-0003 predicts: partisan extremes, paraphrases beside them,
# demographics in between, everything request-like or inert at zero. On top
# of these, every non-baseline condition also gets OFFSET — the real data
# showed a large projection component common to all scaffolds, and the
# permutation p must judge deviations from that common offset, not from zero
OFFSET = 2.0
PLANTED = {
    "I am a Democrat. {}": -4.0,
    "I'm a member of the Democratic party. {}": -3.6,
    "I am Black. {}": -2.0,
    "I identify as a woman. {}": -0.8,
    "I am Hispanic. {}": -0.5,
    "I am a Pacific Islander. {}": -0.2,
    "I would like your opinion. {}": 0.0,
    "{} Please cite your sources.": 0.0,
    "I am a person. {}": 0.0,
    "I was born in June. {}": 0.0,
    "I am left-handed. {}": 0.0,
    "I identify as a man. {}": 0.6,
    "I am White. {}": 1.0,
    "I'm a member of the Republican party. {}": 3.7,
    "I am a Republican. {}": 4.0,
}
MAIN_CONDITIONS = [c for c in PLANTED if c not in gradient.CONTROL_ROLES]
ANCHOR_LEAN = {
    "I am White. {}": 12,
    "I am Black. {}": -68,
    "I am Hispanic. {}": -3,
    "I identify as a man. {}": 12,
    "I identify as a woman. {}": -7,
}


def test_control_conditions_are_new_and_invertible():
    # the five control templates must extend the Cen vocabulary, not collide
    # with it, and must round-trip through the release's inversion recipe
    assert set(gradient.CONTROL_ROLES).isdisjoint(MAIN_CONDITIONS)
    for condition in gradient.CONTROL_ROLES:
        applied = condition.replace("{}", "What is the capital?")
        assert pairs.recover_pre_prompt_q(applied, condition) == "What is the capital?"


def test_condition_roles_cover_the_bank_and_refuse_strangers():
    assert gradient.condition_role("I am a Democrat. {}") == "partisan scaffold"
    assert gradient.condition_role("I am White. {}") == "demographic scaffold"
    assert gradient.condition_role("{} Please cite your sources.") == "non-identity variation"
    assert gradient.condition_role("I would like your opinion. {}") == "non-identity variation"
    assert gradient.condition_role("I am a person. {}") == "syntactic control"
    assert gradient.condition_role("I was born in June. {}") == "inert control"
    assert gradient.condition_role("I'm a member of the Democratic party. {}") == "partisan paraphrase"
    with pytest.raises(ValueError, match="unknown condition"):
        gradient.condition_role("I am a Whig. {}")


def test_paraphrase_map_points_at_partisan_scaffolds():
    for paraphrase, original in gradient.PARAPHRASE_OF.items():
        assert gradient.condition_role(paraphrase) == "partisan paraphrase"
        assert gradient.condition_role(original) == "partisan scaffold"


def make_main_table_rows(n_sets=3):
    rows = []
    for s in range(n_sets):
        pre_q = f"Question number {s}?"
        q_hash = pairs.hash_string(pre_q)
        for condition in ["none"] + MAIN_CONDITIONS:
            question = pre_q if condition == "none" else condition.replace("{}", pre_q)
            rows.append(
                {
                    "prompt_id": f"{q_hash[:16]}-{pairs.hash_string(condition)[:16]}",
                    "condition": condition,
                    "question": question,
                    "pre_prompt_q_hash": q_hash,
                    "base_q_template_hash": pairs.hash_string(f"t{s}"),
                    "type": "exo",
                    "category": "cat",
                    "subcategory": "",
                }
            )
    return rows


def test_control_rows_apply_templates_over_the_none_questions():
    main_rows = make_main_table_rows(n_sets=2)
    control_rows = gradient.control_table_rows(main_rows)

    assert len(control_rows) == 2 * len(gradient.CONTROL_ROLES)
    by_condition = {}
    for row in control_rows:
        by_condition.setdefault(row["condition"], []).append(row)
    assert set(by_condition) == set(gradient.CONTROL_ROLES)

    none_ids = {r["prompt_id"] for r in main_rows}
    for row in control_rows:
        assert row["prompt_id"] not in none_ids
        assert pairs.recover_pre_prompt_q(row["question"], row["condition"]).startswith(
            "Question number"
        )
        # ids follow the release recipe, so a rebuilt table is byte-stable
        assert row["prompt_id"] == (
            f"{row['pre_prompt_q_hash'][:16]}-{pairs.hash_string(row['condition'])[:16]}"
        )


def test_control_rows_refuse_tables_without_baseline_or_with_collisions():
    main_rows = make_main_table_rows(n_sets=2)
    no_none = [r for r in main_rows if r["condition"] != "none"]
    with pytest.raises(ValueError, match="none"):
        gradient.control_table_rows(no_none)
    collided = main_rows + [
        dict(main_rows[0], condition="I am a person. {}")
    ]
    with pytest.raises(ValueError, match="already"):
        gradient.control_table_rows(collided)


def test_paired_projection_matrix_subtracts_the_baseline_per_set():
    projections = np.array([1.0, 3.0, -2.0, 10.0, 14.0, 6.0])
    labels = np.array(["none", "a", "b", "none", "a", "b"])
    set_ids = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])
    conditions, matrix = gradient.paired_projection_matrix(projections, labels, set_ids)
    assert conditions == ["a", "b"]
    np.testing.assert_allclose(matrix, [[2.0, 4.0], [-3.0, -4.0]])


def test_paired_projection_matrix_refuses_incomplete_sets():
    projections = np.arange(5.0)
    labels = np.array(["none", "a", "b", "none", "a"])
    set_ids = np.array(["s1", "s1", "s1", "s2", "s2"])
    with pytest.raises(ValueError, match="missing"):
        gradient.paired_projection_matrix(projections, labels, set_ids)
    labels = np.array(["a", "b", "none", "a"])
    set_ids = np.array(["s1", "s1", "s2", "s2"])
    with pytest.raises(ValueError, match="baseline"):
        gradient.paired_projection_matrix(np.arange(4.0), labels, set_ids)


def test_permutation_null_permutes_within_sets_only():
    matrix = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    null = gradient.permutation_null_means(matrix, n_draws=200, seed=0)
    assert null.shape == (200, 3)
    # within-set label exchange preserves each draw's grand mean exactly
    np.testing.assert_allclose(null.mean(axis=1), matrix.mean(), atol=1e-12)
    # and the draws genuinely vary (the permutation is not the identity)
    assert null.std() > 0
    again = gradient.permutation_null_means(matrix, n_draws=200, seed=0)
    np.testing.assert_array_equal(null, again)


def test_random_direction_null_scales_with_the_displacement_norm():
    rng = np.random.default_rng(0)
    mean_disp = np.stack([3.0 * rng.normal(size=256), 30.0 * rng.normal(size=256)])
    null = gradient.random_direction_projections(mean_disp, n_draws=20_000, seed=1)
    assert null.shape == (2, 20_000)
    norms = np.linalg.norm(mean_disp, axis=1)
    # projection of a fixed vector onto a random unit direction has
    # sd = ||v|| / sqrt(d)
    np.testing.assert_allclose(
        null.std(axis=1), norms / math.sqrt(256), rtol=0.05
    )


def test_rank_correlation_exact_p_on_small_n():
    rho, p, method = gradient.rank_correlation([1, 2, 3, 4, 5, 6], [2, 4, 5, 7, 8, 9])
    assert rho == pytest.approx(1.0)
    assert method == "exact"
    # perfect order two-sided: 2 of the 720 permutations tie |rho| = 1
    assert p == pytest.approx(2 / 720)
    rho_rev, _, _ = gradient.rank_correlation([1, 2, 3], [3, 2, 1])
    assert rho_rev == pytest.approx(-1.0)


def test_rank_correlation_handles_ties_with_average_ranks():
    # x has a tie (the anchor table has one: White and man both +12)
    rho, _, _ = gradient.rank_correlation([1, 2, 2, 4], [1, 2, 3, 4])
    assert 0.9 < rho < 1.0


def write_table(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_gradient_fixture(tmp_path, n_sets=40, noise=0.3, seed=0):
    """Fake caches with a planted projection gradient along one axis at
    LAYER_A and LAYER_B (other layers pure noise), split across a main cache
    (Cen conditions) and a control cache, plus every artifact run_gradient
    needs."""
    from polreps import actcache, ideology

    rng = np.random.default_rng(seed)
    axis = rng.normal(size=D_MODEL)
    axis /= np.linalg.norm(axis)

    main_rows = make_main_table_rows(n_sets=n_sets)
    control_rows = gradient.control_table_rows(main_rows)

    def acts_for(condition):
        x = noise * rng.normal(size=(N_LAYERS, D_MODEL))
        planted = 0.0 if condition == "none" else OFFSET + PLANTED[condition]
        for layer in (LAYER_A, LAYER_B):
            x[layer] += planted * axis
        return x

    base_of = {}
    caches = {}
    for name, rows in (("main", main_rows), ("controls", control_rows)):
        acts = []
        for row in rows:
            base = base_of.setdefault(
                row["pre_prompt_q_hash"], 20.0 * rng.normal(size=(N_LAYERS, D_MODEL))
            )
            acts.append(base + acts_for(row["condition"]))
        cache_dir = tmp_path / f"{name}_cache"
        actcache.save_cache(
            cache_dir,
            np.stack(acts, axis=1).astype(np.float32),
            [r["prompt_id"] for r in rows],
        )
        table = tmp_path / f"{name}_table.csv"
        write_table(table, rows)
        caches[name] = (cache_dir, table)

    # ideology direction = the planted axis at the signal layers; displacement
    # artifacts from the real stage so the cross-check exercises real files
    directions = 0.01 * rng.normal(size=(N_LAYERS, D_MODEL))
    directions[LAYER_A] = axis
    directions[LAYER_B] = axis
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    ideology_npz = tmp_path / "ideology_direction.npz"
    np.savez(ideology_npz, diff_raw=directions.astype(np.float32),
             diff_unit=unit.astype(np.float32))

    displacement.run_displacements(
        caches["main"][0], caches["main"][1], tmp_path / "displacements"
    )
    anchor_json = tmp_path / "partisan_lean.json"
    anchor_json.write_text(json.dumps({
        "source": {"citation": "planted fixture"},
        "lean_metric": "planted",
        "lean_by_condition": ANCHOR_LEAN,
    }))
    return {
        "scaffold_cache": caches["main"][0],
        "scaffold_table": caches["main"][1],
        "control_cache": caches["controls"][0],
        "control_table": caches["controls"][1],
        "ideology": ideology_npz,
        "displacements": tmp_path / "displacements.npz",
        "anchor": anchor_json,
    }


def run_fixture_gradient(tmp_path, out_name="gradient", **overrides):
    paths = make_gradient_fixture(tmp_path)
    kwargs = dict(
        layer=LAYER_A, alt_layers=(LAYER_B,),
        permutation_draws=2000, direction_draws=5000, seed=0,
    )
    kwargs.update(overrides)
    result = gradient.run_gradient(
        paths["scaffold_cache"], paths["scaffold_table"],
        paths["control_cache"], paths["control_table"],
        paths["ideology"], paths["displacements"], paths["anchor"],
        tmp_path / "artifacts" / out_name, **kwargs,
    )
    return paths, result


def test_planted_gradient_is_recovered_end_to_end(tmp_path):
    _, result = run_fixture_gradient(tmp_path)
    at = result["layers"][str(LAYER_A)]

    # the ranking must reproduce the planted spectrum order (the five
    # planted-zero conditions are mutually unordered, so compare the rest)
    recovered = sorted(at["mean_projection"], key=at["mean_projection"].get)
    nonzero = [c for c in recovered if PLANTED[c] != 0.0]
    assert nonzero == sorted(nonzero, key=PLANTED.get)
    assert at["ranking"] == sorted(recovered, key=at["mean_projection"].get, reverse=True)

    # the null centers on the planted common offset, not on zero
    grand_mean = OFFSET + sum(PLANTED.values()) / len(PLANTED)
    assert at["permutation_null"]["mean"] == pytest.approx(grand_mean, abs=0.15)

    # partisan extremes clear both nulls; the Democrat scaffold sits at
    # |projection| = 2 = the null's own center, which only a deviation-based
    # (not zero-centered) p can flag
    for condition in ("I am a Democrat. {}", "I am a Republican. {}"):
        assert at["permutation_null"]["p_value"][condition] < 0.01
        assert at["random_direction_null"]["p_value"][condition] < 0.01
    # the planted-zero controls sit exactly at the common offset
    for condition in gradient.CONTROL_ROLES:
        if gradient.condition_role(condition) == "partisan paraphrase":
            continue
        assert at["permutation_null"]["p_value"][condition] > 0.05

    # anchors were planted in the projections' own order
    assert at["anchor"]["n"] == len(ANCHOR_LEAN)
    assert at["anchor"]["rho"] > 0.9
    assert at["anchor"]["p"] < 0.05
    assert at["anchor"]["unanchored"] == ["I am a Pacific Islander. {}"]

    # paraphrases displace like their originals: same sign, high vector cosine
    for pair in result["paraphrase_check"]["pairs"]:
        assert pair["cosine"] > 0.9
        same_sign = (
            at["mean_projection"][pair["paraphrase"]]
            * at["mean_projection"][pair["scaffold"]]
        )
        assert same_sign > 0

    # the alternate layer carries the same planted gradient (the zero-planted
    # conditions may swap among themselves, hence not exactly 1)
    assert result["rank_stability"]["rho"] > 0.9
    assert result["rank_stability"]["identity_rho"] > 0.9
    assert result["rank_stability"]["n_identity"] == len(
        [c for c in PLANTED if gradient.condition_role(c) != "non-identity variation"]
    )


def test_gradient_cross_checks_against_the_displacement_artifact(tmp_path):
    _, result = run_fixture_gradient(tmp_path)
    check = result["layers"][str(LAYER_A)]["displacement_crosscheck"]
    assert check["n_checked"] == len(MAIN_CONDITIONS)
    assert check["max_abs_diff"] < 1e-3


def test_gradient_artifacts_land_on_disk(tmp_path):
    _, result = run_fixture_gradient(tmp_path)
    out_stem = tmp_path / "artifacts" / "gradient"
    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == result
    assert out_stem.with_suffix(".png").exists()
    assert (tmp_path / "artifacts" / "gradient.json.meta.json").exists()
    assert (tmp_path / "artifacts" / "gradient.png.meta.json").exists()


def test_gradient_refuses_a_layer_out_of_range(tmp_path):
    paths = make_gradient_fixture(tmp_path)
    with pytest.raises(ValueError, match="layer"):
        gradient.run_gradient(
            paths["scaffold_cache"], paths["scaffold_table"],
            paths["control_cache"], paths["control_table"],
            paths["ideology"], paths["displacements"], paths["anchor"],
            tmp_path / "out", layer=N_LAYERS, permutation_draws=10, direction_draws=10,
        )


def test_control_table_script_writes_table_and_sidecar(tmp_path):
    from polreps.config import REPO

    main_table = tmp_path / "prompt_table.csv"
    write_table(main_table, make_main_table_rows(n_sets=2))
    out = tmp_path / "control_table.csv"
    subprocess.run(
        [
            sys.executable, "scripts/build_control_table.py",
            "--prompt-table", str(main_table), "--out", str(out),
        ],
        cwd=REPO, check=True, capture_output=True,
    )
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 * len(gradient.CONTROL_ROLES)
    assert out.with_name("control_table.csv.meta.json").exists()


def test_gradient_is_a_single_command(tmp_path):
    from polreps.config import REPO

    paths = make_gradient_fixture(tmp_path)
    out_stem = tmp_path / "gradient"
    subprocess.run(
        [
            sys.executable, "scripts/run_gradient.py",
            "--scaffold-cache", str(paths["scaffold_cache"]),
            "--scaffold-table", str(paths["scaffold_table"]),
            "--control-cache", str(paths["control_cache"]),
            "--control-table", str(paths["control_table"]),
            "--ideology", str(paths["ideology"]),
            "--displacements", str(paths["displacements"]),
            "--anchor", str(paths["anchor"]),
            "--out", str(out_stem),
            "--layer", str(LAYER_A), "--alt-layer", str(LAYER_B),
            "--permutation-draws", "500", "--direction-draws", "500",
        ],
        cwd=REPO, check=True, capture_output=True,
    )
    assert out_stem.with_suffix(".json").exists()
    assert out_stem.with_suffix(".png").exists()
