import json

import numpy as np
import pytest

from polreps import ideology

N_LAYERS, D_MODEL, SIGNAL_LAYER = 5, 24, 2


def make_ideology_data(n_per_party=60, scale=4.0, noise=1.0, seed=0):
    """Planted content corpus: pure noise everywhere except one layer where
    activations shift along a known axis in proportion to the speaker's
    nominate_dim1 (party = the score's sign)."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=D_MODEL)
    axis /= np.linalg.norm(axis)

    scores = np.concatenate(
        [
            rng.uniform(-0.8, -0.1, size=n_per_party),
            rng.uniform(0.1, 0.8, size=n_per_party),
        ]
    )
    parties = np.array(["D"] * n_per_party + ["R"] * n_per_party)
    acts = noise * rng.normal(size=(N_LAYERS, 2 * n_per_party, D_MODEL))
    acts[SIGNAL_LAYER] += scale * scores[:, None] * axis[None, :]
    return acts.astype(np.float32), parties, scores, axis


def test_diff_in_means_recovers_planted_axis_conservative_positive():
    acts, parties, scores, axis = make_ideology_data(noise=0.0)
    diff = ideology.diff_in_means_directions(acts, parties)

    assert diff.shape == (N_LAYERS, D_MODEL)
    cos = ideology.cosine_rows(diff[[SIGNAL_LAYER]], axis[None, :])[0]
    # R scores are positive, so the planted axis must come back un-flipped
    assert cos > 0.99


def test_diff_in_means_requires_both_parties():
    acts, parties, _, _ = make_ideology_data()
    with pytest.raises(ValueError, match="'R'"):
        ideology.diff_in_means_directions(acts, np.where(parties == "R", "D", "D"))


def test_ridge_recovers_planted_axis_and_agrees_with_diff():
    acts, parties, scores, axis = make_ideology_data(noise=0.5, seed=1)
    ridge, alphas = ideology.ridge_directions(acts, scores)

    assert ridge.shape == (N_LAYERS, D_MODEL)
    assert len(alphas) == N_LAYERS
    cos_axis = ideology.cosine_rows(ridge[[SIGNAL_LAYER]], axis[None, :])[0]
    assert cos_axis > 0.9

    diff = ideology.diff_in_means_directions(acts, parties)
    both = ideology.cosine_rows(diff, ridge)
    # the robustness check the manifest reports. Note it can be high at pure
    # noise layers too — both extractors fit the *same* label noise (party is
    # nearly a function of the score) — so agreement here is necessary, not
    # sufficient; the transfer test is the real check
    assert both[SIGNAL_LAYER] > 0.9


def test_ridge_refuses_mismatched_scores():
    acts, _, scores, _ = make_ideology_data()
    with pytest.raises(ValueError, match="rows"):
        ideology.ridge_directions(acts, scores[:-1])


def test_run_ideology_directions_writes_arrays_manifest_metadata(tmp_path):
    import csv

    from polreps import actcache

    acts, parties, scores, axis = make_ideology_data(noise=0.5)
    prompt_ids = [f"crec-{i:04d}" for i in range(acts.shape[1])]
    cache_dir = tmp_path / "content_cache"
    actcache.save_cache(cache_dir, acts, prompt_ids)

    table = tmp_path / "content_corpus.csv"
    rows = [
        {"prompt_id": pid, "party": party, "nominate_dim1": f"{score:.3f}"}
        for pid, party, score in zip(prompt_ids, parties, scores)
    ]
    rows.reverse()  # order must not matter: the join is by prompt id
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "party", "nominate_dim1"])
        writer.writeheader()
        writer.writerows(rows)

    out_stem = tmp_path / "artifacts" / "ideology_direction"
    manifest = ideology.run_ideology_directions(cache_dir, table, out_stem)

    arrays = np.load(out_stem.with_suffix(".npz"))
    assert arrays["diff_raw"].shape == (N_LAYERS, D_MODEL)
    for key in ("diff_unit", "ridge_unit"):
        np.testing.assert_allclose(
            np.linalg.norm(arrays[key], axis=-1), 1.0, atol=1e-5
        )
    cos = ideology.cosine_rows(arrays["diff_unit"][[SIGNAL_LAYER]], axis[None, :])[0]
    assert cos > 0.95

    on_disk = json.loads(out_stem.with_suffix(".json").read_text())
    assert on_disk == manifest
    assert manifest["diff_ridge_cosine"][SIGNAL_LAYER] > 0.9
    assert manifest["n_layers"] == N_LAYERS
    assert (tmp_path / "artifacts" / "ideology_direction.npz.meta.json").exists()
