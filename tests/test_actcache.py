import json

import numpy as np
import pytest

from polreps import actcache


def toy_acts(n_layers=3, n_prompts=4, d_model=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_layers, n_prompts, d_model))


def test_round_trip_preserves_values_and_order(tmp_path):
    acts = toy_acts()
    ids = ["p0", "p1", "p2", "p3"]

    actcache.save_cache(tmp_path / "cache", acts, ids)
    loaded, loaded_ids = actcache.load_cache(tmp_path / "cache")

    assert loaded_ids == ids
    assert loaded.shape == (3, 4, 8)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, acts.astype(np.float32))


def test_on_disk_layout_is_one_fp32_array_per_layer(tmp_path):
    # ticket 03 writes this format incrementally, so the layout itself is the
    # contract, not just the round-trip
    actcache.save_cache(tmp_path / "cache", toy_acts(), ["a", "b", "c", "d"])

    index = json.loads((tmp_path / "cache" / "index.json").read_text())
    assert index["prompt_ids"] == ["a", "b", "c", "d"]
    assert index["n_layers"] == 3
    assert index["d_model"] == 8

    for layer in range(3):
        arr = np.load(tmp_path / "cache" / f"layer_{layer:02d}.npy")
        assert arr.dtype == np.float32
        assert arr.shape == (4, 8)


def test_load_verifies_prompt_ids_against_expectation(tmp_path):
    actcache.save_cache(tmp_path / "cache", toy_acts(), ["a", "b", "c", "d"])

    # exact match passes
    actcache.load_cache(tmp_path / "cache", expect_prompt_ids=["a", "b", "c", "d"])

    # wrong order and missing rows both refuse loudly
    with pytest.raises(ValueError, match="prompt"):
        actcache.load_cache(tmp_path / "cache", expect_prompt_ids=["b", "a", "c", "d"])
    with pytest.raises(ValueError, match="prompt"):
        actcache.load_cache(tmp_path / "cache", expect_prompt_ids=["a", "b", "c"])


def test_load_refuses_cache_inconsistent_with_index(tmp_path):
    actcache.save_cache(tmp_path / "cache", toy_acts(), ["a", "b", "c", "d"])

    # a missing layer file means an interrupted or corrupted caching run
    (tmp_path / "cache" / "layer_01.npy").unlink()
    with pytest.raises(ValueError, match="layer_01"):
        actcache.load_cache(tmp_path / "cache")


def test_load_refuses_row_count_mismatch(tmp_path):
    actcache.save_cache(tmp_path / "cache", toy_acts(), ["a", "b", "c", "d"])

    truncated = np.load(tmp_path / "cache" / "layer_02.npy")[:2]
    np.save(tmp_path / "cache" / "layer_02.npy", truncated)
    with pytest.raises(ValueError, match="layer_02"):
        actcache.load_cache(tmp_path / "cache")


def test_load_refuses_non_fp32_layer_files(tmp_path):
    actcache.save_cache(tmp_path / "cache", toy_acts(), ["a", "b", "c", "d"])

    as_fp64 = np.load(tmp_path / "cache" / "layer_00.npy").astype(np.float64)
    np.save(tmp_path / "cache" / "layer_00.npy", as_fp64)
    with pytest.raises(ValueError, match="float32"):
        actcache.load_cache(tmp_path / "cache")


def test_save_refuses_duplicate_prompt_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        actcache.save_cache(tmp_path / "cache", toy_acts(), ["a", "a", "c", "d"])
