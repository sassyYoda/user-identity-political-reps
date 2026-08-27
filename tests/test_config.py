"""The model-scoping seam: every cache and artifact path is derived from a
model name, so the Gemma-3 run can never overwrite the Gemma-2 replication
artifacts (ADR-0004 keeps both)."""

import re

from polreps import config


def test_subject_model_is_gemma_3_at_a_pinned_revision():
    assert config.MODEL_NAME == "google/gemma-3-12b-it"
    assert re.fullmatch(r"[0-9a-f]{40}", config.MODEL_REVISION)


def test_replication_model_is_the_completed_gemma_2_run():
    assert config.REPLICATION_MODEL_NAME == "google/gemma-2-9b-it"
    # the revision the milestone-1 numbers were produced at; changing it would
    # orphan every NUMBERS.md row in the Gemma-2 section
    assert config.REPLICATION_MODEL_REVISION == "11c9b309abf73637e4b6f9a3fa1e92e615547819"


def test_model_scoped_paths_differ_per_model():
    subject_cache = config.cache_dir()
    replication_cache = config.cache_dir(config.REPLICATION_MODEL_NAME)
    assert subject_cache == config.ACTIVATIONS / "gemma-3-12b-it" / "main"
    assert replication_cache == config.ACTIVATIONS / "gemma-2-9b-it" / "main"
    assert subject_cache != replication_cache

    assert config.artifacts_dir() == config.ARTIFACTS / "gemma-3-12b-it"
    assert (
        config.artifacts_dir(config.REPLICATION_MODEL_NAME)
        == config.ARTIFACTS / "gemma-2-9b-it"
    )


def test_cache_paths_reveal_their_model():
    # scripts route their outputs by this, so reading the wrong model's cache
    # can never write into another model's artifact directory
    assert config.model_slug_of_cache(config.cache_dir()) == "gemma-3-12b-it"
    assert (
        config.model_slug_of_cache(config.cache_dir(config.REPLICATION_MODEL_NAME))
        == "gemma-2-9b-it"
    )
    assert config.model_slug_of_cache("/tmp/somewhere/else") is None
    assert config.model_slug_of_cache(config.ACTIVATIONS) is None
