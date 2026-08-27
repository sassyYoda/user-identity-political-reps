"""Project-wide constants: the subject model, pinned revision, and layout."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Gemma-3-12B-IT for everything reported (ADR-0004); revision pinned 2026-08-27
# via scripts/check_model_access.py so upstream pushes can't move our numbers
MODEL_NAME = "google/gemma-3-12b-it"
MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"

# The completed Gemma-2-9B-IT run stays as the second-model replication anchor;
# its NUMBERS.md rows are tied to this revision
REPLICATION_MODEL_NAME = "google/gemma-2-9b-it"
REPLICATION_MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"

# Cen et al. 2025 release (arXiv:2509.18446); revision pinned 2026-08-26
DATASET_NAME = "sarahcen/llm-election-data-2024"
DATASET_REVISION = "7bb3c18c2eadfc3f96db0dd394768496f7107a79"

REPO = Path(__file__).resolve().parents[1]
DATA_RAW = REPO / "data" / "raw"
ACTIVATIONS = REPO / "activations"
ARTIFACTS = REPO / "artifacts"


def cache_dir(model_name=MODEL_NAME):
    """activations/<model>/main — scoped so no model's run can clobber another's."""
    return ACTIVATIONS / model_name.split("/")[-1] / "main"


def artifacts_dir(model_name=MODEL_NAME):
    """artifacts/<model>/ for per-model outputs; cross-model artifacts (the
    prompt table, replication figures) stay at the ARTIFACTS root."""
    return ARTIFACTS / model_name.split("/")[-1]


def model_slug_of_cache(cache_dir):
    """The <model> of an activations/<model>/... path, None for anything else.

    Scripts route their default outputs by this, so reading one model's cache
    can never write into another model's artifact directory.
    """
    try:
        rel = Path(cache_dir).resolve().relative_to(ACTIVATIONS.resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def artifacts_stem_for_cache(cache_dir, filename):
    """artifacts/<model>/<filename> for a model-scoped cache path, or None
    when the cache lives elsewhere and the caller must name the output."""
    slug = model_slug_of_cache(cache_dir)
    return None if slug is None else ARTIFACTS / slug / filename


def hf_token():
    """HF token from .env (or the environment); Gemma is gated."""
    load_dotenv(REPO / ".env")
    return os.environ.get("HF_TOKEN")
