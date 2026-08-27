"""Project-wide constants: the subject model, pinned revision, and layout."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Gemma-2-9B-IT for everything reported (ADR-0001); revision pinned 2026-08-26
# via scripts/check_model_access.py so upstream pushes can't move our numbers
MODEL_NAME = "google/gemma-2-9b-it"
MODEL_REVISION = "11c9b309abf73637e4b6f9a3fa1e92e615547819"

REPO = Path(__file__).resolve().parents[1]
DATA_RAW = REPO / "data" / "raw"
ACTIVATIONS = REPO / "activations"
ARTIFACTS = REPO / "artifacts"


def hf_token():
    """HF token from .env (or the environment); Gemma-2 is gated."""
    load_dotenv(REPO / ".env")
    return os.environ.get("HF_TOKEN")
