"""Confirm the HF token in .env can see gated Gemma-2-9B-IT at the pinned revision.

Run after environment setup, before any caching run:

    uv run python scripts/check_model_access.py
"""

import json
import sys

from huggingface_hub import HfApi

from polreps.config import ARTIFACTS, MODEL_NAME, MODEL_REVISION, hf_token
from polreps.runmeta import save_run_metadata


def main():
    token = hf_token()
    if not token:
        sys.exit("No HF_TOKEN found in .env or the environment.")

    info = HfApi(token=token).model_info(MODEL_NAME, revision=MODEL_REVISION)

    ARTIFACTS.mkdir(exist_ok=True)
    check_path = ARTIFACTS / "model_access_check.json"
    check_path.write_text(
        json.dumps(
            {"model": MODEL_NAME, "revision": info.sha, "gated": info.gated}, indent=2
        )
        + "\n"
    )
    save_run_metadata(check_path, seed=None, model_revision=info.sha)

    print(f"ok: {MODEL_NAME} visible at {info.sha} (gated={info.gated})")
    print(f"wrote {check_path.relative_to(ARTIFACTS.parent)} and its .meta.json")


if __name__ == "__main__":
    main()
