"""Sidecar metadata for every produced artifact.

Any number cited in NUMBERS.md must trace back to one of these records: the
seed, the library versions that could change the number, the pinned model
revision, and the git state of the code that ran.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from polreps.config import REPO

# the packages whose API or numerics changes could silently move a result
TRACKED = ("torch", "transformer-lens", "scikit-learn", "datasets", "numpy")


def run_metadata(seed, config=None, model_revision=None):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    # -uno: dirty means tracked code drifted, not that scratch files exist
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "-uno"],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.strip()
    )
    return {
        "seed": seed,
        "config": config or {},
        "model_revision": model_revision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": ".".join(str(v) for v in sys.version_info[:3]),
        "versions": {pkg: version(pkg) for pkg in TRACKED},
        "git_commit": commit,
        "git_dirty": dirty,
    }


def save_run_metadata(artifact_path, seed, config=None, model_revision=None):
    """Write `<artifact>.meta.json` next to the artifact and return its path."""
    artifact_path = Path(artifact_path)
    meta_path = artifact_path.with_name(artifact_path.name + ".meta.json")
    meta_path.write_text(
        json.dumps(run_metadata(seed, config, model_revision), indent=2) + "\n"
    )
    return meta_path
