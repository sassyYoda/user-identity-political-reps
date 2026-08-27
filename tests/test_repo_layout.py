import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def ignored(path):
    return subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO
    ).returncode == 0


# activations and raw data are hundreds of MB and re-derivable; the token must
# never be committable at all
@pytest.mark.parametrize(
    "path", ["data/raw/probe.txt", "activations/probe.npz", "artifacts/probe.png", ".env"]
)
def test_generated_paths_are_gitignored(path):
    assert ignored(path), f"{path} is committable"


def test_artifact_directories_exist():
    for d in ("data/raw", "activations", "artifacts"):
        assert (REPO / d).is_dir(), d
