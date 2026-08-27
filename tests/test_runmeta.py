import json
from datetime import datetime

from polreps import runmeta


def test_sidecar_written_next_to_artifact(tmp_path):
    artifact = tmp_path / "curve.png"
    artifact.write_bytes(b"")

    meta_path = runmeta.save_run_metadata(artifact, seed=0)

    assert meta_path == tmp_path / "curve.png.meta.json"
    assert meta_path.exists()


def test_metadata_records_seed_config_and_revision(tmp_path):
    config = {"n_pairs": 500, "layers": "all"}
    meta_path = runmeta.save_run_metadata(
        tmp_path / "acts.npz", seed=42, config=config, model_revision="abc123"
    )
    meta = json.loads(meta_path.read_text())

    assert meta["seed"] == 42
    assert meta["config"] == config
    assert meta["model_revision"] == "abc123"


def test_metadata_records_environment(tmp_path):
    meta_path = runmeta.save_run_metadata(tmp_path / "x.npz", seed=0)
    meta = json.loads(meta_path.read_text())

    # every package whose API breakage could change a number
    for pkg in ("torch", "transformer-lens", "scikit-learn", "datasets", "numpy"):
        assert pkg in meta["versions"], pkg
        assert meta["versions"][pkg]

    assert meta["python"].startswith("3.")
    # timestamp must parse and carry an offset so runs are orderable across machines
    assert datetime.fromisoformat(meta["timestamp"]).tzinfo is not None


def test_metadata_records_git_state(tmp_path):
    meta = json.loads(runmeta.save_run_metadata(tmp_path / "x.npz", seed=0).read_text())

    assert len(meta["git_commit"]) == 40
    int(meta["git_commit"], 16)
    assert isinstance(meta["git_dirty"], bool)
