"""The replication-figure seam: two probe-curve JSONs (different models,
different layer counts) overlaid on a normalized-depth axis."""

import json

import pytest

from polreps import replication


def fake_curve(n_layers, peak_layer, peak=0.9, base=0.3, chance=0.25):
    """A curve dict in the run_sweep output schema, peaked at peak_layer."""
    accs = [base] * n_layers
    accs[peak_layer] = peak
    variant = {
        "chance": chance,
        "mean_accuracy": accs,
        "shuffled_mean_accuracy": [chance] * n_layers,
    }
    return {"n_layers": n_layers, "multinomial": variant, "binary": None}


def write_curve(path, curve):
    path.write_text(json.dumps(curve) + "\n")
    return path


def test_overlay_normalizes_depth_and_reports_peaks(tmp_path):
    a = write_curve(tmp_path / "a.json", fake_curve(6, peak_layer=3))
    b = write_curve(tmp_path / "b.json", fake_curve(8, peak_layer=2, peak=0.8))
    out_stem = tmp_path / "replication"

    summary = replication.run_replication({"model-a": a, "model-b": b}, out_stem)

    assert (tmp_path / "replication.png").stat().st_size > 0
    assert (tmp_path / "replication.png.meta.json").exists()
    assert (tmp_path / "replication.json.meta.json").exists()

    written = json.loads((tmp_path / "replication.json").read_text())
    assert written == summary
    multi = summary["multinomial"]
    assert multi["model-a"]["peak_layer"] == 3
    assert multi["model-a"]["peak_accuracy"] == 0.9
    assert multi["model-b"]["peak_layer"] == 2
    assert multi["model-b"]["peak_accuracy"] == 0.8

    # depth spans exactly [0, 1] for both models despite different layer counts
    for label, n_layers in (("model-a", 6), ("model-b", 8)):
        depth = multi[label]["depth"]
        assert len(depth) == n_layers
        assert depth[0] == 0.0
        assert depth[-1] == 1.0


def test_overlay_includes_binary_panel_only_when_all_models_have_it(tmp_path):
    with_binary = fake_curve(6, peak_layer=3)
    with_binary["binary"] = {
        "chance": 0.5,
        "mean_accuracy": [0.6] * 6,
        "shuffled_mean_accuracy": [0.5] * 6,
    }
    a = write_curve(tmp_path / "a.json", with_binary)
    b = write_curve(tmp_path / "b.json", fake_curve(8, peak_layer=2))

    summary = replication.run_replication({"a": a, "b": b}, tmp_path / "out")
    assert "binary" not in summary


def test_overlay_refuses_curves_sharing_no_panel(tmp_path):
    # an overlay with nothing to overlay must refuse before writing anything,
    # not leave an empty summary on disk and die inside matplotlib
    no_multi = {"n_layers": 6, "multinomial": None, "binary": None}
    a = write_curve(tmp_path / "a.json", no_multi)
    b = write_curve(tmp_path / "b.json", no_multi)

    with pytest.raises(ValueError, match="panel"):
        replication.run_replication({"a": a, "b": b}, tmp_path / "out")
    assert not (tmp_path / "out.json").exists()


def test_overlay_refuses_curves_with_different_chance(tmp_path):
    # both models ran the same prompt table, so chance must agree; a mismatch
    # means the curves are not comparable and the overlay would mislead
    a = write_curve(tmp_path / "a.json", fake_curve(6, peak_layer=3, chance=0.25))
    b = write_curve(tmp_path / "b.json", fake_curve(8, peak_layer=2, chance=0.10))

    with pytest.raises(ValueError, match="chance"):
        replication.run_replication({"a": a, "b": b}, tmp_path / "out")
