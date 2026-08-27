import csv
import json
import subprocess
import sys

import pytest

from polreps import sweep
from polreps.config import REPO


def write_prompt_table(path, planted, drop=()):
    rows = [
        {"prompt_id": pid, "condition": cond, "base_q_template_hash": group}
        for pid, cond, group in zip(
            planted["prompt_ids"], planted["labels"], planted["groups"]
        )
        if pid not in drop
    ]
    rows.reverse()  # cache order and table order differ; the join must be by id
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "condition", "base_q_template_hash"])
        writer.writeheader()
        writer.writerows(rows)


def test_run_sweep_writes_figure_numbers_and_metadata(planted, tmp_path):
    table = tmp_path / "prompt_table.csv"
    write_prompt_table(table, planted)
    out_stem = tmp_path / "artifacts" / "probe_curve"

    sweep.run_sweep(
        planted["cache_dir"], table, out_stem, binary=("democrat", "republican")
    )

    assert (tmp_path / "artifacts" / "probe_curve.png").stat().st_size > 0
    assert (tmp_path / "artifacts" / "probe_curve.png.meta.json").exists()
    assert (tmp_path / "artifacts" / "probe_curve.json.meta.json").exists()

    curve = json.loads((tmp_path / "artifacts" / "probe_curve.json").read_text())
    multi = curve["multinomial"]
    accs = multi["mean_accuracy"]
    assert len(accs) == 6
    assert accs.index(max(accs)) == planted["signal_layer"]
    # the shuffled-label reference must sit near chance where the real curve peaks
    assert abs(multi["shuffled_mean_accuracy"][planted["signal_layer"]] - multi["chance"]) < 0.15

    binary = curve["binary"]
    assert binary["conditions"] == ["democrat", "republican"]
    assert binary["chance"] == 0.5
    assert binary["mean_accuracy"][planted["signal_layer"]] > 0.9


def test_run_sweep_refuses_label_table_mismatch(planted, tmp_path):
    table = tmp_path / "prompt_table.csv"
    write_prompt_table(table, planted, drop={planted["prompt_ids"][0]})

    with pytest.raises(ValueError, match="prompt"):
        sweep.run_sweep(planted["cache_dir"], table, tmp_path / "out")


def test_sweep_is_a_single_command(planted, tmp_path):
    table = tmp_path / "prompt_table.csv"
    write_prompt_table(table, planted)
    out_stem = tmp_path / "probe_curve"

    subprocess.run(
        [
            sys.executable, "scripts/run_probe_sweep.py",
            "--cache", str(planted["cache_dir"]),
            "--labels", str(table),
            "--out", str(out_stem),
            "--binary", "democrat", "republican",
        ],
        cwd=REPO, check=True, capture_output=True,
    )

    assert (tmp_path / "probe_curve.png").exists()
    assert (tmp_path / "probe_curve.json").exists()
